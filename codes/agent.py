import os
import re
import sys
import time
import signal
import threading
import faulthandler
import numpy as np
import torch
import torch.optim as optim
from torch.multiprocessing import Pool
from tqdm import tqdm
from sklearn import metrics
from itertools import repeat

from worker import Worker, _reset_env, _step_env
from utils import torch_load_cpu, get_inner_model, env_wrapper

class Memory:
    def __init__(self):
        self.steps = {}
        self.eval_values = {}
        self.training_values = {}

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    dist = dist.sqrt()
    return dist

def _die_with_parent(parent_pid=None):
    """Pool-worker initializer: die with the master process (Linux only).

    If an external killer SIGKILLs run.py, its pool workers would otherwise
    be orphaned (reparented to init) and block forever on the dead task
    queue, each pinning ~1 GB of the container's memory budget.  With
    PR_SET_PDEATHSIG the kernel SIGKILLs every worker the moment the master
    dies — even if the parent is dead too and nobody can sweep the group.
    """
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG = 1, SIGKILL = 9
        # Close the fork->prctl race: if the master died before we armed the
        # signal, getppid() has already changed; kill ourselves instead.
        if parent_pid is not None and os.getppid() != parent_pid:
            os.kill(os.getpid(), 9)
    except Exception:
        pass  # non-Linux or restricted environment: parent-process sweep still covers


def worker_run(worker, param, opts, Batch_size, seed, epoch=None):

    # distribute current parameters
    worker.load_param_from_master(param)
    # Set seed for reproducibility (compatible with both gym and gymnasium)
    _reset_env(worker.env, seed=seed)

    # get returned gradients and info from all agents
    out = worker.train_one_epoch(Batch_size, opts.device,
                                 opts.do_sample_for_training, epoch=epoch)

    # V-BSA LTD round-trip: pool-side attribute mutations die with the
    # task's pickle copy, so train_one_epoch carries the updated state out
    # in its return tuple and the master writes it back onto its own Worker
    # object for the next epoch (see the unpack loops below).
    return out


def weiszfeld_geometric_median(points, eps=1e-6, max_iter=100):
    """Compute the geometric median of a set of points using the Weiszfeld algorithm.

    The geometric median minimizes sum_i ||points[i] - median||.
    Used in ensemble defense for continuous action spaces.

    Args:
        points: tensor of shape [K, d] — K points in d-dimensional space
        eps: convergence threshold
        max_iter: maximum iterations
    Returns:
        median: tensor of shape [d]
    """
    if points.size(0) == 1:
        return points[0].clone()

    # Initialize at coordinate-wise median
    median = points.median(dim=0).values

    for _ in range(max_iter):
        diffs = points - median.unsqueeze(0)
        norms = torch.norm(diffs, dim=1)

        # Avoid division by zero: skip points that coincide with the current estimate
        mask = norms > eps
        if mask.sum() == 0:
            break

        weights = 1.0 / norms
        weights[~mask] = 0.0

        new_median = (points * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()

        if torch.norm(new_median - median) < eps:
            return new_median

        median = new_median

    return median


def coordinate_wise_median(points):
    """Coordinate-wise median over worker gradients.

    Uses the standard statistical median: for an even number of workers the
    mean of the two middle values (torch.median returns the lower middle
    value instead, which is not the convention in the Byzantine-robust
    aggregation literature).

    Args:
        points: tensor of shape [num_workers, ...] (gradients stacked on dim 0).

    Returns:
        tensor of shape [...] — the per-coordinate median.
    """
    n = points.size(0)
    sorted_points, _ = torch.sort(points, dim=0)
    if n % 2 == 1:
        return sorted_points[n // 2]
    return (sorted_points[n // 2 - 1] + sorted_points[n // 2]) / 2.0


def coordinate_wise_trimmed_mean(points, trim=1):
    """Coordinate-wise trimmed mean: drop `trim` extreme values per coordinate.

    Args:
        points: tensor of shape [num_workers, ...] (gradients stacked on dim 0).
        trim: number of values removed from each side per coordinate.

    Returns:
        tensor of shape [...] — the trimmed mean.
    """
    n = points.size(0)
    if n - 2 * trim < 1:
        return points.mean(dim=0)
    sorted_points, _ = torch.sort(points, dim=0)
    return sorted_points[trim:n - trim].mean(dim=0)



class Agent:

    def __init__(self, opts):

        # figure out the options
        self.opts = opts

        # setup arrays for distrubuted RL
        self.world_size = opts.num_worker

        # figure out the master
        self.master = Worker(
                id = 0,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units,
                activation = opts.activation,
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
        ).to(opts.device)

        # figure out a copy of the master node for importance sampling purpose
        self.old_master = Worker(
                id = -1,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units,
                activation = opts.activation,
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
        ).to(opts.device)

        # ---- figure out all the actors ----
        self.ensemble = opts.ensemble if hasattr(opts, 'ensemble') else False
        self.num_groups = opts.num_groups if hasattr(opts, 'num_groups') else 5

        self.workers = []
        self.true_Byzantine = []

        if self.ensemble:
            # =================================================================
            # Ensemble mode: strict Byzantine placement per group.
            # Only `compromised_groups` receive Byzantine workers; all other
            # groups remain fully honest.  This gives precise control over
            # which groups are "infiltrated" (m << K/2).
            # =================================================================
            K = self.num_groups
            workers_per_group = self.world_size // K
            assert workers_per_group >= 3, (
                f"Group size must be >= 3 for FedPG-BR to be meaningful; "
                f"got {workers_per_group} ({self.world_size} / {K})")

            # ---- Control panel: which groups are infiltrated ----
            if opts.num_Byzantine > 0:
                # Place Byzantine in the first m groups, 2 per group by default
                byz_per_group = max(1, opts.num_Byzantine // max(1, opts.num_Byzantine // 2))
                if byz_per_group < 2:
                    byz_per_group = 2  # at least 2 per group for statistical stealth
                m = opts.num_Byzantine // byz_per_group
                compromised_groups = list(range(m))
            else:
                compromised_groups = []
                byz_per_group = 0
            # ---------------------------------------------------

            self.group_workers = [[] for _ in range(K)]
            worker_id_counter = 1
            for k in range(K):
                byz_placed = 0
                for i in range(workers_per_group):
                    is_byz = (k in compromised_groups
                              and byz_placed < byz_per_group)
                    if is_byz:
                        byz_placed += 1

                    self.true_Byzantine.append(is_byz)
                    w = Worker(
                        id=worker_id_counter,
                        is_Byzantine=is_byz,
                        env_name=opts.env_name,
                        gamma=opts.gamma,
                        hidden_units=opts.hidden_units,
                        activation=opts.activation,
                        output_activation=opts.output_activation,
                        attack_type=opts.attack_type,
                        max_epi_len=opts.max_epi_len,
                        opts=opts
                    ).to(opts.device)
                    w.group_id = k
                    self.workers.append(w)
                    self.group_workers[k].append(w)
                    worker_id_counter += 1

            group_sizes = [len(g) for g in self.group_workers]
            byz_counts = [sum(1 for w in g if w.is_Byzantine) for g in self.group_workers]
            print(f'Ensemble: {K} groups x {workers_per_group} workers '
                  f'(total {self.world_size})')
            print(f'  Byzan. per group: {byz_counts}  |  '
                  f'Compromised groups: {compromised_groups}')

            # ---- Create K independent master policies, old masters, optimizers ----
            self.group_masters = []
            self.group_old_masters = []
            self.group_optimizers = []
            for k in range(K):
                master_k = Worker(
                    id=-(k + 1),
                    is_Byzantine=False,
                    env_name=opts.env_name,
                    gamma=opts.gamma,
                    hidden_units=opts.hidden_units,
                    activation=opts.activation,
                    output_activation=opts.output_activation,
                    max_epi_len=opts.max_epi_len,
                    opts=opts
                ).to(opts.device)
                self.group_masters.append(master_k)

                old_master_k = Worker(
                    id=-(k + 1 + K),
                    is_Byzantine=False,
                    env_name=opts.env_name,
                    gamma=opts.gamma,
                    hidden_units=opts.hidden_units,
                    activation=opts.activation,
                    output_activation=opts.output_activation,
                    max_epi_len=opts.max_epi_len,
                    opts=opts
                ).to(opts.device)
                self.group_old_masters.append(old_master_k)

                if not opts.eval_only:
                    self.group_optimizers.append(
                        optim.Adam(master_k.logits_net.parameters(), lr=opts.lr_model)
                    )

        else:
            # =================================================================
            # Non-ensemble mode: first N workers are Byzantine (original logic).
            # =================================================================
            self.num_groups = 1
            for i in range(self.world_size):
                is_byz = (i < opts.num_Byzantine)
                self.true_Byzantine.append(is_byz)
                w = Worker(
                    id=i + 1,
                    is_Byzantine=is_byz,
                    env_name=opts.env_name,
                    gamma=opts.gamma,
                    hidden_units=opts.hidden_units,
                    activation=opts.activation,
                    output_activation=opts.output_activation,
                    attack_type=opts.attack_type,
                    max_epi_len=opts.max_epi_len,
                    opts=opts
                ).to(opts.device)
                w.group_id = -1
                self.workers.append(w)

        byz_total = sum(self.true_Byzantine)
        print(f'{self.world_size} workers initialized; '
              f'{byz_total} Byzantine, '
              f'{self.world_size - byz_total} honest.')

        if not opts.eval_only and not self.ensemble:
            # figure out the optimizer
            self.optimizer = optim.Adam(self.master.logits_net.parameters(), lr = opts.lr_model)

        # maxtasksperchild recycles each worker after N tasks, containing
        # per-worker memory growth over long full-mode runs (~260 epochs).
        # NOTE: pool-process persistence does NOT persist Worker object
        # state — every task pickles the master's Worker fresh, and pool-side
        # attribute mutations die with the task.  The V-BSA LTD state
        # therefore survives via an explicit master round-trip
        # (train_one_epoch returns it, worker_run forwards it, the train
        # loops write it back).  The initializer makes workers die with the
        # master (see _die_with_parent).
        self.pool = Pool(self.world_size, maxtasksperchild=400,
                         initializer=_die_with_parent,
                         initargs=(os.getpid(),))
        self.memory = Memory()

    def load(self, load_path):
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.master.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('master', {})})

        if not self.opts.eval_only:
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

        print(' [*] Loading data from {}'.format(load_path))


    def save(self, epoch, run_id):
        print('Saving model and state...')
        torch.save(
            {
                'master': get_inner_model(self.master.logits_net).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'r{}-epoch-{}.pt'.format(run_id,epoch))
        )


    def eval(self):
        # turn model to eval mode
        self.master.eval()

    def train(self):
        # turn model to trainig mode
        self.master.train()

    def _aggregate_gradients(self, gradient, good_set, rule):
        """Aggregate per-worker gradients into a single parameter-gradient list.

        Shared by all aggregation rules.  For the
        non-filtering rules (median / trimmed_mean / geometric_median /
        fedavg) every worker contributes; for fedpg_br only the workers in
        `good_set` contribute (mean over the filtered set).

        Args:
            gradient: gradient[i][idx] = parameter tensor of worker i for
                parameter index idx.
            good_set: boolean tensor of shape [num_workers].
            rule: 'fedpg_br' | 'fedavg' | 'median' | 'trimmed_mean' |
                'geometric_median'.

        Returns:
            mu: list of aggregated tensors, one per parameter.
        """
        num_workers = len(gradient)
        keep = [i for i in range(num_workers) if good_set[i]]
        mu = []
        for idx in range(len(gradient[0])):
            grads = torch.stack([gradient[i][idx] for i in keep])  # [n_keep, *shape]
            if rule in ('fedpg_br', 'fedavg'):
                mu.append(grads.mean(0))
            elif rule == 'median':
                mu.append(coordinate_wise_median(grads))
            elif rule == 'trimmed_mean':
                mu.append(coordinate_wise_trimmed_mean(grads, trim=self.opts.trim))
            elif rule == 'geometric_median':
                flat = grads.reshape(grads.size(0), -1)
                mu.append(weiszfeld_geometric_median(flat).view_as(grads[0]))
            else:
                raise ValueError(f'Unknown aggregation rule: {rule}')
        return mu

    # =========================================================================
    # Ensemble Defense
    # =========================================================================

    def ensemble_predict(self, obs, device, sample=False):
        """Predict action via ensemble of K independently trained group policies.

        For discrete action spaces, uses majority vote.

        Args:
            obs: Observation vector (numpy array).
            device: Torch device for inference.
            sample: If True, sample from policy; if False, take greedy action.

        Returns:
            Predicted action (int).
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(device)

        # ---- Discrete: majority vote ----
        actions = []
        for k in range(self.num_groups):
            act, _ = self.group_masters[k].logits_net(obs_tensor, sample=sample)
            actions.append(act)

        # Count frequencies and pick argmax
        from collections import Counter
        counts = Counter(actions)
        return counts.most_common(1)[0][0]

    def _ensemble_rollout(self, device, max_steps=1000, render=False,
                          save_dir='./', filename='.'):
        """Run one episode using ensemble prediction for action selection."""
        from utils import env_wrapper, save_frames_as_gif

        env = self.master.env
        obs = _reset_env(env)
        done = False
        ep_rew = []
        frames = []
        step = 0

        while not done and step < max_steps:
            step += 1
            if render:
                frames.append(env.render(mode="rgb_array"))

            obs_wrapped = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.ensemble_predict(obs_wrapped, device, sample=False)
            obs, rew, done, _ = _step_env(env, action)
            ep_rew.append(rew)

        if render:
            save_frames_as_gif(frames, save_dir, filename)

        return np.sum(ep_rew), len(ep_rew), ep_rew

    def start_validating_ensemble(self, tb_logger=None, id=0, max_steps=1000,
                                   render=False, run_id=0, mode='human'):
        """Validation using ensemble prediction.

        Same as start_validating() but uses ensemble_predict() to aggregate
        actions from all K group policies.
        """
        print('\nValidating (ensemble)...', flush=True)

        val_ret = 0.0
        val_len = 0.0

        for _ in range(self.opts.val_size):
            epi_ret, epi_len, _ = self._ensemble_rollout(
                self.opts.device,
                max_steps=max_steps,
                render=render,
                save_dir='./outputs/',
                filename=f'gym_{run_id}_{_}.gif'
            )
            val_ret += epi_ret
            val_len += epi_len

        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size

        print('\nGradient step: %3d \t return: %.3f \t ep_len: %.3f' %
              (id, np.mean(val_ret), np.mean(val_len)))

        if tb_logger is not None:
            tb_logger.add_scalar(f'validate/total_rewards_{run_id}',
                                 np.mean(val_ret), id)
            tb_logger.add_scalar(f'validate/epi_length_{run_id}',
                                 np.mean(val_len), id)
            tb_logger.close()

        return np.mean(val_ret)

    # =========================================================================
    # Watchdog around pool.starmap
    # =========================================================================
    def _pool_starmap(self, fn, args, desc, timeout=None):
        """Run pool.starmap with a hang watchdog.

        starmap blocks FOREVER if a pool worker process dies hard
        (OOM-killed / segfaulted): the result of that task never arrives
        and the call never returns.  Run the call in a helper thread; if it
        does not finish within `timeout` seconds, dump every thread's stack
        into the run log (post-mortem evidence of where it hung) and kill
        this whole process group so the orchestrating process can mark the
        run failed instead of freezing as well.
        """
        if timeout is None:
            timeout = getattr(self.opts, 'epoch_timeout', 1800)
        box = {}

        def _target():
            try:
                box['result'] = self.pool.starmap(fn, args)
            except BaseException:
                box['error'] = sys.exc_info()

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            print(f'[HANG] {desc}: pool.starmap did not return within '
                  f'{timeout}s — a pool worker most likely died '
                  f'(OOM-kill/segfault) and its result will never arrive. '
                  f'Dumping thread stacks and aborting this run...',
                  flush=True)
            faulthandler.dump_traceback(all_threads=True)
            sys.stderr.flush()
            # Kill this process together with its pool workers.  Only kill
            # the group when we are its leader (each run is launched in its
            # own process session); otherwise fall back to terminating just
            # the pool.
            if hasattr(os, 'killpg') and os.getpgrp() == os.getpid():
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(10)
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    self.pool.terminate()
                except Exception:
                    pass
                time.sleep(10)
            os._exit(3)  # distinctive exit code: epoch watchdog fired

        if 'error' in box:
            etype, e, tb = box['error']
            raise e.with_traceback(tb)
        return box['result']

    # =========================================================================
    # Ensemble Training
    # Each group trains its own global policy independently using the same
    # foundational aggregation rule.  Groups are trained sequentially.
    # =========================================================================

    def _start_training_ensemble(self, tb_logger=None, run_id=None):
        """Train K independent global policies, one per worker group.

        Implements Algorithm 1 of the FedPG-BR paper (prior work). Workers are
        partitioned into K non-overlapping groups. Each group independently
        trains a global policy using the same foundational aggregation rule
        (FedPG-BR or GOMDP).

        At validation time, actions from all K policies are aggregated via
        majority vote.

        Args:
            tb_logger: Optional TensorBoard SummaryWriter.
            run_id: Integer run identifier for logging.
        """
        opts = self.opts
        K = self.num_groups

        # Shared step counter across all groups
        step = 0
        epoch = 0
        ratios_step = 0

        # Checkpoint resume: continue from the latest complete epoch instead
        # of training from scratch (used to continue runs that were
        # externally interrupted).  The resumed run follows the same
        # epoch-by-epoch schedule; only the np/torch RNG streams are restored
        # approximately, so it is not bit-identical to an uninterrupted run.
        if getattr(opts, 'resume_from', None) is not None:
            epoch, step = self._resume_ensemble(opts.resume_from)
            print(f'[resume] continuing from epoch {epoch}, step {step}',
                  flush=True)

        _max_epochs = getattr(opts, 'max_epochs', None)
        while step <= opts.max_trajectories and (_max_epochs is None
                                                 or epoch < _max_epochs):
            epoch += 1

            print('\n\n')
            print("|", format(f" Training step {step} run_id {run_id} in {opts.seeds}",
                              "*^60"), "|")

            # ---- Accumulators for logging (across all groups) ----
            all_batch_loss = []
            all_batch_rets = []
            all_batch_lens = []
            all_N_good = []
            all_grad_array = []
            epoch_traj_total = 0

            # ============================================================
            # Train each group independently (serial, as requested)
            # ============================================================
            for k in range(K):
                workers_k = self.group_workers[k]
                world_k = len(workers_k)
                master_k = self.group_masters[k]
                old_master_k = self.group_old_masters[k]
                optimizer_k = self.group_optimizers[k]

                # Turn group-k master into training mode
                master_k.train()
                print("Training with lr={:.3e} for group {}/{}".format(
                    optimizer_k.param_groups[0]['lr'], k + 1, K), flush=True)

                # ---- Distribute group-k master params to group-k workers ----
                param_k = get_inner_model(master_k.logits_net).state_dict()

                if opts.FedPG_BR:
                    Batch_size = np.random.randint(opts.Bmin, opts.Bmax + 1)
                else:
                    Batch_size = opts.B

                seeds = np.random.randint(1, 100000, world_k).tolist()

                args = zip(workers_k,
                           repeat(param_k),
                           repeat(opts),
                           repeat(Batch_size),
                           seeds,
                           repeat(epoch))

                results = self._pool_starmap(
                    worker_run, args,
                    desc=f'Group {k+1}/{K} (epoch {epoch})',
                    timeout=opts.epoch_timeout)

                # ---- Collect gradients from group-k workers ----
                gradient = []
                for w, out in zip(workers_k,
                                  tqdm(results, desc=f'Group {k+1}/{K} workers')):
                    grad, loss, rets, lens, vbsa_out = out
                    gradient.append(grad)
                    all_batch_loss.append(loss)
                    all_batch_rets.append(rets)
                    all_batch_lens.append(lens)
                    if vbsa_out is not None:
                        w._vbsa_mu, w._vbsa_disp = vbsa_out[0], vbsa_out[1]

                # ---- Byzantine indices within this group ----
                byzantine_indices_k = [i for i, w in enumerate(workers_k) if w.is_Byzantine]

                # ---- Set old master ----
                old_master_k.load_param_from_master(param_k)

                # ---- Flatten per-worker gradients: [world_k, dim] ----
                mu_vec = None
                for idx, item in enumerate(old_master_k.parameters()):
                    grad_item = []
                    for i in range(world_k):
                        grad_item.append(gradient[i][idx])
                    grad_item = torch.stack(grad_item).view(world_k, -1)
                    if mu_vec is None:
                        mu_vec = grad_item.clone()
                    else:
                        mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)

                # ---- Aggregation rule ----
                if opts.agg_rule == 'fedpg_br':
                    dist = euclidean_dist(mu_vec, mu_vec)
                    V = 2 * np.log(2 * world_k / opts.delta)
                    sigma = opts.sigma
                    threshold = 2 * sigma * np.sqrt(V / Batch_size)
                    alpha = opts.alpha

                    k_prime = (dist <= threshold).sum(-1) > (0.5 * world_k)
                    if torch.sum(k_prime) > 0:
                        mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
                        mu_med_vec = mu_vec[k_prime][
                            euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()
                        ].view(1, -1)
                        Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 1 * threshold
                    else:
                        Good_set = k_prime

                    if (torch.sum(Good_set) < (1 - alpha) * world_k
                            or torch.sum(Good_set) == 0):
                        k_prime = ((dist <= 2 * sigma).sum(-1)
                                   > (0.5 * world_k))
                        if torch.sum(k_prime) > 0:
                            mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
                            mu_med_vec = mu_vec[k_prime][
                                euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()
                            ].view(1, -1)
                            Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 2 * sigma
                        else:
                            Good_set = torch.zeros(world_k, 1).to(opts.device).bool()
                else:
                    # Non-filtering aggregators (fedavg / median / trimmed_mean /
                    # geometric_median): every worker is "good" by construction.
                    Good_set = torch.ones(world_k, 1).to(opts.device).bool()

                N_good = torch.sum(Good_set)
                all_N_good.append(N_good.item())

                # ---- DIAGNOSTIC: log filtered worker indices in infiltrated groups ----
                if opts.agg_rule == 'fedpg_br' and byzantine_indices_k:
                    filtered = (~Good_set.view(-1)).nonzero(as_tuple=True)[0].tolist()
                    byz_set = set(byzantine_indices_k)
                    filtered_byz = [i for i in filtered if i in byz_set]
                    filtered_honest = [i for i in filtered if i not in byz_set]
                    if filtered:
                        print(f"[BSA-FILTER] Group {k}: N_good={N_good.item()}/"
                              f"{world_k}, filtered indices={filtered}, "
                              f"Byz filtered={filtered_byz}, "
                              f"Honest filtered={filtered_honest}")

                # ---- Compute mu (aggregated gradient) ----
                if N_good > 0:
                    mu = self._aggregate_gradients(
                        gradient, Good_set.view(-1), opts.agg_rule)
                else:
                    mu = None

                # ---- DIAGNOSTIC: deviation of aggregated update from honest mean ----
                if byzantine_indices_k and mu is not None:
                    honest_idx = [i for i in range(world_k)
                                  if i not in set(byzantine_indices_k)]
                    mu_flat = torch.cat([p.reshape(-1) for p in mu])
                    honest_mean_flat = mu_vec[honest_idx].mean(dim=0)
                    dev = (torch.norm(mu_flat - honest_mean_flat)
                           / (torch.norm(honest_mean_flat) + 1e-12))
                    print(f"[BSA-DEV] Group {k}: rule={opts.agg_rule}, "
                          f"||mu||={torch.norm(mu_flat).item():.4e}, "
                          f"dev_from_honest_mean={dev.item():.4f}")

                # ---- Gradient update for group-k master ----
                grad_array = []
                if opts.FedPG_BR or opts.SVRPG:
                    if opts.FedPG_BR:
                        b = opts.b
                        N_t = np.random.geometric(
                            p=1 - Batch_size / (Batch_size + b))
                    elif opts.SVRPG:
                        b = opts.b
                        N_t = opts.N

                    for n in tqdm(range(N_t), desc=f'Group {k+1} master'):
                        optimizer_k.zero_grad()
                        (weights, new_logp, _, _, batch_states,
                         batch_actions) = master_k.collect_experience_for_training(
                            b, opts.device, record=True,
                            sample=opts.do_sample_for_training)

                        loss_new = -(new_logp * weights).mean()
                        master_k.logits_net.zero_grad()
                        loss_new.backward()

                        if mu:
                            # Batched old-logp: one [N, OBS_DIM] forward per
                            # batch instead of one autograd forward per state
                            # (a batch is b=8 episodes, i.e. thousands of
                            # states).  Per-row math is identical to the
                            # per-state loop it replaces:
                            #   MlpPolicy.forward(obs, fixed_action=a)[1]
                            #     == log_softmax(logits)[a]
                            # (env_wrapper is the identity).
                            obs_stack = torch.stack([
                                torch.as_tensor(s, dtype=torch.float32)
                                for s in batch_states]).to(opts.device)
                            act_stack = torch.as_tensor(
                                [int(a) for a in batch_actions],
                                device=opts.device)
                            logits_old = \
                                old_master_k.logits_net.logits_net(
                                    obs_stack)
                            old_logp = torch.log_softmax(
                                logits_old, dim=-1
                            ).gather(1, act_stack.unsqueeze(1)).squeeze(1)

                            ratios = torch.exp(old_logp.detach() - new_logp.detach())
                            ratios_step += 1

                            loss_old = -(old_logp * weights * ratios).mean()
                            old_master_k.logits_net.zero_grad()
                            loss_old.backward()
                            grad_old = [item.grad for item in old_master_k.parameters()]

                            if (torch.abs(ratios.mean()) < 0.995
                                    or torch.abs(ratios.mean()) > 1.005):
                                N_t = n
                                break

                            if tb_logger is not None:
                                tb_logger.add_scalar(
                                    f'params/ratios_{run_id}', ratios.mean(), ratios_step)

                            for idx_m, item in enumerate(master_k.parameters()):
                                item.grad = item.grad - grad_old[idx_m] + mu[idx_m]
                                grad_array += (item.grad.data.view(-1).cpu().tolist())

                        optimizer_k.step()

                else:
                    # GOMDP
                    b = 0
                    N_t = 0
                    for idx_m, item in enumerate(master_k.parameters()):
                        item.grad = mu[idx_m]
                        grad_array += (item.grad.data.view(-1).cpu().tolist())
                    optimizer_k.step()

                all_grad_array.extend(grad_array)

                # ---- Accumulate total trajectories for this group ----
                epoch_traj_total += Batch_size * world_k + b * N_t

            # ============================================================
            # End of per-group training — aggregate logging & validate
            # ============================================================

            # Step increment: per-worker trajectory count
            # (total worker trajectories + total master trajectories) / num_workers
            step += round(epoch_traj_total / self.world_size) if self.world_size > 1 else epoch_traj_total

            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t N_good: %s' %
                  (epoch, np.mean(all_batch_loss), np.mean(all_batch_rets),
                   np.mean(all_batch_lens), str(all_N_good)))

            # Tensorboard logging
            if tb_logger is not None:
                tb_logger.add_scalar(f'train/total_rewards_{run_id}',
                                     np.mean(all_batch_rets), step)
                tb_logger.add_scalar(f'train/epi_length_{run_id}',
                                     np.mean(all_batch_lens), step)
                tb_logger.add_scalar(f'train/loss_{run_id}',
                                     np.mean(all_batch_loss), step)
                if all_grad_array:
                    tb_logger.add_scalar(f'grad/grad_{run_id}',
                                         np.mean(all_grad_array), step)

                if run_id not in self.memory.steps.keys():
                    self.memory.steps[run_id] = []
                    self.memory.eval_values[run_id] = []
                    self.memory.training_values[run_id] = []

                self.memory.steps[run_id].append(step)
                self.memory.training_values[run_id].append(np.mean(all_batch_rets))

            # ---- Ensemble validation ----
            eval_reward = self.start_validating_ensemble(
                tb_logger, step, max_steps=opts.val_max_steps,
                render=opts.render, run_id=run_id)
            if tb_logger is not None:
                self.memory.eval_values[run_id].append(eval_reward)

            # ---- Save all group models ----
            if not opts.no_saving:
                self._save_ensemble(epoch, run_id, step)

    def _save_ensemble(self, epoch, run_id, step=None):
        """Save all K group master policies and optimizers.

        Group files are written first and the meta file (epoch / step / RNG
        state) last: `_resume_ensemble` only trusts an epoch whose meta file
        exists, so a run killed mid-save simply falls back to the previous
        epoch.
        """
        print('Saving ensemble models and states...')
        for k in range(self.num_groups):
            torch.save(
                {
                    'master': get_inner_model(
                        self.group_masters[k].logits_net).state_dict(),
                    'optimizer': self.group_optimizers[k].state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': (torch.cuda.get_rng_state_all()
                                       if self.opts.use_cuda else None),
                },
                os.path.join(self.opts.save_dir,
                             f'r{run_id}-epoch-{epoch}-group{k}.pt')
            )
        torch.save(
            {
                'epoch': epoch,
                'step': step,
                'np_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng_state': (torch.cuda.get_rng_state_all()
                                   if self.opts.use_cuda else None),
            },
            os.path.join(self.opts.save_dir,
                         f'r{run_id}-epoch-{epoch}-meta.pt')
        )

    def _resume_ensemble(self, save_dir):
        """Restore the latest complete ensemble checkpoint from `save_dir`.

        Scans for r{run_id}-epoch-{epoch}-meta.pt files, picks the highest
        epoch whose meta AND all K group files exist and load cleanly, then
        restores group masters, optimizers and RNG state.  Returns
        (epoch, step) so the training loop continues where it left off, or
        (0, 0) if no complete checkpoint exists (a run killed during its
        first epoch has not saved anything yet).
        """
        metas = []
        for fn in os.listdir(save_dir):
            m = re.match(r'^r(\d+)-epoch-(\d+)-meta\.pt$', fn)
            if m:
                metas.append((int(m.group(2)), int(m.group(1))))  # (epoch, run_id)
        metas.sort(reverse=True)  # newest epoch first

        for epoch, run_id in metas:
            group_paths = [
                os.path.join(save_dir,
                             f'r{run_id}-epoch-{epoch}-group{k}.pt')
                for k in range(self.num_groups)
            ]
            if not all(os.path.exists(p) for p in group_paths):
                print(f'[resume] epoch {epoch} incomplete (missing group '
                      f'files); trying an older epoch')
                continue
            failed_path = os.path.join(save_dir,
                                       f'r{run_id}-epoch-{epoch}-meta.pt')
            try:
                meta = torch_load_cpu(failed_path)
                for k, path in enumerate(group_paths):
                    failed_path = path
                    data = torch_load_cpu(path)
                    model_actor = get_inner_model(
                        self.group_masters[k].logits_net)
                    model_actor.load_state_dict(
                        {**model_actor.state_dict(),
                         **data.get('master', {})})
                    self.group_optimizers[k].load_state_dict(
                        data['optimizer'])
                    for state in self.group_optimizers[k].state.values():
                        for key, v in state.items():
                            if torch.is_tensor(v):
                                state[key] = v.to(self.opts.device)
                if 'np_rng' in meta:
                    np.random.set_state(meta['np_rng'])
                if 'torch_rng' in meta:
                    torch.set_rng_state(meta['torch_rng'])
                if self.opts.use_cuda and meta.get('cuda_rng_state') is not None:
                    torch.cuda.set_rng_state_all(meta['cuda_rng_state'])
            except Exception as e:
                print(f'[resume] epoch {epoch} restore failed '
                      f'({failed_path}: {e}); trying an older epoch')
                continue

            print(f'[resume] loaded epoch {epoch} from {save_dir}')
            return epoch, meta.get('step', 0) or 0

        print(f'[resume] no complete checkpoints found in {save_dir}; '
              f'starting from scratch')
        return 0, 0

    def _resume_single(self, save_dir):
        """Restore the latest complete single-master checkpoint from `save_dir`
        (non-ensemble loop).  Mirrors _resume_ensemble: picks the highest
        epoch whose meta AND r{run_id}-epoch-{epoch}.pt both exist and load
        cleanly.  Returns (epoch, step), or (0, 0) if no complete checkpoint
        exists.
        """
        metas = []
        for fn in os.listdir(save_dir):
            m = re.match(r'^r(\d+)-epoch-(\d+)-meta\.pt$', fn)
            if m:
                metas.append((int(m.group(2)), int(m.group(1))))  # (epoch, run_id)
        metas.sort(reverse=True)

        for epoch, run_id in metas:
            model_path = os.path.join(save_dir, f'r{run_id}-epoch-{epoch}.pt')
            meta_path = os.path.join(save_dir,
                                     f'r{run_id}-epoch-{epoch}-meta.pt')
            if not os.path.exists(model_path):
                print(f'[resume] epoch {epoch} incomplete (missing model '
                      f'file); trying an older epoch')
                continue
            try:
                meta = torch_load_cpu(meta_path)
                data = torch_load_cpu(model_path)
                model_actor = get_inner_model(self.master.logits_net)
                model_actor.load_state_dict(
                    {**model_actor.state_dict(), **data.get('master', {})})
                self.optimizer.load_state_dict(data['optimizer'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.opts.device)
                if 'np_rng' in meta:
                    np.random.set_state(meta['np_rng'])
                if 'torch_rng' in meta:
                    torch.set_rng_state(meta['torch_rng'])
                if self.opts.use_cuda and meta.get('cuda_rng_state') is not None:
                    torch.cuda.set_rng_state_all(meta['cuda_rng_state'])
            except Exception as e:
                print(f'[resume] epoch {epoch} restore failed '
                      f'({model_path}: {e}); trying an older epoch')
                continue

            print(f'[resume] loaded epoch {epoch} from {save_dir}')
            return epoch, meta.get('step', 0) or 0

        print(f'[resume] no complete checkpoints found in {save_dir}; '
              f'starting from scratch')
        return 0, 0

    def start_training(self, tb_logger = None, run_id = None):
        # ---- Dispatch to ensemble training if enabled ----
        if self.ensemble:
            self._start_training_ensemble(tb_logger, run_id)
            return

        # parameters of running
        opts = self.opts

        # for storing number of trajectories sampled
        step = 0
        epoch = 0
        ratios_step = 0

        # Checkpoint resume for the single-master loop: mirrors the
        # ensemble loop's resume_from handling.
        if getattr(opts, 'resume_from', None) is not None:
            epoch, step = self._resume_single(opts.resume_from)
            print(f'[resume] continuing from epoch {epoch}, step {step}',
                  flush=True)

        _max_epochs = getattr(opts, 'max_epochs', None)

        # Start the training loop
        while step <= opts.max_trajectories and (_max_epochs is None
                                                 or epoch < _max_epochs):
            # epoch for storing checkpoints of model
            epoch += 1

            # Deterministic per-epoch seeding keyed by (epoch, dynamics_index):
            # --dynamics_index yields real independent realizations in the
            # non-ensemble loop.
            _d = getattr(opts, 'dynamics_index', 0)
            _epoch_seed = epoch * 100000 + _d * 1000
            np.random.seed(_epoch_seed)
            torch.manual_seed(_epoch_seed + 1)

            # Turn model into training mode
            print('\n\n')
            print("|",format(f" Training step {step} run_id {run_id} in {opts.seeds}","*^60"),"|")
            self.train()

            # setup lr_scheduler
            print("Training with lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)

            # some emplty list for training and logging purpose
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []

            # distribute current params and Batch_Size to all workers
            # CLONE: state_dict() shares storage with the live parameters,
            # so in-place optimizer updates would retroactively rewrite this
            # broadcast snapshot.
            param = {k: v.clone() for k, v in
                     get_inner_model(
                         self.master.logits_net).state_dict().items()}

            if opts.FedPG_BR:
                Batch_size = np.random.randint(opts.Bmin, opts.Bmax + 1)
            else:
                Batch_size = opts.B

            seeds = np.random.randint(1,100000, self.world_size).tolist()
            args = zip(self.workers,
                       repeat(param),
                       repeat(opts),
                       repeat(Batch_size),
                       seeds,
                       repeat(epoch))

            results = self._pool_starmap(
                worker_run, args, desc=f'epoch {epoch}',
                timeout=opts.epoch_timeout)

            #  collect the gradient(for training), loss(for logging only), returns(for logging only), and epi_length(for logging only) from workers
            for w, out in zip(self.workers, tqdm(results, desc='Worker node')):
                grad, loss, rets, lens, vbsa_out = out

                # store all values
                gradient.append(grad)

                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)
                if vbsa_out is not None:
                    w._vbsa_mu, w._vbsa_disp = vbsa_out[0], vbsa_out[1]

            # make the old policy as a copy of the current master node
            self.old_master.load_param_from_master(param)

            # do Aggregate Algorithm to detect Byzantine worker on master node
            # (non-ensemble loop: filtering only for fedpg_br;
            # median / trimmed_mean / fedavg take every worker's gradient)
            if opts.FedPG_BR and opts.agg_rule == 'fedpg_br':

                # flatten the gradient vectors of each worker and put them together, shape [num_worker, -1]
                mu_vec = None
                for idx,item in enumerate(self.old_master.parameters()):
                    # stack gradient[idx] from all worker nodes
                    grad_item = []
                    for i in range(self.world_size):
                         grad_item.append(gradient[i][idx])
                    grad_item = torch.stack(grad_item).view(self.world_size,-1)

                    # concat stacked grad vector
                    if mu_vec is None:
                        mu_vec = grad_item.clone()
                    else:
                        mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)

                # calculate the norm distance between each worker's gradient vector, shape [num_worker, num_worker]
                dist = euclidean_dist(mu_vec, mu_vec)

                # calculate C, Variance Bound V, thresold, and alpha
                V = 2 * np.log(2 * opts.num_worker / opts.delta)
                sigma = opts.sigma

                threshold = 2 * sigma * np.sqrt(V / Batch_size)
                alpha = opts.alpha

                # to find MOM: |dist <= threshold| > 0.5 * num_worker
                mu_med_vec = None
                k_prime = (dist <= threshold).sum(-1) > (0.5 * self.world_size)

                # computes the mom of the gradients, mu_med_vec, and
                # filter the gradients it believes to be Byzantine and store the index of non-Byzantine graidents in Good_set
                if torch.sum(k_prime) > 0:
                    mu_mean_vec = torch.mean(mu_vec[k_prime],0).view(1,-1)
                    mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1,-1)
                    # applying R1 to filter
                    Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 1 * threshold
                else:
                    Good_set = k_prime # skip this step if k_prime is emplty (i.e., all False)

                # avoid the scenarios that Good_set is empty or can have |Gt| < (1 − α)K.
                if torch.sum(Good_set) < (1 - alpha) * self.world_size or torch.sum(Good_set) == 0:

                    # re-calculate mom of the gradients
                    k_prime = (dist <= 2 * sigma).sum(-1) > (0.5 * self.world_size)
                    if torch.sum(k_prime) > 0:
                        mu_mean_vec = torch.mean(mu_vec[k_prime],0).view(1,-1)
                        mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1,-1)
                        # re-filter with R2
                        Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 2 * sigma
                    else:
                        Good_set = torch.zeros(self.world_size,1).to(opts.device).bool()

            # else will treat all nodes as non-Byzantine nodes
            else:
                Good_set = torch.ones(self.world_size,1).to(opts.device).bool()

            # calculate number of good gradients for logging
            N_good = torch.sum(Good_set)

            # aggregate the detected worker gradients to get mu
            # (non-ensemble loop: median / trimmed_mean /
            # geometric_median dispatch, every Good_set worker contributes)
            if N_good > 0:
                mu = []
                rule = opts.agg_rule
                for idx,item in enumerate(self.old_master.parameters()):
                    grad_item = []
                    for i in range(self.world_size):
                        if Good_set[i]: # only aggregate non-Byzantine gradients
                            grad_item.append(gradient[i][idx])
                    grads = torch.stack(grad_item)
                    if rule in ('fedpg_br', 'fedavg'):
                        mu.append(grads.mean(0))
                    elif rule == 'median':
                        mu.append(coordinate_wise_median(grads))
                    elif rule == 'trimmed_mean':
                        mu.append(coordinate_wise_trimmed_mean(grads,
                                                               trim=opts.trim))
                    elif rule == 'geometric_median':
                        mu.append(weiszfeld_geometric_median(
                            grads.reshape(grads.size(0), -1)).view_as(grads[0]))
                    else:
                        raise ValueError(f'Unknown aggregation rule: {rule}')
            else: # if still all nodes are detected to be Byzantine, check the sigma. If siagma is set properly, this situation will not happen.
                mu = None

            # perform gradient update in master node
            grad_array = [] # store gradients for logging

            if opts.FedPG_BR or opts.SVRPG:

                if opts.FedPG_BR:
                    # for n=1 to Nt ~ Geom(B/B+b) do grad update
                    b = opts.b
                    N_t = np.random.geometric(p= 1 - Batch_size/(Batch_size + b))

                elif opts.SVRPG:
                    b = opts.b
                    N_t = opts.N

                for n in tqdm(range(N_t), desc = 'Master node'):

                    # calculate new gradient in master node
                    self.optimizer.zero_grad()

                    # sample b trajectory using the latest policy (\theta_n) of master node
                    weights, new_logp, batch_rets, batch_lens, batch_states, batch_actions = self.master.collect_experience_for_training(b,
                                                                                                                                 opts.device,
                                                                                                                                 record = True,
                                                                                                                                 sample = opts.do_sample_for_training)


                    # calculate gradient for the new policy (\theta_n)
                    loss_new = -(new_logp * weights).mean()
                    self.master.logits_net.zero_grad()
                    loss_new.backward()

                    if mu:
                        # get the old log_p with the old policy (\theta_0)
                        # but fixing the actions to be the same as the
                        # sampled trajectory — batched, per-row math
                        # identical to the per-state loop it replaces
                        # (log_softmax(logits)[a]); see the group-master
                        # inner loop above.
                        obs_stack = torch.stack([
                            torch.as_tensor(s, dtype=torch.float32)
                            for s in batch_states]).to(opts.device)
                        act_stack = torch.as_tensor(
                            [int(a) for a in batch_actions],
                            device=opts.device)
                        logits_old = self.old_master.logits_net.logits_net(
                            obs_stack)
                        old_logp = torch.log_softmax(
                            logits_old, dim=-1
                        ).gather(1, act_stack.unsqueeze(1)).squeeze(1)

                        # Finding the ratio (pi_theta / pi_theta__old):
                        # print(old_logp, new_logp)
                        ratios = torch.exp(old_logp.detach() - new_logp.detach())
                        ratios_step += 1

                        # calculate gradient for the old policy (\theta_0)
                        loss_old = -(old_logp * weights * ratios).mean()
                        self.old_master.logits_net.zero_grad()
                        loss_old.backward()
                        grad_old = [item.grad for item in self.old_master.parameters()]

                        # early stop if ratio is not within [0.995, 1.005]
                        if torch.abs(ratios.mean()) < 0.995 or torch.abs(ratios.mean()) > 1.005:
                            N_t = n
                            break

                        if tb_logger is not None:
                            tb_logger.add_scalar(f'params/ratios_{run_id}', ratios.mean(), ratios_step)

                        # adjust and set the gradient for latest policy (\theta_n)
                        for idx,item in enumerate(self.master.parameters()):
                            item.grad = item.grad - grad_old[idx] + mu[idx]  # if mu is None, use grad from master
                            grad_array += (item.grad.data.view(-1).cpu().tolist())

                    # take a gradient step
                    self.optimizer.step()

            else: # GOMDP in this case

                b = 0
                N_t = 0

                # perform gradient descent with mu vector
                for idx,item in enumerate(self.master.parameters()):
                    item.grad = mu[idx]
                    grad_array += (item.grad.data.view(-1).cpu().tolist())

                # take a gradient step
                self.optimizer.step()

            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t N_good: %d'%
                (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens), N_good))

            # current step: number of trajectories sampled
            # step += max(Batch_size, b * N_t) if self.world_size > 1 else Batch_size + b * N_t
            step += round((Batch_size * self.world_size  + b * N_t) / (1 + self.world_size)) if self.world_size > 1 else Batch_size + b * N_t

            # Logging to tensorboard
            if(tb_logger is not None):

                # training log
                tb_logger.add_scalar(f'train/total_rewards_{run_id}', np.mean(batch_rets), step)
                tb_logger.add_scalar(f'train/epi_length_{run_id}', np.mean(batch_lens), step)
                tb_logger.add_scalar(f'train/loss_{run_id}', np.mean(batch_loss), step)
                # grad log
                tb_logger.add_scalar(f'grad/grad_{run_id}', np.mean(grad_array), step)
                # optimizer log
                tb_logger.add_scalar(f'params/lr_{run_id}', self.optimizer.param_groups[0]['lr'], step)
                tb_logger.add_scalar(f'params/N_t_{run_id}', N_t, step)

                # Byzantine filtering log
                if opts.FedPG_BR:

                    y_true = self.true_Byzantine
                    y_pred = (~ Good_set).view(-1).cpu().tolist()

                    dist_Byzantine = dist[:opts.num_Byzantine][:,:opts.num_Byzantine]
                    dist_good = dist[opts.num_Byzantine:][:,opts.num_Byzantine:]
                    dist_good_Byzantine = dist[:opts.num_Byzantine][:,opts.num_Byzantine:]

                    if opts.num_Byzantine > 0:
                        tb_logger.add_scalar(f'grad_norm_mean/Byzantine_{run_id}', torch.mean(dist_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_max/Byzantine_{run_id}', torch.max(dist_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_mean/Good_{run_id}', torch.mean(dist_good), step)
                        tb_logger.add_scalar(f'grad_norm_max/Good_{run_id}', torch.max(dist_good), step)
                        tb_logger.add_scalar(f'grad_norm_mean/Between_{run_id}', torch.mean(dist_good_Byzantine), step)
                        tb_logger.add_scalar(f'grad_norm_max/Between_{run_id}', torch.max(dist_good_Byzantine), step)

                        tb_logger.add_scalar(f'Byzantine/precision_{run_id}', metrics.precision_score(y_true, y_pred), step)
                        tb_logger.add_scalar(f'Byzantine/recall_{run_id}', metrics.recall_score(y_true, y_pred), step)
                        tb_logger.add_scalar(f'Byzantine/f1_score_{run_id}', metrics.f1_score(y_true, y_pred), step)

                    tb_logger.add_scalar(f'Byzantine/threshold_{run_id}', threshold, step)
                    tb_logger.add_scalar(f'grad_norm_mean/ALL_{run_id}', torch.mean(dist), step)
                    tb_logger.add_scalar(f'grad_norm_max/ALL_{run_id}', torch.max(dist), step)
                    tb_logger.add_scalar(f'Byzantine/N_good_pred_{run_id}', N_good, step)

                # store per-run history (steps, training and validation returns)
                if run_id not in self.memory.steps.keys():
                    self.memory.steps[run_id] = []
                    self.memory.eval_values[run_id] = []
                    self.memory.training_values[run_id] = []

                self.memory.steps[run_id].append(step)
                self.memory.training_values[run_id].append(np.mean(batch_rets))


            # do validating
            eval_reward = self.start_validating(tb_logger, step, max_steps = opts.val_max_steps, render = opts.render, run_id = run_id)
            if(tb_logger is not None):
                 self.memory.eval_values[run_id].append(eval_reward)

            # save current model
            if not opts.no_saving:
                self.save(epoch, run_id)
                torch.save(
                    {'epoch': epoch,
                     'step': step,
                     'np_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng_state': (torch.cuda.get_rng_state_all()
                                        if opts.use_cuda else None)},
                    os.path.join(self.opts.save_dir,
                                 f'r{run_id}-epoch-{epoch}-meta.pt'))


    # validate the new model
    def start_validating(self,tb_logger = None, id = 0, max_steps = 1000, render = False, run_id = 0, mode = 'human'):
        print('\nValidating...', flush=True)

        val_ret = 0.0
        val_len = 0.0

        for _ in range(self.opts.val_size):
            epi_ret, epi_len, _ = self.master.rollout(self.opts.device, max_steps = max_steps, render = render, sample = False, mode = mode, save_dir = './outputs/', filename = f'gym_{run_id}_{_}.gif')
            val_ret += epi_ret
            val_len += epi_len

        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size

        print('\nGradient step: %3d \t return: %.3f \t ep_len: %.3f'%
                (id,  np.mean(val_ret), np.mean(val_len)))

        if(tb_logger is not None):
            tb_logger.add_scalar(f'validate/total_rewards_{run_id}', np.mean(val_ret), id)
            tb_logger.add_scalar(f'validate/epi_length_{run_id}', np.mean(val_len), id)
            tb_logger.close()

        return np.mean(val_ret)
