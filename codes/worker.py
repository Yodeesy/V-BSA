import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from policy import MlpPolicy
from utils import get_inner_model, save_frames_as_gif
from utils import env_wrapper
import random
import math


def _reset_env(env, seed=None):
    """Reset environment, compatible with both Gym (returns obs) and
    Gymnasium (returns (obs, info))."""
    if seed is not None:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            # Old Gym: reset() doesn't accept seed kwarg
            env.seed(seed)
            result = env.reset()
    else:
        result = env.reset()
    if isinstance(result, tuple):
        return result[0]  # Gymnasium: (obs, info)
    return result  # Old Gym: obs


def _step_env(env, action):
    """Step environment, compatible with both Gym (returns 4-tuple) and
    Gymnasium (returns 5-tuple). Always returns (obs, rew, done, info)."""
    try:
        arr = (action.detach().cpu().numpy() if hasattr(action, 'detach')
               else np.asarray(action))
        finite = np.isfinite(arr).all()
    except Exception:
        finite = True
    if not finite:
        # Diverged (NaN/inf) action: Box2D's physics solver can spin
        # forever on non-finite inputs, wedging the worker in C code where
        # no Python watchdog can reach it — the whole pool then hangs on
        # starmap.  End the episode immediately so the worker always
        # returns in bounded time.
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        return obs, 0.0, True, {}
    result = env.step(action)
    if len(result) == 5:
        # Gymnasium: (obs, rew, terminated, truncated, info)
        obs, rew, terminated, truncated, info = result
        done = terminated or truncated
    else:
        # Old Gym: (obs, rew, done, info)
        obs, rew, done, info = result
    if not np.isfinite(np.asarray(obs)).all():
        # NaN physics state: end the episode before it feeds the policy.
        done = True
    return obs, rew, done, info


def compute_target_grad(logits_net, trig_states, target_action, device):
    """V-BSA steering gradient d_k (Step 2): cross-entropy toward a fixed
    target action, averaged over the entropy-triggered states.  The loss
    is computed on the RAW pre-Tanh logits (a forward hook bypasses the
    Tanh saturation so the gradient does not vanish; LunarLander's last
    layer is Tanh).  Returns a list of cloned per-parameter gradients."""
    logits_net.zero_grad()

    last_layer = list(logits_net.logits_net.children())[-1]
    is_tanh = isinstance(last_layer, torch.nn.Tanh)

    obs_stack = torch.stack([obs_t for obs_t in trig_states])
    targets = torch.full((obs_stack.shape[0],), int(target_action),
                         dtype=torch.long, device=device)

    if is_tanh:
        pre_tanh_logits = []

        def hook_fn(module, input, output):
            pre_tanh_logits.append(input[0])

        hook_handle = last_layer.register_forward_hook(hook_fn)
        _ = logits_net.logits_net(obs_stack)
        hook_handle.remove()
        raw_logits = pre_tanh_logits[0]
        if raw_logits.dim() == 1:
            raw_logits = raw_logits.unsqueeze(0)
        loss_steer = F.cross_entropy(raw_logits, targets)
    else:
        logits = logits_net.logits_net(obs_stack)
        dist = Categorical(logits=logits)
        loss_steer = -dist.log_prob(targets).mean()

    loss_steer.backward()
    return [p.grad.clone() for p in logits_net.parameters()]


class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation='Tanh',
                 output_activation='Identity',
                 attack_type=None,
                 max_epi_len=0,
                 opts=None
                 ):
        super(Worker, self).__init__()

        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        from utils import resolve_env_name
        self.env = gym.make(resolve_env_name(env_name))
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len

        assert opts is not None
        self.opts = opts  # store for worker-side hyperparameter access

        # V-BSA local temporal dispersion (LTD) state: per-parameter EMA
        # mean/dispersion of THIS worker's honest gradients (attack-free
        # baseline).  Lives on the master-side object — pool-side
        # attribute writes die with the task — so train_one_epoch carries
        # the updated state out in its return tuple and the master writes
        # it back.
        self._vbsa_mu = None
        self._vbsa_disp = None

        # get observation dim
        obs_dim = self.env.observation_space.shape[0]
        if not isinstance(self.env.action_space, Discrete):
            raise NotImplementedError(
                f'{env_name}: only discrete action spaces are supported')
        n_acts = self.env.action_space.n

        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim] + hidden_sizes + [n_acts]  # make core of policy network

        # get policy net
        self.logits_net = MlpPolicy(self.sizes, activation, output_activation)

        if self.id == 1:
            print(self.logits_net)

    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param})

    def rollout(self, device, max_steps=1000, render=False, env=None, obs=None, sample=True, mode='human',
                save_dir='./', filename='.'):

        if env is None and obs is None:
            env = self.env
            obs = _reset_env(env)

        done = False
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()

            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample=sample)[0]
            obs, rew, done, _ = _step_env(env, action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        return np.sum(ep_rew), len(ep_rew), ep_rew

    def collect_experience_for_training(self, B, device, record=False, sample=True):
        # make some empty lists for logging.
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_log_prob = []  # for gradient computing

        # reset episode-specific variables
        obs = _reset_env(self.env)  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment
            obs = env_wrapper(self.env_name, obs)
            act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample=sample)

            obs, rew, done, info = _step_env(self.env, act)

            # timestep
            t = t + 1

            # save action_log_prob, reward
            batch_log_prob.append(log_prob)

            ep_rews.append(rew)

            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:

                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                for r in ep_rews[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32)

                # return whitening
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break

                # reset episode-specific variables
                obs, done, ep_rews, t = _reset_env(self.env), False, [], 1

        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype=torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        # Safety: a poisoned episode (e.g. a NaN action that _step_env
        # force-ended) yields NaN log-probs / advantages.  Zero them so one
        # bad timestep cannot turn the loss — and therefore the gradients
        # sent to the master — NaN.  A NaN model is what wedges Box2D in
        # the first place.
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        logp = torch.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens

    def train_one_epoch(self, B, device, sample, epoch=None):
        opts = getattr(self, 'opts', None)

        # ---- Collect worker rollouts ----
        # V-BSA needs the rollout states for the entropy trigger gate.
        need_record = self.is_Byzantine and self.attack_type == 'vbsa-attack'
        if need_record:
            (weights, logp, batch_rets, batch_lens,
             batch_states, batch_actions) = self.collect_experience_for_training(
                B, device, sample=sample, record=True)
        else:
            weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(
                B, device, sample=sample)

        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()

        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()

        vbsa_out = None  # V-BSA LTD round-trip (mu, disp); None elsewhere

        # ---- V-BSA (Viability-aware Behavioral Steering Attack) ----
        # Four-step closed-form pipeline:
        #   Step 1  LTD update  — attack-free per-coordinate EMA of THIS
        #                          worker's honest gradients (mu, disp).
        #                          Runs EVERY epoch, trigger or not.
        #   Step 2  Trigger gate + steering — trigger on the local policy
        #                          entropy H(s) over the rollout states;
        #                          d_k = grad of CE toward the target
        #                          action over the triggered states.
        #   Step 3  Camouflage   — v = g + beta*d_k, rescaled to ||g||_2
        #                          (norm matching BEFORE the box: the
        #                          projection-order fix).
        #   Step 4  Box clamp    — u* = clamp(v, mu - c*disp, mu + c*disp),
        #                          LAST step, so u* is in the box by
        #                          construction.
        # Greedy HOW: fires every epoch (no window gating; the entropy
        # trigger is the only selector).  The LTD state round-trips through
        # the return tuple (pool processes do not persist attribute writes;
        # see Worker.__init__).
        if self.is_Byzantine and self.attack_type == 'vbsa-attack':
            n_acts = self.env.action_space.n
            vbsa_beta = getattr(opts, 'vbsa_beta', 1.0)
            vbsa_c = getattr(opts, 'vbsa_c', 0.85)
            vbsa_alpha = getattr(opts, 'vbsa_alpha', 0.1)
            ent_frac = getattr(opts, 'vbsa_entropy_frac', 0.9)
            ent_thr = ent_frac * math.log(max(2, n_acts))
            gate_mode = getattr(opts, 'vbsa_gate', 'entropy')

            honest_grad = [p.grad.clone() for p in self.parameters()]
            # NaN guard: a poisoned episode can put NaN into the honest
            # grads; the V-BSA math (LTD + payload) runs on the sanitized
            # copy, while the honest-submission path below keeps the
            # existing silent-epoch semantics untouched.
            honest_grad_n = [torch.nan_to_num(g, nan=0.0, posinf=0.0,
                                              neginf=0.0)
                             for g in honest_grad]

            # ---- Step 1: attack-free LTD update (always, trigger or not) ----
            if self._vbsa_mu is None:
                self._vbsa_mu = [g.clone() for g in honest_grad_n]
                self._vbsa_disp = [torch.abs(g) * 0.5 + 1e-6
                                   for g in honest_grad_n]
            else:
                for i, g in enumerate(honest_grad_n):
                    diff = g - self._vbsa_mu[i]
                    self._vbsa_mu[i] = self._vbsa_mu[i] + vbsa_alpha * diff
                    self._vbsa_disp[i] = ((1.0 - vbsa_alpha) *
                                          self._vbsa_disp[i] +
                                          vbsa_alpha * torch.abs(diff))
            vbsa_out = ([m.clone() for m in self._vbsa_mu],
                        [d.clone() for d in self._vbsa_disp])

            # ---- Step 2: entropy trigger gate over the local rollout states ----
            # 'entropy' (default): trigger on LOCAL POLICY ENTROPY
            # H(s) = -sum_a pi(a|s) ln pi(a|s) computed from THIS worker's
            # own logits net (worker-local hesitation, not a vote across
            # groups), threshold entropy_frac * ln(n_acts).
            # 'random' and 'none' are the gate ablations.
            net = self.logits_net.logits_net
            states_t = torch.stack(
                [torch.as_tensor(s, dtype=torch.float32)
                 for s in batch_states]).to(device)
            states_t = states_t[torch.isfinite(states_t).all(dim=1)]
            n_states = states_t.shape[0]
            trig_idx = []
            H_mean = float('nan')
            if n_states > 0:
                with torch.no_grad():
                    logits = net(states_t)
                    probs = torch.softmax(logits, dim=-1)
                    ent = -(probs * torch.log(probs + 1e-8)).sum(-1)
                    H_mean = float(ent.mean())
                if gate_mode == 'entropy':
                    trig_idx = (ent > ent_thr).nonzero(
                        as_tuple=False).view(-1).tolist()
                elif gate_mode == 'random':
                    rng = random.Random(
                        int(getattr(opts, 'seed', 0)) * 10 ** 9
                        + epoch * 1000 + self.id)
                    p = float(getattr(opts, 'vbsa_rand_frac', 0.15))
                    trig_idx = [i for i in range(n_states)
                                if rng.random() < p]
                else:  # 'none': every visited state triggers
                    trig_idx = list(range(n_states))
            trig_states = [states_t[i] for i in trig_idx]
            trig_count = len(trig_states)

            g_norm = math.sqrt(sum(float(torch.sum(g ** 2))
                                   for g in honest_grad_n)) + 1e-10
            grad = [g for g in honest_grad]  # default: honest submission
            target = self._vbsa_target(opts, n_acts)
            d_norm = norm_ratio = inj_ratio = bind = percoord = float('nan')

            if trig_count > 0:
                # ---- Step 2 (cont.): steering gradient d_k ----
                d_k = compute_target_grad(self.logits_net, trig_states,
                                          target, device)
                d_norm = math.sqrt(sum(float(torch.sum(g ** 2))
                                       for g in d_k)) + 1e-10

                # ---- Step 3: fuse + norm camouflage (scale FIRST) ----
                v = [g + vbsa_beta * d
                     for g, d in zip(honest_grad_n, d_k)]
                v_norm = math.sqrt(sum(float(torch.sum(x ** 2))
                                       for x in v)) + 1e-10
                v_cam = [x * (g_norm / v_norm) for x in v]

                # ---- Step 4: local temporal confidence box (clamp LAST) ----
                u = []
                n_clamped = 0
                n_total = 0
                for x, m, s in zip(v_cam, self._vbsa_mu, self._vbsa_disp):
                    lo, hi = m - vbsa_c * s, m + vbsa_c * s
                    n_total += x.numel()
                    n_clamped += int(((x < lo) | (x > hi)).sum())
                    u.append(torch.clamp(x, min=lo, max=hi))
                grad = u
                bind = n_clamped / max(1, n_total)

                u_norm = math.sqrt(sum(float(torch.sum(x ** 2))
                                       for x in u)) + 1e-10
                norm_ratio = u_norm / g_norm
                diff_norm = math.sqrt(sum(
                    float(torch.sum((x - g) ** 2))
                    for x, g in zip(u, honest_grad_n))) + 1e-10
                inj_ratio = diff_norm / g_norm
                percoord = float(sum(
                    (x - g).abs().mean()
                    for x, g in zip(u, honest_grad_n))) / max(
                        1e-12, float(sum(s.abs().mean()
                                         for s in self._vbsa_disp)))

            # ---- DIAGNOSTIC: per epoch, per Byzantine worker ----
            print(f'[VBSA-DIAG] worker {self.id} '
                  f'(group {getattr(self, "group_id", -1)}): '
                  f'epoch={epoch}, target={target}, '
                  f'||g||={g_norm:.4e}, '
                  f'||d_k||={d_norm:.4e}, '
                  f'beta={vbsa_beta:.3f}, '
                  f'||u*||/||g||={norm_ratio:.3f}, '
                  f'||u*-g||/||g||={inj_ratio:.3f}, '
                  f'box_bind={bind:.3f}, '
                  f'percoord={percoord:.3f}, '
                  f'n_trig={trig_count}/{n_states}, '
                  f'H_mean={H_mean:.3f}',
                  flush=True)
        # ---- End V-BSA ----

        # determine if the agent is byzantine (other attack types)
        elif self.is_Byzantine and self.attack_type is not None:
            # pulse-attack (clean control): submit the honest gradient
            # every epoch (the pulse injection window is configured to
            # never fire for the control arms).
            if self.attack_type == 'pulse-attack':
                grad = [item.grad for item in self.parameters()]
            else:
                raise NotImplementedError(
                    f'unknown attack_type {self.attack_type}')

        else:
            # return true gradient
            grad = [item.grad for item in self.parameters()]

        # report the results to the agent for training purpose; vbsa_out is
        # the V-BSA LTD round-trip (mu, disp) or None.
        return (grad, batch_loss.item(), np.mean(batch_rets),
                np.mean(batch_lens), vbsa_out)

    def _vbsa_target(self, opts, n_acts):
        """V-BSA target action (Step 2).

        'auto' (default): group-diverse steering in ensemble mode —
        target = group_id % n_acts, so the K group policies are pulled
        toward different actions and the test-time majority vote loses its
        clean consensus; in non-ensemble mode every Byzantine shares one
        fixed target (action 0) so the per-coordinate bias pushes the same
        direction for median/trimmed.  'fixed': --vbsa_target_action
        everywhere.
        """
        mode = getattr(opts, 'vbsa_target_mode', 'auto')
        if mode == 'fixed':
            return int(getattr(opts, 'vbsa_target_action', 0)) % n_acts
        gid = getattr(self, 'group_id', -1)
        if gid >= 0:
            return gid % n_acts
        return 0

    def to(self, device):
        self.logits_net.to(device)
        return self

    def eval(self):
        self.logits_net.eval()
        return self

    def train(self):
        self.logits_net.train()
        return self

    def parameters(self):
        return self.logits_net.parameters()
