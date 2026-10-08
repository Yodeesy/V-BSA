#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser('FT-FedScsPG')

    ### Overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1',
                        choices=['LunarLander', 'CartPole-v1', 'Acrobot-v1'],
                        help='OpenAI Gym env name for test')
    parser.add_argument('--eval_only', action='store_true',
                        help='used only if to evaluate a pre-trained model')
    parser.add_argument('--no_saving', action='store_true',
                        help='Disable saving checkpoints')
    parser.add_argument('--save_dir', default=None,
                        help='Override the checkpoint output directory '
                             '(default: outputs/<env>/<setup>/<run_name>)')
    parser.add_argument('--no_tb', action='store_true',
                        help='Disable Tensorboard logging')
    parser.add_argument('--render', action='store_true',
                        help='render to view the env')
    parser.add_argument('--mode', type=str, choices = ['human', 'rgb'], default='human',
                        help='render mode')
    parser.add_argument('--log_dir', default = 'logs',
                        help='log folder' )
    parser.add_argument('--run_name', default='run_name',
                        help='Name to identify the experiments')


    # Multiple runs
    parser.add_argument('--multiple_run', type=int, default=1,
                        help='number of repeated runs')
    parser.add_argument('--seed', type=int, default=0,
                        help='Starting point of random seed when running multiple times')


    # Federation and Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=10,
                        help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=0,
                        help = 'number of worker node that is Byzantine')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help = 'atmost alpha-fractional worker nodes are Byzantine')
    parser.add_argument('--attack_type', type=str, default='pulse-attack',
                        choices = ['vbsa-attack', 'pulse-attack'],
                        help = 'the behavior scheme of a Byzantine worker')
    # Ensemble defense
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Enable ensemble defense (K groups, each trains independent global policy)')
    parser.add_argument('--num_groups', '-K', type=int, default=5,
                        help='Number of groups for ensemble defense')

    # Override FedPG-BR hyperparams (for reproduction tuning)
    parser.add_argument('--sigma', type=float, default=None,
                        help='Override FedPG-BR variance bound sigma (default: env-specific)')

    # RL Algorithms (default GOMDP)
    parser.add_argument('--SVRPG', action='store_true',
                        help='run SVRPG')
    parser.add_argument('--FedPG_BR', action='store_true',
                        help='run FT-FedScsPG')

    # Group aggregation rule
    parser.add_argument('--agg_rule', type=str, default=None,
                        choices=['fedpg_br', 'fedavg', 'median', 'trimmed_mean', 'geometric_median'],
                        help='Group aggregation rule for ensemble training. '
                             'Default: fedpg_br if --FedPG_BR is set, else fedavg (legacy behavior).')
    parser.add_argument('--trim', type=int, default=1,
                        help='Number of extreme values trimmed per side per coordinate '
                             '(only used with --agg_rule trimmed_mean)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Override the env-specific max_trajectories')
    parser.add_argument('--epoch_timeout', type=int, default=1800,
                        help='Kill the run if one training epoch (a '
                             'pool.starmap call) does not finish within '
                             'this many seconds (watchdog against dead '
                             'pool workers)')

    parser.add_argument('--dynamics_index', type=int, default=0,
                        help='Non-ensemble per-epoch seed offset: the same '
                             'value reproduces the same epoch-by-epoch '
                             'dynamics in different runs')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Train epochs up to and including this value, '
                             'then stop (default None = run until '
                             'max_trajectories)')
    parser.add_argument('--pulse_epoch', type=int, default=None,
                        help='pulse-attack: reserved for command-line '
                             'compatibility (parsed, not consumed — '
                             'pulse-attack workers submit honest gradients '
                             'at every epoch)')
    parser.add_argument('--pulse_epochs', type=str, default=None,
                        help='pulse-attack: reserved for command-line '
                             'compatibility (parsed, not consumed — '
                             'pulse-attack workers submit honest gradients '
                             'at every epoch)')

    # V-BSA attack hyperparameters.  Only read when
    # --attack_type vbsa-attack; honest and pulse-attack workers ignore them.
    parser.add_argument('--vbsa_beta', type=float, default=1.0,
                        help='vbsa-attack: steering-mix weight beta in '
                             'v = g + beta * d_k (signed default 1.0)')
    parser.add_argument('--vbsa_c', type=float, default=0.85,
                        help='vbsa-attack: local temporal confidence box '
                             'half-width c in [mu - c*disp, mu + c*disp] '
                             '(signed default 0.85)')
    parser.add_argument('--vbsa_alpha', type=float, default=0.1,
                        help='vbsa-attack: LTD EMA rate alpha for the '
                             'per-coordinate mean/dispersion (signed '
                             'default 0.1)')
    parser.add_argument('--vbsa_entropy_frac', type=float, default=0.9,
                        help='vbsa-attack: entropy trigger threshold as a '
                             'fraction of log(n_actions) (default 0.9); '
                             'states with H > frac*log(n_acts) are '
                             'trigger states')
    parser.add_argument('--vbsa_target_mode', type=str, default='auto',
                        choices=['auto', 'fixed'],
                        help="vbsa-attack: 'auto' = group_id %% n_actions "
                             "in ensemble mode / action 0 in non-ensemble "
                             "mode; 'fixed' = --vbsa_target_action "
                             "everywhere (ablation knob)")
    parser.add_argument('--vbsa_target_action', type=int, default=0,
                        help='vbsa-attack: fixed target action when '
                             '--vbsa_target_mode fixed')
    parser.add_argument('--vbsa_gate', type=str, default='entropy',
                        choices=['entropy', 'random', 'none'],
                        help="vbsa-attack: trigger gate. 'entropy' = the "
                             "local-policy-entropy gate H > "
                             "entropy_frac*log(n_acts) (default); "
                             "'random' = independent per-state trigger with "
                             "p = --vbsa_rand_frac (seeded per worker/"
                             "epoch); 'none' = every visited state "
                             "triggers (dense)")
    parser.add_argument('--vbsa_rand_frac', type=float, default=0.15,
                        help='vbsa-attack: per-state trigger probability '
                             'under --vbsa_gate random')


    # Training and validating
    parser.add_argument('--val_size', type=int, default=10,
                        help='Number of episoid used for reporting validation performance')
    parser.add_argument('--val_max_steps', type=int, default=1000,
                        help='Maximum trajectory length used for reporting validation performance')


    # Load pre-trained modelss
    parser.add_argument('--load_path', default = None,
                        help='Path to load pre-trained model parameters')
    parser.add_argument('--resume_from', default=None,
                        help='Resume ensemble training from the latest '
                             'checkpoint in this directory (useful for '
                             'continuing runs that were externally '
                             'interrupted)')


    ### end of parameters
    opts = parser.parse_args(args)

    # Save CLI overrides before env-specific defaults overwrite them
    _sigma_override = opts.sigma
    _max_traj_override = opts.max_trajectories

    opts.use_cuda = False
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = (opts.save_dir if opts.save_dir is not None else
                     os.path.join(
                         'outputs',
                         '{}'.format(opts.env_name),
                         "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
                         opts.run_name
                     )) if not opts.no_saving else None
    opts.log_dir = os.path.join(
        f'{opts.log_dir}',
        '{}'.format(opts.env_name),
        "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_tb else None

    if opts.env_name == 'CartPole-v1':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500
        opts.max_trajectories = 5000
        opts.gamma  = 0.999
        opts.min_reward = 0  # for logging purpose (not important)
        opts.max_reward = 600  # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3
        opts.hidden_units = '16,16'
        opts.activation = 'ReLU'
        opts.output_activation = 'Tanh'

        # batch_size
        opts.B = 16 # for SVRPG and GOMDP
        opts.Bmin = 12 # for FT-FedScsPG
        opts.Bmax = 20 # for FT-FedScsPG
        opts.b = 4 # mini batch_size for SVRPG and FT-FedScsPG

        # inner loop iteration for SVRPG
        opts.N = 3

        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.06

    if opts.env_name == 'LunarLander':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 1000
        opts.max_trajectories = 1e4
        opts.gamma  = 0.99
        opts.min_reward = -1000 # for logging purpose (not important)
        opts.max_reward = 300 # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3 # 8e-4
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'

        # batch_size
        opts.B = 32 # for SVRPG and GOMDP
        opts.Bmin = 26 # for FT-FedScsPG
        opts.Bmax = 38 # for FT-FedScsPG
        opts.b = 8 # mini batch_size for SVRPG and FT-FedScsPG

        # inner loop iteration for SVRPG
        opts.N = 3

        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.07

    if opts.env_name == 'Acrobot-v1':
        # Two-link underactuated swing-up, Discrete(3) actions, 6-dim obs,
        # sparse reward (-1 per step until the tip clears the goal line;
        # gymnasium native env, no version resolution needed).  Hyper-
        # parameters mirror the LunarLander block.  |A| = 3 -> the
        # entropy-gate threshold 0.9*ln3 and the 2:2:1 target split
        # auto-derive from n_acts.  max_trajectories is set high so the
        # epoch cap (--max_epochs) is the binding termination.
        opts.max_epi_len = 500
        opts.max_trajectories = 5e4
        opts.gamma  = 0.99
        opts.min_reward = -500 # for logging purpose (not important)
        opts.max_reward = 0 # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'

        # batch_size
        opts.B = 32 # for SVRPG and GOMDP
        opts.Bmin = 26 # for FT-FedScsPG
        opts.Bmax = 38 # for FT-FedScsPG
        opts.b = 8 # mini batch_size for SVRPG and FT-FedScsPG

        # inner loop iteration for SVRPG
        opts.N = 3

        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.07

    # Allow CLI override of key hyperparams for reproduction tuning
    if _sigma_override is not None:
        opts.sigma = _sigma_override
    if _max_traj_override is not None:
        opts.max_trajectories = _max_traj_override

    # Resolve aggregation rule (default preserves legacy behavior:
    # --FedPG_BR -> FedPG-BR filtering, otherwise GOMDP plain mean)
    if opts.agg_rule is None:
        opts.agg_rule = 'fedpg_br' if opts.FedPG_BR else 'fedavg'
    if opts.agg_rule not in ('fedpg_br', 'fedavg') and not opts.ensemble:
        print(f'NOTE: --agg_rule {opts.agg_rule} in non-ensemble mode '
              'aggregates over all 30 worker gradients (no filtering); '
              'FedPG-BR filtering applies only to --agg_rule fedpg_br.')

    # Checkpoint resume (ensemble and single-model training).  Resuming
    # requires saving enabled; if no explicit --save_dir was given, keep
    # saving into the directory being resumed from so later resumes find
    # the new epochs.
    if opts.resume_from is not None:
        if opts.no_saving:
            print('WARNING: --resume_from needs saving enabled; overriding --no_saving.')
            opts.no_saving = False
        if opts.save_dir is None:
            opts.save_dir = opts.resume_from

    # Resolve the pulse injection epoch set (--pulse_epoch / --pulse_epochs,
    # unioned).  Parsed for command-line compatibility; pulse-attack
    # workers submit honest gradients at every epoch, so the set is not
    # consumed at runtime.  Stored as a sorted list (not a set) so run.py
    # can dump vars(opts) to args.json.
    pulse_epoch_set = set()
    if opts.pulse_epoch is not None:
        pulse_epoch_set.add(opts.pulse_epoch)
    if opts.pulse_epochs is not None:
        for part in opts.pulse_epochs.split(','):
            part = part.strip()
            if part:
                pulse_epoch_set.add(int(part))
    opts.pulse_epoch_set = sorted(pulse_epoch_set)

    assert opts.SVRPG + opts.FedPG_BR <= 1
    print('run GOMDP\n' if opts.SVRPG + opts.FedPG_BR == 0 else ('run FT-FedScsPG\n' if opts.FedPG_BR else 'run SVRPG\n'))
    print(f'Aggregation rule: {opts.agg_rule}\n')

    return opts
