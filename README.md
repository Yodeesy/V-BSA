# When to Intervene? State-Aware Sparse Manipulation in Federated Reinforcement Learning
[![arXiv](https://img.shields.io/badge/arXiv-2610.11523-b31b1b.svg)](https://arxiv.org/abs/2610.11523)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Federated reinforcement learning (FRL) enables distributed agents to collaboratively train decision-making policies, but its decentralized training process also
exposes global policy learning to Byzantine manipulation. Existing poisoning
attacks primarily focus on how to construct malicious updates, while trajectorylevel intervention timing remains largely implicit. In sequential decision making,
however, where an intervention is applied can alter subsequent trajectories and
learning signals. Through controlled experiments, we find that changing the selected trajectory states materially alters attack efficacy even when the maliciousupdate construction is fixed. We therefore identify when as a distinct attack dimension and introduce the Viability-constrained Behavioral Steering Attack (VBSA), which uses local policy uncertainty to select sparse intervention states and
applies envelope-constrained behavioral steering. Across discrete-action benchmarks, V-BSA achieves substantial degradation against robust aggregators and
ensemble defenses with only a fraction of the interventions used by dense poisoning, while revealing task- and aggregation-dependent boundaries. Overall, our results highlight intervention timing as a distinct dimension of sequential robustness
in FRL. The code is available at https://github.com/Yodeesy/V-BSA. 

---

## Contents

```
supplementary_code/
├── README.md
├── requirements.txt
└── codes/
    ├── run.py      # entry point: option parsing, seeds, dispatch to training
    ├── options.py  # CLI options and per-environment hyperparameters
    ├── agent.py    # master: FedPG-BR filter, aggregation rules, training loops
    ├── worker.py   # worker: policy-gradient collection + V-BSA Byzantine attack
    ├── policy.py   # MlpPolicy (discrete action spaces)
    └── utils.py    # checkpoint I/O, environment name resolution
```

## Requirements

* Python >= 3.8
* See `requirements.txt`.  The Gymnasium Box2D extra is required for
  `LunarLander`; install with:

```bash
pip install -r requirements.txt
```

All runs in the paper are CPU-only (`torch` without CUDA).  A machine with
16+ cores is recommended: the distributed training loop keeps 30 worker
processes plus one master process busy.

## Attack modes

* `--attack_type vbsa-attack`: the Byzantine attack.  Per epoch, each
  Byzantine worker runs the four-step pipeline implemented in
  `worker.py::Worker.train_one_epoch`:
  1. **LTD** (local temporal distribution): per-coordinate EMA of the
     honest-gradient mean and dispersion;
  2. **Steering** `d_k`: cross-entropy toward a target action computed on
     the raw pre-Tanh logits, plus the entropy gate (`--vbsa_gate
     entropy`, default) that restricts steering to high-entropy states;
  3. **Camouflage**: `v = g + beta * d_k`, rescaled back to `||g||`;
  4. **Box clamp**: `u* = clamp(v, mu - c*disp, mu + c*disp)`.
* `--attack_type pulse-attack`: silent control — Byzantine workers submit
  honest gradients at every epoch.  Used for the control arms
  (`--pulse_epochs 999`; the `--pulse_epoch` / `--pulse_epochs` flags are
  accepted for command-line compatibility and not consumed).

## Running the experiments

All runs use 30 workers, 10 Byzantine workers, 261 epochs and `--no_tb`.
Each arm follows the pattern

```
python run.py --env_name <ENV> --no_tb --seed 0 --num_worker 30 \
    --num_Byzantine 10 --epoch_timeout 1800 --max_epochs 261 \
    --max_trajectories 50000 --dynamics_index <D> \
    --save_dir <OUT> [aggregation flags] [--attack_type <MODE>]
```

with `<MODE>` = `vbsa-attack` (attack arm) or
`pulse-attack --pulse_epochs 999` (control arm), and `<D>` in `{0, 1, 2}`
(the three dynamics realizations per cell).

### LunarLander ensemble matrix (Table 1)

Four aggregation setups, each run for every `(D, MODE)` pair:

| Tag   | Aggregation flags                                             |
|-------|---------------------------------------------------------------|
| ensf  | `--ensemble --num_groups 5 --FedPG_BR --agg_rule fedpg_br`    |
| enst  | `--ensemble --num_groups 5 --FedPG_BR --agg_rule trimmed_mean --trim 1` |
| ensm  | `--ensemble --num_groups 5 --FedPG_BR --agg_rule median`      |
| sfbr  | `--FedPG_BR` (flat, non-ensemble FedPG-BR)                    |

The `ensf` / `enst` / `ensm` reference runs continue training from a
shared epoch-60 checkpoint (`--resume_from <checkpoint_dir>`); the `sfbr`
runs train the full 261 epochs from scratch.  Checkpoints are not
distributed with this supplement; omit `--resume_from` to train from
scratch, or point it at a directory of your own
`r{run}-epoch-{e}*.pt` files (the resume protocol is described below).

### Cross-environment matrices (CartPole / Acrobot)

The `enst` / `ensm` / `sfbr` flags above are run on
`--env_name CartPole-v1` and `--env_name Acrobot-v1` for each `(D, MODE)`
(the `ensf` rows in the paper are the ensemble FedPG-BR runs on these
environments).  CartPole attack arms add `--vbsa_target_mode fixed
--vbsa_target_action 0` (the two-action target code); Acrobot attack arms
use the default `--vbsa_target_mode auto`.

### Non-ensemble baseline matrix

One run per environment (`LunarLander`, `CartPole-v1`, `Acrobot-v1`) and
aggregator, at `--dynamics_index 0` (and `1`, `2` for the dynamics table;
the flat FedPG-BR rows at `d1`/`d2` are the `sfbr` runs above):

| Tag   | Aggregation flags                          |
|-------|--------------------------------------------|
| favg  | `--agg_rule fedavg`                        |
| med   | `--FedPG_BR --agg_rule median`             |
| trm   | `--FedPG_BR --agg_rule trimmed_mean --trim 1` |
| fpbr  | `--FedPG_BR`                               |

CartPole attack arms add `--vbsa_target_mode fixed --vbsa_target_action 0`;
LunarLander and Acrobot attack arms use the default target mode.

## Checkpoint resume protocol

Every epoch saves `r{run_id}-epoch-{epoch}-meta.pt` last (ensemble runs
also save one `r{run_id}-epoch-{epoch}-group{k}.pt` per group).  With
`--resume_from <dir>`, training scans the directory, picks the highest
epoch whose meta file and all model files exist and load cleanly, and
continues from there.  A run killed mid-save therefore resumes from the
previous complete epoch.

## Reading results

* `save_dir/args.json` records the exact configuration of each run.
* The performance metric `J_T` is the return of the last
  `Gradient step: N return: R` validation line (10 validation episodes,
  greedy policy; the ensemble uses the majority vote over the group
  policies).
* The reported damage `D = J_ctl - J_atk` pairs each attack arm with its
  control arm (same dynamics index).
* Attack diagnostics: `[VBSA-DIAG]` prints the per-epoch steering
  statistics; `[BSA-FILTER]` and `[BSA-DEV]` print the server-side
  filtering outcome and the deviation of the aggregated update from the
  honest-mean update.
