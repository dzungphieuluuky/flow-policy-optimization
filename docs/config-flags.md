# Config & Flags

Source of truth: `src/sjfpo/config.py`. YAML mirrors in `configs/`
(`shared.yaml` run matrix + budgets, `sjfpo/ppo/sac.yaml` algorithm defaults).

## Environment Flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `SJFPO_TINY_MATRIX` | env str | `0` | When `1`: tiny preset (Hopper-v5 × seed 0, short budgets) for smokes. |
| `SJFPO_DISABLE_WANDB` | env str | unset (enabled) | `1` disables wandb via `wandb_logging`; wandb is REQUIRED, init failure is fatal. |
| `IS_KAGGLE` | autodetect | `KAGGLE_KERNEL_RUN_TYPE` set → True | Kaggle run-output dirs. |

## Run Matrices

| Mode | Value | Members |
|---|---|---|
| Full (`RUN_MATRIX`) | 45 runs ×1M steps | `METHODS × ENVS × SEEDS` = 3×3×5, exactly as `test_config.py` asserts. |
| Tiny (`SJFPO_TINY_MATRIX=1`) | 3 runs, short budgets | Hopper-v5 × seed 0. |

`METHODS = (sjfpo, ppo, sac)`, `ENVS = (HalfCheetah-v5, Walker2d-v5, Hopper-v5)`,
`SEEDS = (0..4)`; `TOTAL_STEPS = 1M`, `EVAL_STEPS = 50k`, `SAVE_STEPS = 25k`.
`scripts/sweep.sh --tiny` filters the same cartesian product via
`METHODS/ENVS/SEEDS` env axes.

## Algorithm Dataclasses

`configuration_sjfpo.py::SJFPOConfig` (sole SJFPO source: T=16 state, H=16 predict, full-H execute in training / `ratio=0.5` prefix at eval, top_p=0.3 (K=5), D=128,
`vf_coef=0.1`, Euler 16 train / 16 eval, lr 1e-4, ASPO clip 0.05),
`PPOConfig`, `SACConfig` (skrl 2.1.0 hyperparams). CLI scalars beat YAML beats dataclass defaults.

## Tuning Guide

- **Shave runtime:** `SJFPO_TINY_MATRIX=1`, `scripts/train.py --total_steps N`,
  or shrink the sweep with `METHODS/ENVS/SEEDS` axes.
- **Change binding:** `epsilon_clip / cfm_clamp / log_ratio_cap / vf_coef / top_p / H`
  live in `SJFPOConfig` (mirrored in `configs/sjfpo.yaml`).
