# Config & Flags

Source of truth: `src/sjfpo/config.py`. YAML mirrors in `configs/`
(`shared.yaml` run matrix + budgets, `sjfpo/ppo/sac.yaml` algorithm defaults).

## Environment Flags

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `SJFPO_LEGACY_MUJOCO` | env str `0`/`1` | `1` (None → True) | Compat parse; only the MuJoCo path exists. Any other value is unsupported. |
| `SJFPO_TINY_MATRIX` | env str | `0` | When `1`: tiny preset (Hopper-v5 × seed 0, short budgets) for smokes. |
| `SJFPO_RESUME_ONLY` | env str | `0` | Sweep skips `MISSING`, only resumes `PARTIAL`. |
| `SJFPO_DISABLE_WANDB` | env str | unset (enabled) | `1` disables wandb via `wandb_logging`; wandb is REQUIRED, init failure is fatal. |
| `IS_KAGGLE` | autodetect | `KAGGLE_KERNEL_RUN_TYPE` set → True | Kaggle run-output dirs. |

## Run Matrices

| Mode | Value | Members |
|---|---|---|
| Full (`RUN_MATRIX`) | 45 runs ×1M steps | `METHODS × ENVS × SEEDS` = 3×3×5, exactly as `test_config.py` asserts. |
| Tiny (`SJFPO_TINY_MATRIX=1`) | 3 runs, short budgets | Hopper-v5 × seed 0. |

`METHODS = (sjfpo, ppo, sac)`, `ENVS = (HalfCheetah-v5, Walker2d-v5, Hopper-v5)`,
`SEEDS = (0..4)`; `TOTAL_STEPS = 1M`, `EVAL_INTERVAL = 50k`, `CKPT_INTERVAL = 25k`.
`scripts/train.py --matrix` builds the same cartesian product via
`config.build_filtered_matrix` (`--methods/--envs/--seeds` filter it).

## Algorithm Dataclasses

`SJFPOConfig` (T=16 state, H=16 predict / n=8 execute, K=5, D=128, spectral_dim=64, `λ_A=1.0 λ_S=0.5
λ_sigreg=0.1`, Euler 64 train / 16 eval, lr 1e-4), `PPOConfig`, `SACConfig`
(skrl 2.1.0 hyperparams). CLI scalars beat YAML beats dataclass defaults.

## Tuning Guide

- **Shave runtime:** `SJFPO_TINY_MATRIX=1`, `scripts/train.py --total-steps N`,
  or shrink the matrix with `--methods/--envs/--seeds`.
- **Change binding:** `lambda_a / lambda_s / sigreg_coef / K / spectral_dim / N / H`
  live in `SJFPOConfig` (mirrored in `configs/sjfpo.yaml`); tests assert
  `λ_A=1.0 λ_S=0.5 sigreg_coef=0.1 K=5 spectral_dim=64`.
