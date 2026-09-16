# wandb Logging (Enabled by Default)

wandb integration is **required and on by default**. Every training run logs to
wandb unless explicitly disabled — the only off switch is `SJFPO_DISABLE_WANDB=1`.
A failed `wandb.init` (missing key, no network) is a **fatal error**, never a
silent no-op.

## Install

wandb ships with the main requirements (no separate install step):

```bash
pip install -r requirements-linux.txt  # includes wandb (or requirements-amd.txt on ROCm)
```

## `SJFPO_DISABLE_WANDB` — the Only Off Switch

| Value | Behaviour |
|---|---|
| *(unset)* | Enabled — `wandb.init` runs in `online` mode (requires `wandb login`) |
| `"1"` | Disabled — all helpers no-op, zero overhead |
| anything else | Enabled — only exactly `"1"` disables |

`WANDB_MODE=offline` (wandb-native) captures offline runs; sync later with
`wandb sync wandb/run-<id>/`.

> Historical note: the old `SJFPO_WANDB` tri-state gate (`unset`/`0`/`offline`/`online`)
> is deleted — `SJFPO_DISABLE_WANDB=1` is the single source of truth.

## Usage

```bash
# Online training (default)
PYTHONPATH=src python scripts/train.py \
  --method sjfpo --env Hopper-v5 --seed 0 --total_steps 5000 \
  --out_dir runs/train

# Offline capture (wandb-native mode, still requires the package)
WANDB_MODE=offline METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" TOTAL_STEPS=500 \
  bash scripts/sweep.sh --out-base runs/smoke

# Disabled (the ONLY opt-out)
SJFPO_DISABLE_WANDB=1 PYTHONPATH=src python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 ...
```

## Syncing Offline Runs

After an offline run, push artifacts to wandb cloud:

```bash
wandb sync wandb/run-<id>/
```

## API Surface (6 fns, SDK-direct)

`src/sjfpo/wandb_logging.py` is the sole file that imports wandb (module-level
REQUIRED import; absent package is a hard ImportError). 6 public
functions, all calling the wandb SDK directly (`wandb.init`/`log`/`finish`/
`Table`), no wrappers, no custom step metric:

| Function | Signature | Behaviour |
|---|---|---|
| `wandb_disabled` | `() -> bool` | `True` only when `SJFPO_DISABLE_WANDB == "1"` |
| `init_wandb_run_for_method` | `(*, method, env_name, seed, total_steps, cfg_dict) -> bool` | SDK-direct `wandb.init(project=method, group=f"{method}_{env_name}", name=None, tags=[method, "mujoco", "cpu-first"], job_type="smoke" if total_steps <= 10000 else "train", mode=$WANDB_MODE, reinit="finish_previous", force=True)` with no custom step metric (default `step=global_step` is the x-axis, so charts render as lines like system auto-logging); returns `True` when enabled, `False` when disabled; init errors propagate (fail-fast, no try/except) |
| `log_metrics` | `(data, step) -> None` | Thin disabled-guard around `wandb.log(data, step=step)` |
| `log_eval_summary` | `(mean_return, eval_results, all_eval_returns, eval_count, global_step) -> None` | Shared runner/baselines eval payload (mean/length/max/std/count + Table + improvement in one commit) |
| `log_eval_video` | `(frames, global_step) -> None` | `wandb.Video` of one deterministic demo episode (best-effort; render failures never break training) |
| `finish_run` | `(exit_code=0) -> None` | Thin disabled-guard around `wandb.finish(exit_code=exit_code)` |

Callers (`baselines.py`) import at most 6 symbols and hold a
plain `bool` (`wandb_enabled = init_wandb_run_for_method(...)`).

## Dashboard Structure

| Field | Pattern | Example |
|---|---|---|
| `project` | `{method}` (algo name) | `sjfpo` |
| `group` | `{method}_{env}` | `sjfpo_Hopper-v5` |
| `name` | auto (wandb-generated; seed in config.seed) | `(generated)` |
| `job_type` | `smoke` (total_steps ≤ 10 000) or `train` | `smoke` |

## Metrics Logged

All metrics use `step=global_step` (default wandb x-axis, one commit per step).

| Prefix | Keys |
|---|---|
| `train/*` | `policy_loss`, `value_loss`, `kl_loss`, `total_loss`, `grad_norm_preclip`, `mean_log_ratio`, `flow_rho_mean/std`, `flow_rho_bin_range/lowfreq/highfreq/outside_frac`, `flow_v_star_rms`, `cond_temb/pooled/z_state/c`, `cond_gate_msa/mlp`, `cond_scale_msa/mlp` |
| `gae/*` | `advantages_mean`, `advantages_std`, `returns_mean`, `values_mean` |
| `eval/*` | `mean_return`, `min_return`, `max_return`, `std_return`, `mean_length`, `max_length`, `std_length`, `episode_count`, `random_baseline`, `consecutive_below`, `reward_improvement(+_pct)`, `episodes` Table, `video`, `demo_return`, `video_ok`, `diverged`, `diverge_step` |
| `timing/*` | `fps`, `elapsed_s` |
