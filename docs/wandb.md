# wandb Logging (Enabled by Default)

wandb integration is **required and on by default**. Every training run logs to
wandb unless explicitly disabled — the only off switch is `MFPO_DISABLE_WANDB=1`.
A failed `wandb.init` (missing key, no network) is a **fatal error**, never a
silent no-op.

## Install

wandb ships with the main requirements (no separate install step):

```bash
pip install -r requirements.txt        # includes wandb
```

## `MFPO_DISABLE_WANDB` — the Only Off Switch

| Value | Behaviour |
|---|---|
| *(unset)* | Enabled — `wandb.init` runs in `online` mode (requires `wandb login`) |
| `"1"` | Disabled — all helpers no-op, zero overhead |
| anything else | Enabled — only exactly `"1"` disables |

`WANDB_MODE=offline` (wandb-native) captures offline runs; sync later with
`wandb sync wandb/run-<id>/`.

> Historical note: the old `MFPO_WANDB` tri-state gate (`unset`/`0`/`offline`/`online`)
> is deleted — `MFPO_DISABLE_WANDB=1` is the single source of truth.

## Usage

```bash
# Online training (default)
python scripts/train.py \
  --method mfpo --env Hopper-v5 --total-steps 5000 \
  --out-dir runs/train

# Offline capture (wandb-native mode, still requires the package)
WANDB_MODE=offline python scripts/train.py --matrix \
  --methods mfpo ppo sac --envs Hopper-v5 --seeds 0 --total-steps 500 \
  --out-dir runs/smoke

# Disabled (the ONLY opt-out)
MFPO_DISABLE_WANDB=1 python scripts/train.py --method mfpo --env Hopper-v5 ...
```

## Syncing Offline Runs

After an offline run, push artifacts to wandb cloud:

```bash
wandb sync wandb/run-<id>/
```

## API Surface (6 fns, SDK-direct, 159 LOC)

`src/mfpo/wandb_logging.py` is the sole file that imports wandb (module-level
REQUIRED import; absent package is a hard ImportError). 6 public
functions (5 core + the shared eval-summary helper from fn-consolidation C4),
all calling the wandb SDK directly (`wandb.init`/`log`/`finish`/
`Table`/`define_metric`), no wrappers:

| Function | Signature | Behaviour |
|---|---|---|
| `wandb_disabled` | `() -> bool` | `True` only when `MFPO_DISABLE_WANDB == "1"` |
| `init_wandb_run_for_method` | `(*, method, env_name, seed, total_steps, cfg_dict) -> bool` | SDK-direct `wandb.init(project=method, group=f"{method}_{env_name}", name=None, tags=[method, "mujoco", "cpu-first"], job_type="smoke" if total_steps <= 10000 else "train", mode=$WANDB_MODE, reinit="finish_previous", force=True)` + global `wandb.define_metric("*", step_metric="global_step")`; returns `True` when enabled, `False` when disabled; init errors propagate (fail-fast, no try/except) |
| `log_metrics` | `(data, step) -> None` | Thin disabled-guard around `wandb.log(data, step=step)` |
| `log_eval_details` | `(eval_results, all_eval_returns, global_step) -> None` | Per-episode `wandb.Table(columns=["episode", "return", "length"], data=[list(r) for r in rows])` (list-of-lists; tuple rows normalize identically per probe) + `reward_improvement_pct = (curr - prev) / max(abs(prev), 1e-8)` via `log_metrics` |
| `log_eval_summary` | `(mean_return, eval_results, all_eval_returns, eval_count, global_step) -> None` | Shared runner/baselines eval payload (mean/length/max/std/count aggregates, then `log_eval_details`); single home per fn-consolidation C4 |
| `finish_run` | `(exit_code=0) -> None` | Thin disabled-guard around `wandb.finish(exit_code=exit_code)` |

Callers (`runner.py`, `baselines.py`) import at most 6 symbols and hold a
plain `bool` (`_wandb_enabled = init_wandb_run_for_method(...)`); `wl.wandb` stays
settable for tests. Three obsolete helpers (the old init/table/run wrappers)
are deleted (zero hits repo-wide).

## Dashboard Structure

| Field | Pattern | Example |
|---|---|---|
| `project` | `{method}` (algo name) | `mfpo` |
| `group` | `{method}_{env}` | `mfpo_Hopper-v5` |
| `name` | auto (wandb-generated; seed in config.seed) | `(generated)` |
| `job_type` | `smoke` (total_steps ≤ 10 000) or `train` | `smoke` |

## Metrics Logged

All metrics use `step=global_step`.

| Prefix | Keys |
|---|---|
| `train/*` | `policy_loss`, `value_loss`, `aux_loss`, `sigreg_loss`, `clip_fraction`, `mean_log_ratio`, `clamp_engagement`, `total_loss` |
| `gae/*` | `advantages_mean`, `advantages_std`, `returns_mean`, `values_mean` |
| `eval/*` | `mean_return`, `mean_length`, `max_return`, `std_return`, `episode_count` |
| `timing/*` | `fps`, `elapsed_s` |
