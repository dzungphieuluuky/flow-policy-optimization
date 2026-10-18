# AGENTS — scripts

## Overview
- `train.py` — unified CLI, the real entry point
- `play.py` — load a checkpoint, render to video
- `gen_sjfpo_config.py` — regenerates `configs/sjfpo.yaml` from the dataclass defaults
- `verify_deps.sh` — dep-pin gate, **stale**, see Notes

## Where to Look
- `train.py:apply_flat_yaml(args.sjfpo, ...)` — CLI > flat YAML > dataclass defaults. Only dataclass fields pass. `total_steps` / `eval_steps` / `save_steps` / `obs_dim` / `action_dim` are runner scalars, NOT config fields.
- `train.py:run_single` — cross-checks registry dims in `configs/envs.yaml` against the live space, then calls `train_sjfpo_run`.
- `gen_sjfpo_config.py:render` — the SECTIONS table lists every emitted field in emission order. A new config field must be added there, or it lands in the trailing "Not emitted" list. Run the script after adding the field.

## Conventions
- Exit codes: `0` ok, `1` usage, `2` runtime error, `4` missing artifact.
- `sys.path.insert(src)` in `train.py`. Alt: `PYTHONPATH=src`.
- `.venv/bin/python` is the only interpreter. There is no `pip` module in the venv and no `.venv/bin/pip`. Use `uv pip` for installs.

## Anti-Patterns
- DO NOT add runner scalars to the dataclass. They bypass config by design.
- DO NOT extend `verify_deps.sh` without syncing `pyproject.toml`. Already drifted.
- DO NOT hand-edit `configs/sjfpo.yaml`. It is generated, and `tests/test_config_yaml_parity.py` fails on any drift.
- DO NOT hand-edit `sjfpo_online.ipynb`. It is generated from `sjfpo_online.md` by jupytext. Edit the md and run `jupytext --sync`. Editing the ipynb directly is lost on the next sync.

## Commands
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/gen_sjfpo_config.py
.venv/bin/python -m jupytext --sync sjfpo_online.md
```

## Notes — verified 2026-10-15
- `verify_deps.sh` is broken three ways. This is the only open action item here.
  1. Line 22 installs `-r requirements-local.txt`. That file does not exist and is not tracked. This is the failure you actually get: exit 1, before any version check runs.
  2. Further down, the same script asserts `torch>=2.2,<2.6` and `numpy<2.1`. The repo runs **torch 2.11.0+cu130, numpy 2.2.6**, so both asserts would fail even if (1) were fixed.
  3. `pyproject.toml` sets no upper bound (`torch>=2.2.2`, `numpy>=1.26.4`), so the script, not the project, is what drifted.
- Deleted in `4bd45db`: `run_sweeps.sh`, `cuda_probe.py`, `build_notebook.py`. All three were stale. `run_sweeps.sh` passed removed flags (`critic_max_grad_norm`, `critic_ema_decay`), `cuda_probe.py` read metric keys that are never emitted, and `build_notebook.py` installed the deleted `requirements-linux.txt`.
- Deleted in `debd7e8`: `src/myfpo/baselines.py` and the `--baseline` flag. An FPO or FPO++ arm is now a hand-written `dataclasses.replace` in the caller. The Eq. 8 / Eq. 9 / Eq. 10 config mapping is pinned by `tests/test_ratio_axes.py::test_the_paper_configs_select_the_trust_regions_they_name`, so the paper configs stay auditable without the flag.
- **The notebook pair is jupytext-generated.** Since `1e50aae`, `sjfpo_online.ipynb` is produced from `sjfpo_online.md` by `jupytext --sync`. The sync is idempotent, so a second run reports "Unchanged". Per-cell papermill timings and UUIDs are dropped by design (`cell_metadata_filter: -all`); notebook-level Colab, Kaggle, and papermill metadata survive.
- No CI. `pytest -q tests` is the only gate.

## Verified non-issues — do not "fix" these
- **`wandb` under numpy 2 is fine.** `wandb 0.30.0` imports clean on `numpy 2.2.6`. `np.float_` is gone, but nothing in `src/` references it and `tests/test_tracking.py::test_all_fans_out_to_every_backend` passes. Default `report_to` is `"all"`, not `"wandb"`. Do not add a `--sjfpo.report_to` workaround.
- **YAML parity is tested.** `tests/test_config_yaml_parity.py` exists (3 tests: every YAML key is a real config field, YAML matches dataclass defaults, generator is deterministic).
- **`requirements-linux.txt` no longer exists** — deleted in `747c67c`. No `file://` self-ref anywhere; `requirements.txt` is 48 unpinned names.
- **`.gitignore` does not ignore `*.ipynb`.** New notebooks are visible to git.
- **Nothing references `configs/shared.yaml`,** so its absence costs nothing.
- **`requirements.txt` pins `jax[cuda13]`, not cuda12.** The `sjfpo_online.md` install cell matches it now. Note that plain `jax` also appears at line 6, so a naive grep for the CUDA extra finds both.
- `literature_review.md` (new): full synthesis of flow-matching robot RL landscape (FPO, FPO++, GoRL, GORL, GenPO, OGPO, DBPO, GPC, PSS) with comparison table, reading roadmap, and dependency graph. Referenced by `README.md` and `docs/ablation_reference.md`.
## Removed 2026-10-17 — do not reintroduce without a trigger
- `check_divergence`, `check_regression`, and `interval_straddles` are gone from
  `evaluator.py`. A regression flag was the only caller of the LR decay, so the
  decay went with it: `apply_lr_schedule(optimizer, step_count)` now takes no
  `lr_decays` and multiplies by nothing.
- Config fields `stop_return_frac`, `stop_patience`, `lr_decay`, and
  `max_lr_decays` are gone from `SJFPOConfig` and from the generator SECTIONS.
  `configs/sjfpo.yaml` was regenerated.
- The eval loop has **no early exit**. A run ends at `total_steps` or on a
  non-finite loss, which raises `SJFPOAborted`. The log keys `eval/diverged`,
  `eval/regressed`, `eval/diverge_step`, `eval/regression_step`, and
  `train/lr_decays` are gone.
- `stop_eval_sampling` **stayed**. It no longer selects a stopper, it selects the
  distribution that decides `checkpoints/best`. Do not delete it with the guards.
- `tests/test_eval_interval.py` lost 19 tests, `test_tracking.py` lost 4,
  `test_metric_audit.py` lost 1, `test_fpo_parity.py` lost 1. The interval and
  bootstrap tests stayed, because those functions stayed.
