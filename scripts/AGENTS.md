# AGENTS — scripts

## Overview
- `train.py` — unified CLI, the real entry point
- `play.py` — load a checkpoint, render to video
- `gen_sjfpo_config.py` — regenerates `configs/sjfpo.yaml` from the dataclass defaults
- `verify_deps.sh` — dep-pin gate, **stale**, see Notes

## Where to Look
- `train.py:select_method_cfg` — CLI > flat YAML > dataclass defaults. Only dataclass fields pass. `total_steps` / `eval_steps` / `save_steps` / `obs_dim` / `action_dim` are runner scalars, NOT config fields.
- `train.py:run_single` — cross-checks registry dims in `configs/envs.yaml` against the live space, then calls `train_sjfpo_run`.
- `gen_sjfpo_config.py:render` — the SECTIONS table lists every emitted field in emission order. A new config field must be added there, or it lands in the trailing "Not emitted" list. Run the script after adding the field.

## Conventions
- Exit codes: `0` ok, `1` usage, `2` runtime or divergence, `4` missing artifact.
- `sys.path.insert(src)` in `train.py`. Alt: `PYTHONPATH=src`.
- `.venv/bin/python` is the only interpreter. There is no `pip` module in the venv and no `.venv/bin/pip`. Use `uv pip` for installs.

## Anti-Patterns
- DO NOT add runner scalars to the dataclass. They bypass config by design.
- DO NOT extend `verify_deps.sh` without syncing `pyproject.toml`. Already drifted.
- DO NOT hand-edit `configs/sjfpo.yaml`. It is generated, and `tests/test_config_yaml_parity.py` fails on any drift.
- DO NOT hand-edit `sjfpo_online.ipynb` expecting jupytext to follow. The tracked notebook carries Colab, Kaggle, and papermill metadata that `jupytext --sync` strips. Edit both files.

## Commands
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/gen_sjfpo_config.py
```

## Notes — verified 2026-10-15
- `verify_deps.sh` is broken three ways. This is the only open action item here.
  1. Line 22 installs `-r requirements-local.txt`. That file does not exist and is not tracked. This is the failure you actually get: exit 1, before any version check runs.
  2. Further down, the same script asserts `torch>=2.2,<2.6` and `numpy<2.1`. The repo runs **torch 2.11.0+cu130, numpy 2.2.6**, so both asserts would fail even if (1) were fixed.
  3. `pyproject.toml` sets no upper bound (`torch>=2.2.2`, `numpy>=1.26.4`), so the script, not the project, is what drifted.
- Deleted in `4bd45db`: `run_sweeps.sh`, `cuda_probe.py`, `build_notebook.py`. All three were stale. `run_sweeps.sh` passed removed flags (`critic_max_grad_norm`, `critic_ema_decay`), `cuda_probe.py` read metric keys that are never emitted, and `build_notebook.py` installed the deleted `requirements-linux.txt`.
- Deleted in `debd7e8`: `src/myfpo/baselines.py` and the `--baseline` flag. An FPO or FPO++ arm is now a hand-written `dataclasses.replace` in the caller. The Eq. 8 / Eq. 9 / Eq. 10 config mapping is pinned by `tests/test_ratio_axes.py::test_the_paper_configs_select_the_trust_regions_they_name`, so the paper configs stay auditable without the flag.
- No CI. `pytest -q tests` is the only gate.

## Verified non-issues — do not "fix" these
- **`wandb` under numpy 2 is fine.** `wandb 0.30.0` imports clean on `numpy 2.2.6`. `np.float_` is gone, but nothing in `src/` references it and `tests/test_tracking.py::test_all_fans_out_to_every_backend` passes. Default `report_to` is `"all"`, not `"wandb"`. Do not add a `--sjfpo.report_to` workaround.
- **YAML parity is tested.** `tests/test_config_yaml_parity.py` exists (3 tests: every YAML key is a real config field, YAML matches dataclass defaults, generator is deterministic).
- **`requirements-linux.txt` no longer exists** — deleted in `747c67c`. No `file://` self-ref anywhere; `requirements.txt` is 48 unpinned names.
- **`.gitignore` does not ignore `*.ipynb`.** New notebooks are visible to git.
- **Nothing references `configs/shared.yaml`,** so its absence costs nothing.
- **`requirements.txt` pins `jax[cuda13]`, not cuda12.** The `sjfpo_online.md` install cell matches it now. Note that plain `jax` also appears at line 6, so a naive grep for the CUDA extra finds both.