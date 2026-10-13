# AGENTS — scripts

## Overview
- `train.py` — unified CLI, the real entry point
- `play.py` — load a checkpoint, render to video
- `build_notebook.py` — writes `spectral_joint_flow.ipynb`
- `cuda_probe.py` — GPU probe
- `verify_deps.sh` — dep-pin gate, **stale**, see Notes

## Where to Look
- `train.py:select_method_cfg` — CLI > flat YAML > dataclass defaults. Only dataclass fields pass. `total_steps` / `eval_steps` / `save_steps` / `obs_dim` / `action_dim` are runner scalars, NOT config fields.
- `train.py:run_single` — cross-checks registry dims in `configs/envs.yaml` against the live space, then calls `train_sjfpo_run`.
- `build_notebook.py:embedded_files` — recreates `src/`, `scripts/train.py`, `configs/`. Rebuild after touching any of them.

## Conventions
- Exit codes: `0` ok, `1` usage, `2` runtime or divergence, `4` missing artifact.
- `sys.path.insert(src)` in `train.py`. Alt: `PYTHONPATH=src`.
- `.venv/bin/python` is the only interpreter. `uv pip` is preferred for installs, but `.venv/bin/pip` does exist and works.

## Anti-Patterns
- DO NOT add runner scalars to the dataclass. They bypass config by design.
- DO NOT extend `verify_deps.sh` without syncing `pyproject.toml`. Already drifted.
- DO NOT import `build_notebook.py` as a library. Standalone, zero importers.

## Commands
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/build_notebook.py
```

## Notes — verified 2026-10-13
- `verify_deps.sh` is broken three ways. This is the only open action item here.
  1. Line 22 installs `-r requirements-local.txt`. That file does not exist and is not tracked. This is the failure you actually get: exit 1, before any version check runs.
  2. Further down, the same script asserts `torch>=2.2,<2.6` and `numpy<2.1`. The repo runs **torch 2.11.0+cu130, numpy 2.2.6**, so both asserts would fail even if (1) were fixed.
  3. `pyproject.toml` sets no upper bound (`torch>=2.2.2`, `numpy>=1.26.4`), so the script, not the project, is what drifted.
- `build_notebook.py` writes `spectral_joint_flow.ipynb`, which is not in the tree. The only tracked notebook is `sjfpo_online.ipynb`.
- No CI. `pytest -q tests` is the only gate.

## Verified non-issues — do not "fix" these
- **`wandb` under numpy 2 is fine.** `wandb 0.30.0` imports clean on `numpy 2.2.6`. `np.float_` is gone, but nothing in `src/` references it and `tests/test_tracking.py::test_all_fans_out_to_every_backend` passes. Default `report_to` is `"all"`, not `"wandb"`. Do not add a `--sjfpo.report_to` workaround.
- **YAML parity is tested.** `tests/test_config_yaml_parity.py` exists (3 tests: every YAML key is a real config field, YAML matches dataclass defaults, generator is deterministic).
- **`requirements-linux.txt` no longer exists** — deleted in `747c67c`. No `file://` self-ref anywhere; `requirements.txt` is 48 unpinned names.
- **`.gitignore` does not ignore `*.ipynb`.** New notebooks are visible to git.
- **Nothing references `configs/shared.yaml`,** so its absence costs nothing.
