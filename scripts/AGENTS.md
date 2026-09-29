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
- `.venv/bin/python` is the only interpreter. No `pip` in this venv; use `uv pip`.
- Env switches: `SJFPO_TINY_MATRIX=1`, `SJFPO_DISABLE_WANDB=1`.

## Anti-Patterns
- DO NOT add runner scalars to the dataclass. They bypass config by design.
- DO NOT extend `verify_deps.sh` without syncing `pyproject.toml`. Already drifted.
- DO NOT import `build_notebook.py` as a library. Standalone, zero importers.

## Commands
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/build_notebook.py
```

## Notes — stale, fix in a build-health pass
- `verify_deps.sh` asserts `torch>=2.2,<2.6` and `numpy<2.1`. Repo runs **torch 2.11.0+cu130, numpy 2.2.6**. Running it fails.
- **numpy 2.0 removed `np.float_`. `wandb` still calls it.** Default `report_to="wandb"` raises `AttributeError` at import. Always pass `--sjfpo.report_to tensorboard`. This also fails `tests/test_tracking.py::test_all_fans_out_to_every_backend`.
- `build_notebook.py` writes `spectral_joint_flow.ipynb`, which is not in the tree. The only tracked notebook is `sjfpo_online.ipynb`.
- `configs/shared.yaml` does not exist. Only `configs/envs.yaml` and `configs/sjfpo.yaml`.
- `tests/test_config_yaml_parity.py` does not exist, so YAML parity is **untested**.
- `configs/sjfpo.yaml` has orphaned indented `report_to` and `num_critics` lines near line 56.
- `requirements-linux.txt:100` carries a machine-absolute `file://` self-ref.
- `.gitignore` ignores `*.ipynb`; `sjfpo_online.ipynb` is tracked anyway. New notebooks stay invisible.
- No CI. `pytest -q` is the only gate.
