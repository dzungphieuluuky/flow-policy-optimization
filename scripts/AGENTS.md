# AGENTS — scripts

## Overview
3 entry points: `train.py` (unified CLI) → `runner`/`baselines`; `build_notebook.py` (minimal Colab notebooks: clone + install + `!train.py`, no source inlined); `verify_deps.sh` (dep-pin gate, partly stale — see Notes).

## Structure
```
train.py            argparse (singular + plural flags) → matrix/single dispatch → build_cfg → run_single
build_notebook.py   title/md + clone/install/train cells → 2 source-free .ipynb (8 cells each)
verify_deps.sh      venv bootstrap + inline version asserts (STALE vs pyproject — see Notes)
```

## Where to Look
- `train.py:select_method_cfg` — CLI flags > flat-YAML overlay > dataclass defaults; only dataclass fields pass (`total_steps`/`eval_steps`/`save_steps`/`obs_dim`/`action_dim` are runner scalars, NOT config fields)
- `sweep.sh` axes → `sweep.scan_runs`/`run_pending` (reentrant: scans FS for done runs, dispatches pending sequentially)
- `build_notebook.py:CLONE_CELL` — getpass/GITHUB_TOKEN → `!git clone` (skipped if present) → `%cd`; notebooks never inline `src/`

## Conventions
- Precedence: CLI flags > `--config <yaml>` > dataclass defaults (`configs/{sjfpo,ppo,sac,shared}.yaml`; YAML keys ⊆ dataclass fields per `test_config_yaml_parity.py`)
- Exit codes: 0 ok · 1 usage · 2 runtime/divergence · 4 missing artifact
- `sys.path.insert(src)` in `train.py` (alt: `PYTHONPATH=src`); `.venv/bin/python` is the working interpreter (torch 2.14.0)
- Env switches: `SJFPO_TINY_MATRIX=1` (3-run matrix), `SJFPO_DISABLE_WANDB=1`

## Anti-Patterns
- DO NOT add runner-level scalars (`total_steps`, intervals, dims) as dataclass fields — they bypass config by design
- DO NOT extend `verify_deps.sh` asserts without syncing `pyproject.toml` (they already drifted — see Notes)
- DO NOT import `build_notebook.py` as a library (standalone script; zero importers)

## Commands
```bash
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000
METHODS="sjfpo ppo" ENVS=Hopper-v5 SEEDS="0 1" bash scripts/sweep.sh --python .venv/bin/python
PYTHONPATH=src .venv/bin/python scripts/build_notebook.py   # rebuilds both .ipynb
```

## Notes
- `verify_deps.sh` asserts torch>=2.2,<2.6 + numpy>=1.26,<2.1 but repo runs torch 2.14.0 / numpy 2.5.3 — running it fails; also references missing `requirements-local.txt`. Fix in a build-health pass, not here.
- `requirements-linux.txt:100` carries a machine-absolute `-e file://...` self-ref (ROCm box artifact).
- `.gitignore` ignores `*.ipynb` yet both root notebooks are tracked (force-added earlier) — new notebooks stay invisible to `git status`.
- No CI (no workflows/Makefile); `pytest -q` + notebook rebuild are the gates.
