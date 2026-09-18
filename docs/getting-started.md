# Getting Started

## Prerequisites

- Python ≥ 3.12 (`pyproject.toml` requires it)
- `pip` + venv (`.venv` already present; `requirements-linux.txt` is the Linux
  pin set — `verify_deps.sh` checks pins, though it is partly stale, see Notes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# if you need full deps: pip install -r requirements-linux.txt  # or: pip install -e .
```

## 5-Minute Smoke Run

```bash
# 0. From repo root
cd spectral-joint-policy

# 1. All tests (300 passed)
PYTHONPATH=src .venv/bin/python -m pytest -q

# 2. MuJoCo single-run smoke (2000 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000 --out_dir .omo/evidence/smoke
# Artifacts: .omo/evidence/smoke/sjfpo_Hopper-v5_seed0/checkpoint_*/model.safetensors + eval_history.csv

# 3. Matrix smoke (sweep.sh axes)
METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" TOTAL_STEPS=2000 bash scripts/sweep.sh --out-base .omo/evidence/smoke --python .venv/bin/python

# 4. Notebook (standalone, checked)
python scripts/build_notebook.py
ls -lh spectral_joint_flow.ipynb
```

## Config Flags (env vars)

| Env var | Default | Effect |
|---|---|---|
| `SJFPO_TINY_MATRIX` | `0` | `1` → tiny preset (Hopper-v5 × seed 0, short budgets) |
| `SJFPO_DISABLE_WANDB` | unset (enabled) | `1` → disable wandb logging (only off switch; wandb is required) |
| `IS_KAGGLE` | autodetect | Kaggle run-output dirs; local stays synthetic — no download |

These mirror the axes `RUN_MATRIX` enumerates — see [Config & Flags](config-flags.md).

## Notebooks on Colab

- Open the `.ipynb` from GitHub in Colab, paste a PAT when prompted
  (or set the `GITHUB_TOKEN` secret), run top to bottom.
- The notebook clones this repo, installs `requirements-linux.txt`, then
  calls `scripts/train.py` — eval runs inside training.

## Where to Look Next

- Tree + file roles → [Project Structure](project-structure.md)
- Shapes + flow diagram → [Architecture](architecture.md)
- Per-file jump table → [Module Guide](modules.md)
- Full flag matrix → [Config & Flags](config-flags.md)
