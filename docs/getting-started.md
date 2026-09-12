# Getting Started

## Prerequisites

- Python 3.10+ (repo pinned and tested on 3.10)
- `pip` + venv (`.venv` already present; `requirements-local.txt` is CPU-first —
  `verify_deps.sh` checks pins)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
# if you need full deps: pip install -r requirements-local.txt  # or follow verify_deps.sh
```

## 5-Minute Smoke Run

```bash
# 0. From repo root
cd /run/media/dung-pham/DATA/Personal/fourier-transform-research

# 1. All tests (216 passed)
PYTHONPATH=src .venv/bin/python -m pytest -q

# 2. MuJoCo single-run smoke (200 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --total-steps 200 --out-dir .omo/evidence/smoke
# Artifacts: .omo/evidence/smoke/sjfpo/checkpoint_*/checkpoint.pt + eval_history.csv

# 3. Matrix smoke (explicit filters)
PYTHONPATH=src .venv/bin/python scripts/train.py --matrix --methods sjfpo ppo sac --envs Hopper-v5 --seeds 0 --total-steps 200 --out-dir .omo/evidence/smoke

# 4. Notebooks (both, validated)
python scripts/build_notebook.py
ls -lh spectral_joint_flow.ipynb flow_matching_vs_ppo_sac.ipynb
```

## Config Flags (env vars)

| Env var | Default | Effect |
|---|---|---|
| `SJFPO_TINY_MATRIX` | `0` | `1` → tiny preset (Hopper-v5 × seed 0, short budgets) |
| `SJFPO_DISABLE_WANDB` | unset (enabled) | `1` → disable wandb logging (only off switch; wandb is required) |
| `IS_KAGGLE` | autodetect | Kaggle run-output dirs; local stays synthetic — no download |
| `SJFPO_RESUME_ONLY` | `0` | Sweep cells skip `MISSING` runs when set |

These mirror the axes `RUN_MATRIX` enumerates — see [Config & Flags](config-flags.md).

## Notebooks on Kaggle

- Upload **only** `spectral_joint_flow.ipynb` — it is self-contained (no `src/`
  needed; the builder proves tmpdir self-containment).
- It branches on `IS_KAGGLE` automatically; real-data cells are opt-in.
- Checkpoint helper `push_checkpoints_to_dataset()` is marked optional — wire
  your dataset slug before using.

## Where to Look Next

- Tree + file roles → [Project Structure](project-structure.md)
- Shapes + flow diagram → [Architecture](architecture.md)
- Per-file jump table → [Module Guide](modules.md)
- Full flag matrix → [Config & Flags](config-flags.md)
