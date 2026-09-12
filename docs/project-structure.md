# Project Structure

## Top-Level Tree

```
.
├── .omo/                         # plans, evidence (see .omo/plans/oversized-split-proposals.md for split roadmap)
├── configs/                      # shared/sjfpo/ppo/sac YAML defaults (mirror config.py)
├── docs/                         # ← you are here
├── scripts/
│   ├── build_notebook.py         # generates both ipynbs from src/*.py (jupytext + CELLORDER)
│   ├── train.py                  # unified trainer: single-run + --matrix sweep (sjfpo/ppo/sac)
│   └── verify_deps.sh            # pinned dep check
├── src/sjfpo/                     # production code (MuJoCo-only, flat Box, fp32, no autocast)
│   ├── config.py                 # flags, RUN_MATRIX (45), SJFPO/PPO/SAC configs, build_filtered_matrix
│   ├── runner.py                 # train_run (SJFPO entry)
│   ├── trainer.py                # assemble_update (ASPO + value + Epps-Pulley SIGReg)
│   ├── networks.py               # StateEncoders, SpectralEncoder, DiT flow, Critic, sigreg_loss
│   ├── dct_utils.py              # orthonormal DCT-II/III action chunks
│   ├── evaluator.py              # Evaluator → 3-col CSV [episode,return,length]
│   ├── baselines.py              # skrl PPO/SAC (shared GaussianPolicyImpl + baseline_train_loop)
│   ├── fpo_core.py               # legacy ASPO + euler_sample 1→0 descending
│   ├── gae.py / collector.py / checkpointing.py  # legacy RL loop
│   ├── sweep.py                  # scan_runs / run_pending / format_run_directory_name
│   ├── analysis.py               # analyze_results (bootstrap 2000, IQM — stats locked)
│   ├── wandb_logging.py          # hard REQUIRED wandb gate (only file importing wandb)
│   ├── utils.py                  # safe_float_value
│   └── __init__.py
├── tests/                        # 263 green; test_pipeline_cpu.py (9 invariants) never touched
├── spectral_joint_flow.ipynb     # 61 cells, self-contained, Kaggle-ready (primary deliverable)
├── flow_matching_vs_ppo_sac.ipynb # 47 cells, SJFPO vs PPO/SAC comparison
└── requirements-local.txt        # CPU-first pins (+ pyyaml for configs/)
```

## File Roles at a Glance

| Path | What it does | When you touch it |
|---|---|---|
| `src/sjfpo/config.py` | Single source of truth for flags, matrices, budgets, algorithm configs | Changing sweep size, budgets, `D/spectral_dim/K/H/N` |
| `src/sjfpo/networks.py` | Encoders + spectral + flow + SIGReg | Any shape/dtype or conditioning change |
| `src/sjfpo/runner.py` | `train_run` entry point | Adding an env |
| `scripts/build_notebook.py` | Notebook assembler (`CELLORDER`, jupytext) | Cell order, markdown docs, build bugs |
| `scripts/train.py` | Unified CLI (single / `--matrix` / `--tiny`) | CLI flags, dispatch |

## Ignored / Generated

- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` — local only.
- `.omo/evidence/` is git-ignored; `*.pt` checkpoints never committed.
- `*.ipynb` are generated — never hand-edit, rebuild via `build_notebook.py`.
