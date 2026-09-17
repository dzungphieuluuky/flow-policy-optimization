# Project Structure

## Top-Level Tree

```
.
├── .omo/                         # local plans, evidence (git-ignored)
├── configs/                      # shared/sjfpo/ppo/sac YAML defaults (mirror config.py)
├── docs/                         # ← you are here
├── scripts/
│   ├── build_notebook.py         # generates both minimal Colab ipynbs (clone + install + !train.py)
│   ├── train.py                  # single-run CLI (sweeps live in sweep.sh)
│   └── verify_deps.sh            # pinned dep check
├── src/sjfpo/                     # production code (MuJoCo-only, flat Box, fp32 storage + bf16 autocast)
│   ├── config.py                 # flags, RUN_MATRIX (45), PPO/SAC configs
│   ├── configuration_sjfpo.py      # SJFPO policy config (sole hyperparam source)
│   ├── modeling_sjfpo.py         # SJFPOPolicy (ASPO + value update, owns all modules)
│   ├── train_sjfpo.py            # online train entry used by scripts/train.py
│   ├── networks.py               # WindowEncoder (owned instances), StateConditioner, DiT flow, V(s) Critic
│   ├── dct_utils.py              # orthonormal DCT-II/III action chunks
│   ├── evaluator.py              # Evaluator → 3-col CSV [episode,return,length]
│   ├── baselines.py              # skrl PPO/SAC (shared GaussianPolicyImpl + baseline_train_loop)
│   ├── fpo_core.py               # ASPO + sample(mode=euler) 0→1 ascending
│   ├── gae.py / collector.py / checkpointing.py  # legacy RL loop
│   ├── sweep.py                  # scan_runs / run_pending / format_run_directory_name
│   ├── analysis.py               # analyze_results (bootstrap 2000, IQM — stats locked)
│   ├── wandb_logging.py          # hard REQUIRED wandb gate (only file importing wandb)
│   ├── utils.py                  # device/dtype/compile helpers
│   └── __init__.py
├── tests/                        # 300 green; test_pipeline_cpu.py (5 tests) never touched
├── spectral_joint_flow.ipynb     # 8 cells, clone + train (primary deliverable)
├── flow_matching_vs_ppo_sac.ipynb # 8 cells, sweep.sh + results + results
├── requirements-linux.txt        # Linux pins (+ pyyaml for configs/)
└── requirements-amd.txt          # AMD/ROCm pins
```

## File Roles at a Glance

| Path | What it does | When you touch it |
|---|---|---|
| `src/sjfpo/config.py` | Single source of truth for flags, matrices, budgets, algorithm configs | Changing sweep size, budgets, `D/K/H/N` |
| `src/sjfpo/networks.py` | Encoders + spectral + flow | Any shape/dtype or conditioning change |
| `src/sjfpo/modeling_sjfpo.py` | `SJFPOPolicy` entry point | Adding an env |
| `scripts/build_notebook.py` | Minimal Colab notebooks (clone/install/train cells) | Cell contents, repo slug |
| `scripts/train.py` | Single-run CLI (underscores, draccus) | CLI flags, dispatch |

## Ignored / Generated

- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` — local only.
- `.omo/evidence/` is git-ignored; `*.pt` checkpoints never committed.
- `*.ipynb` are generated — never hand-edit, rebuild via `build_notebook.py`.
