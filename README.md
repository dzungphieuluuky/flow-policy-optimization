# myfpo — Flow-Matching Policy Optimization (FPO++)

**Code for:** *Flow Policy Gradients for Robot Control (FPO++)*  
**ArXiv:** https://arxiv.org/abs/2602.02481  
**Implementation:** Open-loop action chunking, LeRobot / PyTorch, online RL with CFM surrogate ratios and asymmetric trust region (ASPO).

## What this repo does
Trains expressive, multimodal flow-based robot policies using online reinforcement learning without explicit action likelihoods. Key features:
- Per-sample ratio clipping (`FPO++`) and ASPO trust region (`fpo_core.py`)
- Chunked rollout (`rollout.py`) with boundary diagnostics (`rollout_seam_jerk`)
- Config parity (`tests/test_config_yaml_parity.py`) and metric faithfulness (`tests/test_metric_faithfulness.py`)
- Full diagnostic suite (`tests/test_guard_messages.py`: 76 guard-message snapshots)

## Quick start
```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/gen_sjfpo_config.py
```

## Key papers
- FPO: `arXiv:2507.21053` (ICLR 2026)
- FPO++: `arXiv:2602.02481`
- Diffusion Policy: `arXiv:2303.04137`
- GoRL / GORL: `arXiv:2512.02581v1` (ICML 2026)

See `literature_review.md` for full synthesis.
