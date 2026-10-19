# myfpo — Flow-Matching Policy Optimization (FPO++)

**Code for:** *Flow Policy Gradients for Robot Control (FPO++)*  
**ArXiv:** https://arxiv.org/abs/2602.02481  
**Implementation:** Open-loop action chunking, LeRobot / PyTorch, online RL with CFM surrogate ratios and asymmetric trust region (ASPO).

## What this repo does
Trains expressive, multimodal flow-based robot policies using online reinforcement learning without explicit action likelihoods. Key features:
- Per-sample ratio clipping (`FPO++`) and ASPO trust region (`fpo_core.py`)
- Chunked rollout (`rollout.py`) with boundary diagnostics (`rollout_seam_jerk`)
- AdaLN-Zero DiT actor with causal attention, RoPE, and a log-spaced time embedding (`flow.py`)
- TD-MPC2-style critic: 5 members, LayerNorm + Mish, 1% dropout, REDQ min-of-2 targets (`critic.py`)
- Independent actor and critic gradient clipping (`training.py`)
- Config parity (`tests/test_config_yaml_parity.py`) and metric faithfulness (`tests/test_metric_faithfulness.py`)
- Full diagnostic suite (`tests/test_guard_messages.py`: 76 guarded conditions)

## Quick start
```bash
uv sync                                              # installs myfpo editable + deps
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/gen_config.py
```

## Gates
```bash
.venv/bin/python -m pytest -q   # behaviour
uv run ruff check .            # dead code, unused names
```

## Key papers
- FPO: `arXiv:2507.21053` (ICLR 2026)
- FPO++: `arXiv:2602.02481`
- Diffusion Policy: `arXiv:2303.04137`
- GoRL / GORL: `arXiv:2512.02581v1` (ICML 2026)

See `docs/architecture.md` for how each network is built and why, and
`literature_review.md` for the full synthesis.
