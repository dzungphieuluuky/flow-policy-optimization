# myfpo — Flow-Matching Policy Optimization (FPO++)

**Code for:** *Flow Policy Gradients for Robot Control (FPO++)*  
**ArXiv:** https://arxiv.org/abs/2602.02481  
**Implementation:** Open-loop action chunking, PyTorch, online RL with CFM surrogate ratios and a configurable trust region (PPO by default; ASPO via `--myfpo.trust_region_mode aspo`).

## What this repo does
Trains expressive, multimodal flow-based robot policies using online reinforcement learning without explicit action likelihoods. Key features:
- Per-token ratio clipping by default; trust-region mode selectable (`ppo` default, `spo`, `aspo`) (`flow.py`)
- Chunked rollout (`collect.py`) with boundary diagnostics (`rollout_seam_jerk`)
- Stored actions are the flow's raw output; the CFM target fits them in flow space (`action / actor_scale`), with no separate action normalizer. The squash into the env bounds happens on the way to the env (see `actions.py` and `algo.py`)
- AdaLN-Zero DiT actor with causal attention, RoPE, and a log-spaced time embedding (`nn.py`)
- TD-MPC2-style critic: 2 members by default (`--myfpo.num_critics 5` for the full ensemble), LayerNorm + Mish, 1% dropout, REDQ min-of-2 targets (`nn.py`). The action chunk enters through a learned affine, so the chunk's offset survives
- Independent actor and critic gradient clipping (`train.py`)
- Metric faithfulness (`tests/test_metric_faithfulness.py`)
- Full diagnostic suite (`tests/test_guard_messages.py`: 66 guarded conditions)

## Quick start
```bash
uv sync                                              # installs myfpo editable + deps
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
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

See `nn.py` and `config.py` for how each network is built and why.
