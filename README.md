# myfpo — Flow-Matching Policy Optimization (FPO++)

**Code for:** *Flow Policy Gradients for Robot Control (FPO++)*  
**ArXiv:** https://arxiv.org/abs/2602.02481  
**Implementation:** Open-loop action chunking, PyTorch, online RL with CFM surrogate ratios and a configurable trust region (ASPO by default; PPO via `--myfpo.trust_region_mode ppo`).

## What this repo does
Trains expressive, multimodal flow-based robot policies using online reinforcement learning without explicit action likelihoods. Key features:
- Per-token ratio clipping; trust-region mode selectable (`aspo` default, `ppo`, `spo`) (`myfpo.py`)
- Chunked rollout with a 4-step open-loop horizon and time-limit-correct bootstrap on truncation (`myfpo.py`)
- Stored actions are the flow's raw output clamped to the unit box; the CFM target fits it in flow space (`action / actor_scale`). The env receives the affine unscale into its own bounds (`myfpo.py`)
- Flat-MLP actor over the flattened action chunk: window encoder + per-token timestep embedding into a SiLU MLP (`myfpo.py`)
- TD-MPC2-style critic trunk: LayerNorm + Mish, the action chunk enters through a learned affine so the chunk's offset survives (`myfpo.py`)
- Independent actor and critic gradient clipping
- `myfpo.py` is the single-file implementation; `ppo.py` is a plain PPO baseline (`--method ppo`) for comparisons

## Quick start
```bash
uv sync                                              # installs myfpo editable + deps
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/train.py --method ppo --env Hopper-v5 --seed 0 --total_steps 2000  # plain PPO baseline
```

## Gates
```bash
.venv/bin/python -m pytest -q   # behaviour
uv run ruff check .            # dead code, unused names
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --env InvertedPendulum-v5 --seed 3407 --total_steps 2000 \
  --out_dir /tmp/myfpo_smoke --myfpo.report_to none   # runtime smoke
```
The `atol=0` behaviour pins were deleted on 2026-11-15. See
`SIMPLIFICATION_PLAN.md`, Amendment 1.

## Key papers
- FPO: `arXiv:2507.21053` (ICLR 2026)
- FPO++: `arXiv:2602.02481`
- Diffusion Policy: `arXiv:2303.04137`
- GoRL / GORL: `arXiv:2512.02581v1` (ICML 2026)

See `src/myfpo/myfpo.py` for how the whole implementation fits together.
