# flow-policy-optimization — Flow-Matching Policy Optimization (FPO++)

**Code for:** *Flow Policy Gradients for Robot Control (FPO++)*  
**ArXiv:** https://arxiv.org/abs/2602.02481  
**Implementation:** Open-loop action chunking, PyTorch, online RL with CFM surrogate ratios and a configurable trust region (ASPO by default; PPO via `--flow_policy_optimization.trust_region_mode ppo`).

## What this repo does
Trains expressive, multimodal flow-based robot policies using online reinforcement learning without explicit action likelihoods. Key features:
- Per-token ratio clipping; trust-region mode selectable (`aspo` default, `ppo`, `spo`) (`src/flow_policy_optimization/flow_policy_optimization.py`)
- Chunked rollout with a 4-step open-loop horizon and time-limit-correct bootstrap on truncation
- Stored actions are the flow's raw output clamped to the unit box; the CFM target fits it in flow space (`action / actor_scale`). The env receives the affine unscale into its own bounds
- Flat-MLP actor over the flattened action chunk: window encoder + per-token timestep embedding into a SiLU MLP
- TD-MPC2-style critic trunk: LayerNorm + Mish, the action chunk enters through a learned affine so the chunk's offset survives
- Independent actor and critic gradient clipping
- `src/flow_policy_optimization/flow_policy_optimization.py` is the single-file implementation; `src/flow_policy_optimization/ppo.py` is a plain PPO baseline (`--method ppo`) for comparisons

## Repository layout
- `src/flow_policy_optimization/flow_policy_optimization.py` — the FPO++ implementation (single file)
- `src/flow_policy_optimization/ppo.py` — plain PPO baseline
- `scripts/train.py` — single-run training CLI; `scripts/play.py` — policy playback
- `configs/envs.yaml` — environment registry
- `tests/` — pytest suite
- `docs/benchmark-2026-11-17/` — raw eval CSVs and run metadata for the results below
- `flow_policy_optimization_online.md` / `.ipynb` — notebook walkthrough (jupytext pair; edit the `.md`, then `jupytext --sync`)

## Results (Hopper-v5, seed 0, strict fp32, CPU)

| method | steps | last-10 eval mean | best eval | fps |
|---|---|---|---|---|
| flow-policy-optimization | 100k | 160.8 | 200.6 | 439 |
| flow-policy-optimization | 500k | 156.9 | 208.9 | 387 |
| PPO (`--method ppo`) | 100k | 327.0 | 353.4 | 1688 |
| PPO (`--method ppo`) | 500k | 597.8 | 799.9 | 985 |

Honest status: this reimplementation trains end-to-end, but from scratch on
Hopper-v5 the FPO-family flow-PG trails Gaussian PPO. Our curve plateaus at
~150–210 (two partial collapses), matching the post-release literature's
diagnosis ("double drift", DiPOD, [arXiv 2606.13795](https://arxiv.org/abs/2606.13795);
FPO++ collapse on IsaacLab, FlowRL, [arXiv 2603.27450](https://arxiv.org/abs/2603.27450)).
The repo's value is the implementation and the honest benchmark, not a
state-of-the-art policy. See `FINDINGS.md` for the analysis and
`docs/benchmark-2026-11-17/` for the raw eval CSVs and run metadata.

## Quick start
```bash
uv sync                                              # installs flow-policy-optimization editable + deps
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000
.venv/bin/python scripts/train.py --method ppo --env Hopper-v5 --seed 0 --total_steps 2000  # plain PPO baseline
```

## Gates
```bash
.venv/bin/python -m pytest -q   # behaviour
uv run ruff check .            # dead code, unused names
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --env InvertedPendulum-v5 --seed 3407 --total_steps 2000 \
  --out_dir /tmp/flow_policy_optimization_smoke --flow_policy_optimization.report_to none   # runtime smoke
```
The `atol=0` behaviour pins were deleted on 2026-11-15. See
`SIMPLIFICATION_PLAN.md`, Amendment 1.

## Key papers
- FPO: `arXiv:2507.21053` (ICLR 2026)
- FPO++: `arXiv:2602.02481`
- Diffusion Policy: `arXiv:2303.04137`
- GoRL / GORL: `arXiv:2512.02581v1` (ICML 2026)

See `src/flow_policy_optimization/flow_policy_optimization.py` for how the whole implementation fits together.
