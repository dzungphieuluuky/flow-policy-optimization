# Ablation / Comparison Reference

Based on `literature_review.md` (Sections 2, 5, 11.4) and repo implementation (`fpo_core.py` with the metrics, `rollout.py` with the collector and gae).

## Method taxonomy (repo-relevant)

| Category | Representative | Key feature | Limitation | When to use |
|---|---|---|---|---|
| Gaussian RL | PPO / TRPO | Tractable likelihood | Unimodal, diagonal cov | Low-D, real-time |
| Offline Diffusion | Diffusion Policy (`arXiv:2303.04137`) | Multimodal BC | No online RL | Expert demo available |
| End-to-End Flow RL | FPO (`arXiv:2507.21053`), FPO++ (`arXiv:2602.02481`), `myfpo` | CFM surrogate ratio, ASPO | Approximate ratio (see `fpo_core.py:265`) | Multimodal online robot RL |
| Latent-Decoder | GoRL / GORL (`arXiv:2512.02581v1`) | Decoupled optimization | Two-timescale tuning | Stable generative RL |
| One-Step / Drift | OGPO / DBPO (`arXiv:2604.03540`) | Real-time inference (`1-NFE`) | Reduced expressivity | Latency-critical control |
| Inference Planning | GPC (`IEEE LRA 2026`) | Frozen policy + world model | No policy improvement | Test-time adaptation |

## Design trade-offs in `myfpo`
- **Approximate ratio** (`compute_policy_ratio`): Enables end-to-end flow policy without invertible architecture. Cost: biased proxy (documented at `fpo_core.py:265`).
- **Open-loop chunking** (`rollout.py`): Amortizes inference over `action_horizon`. Cost: chunk boundary artifacts (`rollout_seam_jerk`).
- **Per-token ratios** (`FPO++`): Improves scratch training. The repo default is per token (`do_average_cfm_loss_in_chunk=False`). Cost: doubles MC-sample compute.
- **Asymmetric clipping** (`ASPO`): Preserves entropy for `A < 0`. Cost: requires separate clipping thresholds (`cfm_loss_clamp_negative_advantages_max`).

## Ready-made ablations

The config is the `MyfpoConfig` dataclass, so every knob is a `--myfpo.` flag.

**Cross-attention critic instead of the concat head.** The state window enters
as one token per frame, the action chunk as one token per step. Each stream
queries the other, then the last state token and the mean action token go to
the same `Linear -> LayerNorm -> Mish` MLP. The action tokens carry no norm, so
a constant chunk keeps its offset. No mask: the critic reads a chunk it has
already executed. Costs about `8 * hidden_dim**2` parameters per member.

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --myfpo.critic_arch cross_attn
```

**No critic dropout.** TD-MPC2 puts 1% after the first linear to stop the
ensemble converging onto one function. Every member here regresses onto the
same scalar target, so initialisation is the only thing separating them. Watch
`train/value_pred_member_gap_p95` against the default.

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --myfpo.critic_dropout 0.0
```

**One critic instead of the 2-member ensemble.** `subsampled_min` falls back to
a full min when the ensemble is no wider than the draw, so the backup becomes a
single Q. Matches QC, which runs a single Q.

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --myfpo.num_critics 1
```

**The released 5-member ensemble.** The repo's default is 2 members; TD-MPC2
ships 5. `train/value_pred_member_gap_p95` is the diagnostic that says whether
the wider ensemble is worth its parameters.

```bash
.venv/bin/python scripts/train.py --env Hopper-v5 --myfpo.num_critics 5
```

See `literature_review.md` Section 5 (table) and Section 11.4 (detailed comparison) for full citations and mathematical formulations.
