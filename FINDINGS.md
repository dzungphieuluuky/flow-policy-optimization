# flow_policy_optimization — findings from the FPO++ reimplementation (2026-11-17/18)

Snapshot of what was built, measured, and learned. HEAD at time of writing:
`a09d016` (branch `main`, unpushed by 2 commits). All numbers are single-seed
(`--seed 0`) CPU/fp32 runs; see caveats.

## TL;DR

- Reimplemented a minimal, single-file FPO++ (Flow Policy Gradients for Robot
  Control, [arXiv 2602.02481](https://arxiv.org/abs/2602.02481)) in PyTorch,
  benchmarked it against a repaired PPO baseline on Hopper-v5.
- **PPO wins decisively on this benchmark**: at 100k steps PPO is ~2× flow_policy_optimization in
  return (last-10 eval mean 327 vs 161); at 500k steps the gap is ~3.2×
  (598 vs 157). PPO was also ~2.3–4.3× faster per step (1690/985 vs 439/387 fps).
- flow_policy_optimization **plateaus at ~150–210 return** after ~100k steps and shows two
  instability dips (~254k, ~443–467k steps) that partially recover — the exact
  failure mode the post-release literature diagnoses ("double drift", DiPOD;
  FPO++ collapse, FlowRL).
- This is a **research artifact**, not a path to a stronger Hopper policy.

## Engineering work (all committed)

Phase order and commits (single-file `src/flow_policy_optimization/flow_policy_optimization.py`, CleanRL-style):

- `5c6426d` phase 0 — restored the red gates: 7 stale test modules rewritten,
  error messages restored, PPO baseline repaired (`NameError: backends`; eval
  crashed at first eval), ruff green.
- `7766c7c` phase 1 — S1 correctness: Polyak target update per gradient step
  (was per rollout), rollout budget in **chunks** not env steps (idle-row
  bug), obs-normalizer updated inside `run_rollout` before the behavior loss,
  CFM `"sqrt"` reduction was really a masked mean (renamed `mean`, true `sqrt`
  variant added), `reward_mean_last10` over the last 10 evals, config
  validation + `Literal` types, `fork_rng` evals.
- `3b00711` phase 2 — collector ring buffer replaced by a per-step shift;
  removes the `num_action_steps <= num_obs_steps` limit.
- `7edbbb0` phase 3 — 3b batched `.item()` reduction, 3d **V-head** replaces
  the 4-sample actor baseline, 3e device-side `randperm`, 3f dead eval toggle
  deleted. 3a (GPU env placement) and 3c (GAE vectorisation) deferred.
- `0d6d8df` phase 4 — optimizer grouped by `ndim` (actor out layer explicitly
  excluded), role-tagged param groups read by `update()`, resume re-tags
  roles, `Literal` config types, `ROPE_BASE`, `no_grad` in `RunningMeanStd`,
  dead code dropped. Note: grouping drops weight decay from 14 bias tensors
  (deviation from "keep today's numbers"; flagged, unresolved).
- `88c98b5` fused Adam on CUDA (device-guarded, untested locally — no GPU here).
- `976ecf8` strict IEEE fp32 everywhere (cudnn/matmul `fp32_precision="ieee"`,
  `set_float32_matmul_precision("highest")`; `tf32_enabled: false` is recorded
  in `run_metadata.json`).
- `a09d016` ppo `reward_mean_last10` averaged all evals instead of the last 10
  (same bug class as flow_policy_optimization 1f); fixed to `[-10:]`.

Gates at archival: `pytest -q` 184 passed, `ruff check src scripts tests` clean,
smoke runs exit 0.

## Benchmark results (Hopper-v5, seed 0, 50-episode evals, ~8192 steps/eval for flow_policy_optimization, ~2048 for PPO)

| run | steps | mean last-10 eval | median last-10 | best eval | fps | wall |
|---|---|---|---|---|---|---|
| flow_policy_optimization | 100,141 | 160.8 | 174.4 | 200.6 | 439 | 242s |
| flow_policy_optimization | 500,038 | 156.9 | 156.5 | 208.9 @ ~369k | 387 | 1308s |
| PPO | 100,032 | 327.0 (fixed metric; logged 216 pre-fix) | 334.1 | 353.4 | 1688 | 75s |
| PPO | 500,032 | 597.8 | 608.3 | 799.9 @ ~227k | 985 | 525s |

flow_policy_optimization curve: fast climb to ~100k → hard plateau ~150–210 for 400k steps;
slope +0.08/1k steps (noise). Two collapses (eval at ~254k: 207→122; slumps at
~443–467k: 115–146), each partially recovering. Within-eval outcomes are
bimodal (instant-fail ~1–5 return ≈ 3–5% of episodes; survivors cluster at
180–310). PPO: steady climb to ~230k (peak 800), three deep dips
(~123k, ~232k, ~460k), recovered, settled in a ~600 regime.

Determinism check: the 100k and 500k flow_policy_optimization runs are the same seed-0 trajectory;
their curves are bitwise identical at matched steps (prefix property of the
deterministic CPU pipeline).

## Literature context (post-`2602.02481` reports, all consulted 2026-11)

- **DR-APO** ([2610.16618](https://arxiv.org/html/2610.16618v1), Oct 2026):
  Hopper-v5 final return @ 11M steps — PPO 3023, SAC 3158, **FPO 736 (flow) /
  41 (diffusion)**, DR-APO 3016. FPO is last on the flow track across 5 tasks.
- **FlowRL** ([2603.27450v2](https://arxiv.org/html/2603.27450v2), Jun 2026):
  on-policy IsaacLab — PPO best and most stable; FPO++ "occasionally
  collapses" during training (negative-advantage unbounded updates).
- **DiPOD** ([2606.13795v2](https://arxiv.org/html/2606.13795v2), Jun 2026,
  Berkeley): names the mechanism — **double drift**: RL updates loosen the
  ELBO↔likelihood gap, then the proxy (CFM) gradient drifts from the true
  policy gradient. Fix: interleave ELBO-tightening self-distillation (a
  one-line `+ β∇ELBO` regularizer in practice).
- **RFPO** ([2610.10453](https://arxiv.org/abs/2610.10453), Oct 2026): FPO++
  degrades at few-step sampling; not exercised here.
- **RFO** ([2602.03501](https://arxiv.org/html/2602.03501), Feb 2026): argues
  CFM-surrogate PG is sample-inefficient; reparameterization hybrid.
- FPO/FPO++ success cases are all **post-training pretrained flow policies**
  (humanoid motion tracking, manipulation finetune, dLLM reasoning) — not
  from-scratch MuJoCo locomotion.

Our result sits on the consensus: on dense, from-scratch locomotion the
flow-PG family trails PPO, and its known collapse mode reproduces.

## Honest caveats

- Single seed; no replications.
- This is a single-file reimplementation that *deviates* from published FPO++
  (token-level ratios, per-step Polyak, V-head value baseline, Hopper-v5 with
  defaults, 8-ish denoise steps); treat as a datapoint on this variant.
- PPO eval grid is 4× denser than flow_policy_optimization's (every ~2048 vs ~8192 steps); best-
  checkpoint selection favors PPO slightly (does not affect last-10 means).
- Untested on CUDA (no GPU on the dev box): the fused-Adam path and the fp32
  flags are config-only, verified on CPU only.
- Hopper-v5 with default termination and our reward range: absolute numbers
  are not comparable to published tuned PPO (~3000).

## What this project demonstrates

- Reimplemented an ML research paper end-to-end (single file, CleanRL style).
- Found and fixed correctness bugs in a distributed codebase (target-critic
  lag, rollout accounting, normalizer ordering, eval-return aggregation).
- Built benchmarking + evaluation rigor (fp32 audit, seed-locked runs,
  artifact/checkpoint lineage, wandb offline + eval CSVs).
- Diagnosed an algorithmic failure mode and tied it to the literature
  (ASPO → negative-advantage unboundedness → double drift).
- Ran the honest negative result: on this benchmark, FPO++ ≤ PPO; learned
  when a family is worth 2–4× more compute for no return.

## Suggested resume framing

- *Reimplemented FPO++ (flow-matching policy gradients, CoRL 2026 oral) as a
  minimal CleanRL-style PyTorch codebase; benchmarked vs a repaired PPO
  baseline on Hopper-v5; audited and fixed correctness issues (target-critic
  staleness, rollout budget accounting, observation-normalizer training
  order, eval metric aggregation); benchmarked honest negative result
  (FPO++ 157 vs PPO 598 mean return) and matched it to the post-release
  literature (DiPOD double-drift diagnosis; FlowRL collapse findings; DR-APO
  Hopper table).*
