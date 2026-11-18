# Drifting-Optimized Robot Policies: A Three-Track Research Proposal

**Authors:** flow_policy_optimization team (working proposal)
**Date:** 2026-11-13
**Scope:** three coupled research tracks exploiting *Generative Modeling via Drifting* (Deng, Li, Li, Du, He — arXiv:2602.04770v2) for robot policy learning.

---

## 0. Motivation and positioning

The recent one-step generative paradigm shift — Drifting Models (Deng et al., arXiv:2602.04770v2; **ImageNet-256: FID 1.54 latent / 1.61 pixel at 1-NFE**) — replaces iterative inference-time integration with **training-time evolution of the pushforward distribution**. A drifting field `V(p, q; x)` attracts generated samples toward data and repels them from the current generated distribution; `V` vanishes iff `q = p`; training minimizes `‖f(ε) − stopgrad(f(ε) + V)‖²`. The field is anti-symmetric by construction (`V_pq = −V_qp`), which is load-bearing: their ablation shows breaking anti-symmetry collapses generation quality.

The paradigm is spreading fast across domains — super-resolution (DriftSR), speech (DriftSE), CFD surrogates (arXiv:2606.07481, ~200× speedup over diffusion), autoregressive generation (Drift-AR), low-dose CT (RDDM) — but **no work applies it to robot action policies**, and none combines it with RL. That absence is the opening.

Why this matters now, specifically:

1. **Inference is the deployment bottleneck.** Flow/diffusion action heads (π₀, flow_policy_optimization's flow policy with `num_steps_noise_start=10`) cost 8–20 forward passes per control cycle. At 30–50 Hz manipulation this is the binding throughput constraint, and it dominates every flow_policy_optimization eval.
2. **RL for flow policies is structurally fragile.** FPO/FPO++ (arXiv:2507.21053, arXiv:2602.02481) needs CFM-ratio ELBO surrogates with finite-MC bias; their late-training instability (and flow_policy_optimization's own 3407 run oscillation/collapse after ~320k steps) demonstrates the failure mode empirically.
3. **Drifting needs no densities, no samplers, no chains.** Its attraction/repulsion field is defined pointwise on samples — exactly the interface RL's replay buffer provides.

The frozen-conclusion of this proposal: drifting is a **kernelized, density-free discrepancy minimizer with a stop-gradient**, usable as (a) a one-step *generator* (Track A), (b) an *improvement operator* over replay data (Track B), and (c) an analyzable dynamical system with a stability/guidance theory (Track C).

### Verified baseline facts used throughout (all from this session's reads)

| Fact | Source |
|---|---|
| Drifting Model: MIT/Harvard, Kaiming He co-author; FID 1.54/1.61 @1-NFE | arXiv:2602.04770v2 |
| W-Flow: Sinkhorn-divergence Wasserstein gradient flow, FID ~1.3 (claimed) | arXiv:2605.11755 |
| FPO/FPO++ ratios from CFM losses, per-sample in FPO++; ASPO; humanoid/locomotion | arXiv:2507.21053, arXiv:2602.02481 |
| GoRL: decoupled latent policy + flow decoder, ICML 2026, HopperStand >870/180M steps | arXiv:2512.02581v2 |
| QF3: velocity-space per-dimension clip, 10× wall-clock vs FPO++, 29-DoF G1 from scratch | arXiv:2610.08789 |
| QF3 is homogeneous-α; per-dimension/heterogeneous α unexplored | this session's read, arXiv:2610.08789 |
| flow_policy_optimization repo: flow (OT-CFM-like) policy, ~10 Euler steps default, on-policy FPO++, paired eval seeds | repo audit this session |
| flow_policy_optimization MuJoCo registry: Hopper/HalfCheetah/Walker2d/Swimmer/Pusher/Ant/Humanoid (+ Adroit, ManiSkill) | configs/envs.yaml |
| flow_policy_optimization eval is deterministic per `eval_seed = 10000 + seed` (paired) | src/flow_policy_optimization/training.py:1078 |
| FAFM already does flow-matching **in DCT-coefficient space** + Sobolev regularization (imitation) | arXiv:2606.20135 |
| FAFM/FGO/FreqPolicy/MPD/CSP: frequency-domain action representation is crowded in imitation | survey + arXiv reads this session |
| BEHAVIOR/LeHome 2026 1st-place solutions use correlated ε ~ N(0, βΣ+(1−β)I) | arXiv:2512.06951, arXiv:2606.27163 |
| WarmPrior: non-isotropic prior for flow policies + prior-space RL | arXiv:2605.13959 |
| Structured-Diffuser: task-conditioned non-isotropic noise | arXiv:2509.25685 |
| Lattice: full-covariance structured exploration noise in latent space (SAC/PPO) | NeurIPS 2023 |

---

## Part 1 — Track A (deployment track): One-Step Drifting Action Policies for Real-Time Robot Control

**Title.** *One-Step Drifting Policies for Real-Time Robot Control.*

**One-line claim.** A drifting-distilled action head matches a 10-step flow teacher's returns at ~10× lower inference cost on locomotion and manipulation, at equal or better temporal smoothness.

### 1.1 Problem

Flow/diffusion action heads cost `K` velocity-field evaluations per control cycle (flow_policy_optimization default `K=10`; deployments often use `K=4`, which degraded action quality in our Humanoid-v5 3407 run: returns flatlined near the random-policy band). Real-time manipulation at 30–50 Hz and dynamic locomotion at 100+ Hz cannot afford this; RL on-policy loops multiply the cost by every rollout.

### 1.2 Method

1. **Teacher**: any flow_policy_optimization/flow/pi₀-style flow policy checkpoint (existing checkpoints reused).
2. **Student**: single-block transformer DiT or lightweight MLP-DiT with AdaLN, action-chunk output `[B, H, dA]`, **no integration**. Trained with the Drifting loss (Eq. 6 of the drifting paper):
   `L = E_ε[ ‖f_θ(ε) − stopgrad(f_θ(ε) + V(f_θ(ε)))‖² ]`
   with positives = teacher-action chunks from teacher rollouts in s, negatives = student samples in the same batch.
3. **Drift field instantiation** (two options, ablated):
   - *Raw action space*: RBF kernel `k(x, y) = exp(−‖x−y‖²/ℓ²)` over whitened action chunks (running covariance from replay, shrunk `β` toward identity per the BEHAVIOR/LeHome recipe).
   - *Feature space*: critic encoder `ϕ(s, a)` from flow_policy_optimization's frozen CriticMember as the feature extractor (the drifting paper shows feature drift >> raw space for high-dimensional targets; action chunks are 44–68-dim, where raw RBF is already well-conditioned with whitening).
4. **CFG analog at training time** (optional, reward-conditioned): positives = teacher actions with above-median advantage; negatives = student samples. Drifting field learns a reward-skewed transport directly — 1-NFE reward guidance at inference (see also Track C §3.2).

### 1.3 Experiments (all in flow_policy_optimization unless noted)

- **Environments**: Hopper-v5, HalfCheetah-v5, Walker2d-v5 (core); Ant-v5, Humanoid-v5 (hard). Stretch: AdroitHandDoor-v1 (dexterous).
- **Baselines**: (i) teacher (10-step), (ii) consistency-style distillation (drift vs CFM-consistency comparison), (iii) shortcut-model distillation, (iv) DCT-space FM (FAFM-style) student.
- **Primary metrics**: normalized return, action jerk / LDLJ, wall-clock per policy call (ms), success rate; flow_policy_optimization's paired-seed eval (`eval_seed = 10000 + seed`) for every comparison.
- **Key ablations**: raw vs feature space; kernel bandwidth ℓ; β shrinkage; chunk length H ∈ {4, 8}; teacher N (4 vs 10).

### 1.4 Hypotheses

- **H1.** Student ≥ 95% of teacher returns at 1 NFE (vs 4-step FM degraded in our runs).
- **H2.** ≥ 5× rollout wall-clock speedup at matched eval return.
- **H3.** Jerk/LDLJ improves (smoother trajectories than 4-step FM).
- **H4.** Feature-space drift > raw-space drift for Humanoid-v5 (17-dim); parity at ≤ 6-dim.

### 1.5 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Action-space RBF kernel degenerate (all actions equidistant) | whitening + shrunk covariance; feature space fallback |
| Equilibrium only approximate at small batch | noise std monitoring; drift-norm logging (drift-vs-return correlation, their Fig. 4 analog) |
| Student learns teacher *including its failure modes* | H4 + reward-conditioned variant; BC-ceiling acknowledgment |
| Extra training cost of two-stage pipeline | report wall-clock amortization (10× rollout speedup pays back after ~1/9 the teacher's training runs) |

### 1.6 Timeline (≈ 8 weeks)

W1–2: student architecture + drift kernel + unit tests (drift-norm logging); W3–4: Hopper/Walker2d distillation + latency; W5–6: Humanoid-v5 + ablations; W7: teacher→student VLA/pi₀ smoke (LIBERO via ManiSkill if licensed); W8: write-up.

---

## Part 2 — Track B (algorithmic flagship): Advantage-Weighted Drifting — Density-Free Policy Improvement

**Title.** *Policy Improvement Without Likelihoods: Advantage-Weighted Drifting for One-Step Reinforcement Learning.*

**One-line claim.** Drifting trained against the advantage-weighted replay distribution matches or exceeds density-ratio policy-gradient methods (FPO/FPO++) and advantage-weighted regression at equal interaction budgets — with no likelihoods, no ratios, no sampler backprop.

### 2.1 Problem

Every current RL-improvement operator for generative policies pays one of three taxes:

1. **Likelihood tax** (FPO/FPO++, DPPO/ReinFlow): CFM/ELBO ratios with finite-MC bias, late-collapse instability — empirically observed in both the FPO++ paper and our 3407 run.
2. **Sampler tax** (Diffusion-QL, Reparameterized OFM-RL): backprop through ODE steps; memory/variance blow-up; QF3 avoids it with a velocity clip but stays TD3-flavored and needs the critic at every step.
3. **Behavioral-ceiling tax** (AWR/AWAC/IDQL-style extraction): improvement bounded by behavior support; extraction is itself a distillation problem (current flow_policy_optimization update inherits this: ratio clip acts on CFM losses, effectively a distillation of the behavior policy).

Drifting offers a fourth route: **iterated distribution matching**, which is what RL over replay is anyway — but with a kernelized attraction/repulsion rather than a parametric ratio.

### 2.2 Method

**Update rule.** Given batch from replay, advantages `A(s,a)` (normalized as flow_policy_optimization already does), define the tempered positive distribution
`p*(a | s) ∝ p_replay(a | s) · exp(A(s,a)/β)`.
Then train `f_θ` on the drifting loss with positives sampled from `p*` (importance-weighted replay samples, or top-quantile elite set) and negatives = fresh student samples:

```
L = E_{(s, a+, A), ε} [ ‖f_θ(s, ε) − stopgrad(f_θ(s, ε) + V(p*, q; f_θ(s, ε)))‖² ]
```

`q = student pushforward`; `V` = drifting field (Track A instantiation). Crucially:

- **No densities, ratios, clip, or ODE backprop** enter the update.
- Equilibrium `q = p*` is well-defined and *is the improved policy* — the drift drives the student onto the advantage-filtered distribution, and iterating gives iterated AWR-style improvement, now inside a one-step generator.
- **Safety floor**: mix replay actions into the positive set with fixed floor weight `w_min` (avoids collapse to elite minority when β is large); β ∈ [0.05, 1].
- **Baseline drift anchor** (optional): one distillation block toward the raw replay distribution per K steps — bounds drift staleness as buffer evolves.
- **KL trust region** (optional): blend toward the previous student (EMA-target drifting), mirroring flow_policy_optimization's EMA; prevents large-step jumps when advantages are noisy.

**Implementation slots in flow_policy_optimization.** The batch already carries `raw_advantages`, `normalized_advantages`, `raw_returns`, `chunk_truncated`, `action_is_pad`. The update replaces `policy_loss` (FPO ratio surrogate) with `drift_loss` — a ~150-line sibling of `policy_loss` in `modeling.py`. EMA, advantage normalizers, GAE pipeline unchanged. flow_policy_optimization's paired eval protocol gives trustworthy absolute comparisons against the current FPO++ run at matched seeds.

### 2.3 Theory targets

**T1 (equilibrium).** Under the kernelized anti-symmetric field with positives from `p*`, any fixed point of `q = p*`-attraction and `q`-repulsion satisfies `q = p*` up to kernel bandwidth (formalize the drifting paper's Appendix C.1 identifiability heuristic for tempered targets).

**T2 (improvement direction).** Show the drift gradient aligns with a **first-order improvement step** — e.g., bound `V(p*, q)` in terms of the advantage-weighted residual, relating drift magnitude to `AWR`-style exponential weighting error bounds. Realistic target: a heuristic plus a controlled 2D/tabular case (gridworld with known modes), not a full convergence proof.

**T3 (stability).** Drift-norm dynamics under advantage noise: bound oscillation frequency of `‖V‖` as advantage variance grows; show the safety floor forces contraction toward replay support (prevents reward-hacking blow-up — the failure that motivates QF3's velocity clip and Flow-DPPO's divergence mask).

**T4 (connection to Sinkhorn/Wasserstein gradient flow).** Position drift as a stop-gradient Sinkhorn-WGF (W-Flow, arXiv:2605.11755) with a task-specific energy `E(a) = −A(s, a)/β`; gives the method a variational reading and ablation axes (Sinkhorn vs drift-Sinkhorn baseline).

### 2.4 Experiments

- **Phase 1 (offline, no new infra):** static buffers from flow_policy_optimization runs (or MuJoCo DMC-style datasets): Hopper, Walker2d, HalfCheetah, Ant; compare vs AWR, advantage-weighted FM extraction, IQL-style extraction, flow_policy_optimization FPO ratio (replayed updates). Metrics: normalized return vs #updates, drift-norm trajectory vs final performance.
- **Phase 2 (online-to-online):** flow_policy_optimization with the drift update replacing the PPO surrogate on Hopper-v5, then Walker2d-v5, HalfCheetah-v5; baselines: FPO++ (current), FPO++ + advantage-floor (ablation), Flow-DPPO-style KL mask (if implementable in scope).
- **Phase 3 (hard regime):** Humanoid-v5 (sparse-ish healthy reward); Ant-v5 (8-dim, easier). Success criterion: match FPO++ return at 10× lower per-update inference cost; exceed in wall-clock-to-target.
- **Phase 4 (manipulation, stretch):** LIBERO via ManiSkill or MetaWorld registration (repo already registers Meta-World per commit history); requires Track A's student.

### 2.5 Hypotheses

- **H1.** Track B ≥ AWR/extraction at equal steps, by ≥ 10 points on normalized return on ≥ 2/4 locomotion tasks, with smaller inter-seed variance (drift smoother than clip).
- **H2.** Track B ≥ FPO++ on wall-clock-to-target (≥ 3×) at matched final return on Hopper/Walker2d; within 5% of FPO++ final return on Humanoid-v5.
- **H3.** Drift-norm monotone decreases in stable runs; divergence correlation holds (r > 0.7 with return improvement); divergence predicts late-collapse onset (validates T3).
- **H4.** Safety floor w_min ≥ 0.1 eliminates rare reward-hacked explosions; below that, occasional catastrophic seeds appear (empirical phase diagram).

### 2.6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Drift equilibrium misidentified as gradient descent (drift is not a gradient of a divergence) | measure drift norm, angle; control lr; T3 stability analysis; fall back to small lr + EMA |
| β selection: large β ⇒ mode collapse onto elite; small β ⇒ no improvement | β sweep; advantage clip; floor w_min |
| Kernel degeneracy in 17-dim Humanoid chunks | whitening; Mahalanobis kernel; feature space from frozen critic |
| Importance weight variance | normalized advantages (flow_policy_optimization already does); top-quantile clipping; ESS monitoring |
| Phase 2 integration risk in flow_policy_optimization training loop | minimal-change path: add `update_impl: {fpo, drift}` switch; keep FPO path intact |

### 2.7 Timeline (≈ 12–14 weeks)

W1–2: drift kernel + whitening + diagnostics in flow_policy_optimization; W3–4: offline A/B vs AWR; W5–8: online Walker2d/Hopper + ablations; W9–10: Humanoid-v5; W11–12: theory experiments (T1, T3 diagnostics); W13–14: paper writing.

---

## Part 3 — Track C (theory + deployability): Equilibrium Stability and Reward-Guided Drifting for Deployment-time Steering

**Title.** *Stable, Steerable, One-Step: Equilibrium Theory and Reward Guidance for Drifting Robot Policies.*

**One-line claim.** (i) A stability/identifiability analysis of drifting fields in action space, (ii) a reward-conditioned drifting head giving 1-NFE guidance at deployment without retraining, (iii) safety constraints as hard repulsion terms with measurable guarantees.

### 3.1 Part C1 — Theory (equilibrium & stability)

**C1.1 Identifiability in action space.** Extend the drifting paper's Appendix C.1 heuristic: under anti-symmetric kernelized attraction/repulsion with whitened ℓ2 distance and temperature τ, zero drift implies approximate distribution match up to kernel bandwidth; characterize the residual bias as a function of τ and bandwidth; quantify under action-chunk covariance shrinkage (ties to Track A kernel design).

**C1.2 Drift as Wasserstein dynamics.** Derive the continuity-equation reading of drift updates `x ← x + V(x)` (particles, no SDE); relate to W-Flow's Sinkhorn gradient flow with energy = negative log-density under `p`; obtain a stability condition on τ (contractivity rate of the drift map); determine whether the training dynamics contracts the Wasserstein distance to `p` geometrically under idealized assumptions (tabular/2D exact analysis, then scaling arguments).

**C1.3 Mode coverage theorem (conditional).** Show that anti-symmetry + full-rank repulsion prevents mode starvation under advantage weighting — i.e., the repulsion term prevents collapse onto the top mode that pure exponential weighting exhibits (the failure mode explicitly flagged in the manipulation literature). Give conditions: anchor floor w_min > 0 ⇒ sup-norm support of q retains lower bound on every mode of p* with mass above w_min.

**Deliverable:** one theory paper (ICML/NeurIPS-style), feeding Tracks A/B with kernel/temperature design rules.

### 3.2 Part C2 — Reward-guided drifting at inference (deploy-time)

**Concept.** Train a single reward-conditioned drift field `V(x; s, r)` where reward (or advantage) enters the *positive* set construction, following the drifting paper's CFG mechanism: positives = high-return-labeled chunks; negatives = generated or low-return chunks; at deployment, **any reward scalar can be dialed** (analogous to guidance scale) — the student drifts toward high-reward actions at 1-NFE. This is strictly stronger than reward-guided flow (which needs guidance at every integration step, K× the cost) and simpler than RLHF-style retraining pipelines.

**Method.**
- Teacher/student: reuse Track A student; add reward head or reuse advantage
  from flow_policy_optimization's critic (no new network if desired; advantage normalization reused).
- Training: positives from replay with `A > 0` cutoff or advantage-weighted sampling; reward-conditioned embedding (FiLM/adaLN) of a discretized reward bucket; CFG-style training-time guidance sampling between unconditional and conditional modes (per the drifting paper's CFG section — negative samples from off-class positives at matched reward).
- Inference: given state, sample ε; one forward pass; outputs high-reward chunk. Reward scale is a free inference-time dial.

**Baselines.** reward-guided FM/diffusion teachers (same K), vanilla drifting (unconditional), FPO++ retrained per reward.

**Hypotheses.**
- C2-H1: 1-NFE guided student matches K-step guided teacher on MetaWorld tool-hang-class tasks (or LIBERO where available) at ≤ 20% of its compute.
- C2-H2: reward dial trades success rate smoothly (guidance curve), no reward hacking within [β_min, β_max].
- C2-H3: steerability survives distribution shift (eval on states far from replay).

### 3.3 Part C3 — Safety constraints as hard repulsion

- Add a repulsion term from a **constraint value function** `V_c(s, a)` (learned from constraint-violating replay): `V_safe = V − λ · ∇_a V_c`-style repulsion direction, or simply an augmented negative set of constraint-violating actions. At inference this is free.
- Analyze: hard-vs-soft constraint trade-off; measure constraint violation rate vs λ; compare vs action-clamping (flow_policy_optimization's existing `bound_to_unit`) and vs QF3-style clip.
- Deliverable: a checked, controllable safety dial for deployment — directly relevant to humanoid demos and real-hardware pitch.

### 3.4 Timeline (≈ 10 weeks)

W1–3: equilibrium identifiability experiments (tabular gridworld, 2D, small MuJoCo) + write-up of C1; W4–6: reward-conditioned student + CFG analog; W7–8: safety repulsion + ablations; W9–10: paper assembly + coordination with Tracks A/B.

---

## Part 4 — Unified framing & venue strategy

**Story.** "Drifting models" as one-step discrepancy minimizers unify three contributions: a deployment artifact (Track A), an RL algorithm (Track B), and a theory/deployment-time study (Track C). Together they are one program: make 1-NFE generative policies practical and safe for robot RL.

**Venues.**
- Track A → CoRL / RA-L / ICRA (robot deployment evidence).
- Track B → NeurIPS / ICLR / ICML (algorithm + theory heuristic + benchmarks).
- Track C → ICML / NeurIPS (theory track) or ICRA workshop (reward guidance); safety results fold naturally into Track B's paper.

**Why now / why this team.** The flow_policy_optimization codebase (flow policy, paired-seed eval, MuJoCo registry, Adroit/MetaWorld registration path, ManiSkill GPU env support) is the correct laboratory; μfpo's FPO++ instability experience is direct motivation, and the paired-seed protocol eliminates the seed-variance confound that plagues this literature.

**Go/no-go gates.**
- Gate 1 (end of Track A W4): student retains ≥ 90% teacher returns on Walker2d-v5 → proceed to Humanoid-v5/Manipulation; else narrow to locomotion-only scope.
- Gate 2 (end of Track B W6): advantage-weighted drift ≥ AWR baseline at equal budgets on 2/3 core tasks → proceed to full write-up; else pivot Track B to offline-only (smaller paper, still publishable).
- Gate 3 (Track C W6): reward guidance at 1-NFE beats 4-step guided FM by ≥ clear margin on one manipulation task → proceed; else fold guidance as Section into Track A paper.

## Part 5 — Related work and precise deltas

- **Drifting Models** (arXiv:2602.04770v2): the base paradigm; we are the first to port to robot actions + RL.
- **W-Flow** (arXiv:2605.11755): Sinkhorn-Wasserstein one-step; stronger theory, weaker empirical (FID 1.29 vs 1.54 as reported); we adopt its variational lens for T4, not its algorithm.
- **FPO/FPO++** (arXiv:2507.21053; arXiv:2602.02481): ratio-from-CFM; density-based; our Track B replaces the improvement operator, keeping everything else (GAE, EMA, normalizers) fixed where possible.
- **QF3** (arXiv:2610.08789): velocity-space clip; complementary — a future combination is natural (drift loss + velocity clip) but out of scope here.
- **GoRL** (arXiv:2512.02581v2): decoupled latent policy; orthogonal axis; our Track B keeps one network.
- **FAFM** (arXiv:2606.20135), **FGO**, **FreqPolicy**: frequency-domain action modeling — deliberately *not* our axis (crowded, wrong inductive bias for sparse-reward RL).
- **BEHAVIOR/LeHome solutions** (arXiv:2512.06951; arXiv:2606.27163), **WarmPrior** (arXiv:2605.13959), **Structured-Diffuser** (arXiv:2509.25685), **Lattice** (NeurIPS 2023): correlated-source/prior work — we use the *whitened shrunk-covariance* trick in Track A but not as the contribution; our contribution is the improvement operator.
- **AWR/AWAC/IQL/IDQL**: the weighted-regression family Track B must beat; we position advantage-weighted drift as the principled collapse of that family into a one-step generator.

## Part 6 — Open risks across all tracks

1. **Drift heuristic brittleness** — Mitigation: monitor `‖V‖`, kernel temp ablate, feature-space fallback, C1 analysis.
2. **Evaluation confounds** — Mitigation: flow_policy_optimization's paired eval seeds, ≥ 5 seeds per claim, report both normalized return and raw return, pre-registered kernel/τ for final comparisons.
3. **Over-claiming speed** — Mitigation: measure actual per-call latency, not Euler-step counts; report LDLJ as a real robot metric.
4. **Community race** (He group is fast) — Mitigation: ship Track A within ~6 weeks of Gate 1; Track B preprint by week 14.
5. **Reviewers expecting densities** — Mitigation: T4 variational framing + explicit “no-likelihood” honesty section; the AWR baseline makes the claim legible.

## Appendix A — flow_policy_optimization integration map (file-level)

- `src/flow_policy_optimization/fpo_core.py`: `compute_trust_region_surrogate` untouched for Track B; new `drift_update` sibling.
- `src/flow_policy_optimization/modeling.py`: new `student` actor type (one-step); `sample_chunk` gains a `one_step=True` path; EMA already generic (actor+encoder) — reused.
- `src/flow_policy_optimization/rollout.py`: no changes required; rollouts call student if configured; paired eval protocol reused verbatim.
- `src/flow_policy_optimization/training.py`: `policy_loss` replaced/extended with drift arm; `eval_sampling` gains `"drift_1step"`; scheduler/RMS/etc. untouched.
- `scripts/pins/`: new pin for drift equilibrium check (synthetic 2D mode-separation test).

## Appendix B — Minimal reproducibility checklist

- 5+ seeds per headline number; paired eval; confidence intervals; artifact tag with code + configs + raw logs.
- Pre-registered drift-kernel (`τ`, bandwidth, whitening β) per Track A/B experiment.
- Release flow_policy_optimization fork with drift update behind `--flow_policy_optimization.update_impl drift`.
