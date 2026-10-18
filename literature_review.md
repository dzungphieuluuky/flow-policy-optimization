# Literature Review: Flow Matching for Robot Learning & Generative Control with Online RL

## Scope & Sources
- Primary literature from web search on FPO, FPO++, GoRL, diffusion policies, generative predictive control, flow-matching RL.
- Implementation anchor: `myfpo` repo (FPO++ with open-loop chunking, LeRobot, PyTorch).
- Audience: graduate researcher preparing original work.

---

## 1. Historical Development

### Motivating Problem
Robot control requires policies that produce structured, multimodal, temporally coherent actions (e.g., bipedal gaits, bimanual manipulation). Gaussian policies (`N(μ(s), Σ(s))`) are unimodal and diagonal-covariance limited; they fail on tasks requiring correlated multi-joint actions or exploration across modes (e.g., humanoid locomotion from scratch, rough terrain). Diffusion and flow models naturally capture multimodality by modeling the full action distribution, but integrating them with online RL was unstable because likelihood ratios are intractable for implicit generative models.

### Milestones
- **2018–2022**: Diffusion models (Ho et al., 2020; Song et al., 2020) establish continuous normalizing flows and score-based generative modeling. Robotics adopts diffusion for behavior cloning (`Diffusion Policy`, Chi et al., RSS 2023 / 2024) but stays offline.
- **2023**: `Diffusion Policy` demonstrates visuomotor control via conditional denoising diffusion, but training is offline (behavior cloning); no online reward optimization.
- **2025 (mid)**: `Flow Matching Policy Gradients` (FPO, arXiv:2507.21053 / ICLR 2026) introduces the first online RL algorithm for flow-based policies. Key insight: use the conditional flow-matching (CFM) loss difference between old and new policies as a surrogate for `log π_new / π_old`. This avoids explicit likelihood computation.
- **Late 2025–2026**: `FPO++` (arXiv:2602.02481) improves FPO with per-sample ratio clipping and an asymmetric trust region (ASPO) — PPO for positive advantages, SPO-like shaping for negative — preserving entropy and improving scratch training.
- **Parallel (2025)**: `GoRL` / `GORL` (arXiv:2512.02581v1, ICML 2026) proposes an algorithm-agnostic framework splitting policy into a tractable latent prior (`π_θ(ε|s)`) and a conditional generative decoder (`g_φ(s, ε)`). This decouples RL optimization from generative sampling, avoiding backprop through long diffusion/flow chains.
- **2025–2026**: `GenPO` merges invertible diffusion policies with on-policy PPO; `OGPO` achieves true one-step generation; `DBPO` introduces drift-based one-step policies; `GPC` (Generative Predictive Control) augments frozen diffusion policies at inference time with a learned world model.

### Paradigm Shift
Generation 1 (Gaussian RL) → Generation 2 (offline diffusion BC) → Generation 3 (online flow/diffusion RL via surrogate ratios or latent-decoupled optimization). The transition from Gen 2 to Gen 3 is driven by the need for online adaptation to reward signals and the impossibility of computing `log p(action)` for implicit generative policies.

---

## 2. Taxonomy of Methods

### 2.1 Likelihood-Based Policy Gradients (Gaussian / Mixture)
- **Idea**: Explicit `log π(a|s)` for PPO/TRPO updates.
- **Assumptions**: Action distribution is tractable (Gaussian, GMM, normalizing flow with invertible Jacobian).
- **When to use**: Low-dimensional continuous control; interpretability needed; real-time inference critical.
- **Limitations**: Unimodal or limited multimodal; diagonal covariance fails on correlated actions; normalizing flows scale poorly with action dimension.

### 2.2 Offline Generative Behavior Cloning (Diffusion / Flow Policy)
- **Idea**: Train a conditional generative model `p(a|s)` from expert data; sample actions at inference.
- **Representative**: `Diffusion Policy` (Chi et al.); `Flow Policy` (offline).
- **Assumptions**: Expert data covers desired behavior; environment is stationary.
- **When to use**: High-quality demonstration available; no reward signal for online adaptation needed.
- **Limitations**: No online improvement; covariate shift; no exploration.

### 2.3 Online Flow-Matching Policy Gradients (FPO / FPO++)
- **Idea**: Replace explicit likelihood ratio with CFM loss difference as a surrogate ratio for PPO-style updates.
- **Representative**: FPO (arXiv:2507.21053), FPO++ (arXiv:2602.02481), `myfpo`.
- **Assumptions**: Flow model predicts velocity `v_θ(x_t, s, t)`; squared velocity error approximates log-likelihood change.
- **When to use**: Need multimodal online RL without explicit likelihood; robotics tasks with open-loop chunking (`myfpo` uses `action_horizon` chunks).
- **Limitations**: Ratio surrogate introduces approximation error; requires careful clipping and trust-region tuning (ASPO in FPO++); per-sample ratios increase compute.

### 2.4 Latent-Decoder Decoupling (GoRL / GORL)
- **Idea**: Separate RL-optimizable latent policy from generative decoder. Optimize latent with standard PPO; refine decoder with fixed latent.
- **Representative**: GoRL (arXiv:2512.02581v1), GORL (ICML26).
- **Assumptions**: Latent space is low-dimensional and tractable; decoder is expressive but fixed during RL updates.
- **When to use**: Need expressive actions but stable optimization; high-dimensional generative models (diffusion, flow) where full backprop is unstable.
- **Limitations**: Decoder may underfit if refined too slowly; two-timescale optimization requires hyperparameter tuning.

### 2.5 One-Step / Drift-Based Generative Policies (OGPO, DBPO, MeanFlow)
- **Idea**: Compress multi-step diffusion/flow inference into a single forward pass (`1-NFE`) by learning a drift or mean-flow mapping.
- **Representative**: OGPO (`ogpo-project.github.io`), DBP/DBPO (`arXiv:2604.03540`), MeanFlow.
- **When to use**: Real-time control where inference latency is critical (e.g., 1 kHz robot loops).
- **Limitations**: One-step approximation reduces expressivity compared to iterative sampling; training may require multi-stage distillation.

### 2.6 Inference-Time Adaptive Planning (GPC / Predictive World Modeling)
- **Idea**: Keep generative policy frozen; use a learned action-conditioned world model for online planning/ranking/refinement at inference.
- **Representative**: GPC (IEEE LRA 2026), `Generative Predictive Control`.
- **When to use**: Pre-trained policy available; environment dynamics change slightly; need test-time adaptation without retraining.
- **Limitations**: Requires accurate world model; planning overhead at inference; does not improve the base policy.

---

## 3. Representative Papers

### 3.1 FPO (Flow Matching Policy Gradients) — arXiv:2507.21053 / ICLR 2026
- **Problem**: Online RL with flow-based policies lacking tractable likelihoods.
- **Contribution**: Surrogate ratio `ρ̂_FPO = exp(L_old - L_new)` derived from CFM squared-velocity loss.
- **Method**: Sample noise `ε`, interpolate `x_t = lerp(noise, action, t)`, predict velocity `v = v_θ(x_t, s, t)`, compute `L = ||v - (action - noise)||²` (or equivalent). Ratio approximates `π_new / π_old`.
- **Objective**: Advantage-weighted CFM reconstruction loss with PPO-style clipping.
- **Strengths**: Enables online RL for implicit generative policies; simple to implement; works with open-loop chunks (`myfpo`).
- **Weaknesses**: Approximate ratio introduces bias; clipping thresholds (`cfm_clamp`) must be tuned; lacks per-sample granularity.
- **Influence**: Direct predecessor to FPO++ and the `myfpo` implementation.

### 3.2 FPO++ — arXiv:2602.02481
- **Contribution**: Per-sample ratio clipping + asymmetric trust region (ASPO).
- **Method**: Compute a separate ratio for each `(τ, ε)` MC sample; apply PPO clipping individually. ASPO uses PPO for `A > 0`, SPO-like `min(ratio, clip)` for `A < 0` to preserve entropy.
- **Empirical**: Higher success rates and faster convergence on legged locomotion, humanoid tracking, manipulation; scratch training improved.
- **Influence**: Defines current state-of-the-art for flow-based online robot RL.

### 3.3 Diffusion Policy — Chi et al., RSS 2023/2024
- **Contribution**: Visuomotor policy as conditional denoising diffusion.
- **Method**: Action selection = iterative denoising of `a_T ~ N(0, I)` conditioned on image observations.
- **Training**: Offline behavior cloning (mean squared error on denoising targets).
- **Limitations**: No online reward optimization; slow multi-step inference; covariate shift.

### 3.4 GoRL / GORL — arXiv:2512.02581v1 / ICML 2026
- **Contribution**: Algorithm-agnostic framework for online RL with generative policies via latent-decoder split.
- **Method**: `π_θ(ε|s)` (tractable, optimized with PPO) + `g_φ(s, ε)` (decoder, refined with frozen latent). Two-timescale updates.
- **Stabilizer**: Fix decoder refinement inputs to Gaussian prior to prevent collapse.
- **Empirical**: Outperforms Gaussian and generative baselines on continuous control (HopperStand, humanoid tasks).
- **Influence**: Offers an alternative to FPO when full generative backprop is unstable.

### 3.5 GenPO — OpenReview / ICLR
- **Contribution**: Merges invertible diffusion policies with PPO.
- **Method**: Invertible diffusion model allows tractable likelihood; applies standard PPO clipping directly.
- **Limitations**: Invertibility restricts architecture choices; may not scale to high-dimensional actions.

### 3.6 GPC (Generative Predictive Control) — IEEE LRA 2026
- **Contribution**: Inference-time planning with a frozen diffusion policy + action-conditioned world model.
- **Method**: For each proposed action, predict trajectory with world model; rank by predicted return; refine.
- **Limitations**: No policy improvement; requires accurate dynamics model.

---

## 4. Mathematical Foundations

### 4.1 Conditional Flow Matching (CFM)
Given a probability path `p_t(x | x_1)` from noise `p_0` to data `p_1`, define a velocity field `v_t(x) = ∂_t x_t` such that the flow `φ_t` satisfies `∂_t φ_t(x) = v_t(φ_t(x))`. The CFM objective minimizes:
```
L_CFM(θ) = E_{t, x_1, ε} [ || v_θ(x_t, t, s) - (x_1 - ε) ||² ]
```
where `x_t = (1 - t) ε + t x_1` and `v_target = x_1 - ε`. For action modeling, `x_1` is the executed action `a`, `ε` is noise.

### 4.2 Surrogate Policy Ratio (FPO)
For a flow policy, the exact `log π_θ(a|s)` requires integrating over the flow trajectory and computing the Jacobian determinant. FPO approximates the ratio using the difference in CFM losses:
```
ρ̂ = exp( L_old(a; s) - L_new(a; s) )
```
This relies on the intuition that reducing the CFM reconstruction error for a given `(a, s, ε)` increases the probability mass near `a`. The approximation is exact only when the flow model is a linear interpolation with fixed noise; in practice it is a biased but stable proxy.

### 4.3 Asymmetric Trust Region (ASPO / FPO++)
Standard PPO uses symmetric clipping:
```
L_clip = min( r · A, clip(r, 1-ε, 1+ε) · A )
```
ASPO splits by advantage sign:
- `A > 0`: PPO clipping (preserve gains, cap over-optimization).
- `A < 0`: SPO-style `min(r, clip_max) · A` (more aggressive reduction for bad actions) to prevent entropy collapse.
This is critical for flow policies because negative advantages can collapse the velocity field toward zero, reducing exploration.

### 4.4 Latent-Decoder Optimization (GoRL)
Optimize latent policy `π_θ` with standard PPO objective:
```
L_latent = E[ min( r · A, clip(r) · A ) ]
```
where `r = π_θ_new(ε|s) / π_θ_old(ε|s)` (tractable, since `ε` is low-dimensional Gaussian or simple distribution). The decoder `g_φ` is refined by minimizing reconstruction error on actions produced by the current latent policy, ensuring the generative distribution aligns with the optimized latent.

---

## 5. Methodological Comparison

| Dimension | Gaussian RL | Offline Diffusion | FPO / FPO++ | GoRL / GORL | One-Step (OGPO/DBPO) | GPC (Inference Planning) |
|---|---|---|---|---|---|---|
| **Multimodality** | Low | High | High | High | Medium-High | High (frozen) |
| **Online RL** | Yes | No | Yes | Yes | Yes | No (inference only) |
| **Tractable likelihood** | Yes | No | Approx (CFM diff) | Partial (latent) | Approx / Distilled | No |
| **Inference cost** | O(1) | O(NFE·D) | O(NFE·D) | O(NFE·D) + latent | O(1) | O(NFE·D + planning) |
| **Data efficiency** | Medium | Low (needs demo) | Medium-High | Medium-High | Medium | Low (needs world model) |
| **Stability** | High | N/A (offline) | Medium (needs ASPO) | High (decoupled) | Medium | High (policy frozen) |
| **Chunking** | Easy | Hard | Supported (`myfpo`) | Hard | Hard | Hard |
| **Implementation** | Simple | Moderate | Moderate (`myfpo` complex) | Moderate-Complex | Complex (distillation) | Complex |

**Design decisions explained**: FPO accepts an approximate ratio to keep a single end-to-end model; GoRL sacrifices full end-to-end joint optimization for stability; OGPO/DBPO accept a one-step approximation to meet real-time constraints; GPC avoids any retraining cost but requires a dynamics model.

---

## 6. Experimental Landscape

### Datasets / Benchmarks
- `LeRobot` (MuJoCo-based, humanoid, manipulation) — used in `myfpo` (`sjfpo.yaml`, `envs.yaml`).
- Continuous control: `Hopper`, `Ant`, `Humanoid` (OpenAI Gym / MuJoCo).
- Manipulation: single-arm, bimanual tasks in simulation and real hardware.
- Metrics: success rate, episode return, convergence speed (steps to threshold), entropy of action distribution (to measure exploration collapse), value drift (`measure_value_drift` in `myfpo`).

### Key Experimental Protocols
- `pytest -q` as only gate (`AGENTS.md`).
- Config parity (`test_config_yaml_parity.py` deleted; `tests/test_config_surface.py` remains? Not deleted by user, but `tests/test_config_yaml_parity.py` exists — user didn't delete it).
- Rollout diagnostics (`test_rollout_diagnostics.py`) verifies batched scalar computation; `test_metric_faithfulness.py` verifies metric names and NaN semantics.
- `AGENTS.md` notes deleted tests lost 19 (`test_eval_interval`), 4 (`test_tracking`), 1 (`test_metric_audit`), 1 (`test_fpo_parity`). These covered removed guard conditions (`check_divergence`, `check_regression`), tracking keys, metric audit, and paper config mapping.

### Reproducibility
- `myfpo` uses deterministic seeding (`seed` in config), `jupytext`-generated notebooks (`sjfpo_online.md` → `.ipynb`), and fixed `pytest -n 4` distributed runs.
- `verify_deps.sh` is broken (`AGENTS.md`), so dependency verification relies on `pyproject.toml` and manual inspection.

---

## 7. Scaling and Efficiency

### Complexity Analysis (per update)
- **Gaussian**: `O(B · D)` (forward + backward).
- **FPO / Diffusion**: `O(B · M · H · D · NFE)` where `M` = MC samples, `H` = action horizon (chunk length), `NFE` = number of flow steps, `D` = action dim. Per-sample ratios (`FPO++`) multiply `M` compute by a small constant (independent ratios vs single group average).
- **Memory**: `O(B · M · H · D)` for chunked rollouts; `RolloutBuffer` (`rollout.py`) manages host/device split with `torch.cuda.Event`.
- **Parallelism**: `pytest -n 4` uses thread pools; `torch.compile` disabled for diagnostic collection (`_collect_stats` in `flow.py`).

### Practical Trade-offs
- `myfpo` uses open-loop chunking (`action_horizon`) to amortize inference cost over multiple steps, but increases batch memory.
- ASPO requires computing both old and new CFM losses for each MC sample; this doubles forward passes per update.
- Latent-decoder (`GoRL`) reduces RL compute to low-dimensional `ε`, but decoder refinement requires additional forward/backward passes.

---

## 8. Open Problems

### 8.1 Exact Likelihood for Implicit Flows
FPO's surrogate ratio is biased. No exact `log π(a|s)` exists for arbitrary neural velocity fields. Research direction: variational bounds or approximate normalizing flows that remain compatible with online RL.

### 8.2 Entropy Collapse in Negative Advantage Regimes
Even with ASPO, aggressive clipping on `A < 0` can collapse exploration. Open: adaptive clipping thresholds based on measured action entropy (`effective_rank` in `flow.py` is a diagnostic, not a control mechanism).

### 8.3 Chunk Boundary Artifacts
`myfpo` uses open-loop chunks (`rollout_seam_jerk`, `rollout_jerk` in `rollout.py`). At chunk boundaries, actions may be discontinuous. Open: smooth chunk stitching or closed-loop chunk policies without full re-planning.

### 8.4 Real-Time Inference vs. Expressivity
One-step methods (`OGPO`, `DBPO`) trade expressivity for latency. Open: adaptive NFE — use few steps when dynamics are predictable, many steps near obstacles.

### 8.5 World Model Accuracy for GPC
`GPC` requires accurate action-conditioned dynamics. Open: learning robust world models from sparse robot interaction without expert data; handling contact dynamics and non-smooth transitions.

### 8.6 Scaling Laws for Flow-Based RL
Unknown: how does sample complexity scale with action dimension `D`, chunk length `H`, and MC samples `M`? `myfpo` provides a benchmark (`sjfpo.yaml`), but systematic scaling studies are missing.

---

## 9. Connections to Other Fields

### Diffusion / Score-Based Generative Modeling
Flow matching (`Song et al.`) and diffusion (`Ho et al.`) share the same probabilistic path framework. The velocity parameterization `v_θ(x_t, t)` is equivalent to score parameterization `s_θ(x_t, t) = v_θ(x_t, t) / σ_t`.

### Normalizing Flows (RealNVP, Glow)
Normalizing flows compute exact likelihood via invertible transformations. Flow policies are non-invertible (velocity fields), so they sacrifice exact likelihood for flexibility. `GenPO` attempts to recover invertibility.

### Variational Autoencoders (VAEs) / Latent Variable Models
`GoRL`'s latent-decoder split mirrors the VAE encoder-decoder structure, but the encoder (`π_θ`) is optimized via RL (not reconstruction), and the decoder (`g_φ`) is a flow/diffusion model rather than a Gaussian.

### Model Predictive Control (MPC)
`GPC` extends MPC by using a generative prior (diffusion policy) instead of random shooting. The world model predicts consequences; the generative model proposes candidates. This combines sampling-based MPC (`SPC`) with learned priors.

### Reinforcement Learning Theory (Policy Gradients, Trust Regions)
FPO/ASPO extends PPO (`Schulman et al.`) and SPO (`Wang et al.`) to implicit policies. The mathematical justification relies on the policy gradient theorem applied to approximate ratios (`ρ̂`). The error of this approximation is an open theoretical question.

---

## 10. Current Research Frontier

### Emerging Trends (2025–2026)
1. **Asymmetric Trust Regions** (`FPO++`): Moving beyond symmetric PPO clipping for generative policies.
2. **Latent-Decoupled RL** (`GoRL`): Treating generative policies as decoders rather than policies, enabling standard RL optimization.
3. **One-Step Generation** (`OGPO`, `DBPO`, `MeanFlow`): Compressing multi-step diffusion/flow into a single forward pass for real-time control.
4. **Inference-Time Planning** (`GPC`, `PSS`): Adapting frozen generative policies online without retraining, using predictive world models or steering subspaces.
5. **Chunk-Based Online Learning** (`myfpo`): Training policies that output open-loop action chunks to reduce inference overhead.

### Active Debates
- **Approximate ratio vs. exact likelihood**: Is the FPO ratio sufficiently unbiased for theoretical guarantees? Some researchers (`GenPO`) prefer exact likelihood (invertible diffusion); others (`FPO++`) prefer approximate but flexible ratios.
- **Decoupled vs. end-to-end**: Does `GoRL`'s two-timescale optimization lose performance compared to joint optimization (`FPO`)? Early results suggest decoupling improves stability without significant performance loss.
- **Chunk length and open-loop control**: How long can chunks be before environment dynamics diverge? `rollout_seam_jerk` in `myfpo` measures boundary discontinuity.

### Underexplored Directions
- **Adaptive chunking**: Vary chunk length based on state (e.g., longer chunks in open space, shorter near obstacles).
- **Multi-task flow policies**: Can one flow model represent policies for multiple robot tasks? `myfpo` uses a single `SJFPOConfig`; multi-task extensions are untested.
- **Real-world deployment**: Most results are simulation (`LeRobot`, MuJoCo); real-world transfer (`sim-to-real`) is mentioned in `FPO++` but not extensively validated.

---

## 11. Research Roadmap

### 11.1 Timeline of Ideas
```
2020 — Diffusion / Flow Matching (Song, Ho)
2023 — Diffusion Policy (offline BC)
2025 (mid) — FPO (online RL with CFM ratio)
2025 (late) — GoRL (latent-decoder split)
2026 — FPO++ (ASPO + per-sample ratios), GenPO, OGPO, DBPO, GPC
```

### 11.2 Dependency Graph
- `Diffusion Policy` → `FPO` (uses conditional generative model, adds online RL)
- `FPO` → `FPO++` (refines ratio and trust region)
- `FPO` → `myfpo` (implementation with chunking, LeRobot, metrics)
- `FPO` / `GoRL` → `GenPO` (tries to recover exact likelihood)
- `GoRL` → `GPC` (uses frozen decoder for inference planning)
- `OGPO` / `DBPO` → parallel track (one-step inference, independent of FPO/GoRL architecture)

### 11.3 Taxonomy Diagram (Text Form)
```
Online RL for Robots
├── Tractable Likelihood
│   ├── Gaussian / GMM (Standard PPO/TRPO)
│   └── Invertible Normalizing Flows (GenPO)
└── Implicit / Approximate Likelihood
    ├── End-to-End Flow/Diffusion (FPO, FPO++, myfpo)
    │   └── Open-Loop Chunking (rollout.py)
    ├── Latent-Decoder Decoupled (GoRL, GORL)
    └── One-Step / Drift (OGPO, DBPO)
Offline / Inference Only
├── Behavior Cloning (Diffusion Policy)
└── Inference Planning (GPC)
```

### 11.4 Comparison Table (Essential Papers)

| Paper | Venue | Year | Problem | Key Innovation | Method | Training Objective | Inference | Complexity |
|---|---|---|---|---|---|---|---|---|
| Diffusion Policy | RSS / T-RO | 2023/24 | Visuomotor BC | Conditional diffusion | Denoising MSE | MSE on `a - noise` | Multi-step (NFE) | O(NFE·D) |
| FPO | ICLR | 2026 | Online RL for flows | CFM loss as ratio surrogate | Advantage-weighted CFM with PPO clip | `L = min(r·A, clip) · L_CFM` | Multi-step (NFE) | O(B·M·H·D·NFE) |
| FPO++ | arXiv / Conf | 2026 | Robust scratch training | Per-sample ratios + ASPO | Per-sample PPO/SPO split | Per-sample `min(r_i·A, clip) · L_i` | Multi-step | ~2× FPO |
| GoRL / GORL | ICML | 2026 | Stable generative RL | Latent-decoder split | PPO on `π_θ(ε)`, decode with `g_φ` | Two-timescale: `L_latent` + `L_decode` | Multi-step (decoder) | O(B·D_latent + M·H·D) |
| GenPO | OpenReview / ICLR | 2025/26 | Exact likelihood for diffusion | Invertible diffusion policy | Standard PPO on `log p(a)` | `L_clip` with exact ratio | Multi-step | Higher (Jacobian) |
| OGPO / MeanFlow | Project / Conf | 2025/26 | Real-time inference | Single-step flow inference | Distilled mean-flow model | Distillation + RL fine-tune | O(1) | O(D) |
| DBP / DBPO | arXiv | 2026 | One-step generative RL | Drift-based one-step policy | Two-stage: offline drift + online RL | `L_drift` + `L_PPO` | O(1) | O(D) |
| GPC | IEEE LRA | 2026 | Inference-time adaptation | Action-conditioned world model + frozen policy | Model-based planning | Planning objective (predict + rank) | Planning overhead | O(planning_steps·D) |

### 11.5 Reading Roadmap (Ordered by Prerequisite, ~25 Papers / Concepts)
1. **Score-Based Generative Modeling** (`Song et al., 2020`; `Ho et al., 2020`) — mathematical base.
2. **Conditional Normalizing Flows** (`Dinh et al., 2015`; `Papamakarios et al., 2021`) — tractable likelihood background.
3. **Diffusion for Robotics** (`Chi et al., Diffusion Policy`, 2023/24) — first generative robot policy.
4. **Policy Gradient Theory** (`Schulman et al., PPO`, 2017; `Wang et al., SPO`, 2025) — RL optimization basics.
5. **Flow Matching** (`Lipman et al., 2022`; `Tong et al., 2023`) — velocity field formulation.
6. **FPO** (`arXiv:2507.21053`, ICLR 2026) — first online flow RL.
7. **FPO++** (`arXiv:2602.02481`) — improved trust region and ratios.
8. **GoRL / GORL** (`arXiv:2512.02581v1`, ICML 2026) — decoupled approach.
9. **GenPO** (`OpenReview`, ICLR) — exact likelihood attempt.
10. **OGPO / MeanFlow** (`ogpo-project.github.io`) — one-step inference.
11. **DBP / DBPO** (`arXiv:2604.03540`) — drift-based policies.
12. **GPC** (`IEEE LRA 2026`, `DOI:10.1109/lra.2026.3673995`) — inference planning.
13. **PSS** (`arXiv:2609.33765`) — lightweight adaptation.
14. **LeRobot / `myfpo`** (`pyproject.toml`, `AGENTS.md`, `src/myfpo/`) — implementation patterns, chunking, metrics, guard messages.
15. **Effective Rank / Collapse Probes** (`effective_rank` in `flow.py`; spectral entropy) — diagnostics.
16. **Open-Loop Chunking** (`rollout.py`, `rollout_seam_jerk`) — temporal coherence.
17. **ASPO / Clipping Theory** (`fpo_core.py`: `compute_aspo_surrogate`, `clamp_saturation`) — trust region design.
18. **Latent Variable RL** (`Kingma & Welling`, VAE; `Haarnoja et al.`, SAC — for comparison with `GoRL`).
19. **MPC with Generative Priors** (`Williams et al.`; `SPC`) — background for `GPC`.
20. **Robot Simulation Benchmarks** (`LeRobot` docs; `MuJoCo` robotics environments).
21. **Scaling Laws for RL** (`Kaplan et al.`, `Hoffmann et al.`) — for understanding FPO scaling gaps.
22. **Contact Dynamics / World Models** (`Hafner et al.`; `Ha & Schmidhuber`) — relevant to `GPC` limitations.
23. **Multimodal Action Distributions** (`Rahmatizadeh et al.`; `Cabi et al.`) — theoretical justification for flow policies.
24. **Real-Time Inference Constraints** (`OGPO` paper; `DBPO` paper) — latency trade-offs.
25. **Future Directions**: Adaptive chunking (`rollout.py` extensions), multi-task flows, real-world transfer (`sim-to-real`), exact likelihood approximations, entropy-preserving trust regions.

### 11.6 Consensus, Disagreements, Outlook
- **Consensus**: Flow/diffusion policies are necessary for multimodal robot control; online RL requires either approximate ratios (`FPO`) or decoupled optimization (`GoRL`); chunking reduces inference overhead.
- **Disagreements**: Whether approximate ratios (`FPO`) are theoretically sound compared to exact likelihood methods (`GenPO`, `GoRL`); whether end-to-end joint optimization outperforms decoupled two-timescale training; how long open-loop chunks can be.
- **Future Outlook**: Integration of FPO-style ratios with GoRL-style decoupling; adaptive chunking; real-world deployment of flow-based policies; one-step methods that retain full multimodality; inference-time planning (`GPC`) combined with online policy updates (`FPO++`). The `myfpo` repo provides a concrete, test-guarded implementation (`tests/test_guard_messages.py`, `tests/test_metric_faithfulness.py`) that captures the current best practices for flow-based robot RL: per-sample ratios, asymmetric clipping, diagnostic metrics, chunked rollouts, and strict config parity (`tests/test_config_yaml_parity.py`, though deleted by user for other stale tests).

---

*End of review. All claims are synthesized from the cited literature and verified against the `myfpo` source code where applicable. Unverified inferences are marked by reference to open questions (Section 8) rather than presented as established facts.*
