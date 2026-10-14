# Literature Review: Flow Matching for Robot Learning & Online RL

## 1. Historical Development

### The Motivation: From Gaussian to Multimodal Action Policies

Robot control policies traditionally relied on Gaussian or mixture-of-Gaussian parameterizations. These approaches suffer from a fundamental limitation: they assume unimodal (or limited multimodal) action distributions. In dynamic, contact-rich environments—legged locomotion over uneven terrain, humanoid balance correction, dexterous manipulation with obstacle avoidance—the true optimal action distribution is often highly multimodal. A Gaussian policy collapses to the mean, producing suboptimal compromise actions that may never reach any mode of the true distribution.

The rise of diffusion and flow-matching models in computer vision (Ho et al., 2020; Saharia et al., 2022) demonstrated that these generative frameworks could model complex continuous distributions in high-dimensional spaces. By 2023, researchers began applying these ideas to robotics: Chi et al. (2023, 2024b) showed diffusion policies for behavior cloning, and Janner et al. (2022) demonstrated diffusion policies for imitation learning.

### The Transition: From Behavior Cloning to Reinforcement Learning

Initial applications were supervised—behavior cloning (BC) from expert demonstrations. This created a dependency bottleneck: high-quality demonstrations are expensive, and policies could not exceed demonstrator performance. The key insight emerged around 2024-2025: flow models could be trained not just via supervised denoising but via online reinforcement learning, using the flow matching loss itself as a surrogate for likelihood ratios.

**FPO (Yi et al., 2026/arXiv:2507.21053)** introduced Flow Policy Optimization—the first algorithm to bring flow matching into the policy gradient framework. FPO treats the conditional flow matching (CFM) loss difference as a proxy for the log-likelihood ratio required by PPO-style algorithms. This avoids explicit divergence computation, which is computationally prohibitive for flow models.

**FPO++ (Choi et al., arXiv:2602.02481)** refined FPO for challenging robotics tasks, introducing per-sample ratios (finer-grained trust region) and an asymmetric trust region (ASPO) combining PPO clipping for positive advantages with SPO constraints for negative advantages. FPO++ demonstrated the first successful sim-to-real transfer of flow policies trained purely via RL on humanoid robots.

### The Spectral Extension: SJFPO (This Repo)

The spectral-joint-policy repository represents a specific instantiation of FPO++ with a novel base measure: instead of white Gaussian noise, the flow's prior uses a `1/f^beta` spectral profile (`freq_weighted_time` mode). This aligns the interpolant and velocity target with a non-white base measure `N(0, L L^T)`, a contribution distinct from the core FPO++ algorithm.

---

## 2. Taxonomy of Methods

### Category A: Behavior Cloning with Flow/Diffusion Policies
**Central idea**: Train generative policies via supervised denoising on expert demonstrations.  
**Assumptions**: Expert data covers the relevant state-action space; the task is static enough for fixed-horizon planning.  
**Representative**: Diffusion Policy (Chi et al., 2023), Flow Matching for Robot Policies, π0 (Black et al.).  
**Advantages**: Stable training, multimodal expressiveness, clear regression targets.  
**Limitations**: Cannot exceed demonstrator performance; requires expert data; slow dynamics only.  
**Relationship**: Foundation for RL-based methods—most RL flow policies initialize from BC pretraining.

### Category B: Online Policy Gradient with Flow Matching (FPO Family)
**Central idea**: Replace Gaussian likelihood ratios with CFM loss ratios; train purely from rewards.  
**Assumptions**: The flow model can represent the policy; advantages can be estimated via standard RL methods.  
**Subcategories**:  
- **FPO (Yi et al., 2026)**: Basic PPO-compatible flow gradient  
- **FPO++ (Choi et al., 2026)**: Per-sample ratios + ASPO trust region  
- **SJFPO (This repo)**: FPO++ + spectral noise prior (`freq_weighted_time`)  
**Advantages**: Expressive multimodal policies; trains from scratch; no likelihood computation needed.  
**Limitations**: Gradient variance challenges; requires careful trust region tuning; spectral claims need explicit flag setting.  

### Category C: Sampling-Based Predictive Control + Flow (GPC)
**Central idea**: Use flow policies to warm-start sampling-based predictive control (SPC) in a supervised loop.  
**Assumptions**: Simulation is cheap; dynamics are well-modeled; temporal consistency is critical.  
**Representative**: Generative Predictive Control (arXiv:2502.13406).  
**Advantages**: Handles fast dynamics; no expert demonstrations needed; supervised objective stabilizes training.  
**Limitations**: Requires simulation; not pure RL (uses SPC as data generator); warm-start level α is a critical hyperparameter.

### Category D: Variable-Time and Streaming Flow Policies
**Central idea**: Action chunks have variable duration; or actions stream during flow sampling.  
**Subcategories**:  
- **Variable-time planning** (arXiv:2507.15073): Incorporates planning horizon into the flow chunk.  
- **Streaming Flow Policy** (Jiang et al., 2025): Treats action trajectories as flow trajectories; streams actions during integration.  
**Advantages**: Better temporal consistency; faster inference; adaptive to task difficulty.  
**Limitations**: More complex architecture; interpolation back to control frequency needed.

### Category E: Dynamics-Integrated Flow Policies
**Central idea**: Jointly train dynamics and policy models with shared intermediate flow samples.  
**Representative**: Dynamics-Aligned Flow Matching Policy (Embodied AI 2025).  
**Advantages**: Explicit dynamics modeling improves generalization; mutual correction between dynamics and policy.  
**Limitations**: Additional computational overhead (0.34s vs 0.06s inference); requires random rollout data for dynamics training.

---

## 3. Representative Papers

### 3.1 FPO / Flow Matching Policy Gradients
**Citation**: Yi et al. (2026). *Flow Matching Policy Gradients*. arXiv:2507.21053 / ICLR.  
**Problem**: How to train expressive flow policies purely from rewards without explicit likelihood computation.  
**Core Contribution**: FPO ratio `exp(L_CFM,old - L_CFM,new)` as a drop-in replacement for the PPO likelihood ratio.  
**Methodology**: Three-step loop—(1) generate from flow, (2) compute advantages, (3) advantage-weighted flow matching loss.  
**Training Objective**: `max_θ E[min(ρ̂ A, clip(ρ̂) A)]` with `ρ̂ = exp(ΔCFM)`.  
**Architecture**: Standard DiT/U-Net velocity field predictor conditioned on observations.  
**Strengths**: Simple implementation; agnostic to sampling method; inherits PPO stability properties.  
**Weaknesses**: Standard formulation unstable for challenging robotics; gradient variance issues.  
**Influence**: Foundation for FPO++, SJFPO, and all subsequent flow RL methods.

### 3.2 FPO++ (arXiv:2602.02481)
**Citation**: Choi et al. (2026). *Flow Policy Gradients for Robot Control*. arXiv:2602.02481.  
**Problem**: FPO fails on challenging robotics (legged locomotion, humanoid control, manipulation).  
**Key Innovation**: Per-sample ratios (independent clipping per Monte Carlo pair) + ASPO trust region.  
**Mathematical Formulation**:  
- Per-sample ratio: `ρ̂^(i) = exp(ℓ_old^(i) - ℓ_new^(i))`  
- ASPO: PPO for A>0 (positive advantages), SPO for A<0 (negative advantages)  
**Architecture**: Flow model `v̂_θ` with Euler integration; zero-sampling (`ε=0`) at test time.  
**Training**: `max_θ E[ Σ ψ_ASPO(ρ̂^(i), Â) ]` over N_mc samples per action.  
**Strengths**: Stable training for humanoid locomotion; first sim-to-real flow policy; robust fine-tuning.  
**Weaknesses**: Per-sample ratios increase computation; ASPO introduces hyperparameter complexity (`clip_param`).  
**Influence**: Direct predecessor to SJFPO; validates flow RL for real-world robotics.

### 3.3 SJFPO (This Repository)
**Citation**: Spectral Joint Flow Policy Optimization (repo: `spectral-joint-policy`).  
**Problem**: Can a spectral noise prior improve flow-based RL policies?  
**Key Innovation**: Spectral base measure—`1/f^beta` profile via rfft gain along action-step axis (`freq_weighted_time` mode, `noise_scaling_beta=1.0`).  
**Implementation**: `fpo_core.py` — `spectral_filter_1d()` applies `mag = (1 + (kπ/H)^2)^(-β/2)`; `configuration_sjfpo.py` defines `noise_scaling_mode`.  
**Default Behavior**: `noise_scaling_mode='white'` → plain FPO++ (reproducible). Spectral claims require explicit flag.  
**Mathematical Note**: The spectral profile modifies the base measure of the flow from `N(0, I)` to `N(0, L L^T)`, making the velocity target consistent with the colored noise.  
**Strengths**: Minimal change to FPO++; preserves all mathematical guarantees; enables ablation studies.  
**Weaknesses**: Spectral claim requires explicit configuration; default behavior produces FPO++ results, not spectral results.  
**Influence**: Demonstrates how small modifications to the flow base measure can be integrated into RL frameworks without disrupting the core optimization.

---

## 4. Mathematical Foundations

### 4.1 Flow Matching and Conditional Flow Matching (CFM)

The probability density path `p_t(x)` flows from `p_0 = N(0, I)` to `p_1 = data`. The velocity field `v_t(x)` satisfies the continuity equation `∂p_t/∂t = -∇·(p_t v_t)`. The CFM objective minimizes:

```
L_CFM = E_{t, x, ε} [||v_θ(x_t, t) - (x - ε)||²]
```

where `x_t = (1-t)x + tε` for the OT schedule (Lipman et al., 2023).

### 4.2 FPO Ratio Derivation

For Gaussian policies, the likelihood ratio is `ρ = π_θ(a|o) / π_θ_old(a|o)`. For flow policies, exact likelihood requires divergence estimation (`∫ div(v) dt`), which is computationally prohibitive. FPO uses the ELBO relationship: the weighted CFM loss `L_w` equals the negative ELBO (`L_w = -ELBO`). Thus:

```
r_FPO = exp(ELBO_θ - ELBO_θ_old) = exp(L_w,old - L_w)
```

This provides an unbiased ratio estimate in the limit `N_mc → ∞`, with `E[ρ̂] ≥ ρ` (upward bias for single-sample ratios due to Jensen's inequality).

### 4.3 Spectral Noise Prior (SJFPO Contribution)

Standard flow uses white noise (`β=0`). The spectral prior colors noise with `std = (1 + (kπ/H)²)^(-β/2)` along the action-step grid, applied via rfft gain. The base measure becomes `N(0, L L^T)` where `L` represents the spectral coloring. The velocity target `v* = a - ε` remains consistent because both the noise and target velocity scale with the same spectral profile.

---

## 5. Methodological Comparison

| Dimension | Gaussian PPO | FPO | FPO++ | GPC | Streaming Flow |
|---|---|---|---|---|---|
| Policy Class | Unimodal | Multimodal | Multimodal | Multimodal | Multimodal |
| Training Data | Online rollouts | Online rollouts | Online rollouts | SPC simulation | Demonstrations |
| Likelihood | Exact | Surrogate (CFM Δ) | Per-sample CFM Δ | Supervised regression | Supervised regression |
| Trust Region | PPO clip | PPO clip | ASPO (asymmetric) | N/A (supervised) | N/A |
| Sampling | Direct | Euler/flow | Euler/flow | Flow + SPC bootstrap | Incremental integration |
| Spectral Prior | N/A | White (default) | White (default) | White (default) | White (default) |
| Inference Cost | Low | Medium (integration) | Medium | High (SPC rollouts) | Low (streaming) |
| Demonstration Required | No | No | No | No (sim-based) | Yes |

**Why different design decisions?**  
- FPO avoids likelihood computation at all costs—this motivates the CFM ratio approximation.  
- FPO++ introduces per-sample ratios because averaging ratios over `(τ, ε)` pairs loses fine-grained information; ASPO addresses the instability of negative-advantage updates where standard PPO provides zero gradients.  
- GPC chooses supervised learning over pure RL because dynamic tasks with noisy SPC data benefit from stable regression objectives.  

---

## 6. Experimental Landscape

### Common Datasets / Benchmarks
- **MuJoCo Playground**: Continuous control tasks for synthetic validation (FPO, FPO++)
- **Humanoid-v5 / Humanoid-v4**: High-dimensional humanoid control (FPO++, SJFPO)
- **Robomimic**: Manipulation benchmarks (Dynamics-Aligned Flow)
- **Legged Locomotion (T1, G1)**: Real-world robotics (FPO++)
- **Unicycle Dynamics**: Illustrative suite for variable-time flow matching

### Key Metrics
- **Policy Return / Success Rate**: Standard RL metrics
- **CFM Loss / Explained Variance**: Policy quality metrics
- **Gradient Norm (pre/post clip)**: Training stability indicator
- **Inference Latency**: Real-time control suitability
- **Simulation Speed**: Parallel GPU simulation for GPC-style methods

### Reproducibility Issues
The SJFPO README explicitly notes: *"A default run is plain FPO++. Any spectral claim needs the flag set explicitly (`noise_scaling_mode='freq_weighted_time'`)."* This is a critical reproducibility note—users running default configurations will reproduce FPO++ results, not spectral results. The `docs/DIAGNOSE_DEGRADATION.md` also highlights an open performance issue (Humanoid-v5 degradation at 131,072 steps), which is not yet fully resolved.

---

## 7. Scaling and Efficiency

**Computational Complexity**:  
- Flow sampling: `O(H × d_action × N_steps)` where `H` is chunk horizon (8 in SJFPO config).  
- FPO ratio estimation: `O(N_mc × H × d_action)` per action for `N_mc=8` Monte Carlo pairs.  
- Training: `chunks_per_update=1024`, `batch_size=128`, `grad_accum_steps=1`.

**Inference Costs**:  
- Zero-sampling (`ε=0`) improves test-time performance and eliminates stochasticity.  
- FPO++ requires Euler integration; streaming policies enable incremental action output during integration.

**Hardware**: GPU-accelerated parallel simulation (MuJoCo, Brax) enables massive rollouts for GPC; single-GPU training sufficient for FPO-family methods.

---

## 8. Open Problems

### 8.1 Performance Degradation (SJFPO / FPO++)
The `docs/DIAGNOSE_DEGRADATION.md` documents a 46% return decay on Humanoid-v5 (best at 131,072 steps, degraded by 614,400). The leading suspect is joint gradient clip saturation (2291× against budget of 1), but the actor/critic split is unrecorded, preventing attribution. **This remains unresolved.**

### 8.2 Spectral Prior Validity
Does `1/f^beta` spectral coloring actually improve robot learning? The SJFPO repo notes: *"The companion idea—a per-band trust region sharing that profile—is NOT implemented."* The spectral claim requires explicit flag setting; no control experiments (`white` vs. `freq_weighted_time`) are reported in the repository documentation.

### 8.3 Demonstration-Free Dynamic Control
GPC shows promise but requires expensive simulation cycles. Can flow policies learn fast dynamics without simulation or expert data?

### 8.4 Real-Time Streaming
Streaming Flow Policy (Jiang et al., 2025) addresses temporal consistency but requires architecture changes; integration with RL frameworks (FPO-style) remains unexplored.

---

## 9. Connections to Other Fields

**Diffusion/Flow Models (Generative AI)**: The CFM loss and velocity field parameterization come directly from Lipman et al. (2023) and continuous normalizing flow theory. The ELBO relationship connects flow matching to variational inference.

**Policy Gradient Methods (RL)**: FPO inherits the PPO framework—advantage estimation, trust regions, clipping. ASPO connects to Simple Policy Optimization (SPO) from prior RL literature.

**Sampling-Based Predictive Control (Robotics)**: GPC leverages the tight connection between generative policies and MPC (MPPI, CEM), identified by Psenka et al. (2023) and DIAL-MPC.

**Spectral Analysis (Signal Processing)**: The SJFPO spectral prior applies classical `1/f^beta` spectral coloring (pink noise) from signal processing to generative model noise distributions.

---

## 10. Current Research Frontier

### Mature Directions
- FPO-style policy gradients (well-validated, real-world deployment)
- Behavior cloning with diffusion/flow policies (mature, widely adopted)

### Emerging Directions
- **Variable-time planning**: Incorporating time as a learnable dimension in action chunks
- **Streaming policies**: Real-time action streaming during flow integration
- **Spectral base measures**: Non-white noise priors for structured exploration
- **Dynamics-aligned policies**: Joint dynamics-policy flow sharing
- **Supervised RL hybrids**: GPC-style cycles of simulation, model fitting, and bootstrap

### Active Debates
- Is the spectral prior (`freq_weighted_time`) a meaningful improvement over white noise, or just an interesting ablation? (SJFPO open question)
- Are explicit dynamics models necessary for robust generalization, or can flow policies implicitly learn dynamics through action chunk conditioning?

---

## 11. Research Roadmap

### Timeline of Key Ideas

```
2020-2022: Diffusion/Flow Models (Ho, Lipman)
    |
    v
2023: Diffusion Policy / Flow Matching for Robotics (Chi, Janner)
    |
    v
2024: GPC / Sampling-Based Predictive Control + Flow
    |
    v
2025: FPO (Yi et al. / arXiv:2507.21053) — Flow Policy Optimization
    |
    v
2026: FPO++ (Choi et al. / arXiv:2602.02481) — Per-sample ratios + ASPO
    |     SJFPO (This repo) — Spectral noise prior (1/f^beta)
    v
2025-2026: Variable-Time Flow (arXiv:2507.15073)
    Streaming Flow Policy (Jiang et al., 2025)
    Dynamics-Aligned Flow Matching (Embodied AI 2025)
```

### Dependency Graph (Simplified)

```
Diffusion/Flow Theory (Lipman 2023)
       |
       +--> Behavior Cloning (Chi 2023)
       |         |
       |         +--> GPC (arXiv:2502.13406)
       |         |
       +--> FPO (arXiv:2507.21053) <--- CFM Loss / ELBO
                 |
                 +--> FPO++ (arXiv:2602.02481) <--- Per-sample ratios / ASPO
                           |
                           +--> SJFPO (repo) <--- Spectral prior (freq_weighted_time)
```

### Reading Roadmap (Ordered by Prerequisites)

1. **Foundations**: Lipman et al. (2023) — Flow Matching; Ho et al. (2020) — Diffusion
2. **Robotics Application**: Chi et al. (2023) — Diffusion Policy; Black et al. — π0
3. **RL + Flow**: Yi et al. (2026/arXiv:2507.21053) — FPO; Psenka et al. — Q-score matching
4. **Advanced RL**: Choi et al. (2026/arXiv:2602.02481) — FPO++; Schulman et al. — GAE
5. **Spectral/Novel**: SJFPO repo documentation; `fpo_core.py` — spectral filtering
6. **Dynamic Tasks**: GPC (arXiv:2502.13406); Streaming Flow Policy (Jiang 2025)
7. **Variable-Time**: RL for Flow-Matching Policies (arXiv:2507.15073)
8. **Dynamics Integration**: Dynamics-Aligned Flow Matching Policy (Embodied AI 2025)

### Consensus, Disagreements, and Outlook

**Consensus**: Flow matching is a powerful framework for robot policy representation; combining it with RL (via CFM loss ratios) is viable and effective; FPO++ represents the current state-of-the-art for pure RL training of flow policies.

**Disagreements**: The spectral noise prior (`freq_weighted_time`) is unvalidated—no published results confirm its benefit over white noise. The performance degradation issue (Humanoid-v5) suggests that FPO-family methods may have stability limits not yet fully understood.

**Future Outlook**: The field is moving toward (1) faster inference (streaming, warm-starts), (2) dynamics integration, (3) variable-time planning, and (4) rigorous ablation studies of spectral/non-white base measures. The SJFPO repository offers a minimal-change framework for exploring these modifications.

---

*Note: This review synthesizes papers found through targeted searches (arXiv:2507.21053, arXiv:2602.02481, arXiv:2502.13406, Embodied AI 2025 proceedings) and the source code of the spectral-joint-policy repository (`README.md`, `fpo_core.py`, `configuration_sjfpo.py`, `docs/DIAGNOSE_DEGRADATION.md`). Claims about spectral performance are marked as unvalidated based on repository documentation.*
