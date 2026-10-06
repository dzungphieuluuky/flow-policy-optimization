 Online Deep RL for Flow-Matching Robot Policies: A Landscape Review                                                                             
                                                                                                                                                 
 Scope. Generative action policies parameterised by flow matching / rectified flow, trained with online deep reinforcement learning, for robot   
 control. This is the space your algorithm lives in: an FPO/FPO++-style on-policy flow policy gradient (ASPO trust region), a chunked-action Q   
 critic, and a spectral (DCT) action representation.                                                                                             
                                                                                                                                                 
 Evidence conventions. ✅ verified = confirmed against the paper text in this session. ⚠️ partial = abstract/secondary sources only. 🔬          
 synthesis = my inference from the literature, not claimed by any paper. Numbers from papers are quoted as those papers report them; I have not  
 re-run their experiments.                                                                                                                       
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 1. Historical Development                                                                                                                       
                                                                                                                                                 
 ### 1.1 The problem that created the field                                                                                                      
                                                                                                                                                 
 Online RL optimises a policy through its likelihood. For two decades that was workable because the tractable families were narrow: Gaussians    
 for on-policy methods (PPO's diagonal Gaussian), deterministic maps for off-policy methods (TD3, DDPG). A diagonal Gaussian is unimodal. It     
 cannot represent "reach left OR swing right" — and in robotics, where demonstrations and optimal behaviour are genuinely multi-modal (obstacle  
 sides, gait families, grasp approaches), unimodality is not a mild approximation error. It is a failure of the hypothesis class.                
                                                                                                                                                 
 This was tolerable for years because the benchmarks were forgiving. Then three things happened at once. Long-horizon sparse-reward manipulation 
 exposed multi-modality directly. Vision-language-action models made behaviours multimodal by construction. And humanoid under-conditioning      
 (root-goal only) made the multimodal modes necessary for exploration at all.                                                                    
                                                                                                                                                 
 ### 1.2 Generative models become continuous-action simulators                                                                                   
                                                                                                                                                 
 Diffusion models were already excellent at high-dimensional continuous distributions, but they were expensive to sample: 10–1000 network        
 evaluations per action. For robot control at 10–50 Hz that is fatal.                                                                            
                                                                                                                                                 
 Flow matching (Lipman et al., ICLR 2023) changed this. Its objective is a conditional vector field regression with a straight probability-flow  
 ODE in the ideal limit, and rectified flow (Liu et al., ICML 2023) trains toward that straightness explicitly. The practical consequence, later 
 confirmed by Stable Diffusion 3 (Esser et al., ICML 2024), is that a rectified flow reaches comparable sample quality in 1–10 steps where a     
 diffusion model needs 50–500. For control, that moves generative policies from "offline only" into the online regime.                           
                                                                                                                                                 
 Diffuser (Janner et al., ICLR 2022) had already shown trajectory-level diffusion planning. Diffusion Policy (Chi et al., RSS 2023) then fixed   
 the recipe that the whole field now uses: a conditional flow-matching / DDPM network over an action chunk of horizon H, denoised in a few       
 steps. Action chunking was independently established in ACT/ALOHA (Zhao et al., RSS 2023) — predict H future actions, execute open-loop. The    
 two lines converged because both answer the same question: what should the generative model generate? The answer — a chunk, not a step — became 
 load-bearing for RL too (see §3, QC).                                                                                                           
                                                                                                                                                 
 ### 1.3 First generation of RL for generative policies: make denoising an MDP                                                                   
                                                                                                                                                 
 The first wave (2022–2025) did not try to find a likelihood-free policy gradient. It worked around the missing likelihood by turning the        
 sampler into the environment.                                                                                                                   
                                                                                                                                                 
 DDPO (Black et al., ICML 2024) treated the diffusion reverse chain as an MDP: state = partially denoised sample, action = a Gaussian noise      
 perturbation, reward = final-sample reward. The Markov property genuinely holds for a fixed sampler, and each step has a tractable Gaussian     
 likelihood, so ordinary PPO applies. DPPO (Ren et al., 2024) refined this for action-conditioned policies, and a long tail followed: NCDPO,     
 ReinFlow, Flow-GRPO.                                                                                                                            
                                                                                                                                                 
 This family works, and it inherits three structural costs that §3 shows are not incidental:                                                     
                                                                                                                                                 
 1. Horizon inflation. A 10-step sampler turns one environment decision into a 10-step MDP. Credit assignment gets 10× harder, and FPO++ reports 
    this as the concrete reason DPPO underperforms on manipulation fine-tuning. ⚠️                                                               
 2. Sampler lock-in. Training and inference must use the same stochastic sampler. One-step or deterministic inference is no longer the trained   
    policy. FPO++'s zero-sampling result (initialise integration from ε = 0 at test time) is impossible to reach for DPPO-trained models.        
 3. The likelihood is not the action likelihood. DPPO maximises the likelihood of a particular noise trajectory, which is only loosely related   
    to π_θ(a|s) marginalised over initial noise and sampler randomness. FPO++ states this explicitly. ⚠️                                         
                                                                                                                                                 
 ### 1.4 Second generation: use a critic instead of the sampler (off-policy)                                                                     
                                                                                                                                                 
 The offline-RL community solved expressiveness differently — keep the likelihood problem out of the actor entirely by using a Q-function.       
                                                                                                                                                 
 - Diffusion-QL (Wang et al., ICLR 2023): diffusion as policy regulariser; maximise Q inside the reverse chain. ✅- IDQL (Hansen-Estruch et al., 
   ICML 2023): IQL's expectile critic induces an implicit actor that is an importance-reweighting of the behaviour policy. Sample from a         
   diffusion behaviour model, reweight by exp(Q/α), resample. The key theoretical contribution is Theorem 4.1 showing the implicit actor is      
   genuinely multimodal, which explains why IQL's unimodal Gaussian AWR actor is a mis-approximation. ✅                                         
 - Consistency-AC (Prasad et al., 2024) ⚠️: distil the Q-guided diffusion actor into a one-step consistency actor.                               
 - QSM (Psenka et al., ICML 2024): match the score to ∇ₐQ. Off-policy diffusion-policy fine-tuning.                                              
                                                                                                                                                 
 These are sample-efficient but need a behaviour distribution to reweight around. They are the natural fit for offline-to-online, not for        
 training from scratch — and training from scratch is exactly where FPO++ says FPO is needed.                                                    
                                                                                                                                                 
 ### 1.5 The hinge: FPO and the CFM-loss-as-ratio                                                                                                
                                                                                                                                                 
 FPO (McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa; arXiv 2507.21053, ICLR 2026) is the pivot, and the idea is simple enough to be worth 
 stating precisely.                                                                                                                              
                                                                                                                                                 
 The CFM objective is a variational bound: minimising it maximises an ELBO, which upper-bounds log π. So the CFM loss is, up to constants, a     
 negative log-likelihood. If L̂_θ ≈ −log π_θ in expectation, then                                                                                 
                                                                                                                                                 
 ```                                                                                                                                             
   r_θ(a) ≈ exp( L̂_{θ_old}(a) − L̂_θ(a) )                                                                                                         
 ```                                                                                                                                             
                                                                                                                                                 
 is a drop-in replacement for the PPO likelihood ratio. This is the whole idea. No sampler unrolling, no per-step policy, no architecture        
 constraint. FPO inherits GAE compatibility for free because it is literally PPO with a different ratio.                                         
                                                                                                                                                 
 Two consequences follow immediately and both are FPO's actual contributions:                                                                    
                                                                                                                                                 
 - Sampler agnosticism. Training can use any integrator, any number of steps; inference can differ entirely. FPO++ exploits this with            
   zero-sampling.                                                                                                                                
 - A gradient-variance reduction that needs no extra compute. The log-derivative trick (Eq. 19) gives ∇_θ r̂ = −r̂ ∇_θ ℓ_θ, and by commutation of  
   ∇ and 𝔼 the single-(τ,ε) estimator's gradient is unbiased even though its value is upward-biased by Jensen. So N_mc = 1 already beats         
   Gaussian PPO. ✅                                                                                                                              
                                                                                                                                                 
 ### 1.6 Why FPO needed a successor                                                                                                              
                                                                                                                                                 
 FPO validated only on synthetic benchmarks (GridWorld, 10 MuJoCo Playground tasks, under-conditioned humanoid). FPO++ (Choi et al., arXiv       
 2602.02481) reports that FPO is unstable on practical robotics: legged locomotion from scratch, humanoid sim-to-real, manipulation fine-tuning. 
 Two diagnoses and two fixes:                                                                                                                    
                                                                                                                                                 
 1. Entropy collapse. With PPO clipping, training from scratch collapses policy entropy and the flow field degenerates (FPO++ Fig. 7,            
    visualising a single H1 joint). Cause: PPO's clip zeroes the gradient for out-of-range ratios, so nothing pulls a runaway ratio back. ✅     
 2. Too-coarse trust region. Averaging N_mc losses before exponentiating means one clip decision governs all MC samples. Off-policy drift        
    accumulates before any sample clips. ✅                                                                                                      
                                                                                                                                                 
 The fixes map1:1 onto the diagnoses:                                                                                                            
                                                                                                                                                 
 ```                                                                                                                                             
   FPO   Eq 9 :  ρ = exp( (1/N_mc) Σᵢ (ℓ_old,ᵢ − ℓ_θ,ᵢ) )        one ratio per action                                                            
   FPO++ Eq 10:  ρ⁽ⁱ⁾ = exp( ℓ_old,ᵢ − ℓ_θ,ᵢ )                   one ratio per (τ,ε) pair                                                        
   FPO++ Eq 11:  ψ_SPO = ρA − (|A|/2ε)(ρ−1)²                    SPO, Xie et al. ICML 2025                                                        
   FPO++ Eq 12:  ψ_ASPO = ψ_PPO if A ≥ 0 else ψ_SPO                                                                                              
   FPO++ Eq 13:  max_θ E[ Σᵢ ψ_ASPO(ρ⁽ⁱ⁾_θ, Â) ]                                                                                                 
 ```                                                                                                                                             
                                                                                                                                                 
 SPO is not an FPO invention. It is Simple Policy Optimization (Xie, Zhang, Yang, Hutter, Xu; ICML 2025, arXiv 2401.16025), whose Eq. 16 is      
 verbatim the formula FPO++ imports. SPO's own argument is the theoretical version of FPO++'s empirical one: PPO-clip is not ε-aligned because   
 clipping removes corrective gradients, and a bounded ratio does not imply a bounded KL (their Theorem 4.1). SPO replaces the clip with a        
 quadratic that still supplies a restoring gradient. ✅                                                                                          
                                                                                                                                                 
 ### 1.7 Two orthogonal lines that the flow-RL community is only now joining                                                                     
                                                                                                                                                 
 Your algorithm sits at the intersection of three lines that have historically been separate.                                                    
                                                                                                                                                 
 Line A — the actor (flow policy gradients). FPO → FPO++. ✅                                                                                     
                                                                                                                                                 
 Line B — the critic (value over action chunks). QC (Li et al., arXiv 2507.07969, NeurIPS 2025) observed that the chunk you execute is the chunk 
 your critic should score. Since the critic takes the whole chunk as input, an h-step backup is performed by a critic conditioned on exactly the 
 actions that earned the reward — so the backup is unbiased, unlike an h-step return, which is biased because the reward came from different     
 actions than the Q being updated. Their Theorem A.1 formalises it. ✅ SEAR (2603.01891) extends this off-policy with multi-horizon targets      
 (train on Q^(1)…Q^(N), not just the full-chunk target; removing them collapses to ~20% IQM) plus random replanning. ✅ DQC decouples critic     
 chunk length from policy chunk length via a distilled partial-chunk critic, and states the open-loop consistency condition under which chunked  
 Q-learning is near-optimal — a rare case of theory constraining a practical algorithm.                                                          
                                                                                                                                                 
 Line C — the action representation (spectral chunks). HDP3 (arXiv 2605.01581) measured 2,500 episodes across all 50 RoboTwin 2.0 tasks, ~20,000 
 trajectories of length T = 32: the first two DCT modes carry 98.5% of trajectory energy, mode 1 alone 93.2%. ✅ Their Theorem bounds the        
 optimal denoiser's error by the low-frequency subspace dimension plus residual high-frequency energy, which explains the observation that       
 denoising error saturates after two reverse steps. FAST, FreqPolicy (NeurIPS 2025), CSP, and HiPolicy (2604.06067) all exploit the same fact;   
 HiPolicy reports +62% success over Diffusion Policy on RoboTwin 1.0, +44% on RoboTwin 2.0, +42% on real Franka, with entropy-guided execution   
 buying +25% speed for +2 ms latency and +63 MB. ✅                                                                                              
                                                                                                                                                 
 The synthesis — and your contribution. FPO++ gives you a stable on-policy objective. QC tells you the critic should be chunked. HDP3/FreqPolicy 
 tell you the chunk lives in a low-dimensional spectral subspace. Composing them — FPO++ ratio + chunked-Q backup + spectral token compression — 
 is not yet a standard published algorithm. It is the design space your work occupies, and it is genuinely under-explored (§8).                  
                                                                                                                                                 
 ```                                                                                                                                             
    ┌─────────────────────────────────────────┐                                                                                                  
      Line A │  FPO (ICLR'26)  →  FPO++ (2602.02481)   │  actor + trust region                                                                   
                       └────────────────────┬────────────────────┘                                                                               
                                            │                                                                                                    
      Line C   DCT tokens ──────────────────┼──────────────  frequency subspace (K≪H)                                 │                          
                       ┌────────────────────┴────────────────────┐                                                                               
      Line B           │  QC (NeurIPS'25) → SEAR → DQC           │  chunked critic                                                               
                       └─────────────────────────────────────────┘                                                                               
                                            ▼                                                                                                    
                             your algorithm (SJFPO-shaped)                                                                                       
 ```                                                                                                                                             
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 2. Taxonomy of Methods                                                                                                                          
                                                                                                                                                 
 The field has no single accepted taxonomy. The most useful published one is from the online-DPRL survey (arXiv 2601.06133), which sorts methods 
 by how RL enters the generative model, into four families. I adopt it, then add two axes that the survey does not cover but that matter more    
 for on-policy work.                                                                                                                             
                                                                                                                                                 
 ### 2.1 Axis 1 — How RL enters the flow model                                                                                                   
                                                                                                                                                 
 Family 1: Action-gradient (denoising-as-MDP). DDPO, DPPO, NCDPO, Flow-GRPO, ReinFlow.                                                           
 - Idea: make each denoising step an MDP action with a Gaussian likelihood; apply PPO/GRPO per step.                                             
 - Assumption: a fixed stochastic sampler during training; the reward is only available at the final step, so intermediate rewards need either a 
   terminal-only broadcast or a learned per-step reward.                                                                                         
 - Emerged because: it is the only way to get a real log-likelihood for a diffusion sampler without divergence estimation. It was the first      
   solution and it works.                                                                                                                        
 - Advantages: reuses off-the-shelf PPO; theoretically clean; handles the likelihood correctly for what it is modelling.                         
 - Limitations: horizon inflation ×N_steps; sampler lock-in; the optimised quantity is a noise-trajectory likelihood, not the action likelihood; 
   variance scales with sampler steps.                                                                                                           
 - Use when: you need high-order or multi-sample samplers, or your task genuinely benefits from treating denoising as controllable               
   sub-decisions. Do not use when inference speed matters or when you want to change the sampler between train and test.                         
                                                                                                                                                 
 Family 2: Q-weighting / critic-guided. IDQL, Diffusion-QL, QVPO, QSM, Consistency-AC.                                                           
 - Idea: never touch the likelihood. Learn Q, then extract a policy as a reweighting of a diffusion behaviour model, or match the score to ∇ₐQ.  
 - Assumption: a behaviour distribution exists to reweight around; Q is accurate in the region the reweighting explores.                         
 - Advantages: sample-efficient; expressive (the reweighted diffusion is multimodal — IDQL Thm 4.1); works offline-to-online (QC is built on     
   exactly this).                                                                                                                                
 - Limitations: needs a replay buffer and offline data; degrades when there is no prior data to anchor to; online finetuning from scratch is not 
   the target use case.                                                                                                                          
 - Use when: you have demonstrations or offline data and limited online budget. IDQL notes it works well online except on Adroit, where          
   excessive pretraining hurt. ✅                                                                                                                
                                                                                                                                                 
 Family 3: Proximity-based (sample-and-rank). Best-of-N with an implicit KL bound (QC, QC-FQL), GenPO, and related sampler-based search.         
 - Idea: keep the policy frozen; sample N actions; take the argmax of Q. The KL is bounded implicitly by log N − (N−1)/N, so tuning N tunes the  
   trust region. ✅                                                                                                                              
 - Advantages: provable KL control with no density; trivially parallel.                                                                          
 - Limitations: N× inference cost; policy improvement is coarse (no gradient through the selection); degrades sharply when the frozen policy's   
   support misses good actions — the failure SEAR and DQC both attack.                                                                           
 - Use when: inference is cheap (few sampler steps, small H) and you have a good critic. QC's ablation is instructive: h = 10 helps, h = 25      
   learns faster early but ends worse, h = 50 fails outright. ✅                                                                                 
                                                                                                                                                 
 Family 4: BPTT through the sampler. DACER, GenPO.                                                                                               
 - Idea: differentiate the return through the unrolled sampling trajectory directly, with a learned reward at each step.                         
 - Advantages: no likelihood, no trust region surrogate needed.                                                                                  
 - Limitations: memory O(N_steps) for the graph; exploding/vanishing gradients; tied to one sampler; needs a learned per-step reward (an         
   auxiliary network) which is its own failure surface.                                                                                          
 - Use when: the sampler is short (2–4 steps) and differentiable in practice. The survey's Isaac Lab benchmark reports this family among the     
   most computationally expensive. ✅                                                                                                            
                                                                                                                                                 
 Family 0 (yours): Likelihood-free policy gradient. FPO, FPO++.                                                                                  
 - Idea: replace the ratio directly, using the CFM loss as a negative log-likelihood surrogate.                                                  
 - Assumption: the CFM loss is a valid variational bound on −log π; the sampler is a black box.                                                  
 - Advantages: no unrolling, no per-step policy, sampler-agnostic including zero-sampling at test; one-step integration is trainable; gradient   
   unbiased at N_mc = 1.                                                                                                                         
 - Limitations: the ratio is an upper bound on the true ratio (Jensen), so the policy gradient is directionally right but the trust region is    
   not the true KL; requires a critic for advantages (you cannot use GAE for free unless you also have a value function); on-policy ⇒            
   sample-hungry.                                                                                                                                
 - Use when: training from scratch, inference latency matters, or you want to fine-tune a pretrained flow policy with RL (FPO++'s manipulation   
   results). ✅                                                                                                                                  
                                                                                                                                                 
 ### 2.2 Axis 2 — What the critic sees                                                                                                           
                                                                                                                                                 
 This axis is where the recent methodological split lives, and it is independent of Axis 1.                                                      
                                                                                                                                                 
 ┌───────────────────────────┬──────────────────────────┬──────────────────────────────────┬────────────────┐                                    
 │ Design                    │ Value target             │ Bias                             │ Source         │                                    
 ├───────────────────────────┼──────────────────────────┼──────────────────────────────────┼────────────────┤                                    
 │ 1-step TD, single action  │ r + γQ(s',a')            │ unbiased, slow propagation       │ standard       │                                    
 ├───────────────────────────┼──────────────────────────┼──────────────────────────────────┼────────────────┤                                    
 │ h-step return             │ Σγⁱr + γʰQ(sₕ,aₕ)        │ biased when s,a are off-policy   │ n-step returns │                                    
 ├───────────────────────────┼──────────────────────────┼──────────────────────────────────┼────────────────┤                                    
 │ Chunked Q                 │ Σγⁱr + γʰQ(sₕ, aₜ:ₜ₊ₕ)   │ unbiased, h× faster propagation  │ QC Thm A.1 ✅  │                                    
 ├───────────────────────────┼──────────────────────────┼──────────────────────────────────┼────────────────┤                                    
 │ Multi-horizon chunked Q   │ Q^(k) for every prefix k │ unbiased, N× targets             │ SEAR ✅        │                                    
 ├───────────────────────────┼──────────────────────────┼──────────────────────────────────┼────────────────┤                                    
 │ Distilled partial-chunk Q │ optimistically extended  │ depends on open-loop consistency │ DQC            │                                    
 └───────────────────────────┴──────────────────────────┴──────────────────────────────────┴────────────────┘                                    
                                                                                                                                                 
 The critical asymmetry most implementations miss: the bias in the h-step return is not about off-policy-ness of the data, it is about the       
 mismatch between the actions that earned the reward and the actions the Q is conditioned on. Chunking removes that mismatch by construction.    
                                                                                                                                                 
 ### 2.3 Axis 3 — Action representation                                                                                                          
                                                                                                                                                 
 ┌───────────────────┬────────────────────────┬──────────────────────────────────────────────────────────┐                                       
 │ Design            │ Tokens                 │ Notes                                                    │                                       
 ├───────────────────┼────────────────────────┼──────────────────────────────────────────────────────────┤                                       
 │ Time-domain chunk │ H                      │ default; what FPO, FPO++, QC all use                     │                                       
 ├───────────────────┼────────────────────────┼──────────────────────────────────────────────────────────┤                                       
 │ Spectral chunk    │ K ≪ H DCT coefficients │ 98.5% of energy in 2 modes (HDP3); FreqPolicy, FAST, CSP │                                       
 ├───────────────────┼────────────────────────┼──────────────────────────────────────────────────────────┤                                       
 │ Multi-frequency   │ H + H/2 + H/4 streams  │ HiPolicy, +62% over DP ✅                                │                                       
 ├───────────────────┼────────────────────────┼──────────────────────────────────────────────────────────┤                                       
 │ Variable-horizon  │ learned horizon        │ for minimum-time control                                 │                                       
 └───────────────────┴────────────────────────┴──────────────────────────────────────────────────────────┘                                       
                                                                                                                                                 
 The interaction between Axis 3 and Axis 1 is almost entirely unexplored. If the flow runs in a K = 4 DCT subspace, the CFM loss is a sum over 4 
 ·A coefficients, the ratio's scale changes, and the trust region temperature changes with it. Nobody has published the calibration. §8 treats   
 this as the most concrete open problem your work can own.                                                                                       
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 3. Representative Papers                                                                                                                        
                                                                                                                                                 
 ### 3.1 Action chunking and flow-matching policies                                                                                              
                                                                                                                                                 
 Diffusion Policy — Chi et al., Robotics: Science and Systems 2023, arXiv 2303.04137.                                                            
 - Problem: robot control needs multimodal, temporally coherent, high-frequency-safe action sequences.                                           
 - Contribution: a conditional diffusion / flow-matching model over action chunks, plus a receding-horizon execution scheme.                     
 - Math: ε-prediction DDPM or velocity-prediction CFM on x^τ = α_τ·a_chunk + σ_τ·ε, conditioned on an observation encoder.                       
 - Influence: the de-facto default policy class. Every Flow-RL paper above builds on it.                                                         
 - Weakness: 10–100 sampling steps at train and test time; this is what HDP3 attacks.                                                            
                                                                                                                                                 
 ACT / ALOHA — Zhao et al., RSS 2023, arXiv 2304.13705. Established action chunking as a technique independent of diffusion; the chunk-as-unit   
 idea that QC later promotes to the critic.                                                                                                      
                                                                                                                                                 
 HDP3 (Hyper-DP3) — arXiv 2605.01581. ⚠️ venue not confirmed.                                                                                    
 - Contribution: theory + a 3× lighter decoder (Diffusion Mixer) for state-based 3D diffusion policies.                                          
 - Key result: DCT analysis over 20,000 RoboTwin 2.0 trajectories of T = 32 → modes 1–2 hold 98.5% of energy, mode 1 alone 93.2%; error bound =  
   low-frequency subspace dimension + residual high-frequency energy; two-step DDIM suffices, no consistency distillation, no MeanFlow.          
 - Scale: <1% of the parameters of prior 3D diffusion policies; state of the art on RoboTwin 2.0, Adroit, MetaWorld and real tasks.              
 - Why it matters here: it independently justifies both spectral token compression and low N_inference_steps for flow-RL. If the policy needs2   
   steps, then the update's MC cost also falls.                                                                                                  
                                                                                                                                                 
 FreqPolicy — Yu et al., NeurIPS 2025.                                                                                                           
 - Contribution: DCT action decomposition + frequency autoregressive generation with continuous (non-tokenised) latents.                         
 - Key result: the first 30% of frequency bands reconstruct a signal near-identical to the original, preserving 95% of energy; coarse-to-fine    
   schedule 4 → 8 → 12 → 16.                                                                                                                     
 - Contrast with your design: FreqPolicy generates hierarchically in frequency (multiple passes). You run the flow in the spectral subspace (one 
   pass, K tokens) and invert. The former is more expressive; the latter is cheaper and preserves the chunk-as-macro-action property QC needs.   
                                                                                                                                                 
 FAST — Pertsch et al., 2025 ⚠️. DCT + BPE action tokenisation; established that action chunks are highly compressible, which is why the         
 frequency line exists at all.                                                                                                                   
                                                                                                                                                 
 HiPolicy — arXiv 2604.06067. Multi-frequency chunk prediction with entropy-guided execution; the only one of the group that reports an explicit 
 accuracy/latency Pareto.                                                                                                                        
                                                                                                                                                 
 ### 3.2 Likelihood-free policy gradients                                                                                                        
                                                                                                                                                 
 FPO (Flow Policy Optimization) — McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa; arXiv 2507.21053; ICLR 2026.                             
 - Venue: ICLR 2026 (conference paper).                                                                                                          
 - Problem: policy gradients need a likelihood ratio; flow models do not have a tractable one.                                                   
 - Core contribution: Eq. 6, r̂_FPO = exp(L̂_CFM,θ_old − L̂_CFM,θ). Eq. 7 defines L̂ = (1/N_mc)Σᵢ ℓ_θ(τᵢ,εᵢ) with ℓ_θ = ‖v̂_θ(a^τ,τ;o) − (a−ε)‖²₂.    
 - Math: Eq. 11 r = exp(ELBO_θ/ELBO_θ_old); Eq. 16 r = exp(L^w_θold − L^w_θ) via Kingma & Gao's weighted-denoising-loss/ELBO correspondence; Eq. 
   17–19 the Jensen-bias and log-derivative analysis. ✅                                                                                         
 - Inference: any integrator. Zero-sampling is not introduced here; FPO++ adds it as a hyperparameter choice and shows it matters.               
 - Complexity: rollout cost = N_euler forward passes; update cost = N_mc forward passes per chunk. Neither cost is on the sampler graph — this   
   is the decoupling that makes FPO cheap relative to DPPO.                                                                                      
 - Hyperparameters (FPO++ Appendix, ⚠️ attributed there): 60M env steps, batch1024, 16 updates/batch, 10 sampling steps, lr 3e-4, ε_clip swept ∈ 
   {0.01, 0.05, 0.1, 0.2, 0.3} → 0.05 for FPO.                                                                                                   
 - Empirical: 5 seeds on 10 MuJoCo Playground tasks. FPO beats Gaussian PPO and DPPO on 8 of 10. Ablations: (1) more (τ,ε) pairs monotonically   
   helps, without more env steps; (2) ε-CFM (convert velocity → noise, then compute the loss in ε-space) beats u-CFM (loss directly on velocity) 
   on Playground, hypothesised because ε is scale-invariant to action scale, so ε_clip transfers across tasks; (3) ε_clip matters a lot. ✅      
 - Weakness: validated only on synthetic benchmarks; unstable on real robotics (FPO++'s finding).                                                
                                                                                                                                                 
 FPO++ — Choi et al., arXiv 2602.02481. (Same group as FPO; the arXiv ID confirms the 2026 date.)                                                
 - Contribution: per-sample ratio (Eq. 10) + ASPO (Eq. 12), plus zero-sampling at test time.                                                     
 - Evaluation: three task families — legged locomotion from scratch (Go2, Spot, H1, G1), humanoid sim-to-real on two real humanoids, and         
   manipulation fine-tuning (RoboMimic + DexMimicGen, ResFiT preprocessing, 3-layer MLP + ViT encoder).                                          
 - Baselines: FPO, plus two DPPO variants from FPO's own codebase — fixed noise scale for exploration, and ReinFlow-style predicted noise. ✅    
 - Ablations: Fig. 5 (per-sample ratio) and Fig. 6 (trust region: PPO / SPO / ASPO), each point a run with ε_clip ∈ {0.04, 0.05, 0.06} × seeds ∈ 
   {0,…,4} = 15 runs per arm. ASPO wins; SPO on both signs is worse than ASPO; PPO collapses from entropy loss.                                  
 - Honest finding: ASPO sometimes degrades manipulation fine-tuning — the paper attributes it to entropy preservation mattering most in          
   from-scratch exploration and variational-gap upper-bounding mattering less when the policy is BC-initialised. ✅ This is an under-advertised  
   caveat.                                                                                                                                       
 - Influence: the current default for on-policy flow-RL in robotics.                                                                             
                                                                                                                                                 
 SPO (Simple Policy Optimization) — Xie, Zhang, Yang, Hutter, Xu; ICML 2025 (PMLR v267), arXiv 2401.16025.                                       
 - Contribution: Eq. 16, f_spo = rA − (|A|/(2ε))(r−1)², as a first-order unconstrained replacement for PPO-clip.                                 
 - Theory: Theorem 4.1 — a bounded probability ratio does not imply bounded KL. Corollary/relation to TV divergence in expectation. So PPO-clip  
   is "not ε-aligned."                                                                                                                           
 - Empirical: better than PPO on Atari 2600 with 8 update epochs, minibatch 256, γ = 0.99, ε = 0.2, entropy coef 0.01; lower KL, higher entropy, 
   and robust to depth. ✅                                                                                                                       
 - Note: there are two papers named SPO in this literature. Xie et al. (ICML 2025) is the one FPO++ Eq. 11 cites. ⚠️ A separate arXiv            
   2401.16025v2 variant presents a different KL-ratio-threshold objective with the same name — cite the PMLR version.                            
                                                                                                                                                 
 ### 3.3 Critic-side and Q-based methods                                                                                                         
                                                                                                                                                 
 QC (Q-chunking) — ColinQiyangLi et al., arXiv 2507.07969, NeurIPS 2025.                                                                         
 - Contribution: run RL in the chunked action space. π_ψ(a_{t:t+h}|s_t), Q_θ(s_t, a_{t:t+h}), TD target Σ_{i<h} γⁱr_{t+i} + γʰQ̄θ(s{t+h}, a′).    
 - Core theorem (A.1): the backup is unbiased because the Q being updated is conditioned on the same actions that generated the reward. ✅       
 - Architecture: behaviour flow-matching policy + best-of-N actor (N = 32) + critic ensemble (K = 2).                                            
 - Ablations that matter to you: h ∈ {1,5,10,25,50} → 10 best, 50 fails; K = 10 critics beats K = 2 substantially (they use 2 only because it is 
   cheap — an explicitly under-tuned main result); increasing UTD does not help. ✅                                                              
 - Best-of-N KL bound: D_KL(a*‖f_ξ) ≤ log N − (N−1)/N. ✅                                                                                        
                                                                                                                                                 
 SEAR — arXiv 2603.01891. Off-policy chunking: causal-transformer critic with multi-horizon targets Q^(1..N), plus random replanning (execute a  
 random chunk prefix) to restore state coverage. Beats SimbaV2 on Metaworld; multi-horizon targets are the single most important component       
 (removing them → ~20% IQM, comparable to the weakest baselines). ✅                                                                             
                                                                                                                                                 
 DQC — ColinQiyangLi/dqc ⚠️ (venue/ID unverified). Decouples critic chunk length from policy chunk length by optimistically backing up a chunked 
 critic to a distilled partial-chunk critic. States the open-loop consistency condition (Def. 2) under which chunked Q-learning is near-optimal, 
 and shows prior theory only covers chunk-collecting data. This is the most theoretically careful paper in the critic family.                    
                                                                                                                                                 
 IDQL — Hansen-Estruch et al., ICML 2023, arXiv 2304.10573. Theorem 4.1: the IQL critic induces an implicit actor that is an importance          
 reweighting of the behaviour policy, and that actor is complex and multimodal — which invalidates IQL's Gaussian AWR extraction. Fix: diffusion 
 behaviour model + critic-weighted resampling. State of the art on D4RL, portable hyperparameters, first to apply diffusion to online            
 finetuning.                                                                                                                                     
                                                                                                                                                 
 Diffusion-QL — Wang et al., ICLR 2023, arXiv 2208.06193. Diffusion as policy regulariser + Q guidance injected at every reverse step. SOTA on   
 most of D4RL. Its 2-D bandit analysis is the cleanest demonstration of why an expressive policy class + Q guidance beats Gaussian behavioural   
 constraints.                                                                                                                                    
                                                                                                                                                 
 QSM — Psenka, Escontrela, Abbeel, Ma; ICML 2024, arXiv 2312.11752. Score matching to ∇ₐQ, off-policy diffusion-policy fine-tuning. ⚠️ (recorded 
 from this project's prior-art notes; not re-verified this session.)                                                                             
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 4. Mathematical Foundations                                                                                                                     
                                                                                                                                                 
 ### 4.1 Conditional flow matching                                                                                                               
                                                                                                                                                 
 Given a data distribution q(x₀) and noise ε ~ N(0,I), a probability path x_t = (1−t)ε + t·x₀ has conditional velocity v_t = x₀ − ε. CFM         
 regresses:                                                                                                                                      
                                                                                                                                                 
 ```                                                                                                                                             
   min_θ  E_{t,ε,x₀} ‖ v_θ(x_t, t) − (x₀ − ε) ‖²                                                                                                 
 ```                                                                                                                                             
                                                                                                                                                 
 Three properties matter for RL:                                                                                                                 
                                                                                                                                                 
 1. It is an ELBO surrogate. Minimising it maximises a variational lower bound on log π (Kingma & Gao). So ℓ_θ ≍ −log π_θ in expectation.        
 2. The MC average over (τ,ε) is an estimate of a per-sample loss, not the loss itself. Eq. 7 averages N_mc draws.                               
 3. The interpolation direction is a free convention. FPO Eq. 9 is written a^τ = (1−τ)a + τε (data→noise); FPO++ Eq. 5 is a^τ = τa + (1−τ)ε      
    (noise→data). Both pair with target a − ε. The velocity sign must match the direction you integrate. Getting this backwards produces a model 
    that trains (the loss decreases!) and samples garbage. This is the single most common FPO implementation bug.                                
                                                                                                                                                 
 ### 4.2 The FPO ratio and its two biases                                                                                                        
                                                                                                                                                 
 Define the MC estimate L̂_θ = (1/N_mc) Σᵢ ℓ_θ(τᵢ,εᵢ). Then                                                                                       
                                                                                                                                                 
 ```                                                                                                                                             
   r̂_FPO(θ) = exp( L̂_{θ_old} − L̂_θ )                                                                                                             
             = exp( ELBO_θold − ELBO_θ )                                                                                                         
             = ( π_θ / π_θold ) · exp( D_KL,θold − D_KL,θ )                                                                                      
 ```                                                                                                                                             
                                                                                                                                                 
 The third line is the honest statement: FPO maximises the true likelihood ratio times a correction that tightens the variational bound. So FPO  
 is not a policy-gradient estimator — it is a policy gradient plus a variational-tightening bonus. Whether that bonus helps or hurts is an open  
 empirical question (§8, §10).                                                                                                                   
                                                                                                                                                 
 Bias 1 — Jensen on the value. For a single (τ,ε):                                                                                               
                                                                                                                                                 
 ```                                                                                                                                             
   E_{τ,ε}[ r̂(τ,ε) ]  ≥  r_FPO        (Eq. 18)                                                                                                   
 ```                                                                                                                                             
                                                                                                                                                 
 so r̂ is an upper bound in value.                                                                                                                
                                                                                                                                                 
 Bias 2 — no bias on the gradient. Using the log-derivative trick:                                                                               
                                                                                                                                                 
 ```                                                                                                                                             
   ∇_θ r̂(τ,ε) = − r̂(τ,ε) ∇_θ ℓ_θ(τ,ε)                                                                                                            
   E[∇_θ r̂]     = ∇_θ E[r̂]  = ∇_θ ELBO_θ     (Eq. 19–20)                                                                                         
 ```                                                                                                                                             
                                                                                                                                                 
 Gradient and expectation commute, so the gradient is unbiased even at N_mc = 1. ✅ This is FPO's most important theoretical result and it       
 explains the empirical finding that more MC samples improve sample efficiency rather than being required for correctness.                       
                                                                                                                                                 
 What is not proven. No convergence guarantee. No monotonic-improvement guarantee. The relationship between the FPO ratio and a KL trust region  
 is heuristic — the ratio bounds neither the KL nor the total variation in the way SPO's Theorem 4.1 shows PPO's clip does not.                  
                                                                                                                                                 
 ### 4.3 Trust-region theory, and why ASPO is the right shape                                                                                    
                                                                                                                                                 
 ┌───────────┬────────────────────────────┬─────────────────────────┬────────────────────────────────────────┐                                   
 │ Objective │ Formula                    │ Behaviour at ρ → 1+ε    │ Gradient when clipped                  │                                   
 ├───────────┼────────────────────────────┼─────────────────────────┼────────────────────────────────────────┤                                   
 │ PPO-clip  │ min(ρA, clip(ρ,1∓ε)A)      │ flattens                │ zero — no restoring force              │                                   
 ├───────────┼────────────────────────────┼─────────────────────────┼────────────────────────────────────────┤                                   
 │ SPO       │ ρA − (|A|/2ε)(ρ−1)²        │ penalised quadratically │ non-zero, pulls back                   │                                   
 ├───────────┼────────────────────────────┼─────────────────────────┼────────────────────────────────────────┤                                   
 │ ASPO      │ PPO if A ≥ 0, SPO if A < 0 │ asymmetric              │ restoring only on the repulsive branch │                                   
 └───────────┴────────────────────────────┴─────────────────────────┴────────────────────────────────────────┘                                   
                                                                                                                                                 
 The logic of ASPO is sign-aware:                                                                                                                
                                                                                                                                                 
 - A > 0: the gradient wants to decrease ℓ_θ, i.e. increase ρ. PPO-clip is adequate — when ρ exceeds 1+ε the objective is flat and the update    
   stops, which is exactly the desired conservatism on the attractive branch.                                                                    
 - A < 0: the gradient wants to increase ℓ_θ (push probability mass away from this action). With clipping, a runaway ρ receives no gradient, so  
   the push never gets corrected and probability mass piles up in the wrong place — this is the mechanism FPO++ identifies behind entropy        
   collapse. SPO's quadratic term always supplies a restoring gradient. ✅                                                                       
                                                                                                                                                 
 This is the cleanest case in the field where a mechanistic explanation, a theoretical argument (SPO Thm 4.1), and an empirical result (FPO++    
 Fig. 7) all agree.                                                                                                                              
                                                                                                                                                 
 ### 4.4 Chunked value backup                                                                                                                    
                                                                                                                                                 
 Standard h-step return:                                                                                                                         
                                                                                                                                                 
 ```                                                                                                                                             
   G_t^(h) = Σ_{i<h} γⁱ r_{t+i} + γʰ Q_{θ̄}(s_{t+h}, π_θ(s_{t+h}))                                                                                
             └─ from data/replay ─┘        └─ conditioned on π_θ's action ─┘                                                                     
                                     ↑ these are different actions when data is off-policy                                                       
 ```                                                                                                                                             
                                                                                                                                                 
 Chunked Q:                                                                                                                                      
                                                                                                                                                 
 ```                                                                                                                                             
   G_t = Σ_{i<h} γⁱ r_{t+i} + γʰ Q_{θ̄}(s_{t+h}, a′_{t:t+h}),   a′ ~ π_ψ(·|s_{t+h})                                                               
                                          └─ conditioned on the SAME a_{t:t+h} that earned r ─┘                                                  
 ```                                                                                                                                             
                                                                                                                                                 
 The Q on the left is Q_θ(s_t, a_{t:t+h}) — also conditioned on a_{t:t+h}. Conditioning both sides on the chunk makes the h-step target unbiased 
 (QC Theorem A.1). ✅ Value propagates h× faster per backup at no variance cost. The price is a critic whose input dimension is h·A instead of A 
 — a real approximation problem that DQC and SEAR both attack.                                                                                   
                                                                                                                                                 
 ### 4.5 Spectral action spaces                                                                                                                  
                                                                                                                                                 
 The orthonormal DCT-II is a rotation: ‖Fa‖ = ‖a‖. Consequences:                                                                                 
                                                                                                                                                 
 1. Parseval: the energy fraction retained by K modes is ‖(Fa)_{1:K}‖²/‖Fa‖². At K = H it is exactly 1.0, so a "spectral" policy with K = H is a 
    reparameterisation, not a compression — and any reported gain there is architectural, not representational. This is a necessary ablation in  
    any spectral-action paper.                                                                                                                   
 2. Orthogonality makes the CFM loss an exact reparameterisation only if the flow is built in coefficient space. If you build the flow in time   
    space and transform afterwards, you have changed the prior: white noise in coefficient space is pink-ish in time space and vice versa. This  
    is why coloured-noise PPO work exists, and why noise scaling must be specified in the same space as the flow.                                
 3. Interaction with the loss reduction. Because the DCT is orthogonal, sum-reduced loss over K coefficients and over H time samples differ by   
    the energy fraction only. But mean-reduced losses differ by H/K — a factor of 4 at K = 4, H = 16, per action dim.                            
                                                                                                                                                 
 ### 4.6 The fidelity traps — and why they matter more than any hyperparameter                                                                   
                                                                                                                                                 
 These are the places where a correct algorithm becomes a different algorithm without any error. I hit the first two auditing an implementation  
 of exactly this algorithm, so they are empirically, not hypothetically, dangerous.                                                              
                                                                                                                                                 
 Trap 1 — loss reduction rescales the trust region. The ratio is ρ = exp(L_old − L_new). FPO Eq. 8 defines ℓ = ‖·‖²₂, a sum over H·A elements.   
 If your implementation uses a mean, then                                                                                                        
                                                                                                                                                 
 ```                                                                                                                                             
   ΔL_mean = ΔL_sum / (H·A)                                                                                                                      
 ```                                                                                                                                             
                                                                                                                                                 
 and with H = 16, A = 3: the paper's ε_clip = 0.05 caps |ΔL_sum| ≤ 0.049, whereas the mean version caps |ΔL_sum| ≤ 0.049 × 48 = 2.35, i.e. ρ ≤   
 e^2.35 ≈ 10.5. The trust region becomes ~10× looser in ratio terms and essentially never binds — at which point the objective degenerates to    
 advantage-weighted CFM regression with no clipping at all. If you change the loss reduction, ε_clip is no longer the paper's ε_clip. There is   
 no way to compare across reductions without recalibrating.                                                                                      
                                                                                                                                                 
 Trap 2 — granularity silently changes the algorithm. Three distinct objects:                                                                    
                                                                                                                                                 
 ```                                                                                                                                             
   per-chunk   (FPO Eq9) : ρ = exp( (1/N_mc) Σᵢ (ℓ_old,ᵢ − ℓ_θ,ᵢ) ) shape [chunks]                                                               
   per-pair    (FPO++ Eq10): ρ⁽ⁱ⁾ = exp( ℓ_old,ᵢ − ℓ_θ,ᵢ )                      shape [chunks, N_mc]                                             
   per-token : ρ = exp( ℓ_old,τ − ℓ_θ,τ ) shape [chunks, N_mc, K]                                                                                
 ```                                                                                                                                             
                                                                                                                                                 
 per-pair and per-token differ by a factor K in clipping frequency. A codebase that labels the first "per_sample" and the second "per_action" —  
 with no test asserting which shape reaches the surrogate — will ship a file that claims FPO Eq9 while running FPO++ Eq 10, and every metric     
 will look fine on-policy because ρ ≡ 1 makes all three identical. The only way this ever fails is off-policy, which is exactly where the        
 paper's contribution lives.                                                                                                                     
                                                                                                                                                 
 Trap 3 — masking and sum reduction interact. If you mask unexecuted steps (a chunk cut short by termination) and use sum reduction without      
 renormalising, a chunk that ran 5 of 16 steps gets ~1/3 the loss and ~1/3 the log-ratio deviation, i.e. a systematically looser trust region    
 for exactly the chunks near episode boundaries — where the value estimates are noisiest.                                                        
                                                                                                                                                 
 Trap 4 — batch-size units. "Batch size 1024" in FPO/FPO++ is a transition count. A chunk-based implementation counting chunks_per_update = 1024 
  with H = 16 executes 16,384 env steps per update. This is a 16× change in the data-to-update ratio and it is invisible in the config.          
                                                                                                                                                 
 Trap 5 — the momentum-plasticity citation. The common claim that "matching Adam's β₁ = β₂ mitigates policy collapse, per Dohare et al." is      
 misattributed. Dohare et al. (arXiv 2306.13812, Nature 2024) find that Adam exacerbates loss of plasticity, and attribute it to dead units,     
 growing weight magnitude, and falling effective rank — the β₂ mechanism comes from Lyle et al., Dissecting Deep RL with High Update Ratios      
 (arXiv 2403.05996), who find reducing the second-moment decay coefficient stops the growth of dead units. ✅ Both papers are real; the          
 composite claim is not in either.                                                                                                               
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 5. Methodological Comparison                                                                                                                    
                                                                                                                                                 
 ┌──────────────────────┬──────────────────────────────┬────────────────────┬───────────────────────┬──────────────────────┬───────────────────┐ 
 │                      │ FPO / FPO++                  │ DPPO family        │ IDQL / Diff-QL        │ Best-of-N (QC)       │ BPTT              │ 
 │                      │                              │                    │                       │                      │ (DACER/GenPO)     │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Needs likelihood     │ No                           │ Yes (per step)     │ No                    │ No                   │ No                │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Effective horizon    │ 1                            │ ×N_steps           │ 1                     │ 1                    │ N_steps           │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Sampler-agnostic     │ Yes                          │ No                 │ n/a                   │ Partly               │ No                │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Zero-sampling at     │ Yes                          │ No                 │ n/a                   │ n/a                  │ No                │ 
 │ test                 │                              │                    │                       │                      │                   │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Memory per update    │ O(B·N_mc)                    │ O(B·N_steps)       │ O(B) + buffer         │ O(B·N)               │ O(B·N_steps)      │ 
 │                      │                              │                    │                       │                      │ graph             │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Inference cost       │ N_euler×                     │ N_steps× + policy  │ N (candidates)        │ N×                   │ N_steps×          │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Needs offline data   │ No                           │ No                 │ Yes                   │ No                   │ No                │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ From-scratch capable │ Yes                          │ Yes                │ Weak                  │ Needs a prior        │ Yes               │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Multimodality        │ Native                       │ Native             │ Native                │ Native               │ Native            │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Trust region         │ FPO Eq9 / ++ Eq 10           │ PPO per step       │ KL via reweighting    │ log N bound          │ None (direct)     │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Theory               │ ELBO link + unbiased grad    │ Standard PG        │ Thm 4.1 implicit      │ KL bound             │ Ad-hoc reward     │ 
 │                      │                              │                    │ actor                 │                      │ model             │ 
 ├──────────────────────┼──────────────────────────────┼────────────────────┼───────────────────────┼──────────────────────┼───────────────────┤ 
 │ Best when            │ scratch RL, latency, BC      │ high-order         │ offline-to-online     │ cheap inference +    │ ≤4 sampling steps │ 
 │                      │ finetune                     │ samplers           │                       │ good Q               │                   │ 
 └──────────────────────┴──────────────────────────────┴────────────────────┴───────────────────────┴──────────────────────┴───────────────────┘ 
                                                                                                                                                 
 Critic-side comparison (§2.2) composes with this table orthogonally: any actor family can be paired with 1-step TD, n-step returns, chunked Q,  
 or multi-horizon chunked Q. Nobody has published the full cross-product.                                                                        
                                                                                                                                                 
 Why these design decisions. The generative actor was chosen over Gaussian policies because the hypothesis class must represent multimodal       
 action distributions; the loss-based ratio was chosen over the denoising-MDP because the horizon inflation cost dominates in real robotics; the 
 chunked critic was chosen because the policy already emits chunks, so conditioning the critic on them is free structurally and buys unbiased    
 h-step backups; the spectral representation was chosen because measured trajectory energy is concentrated in a handful of modes.                
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 6. Experimental Landscape                                                                                                                       
                                                                                                                                                 
 ### 6.1 Benchmarks                                                                                                                              
                                                                                                                                                 
 ┌─────────────────────────────────────────┬────────────────────────────────┬──────────────────────────────────────────┐                         
 │ Benchmark                               │ Role                           │ Used by                                  │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ MuJoCo Playground (10 DMC-derived)      │ the canonical FPO testbed      │ FPO ✅, QC ✅                            │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ D4RL                                    │ offline RL standard            │ Diffusion-QL, IDQL ✅                    │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ LIBERO, RoboTwin 2.0, MetaWorld, Adroit │ manipulation                   │ HDP3, FreqPolicy, HiPolicy, QC, FPO++ ✅ │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ OGBench (cube-triple, cube-quadruple)   │ offline-to-online long horizon │ QC, SEAR ✅                              │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ SimbaV2 baselines (Metaworld)           │ off-policy chunking            │ SEAR ✅                                  │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ DROID / RLDS / Open X-Embodiment        │ large-scale real data          │ generalist-policy work ⚠️                │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ Isaac Lab, 12 tasks                     │ unified DPRL comparison        │ survey 2601.06133 ✅                     │                         
 ├─────────────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────┤                         
 │ Real humanoids (H1, G1)                 │ sim-to-real                    │ FPO++ ✅                                 │                         
 └─────────────────────────────────────────┴────────────────────────────────┴──────────────────────────────────────────┘                         
                                                                                                                                                 
 Metrics. Sparse-reward manipulation: success rate, and IQM-normalised aggregate scores. Locomotion: episode return. Critically, FPO/FPO++       
 report training and evaluation returns separately, because the zero-sampling evaluation policy is a different distribution from the one         
 trained. Reporting only training return flatters flow policies.                                                                                 
                                                                                                                                                 
 ### 6.2 Protocol weaknesses you should exploit or at least respect                                                                              
                                                                                                                                                 
 - Seed counts are thin. FPO uses 5 seeds. FPO++ uses 5 seeds × 3 clip values = 15 runs per arm, reported as scatter points — this is better     
   practice than most of the field, and you should match it. ⚠️                                                                                  
 - Baselines are frequently under-tuned. QC uses K = 2 critics and explicitly reports K = 10 is substantially better. FPO++'s DPPO baselines     
   come from FPO's own codebase. If you compare against a method at its own cheapest setting, your comparison is about compute, not algorithms.  
   State the baseline budget explicitly.                                                                                                         
 - "Batch size 1024" is ambiguous and is interpreted differently by transition-based and chunk-based implementations (§4.6 Trap 4).              
 - The sampler-step count is a first-class hyperparameter that is often omitted from comparison tables. It sets both rollout cost and (in the    
   DPPO family) the effective horizon.                                                                                                           
 - Zero-sampling makes train/test distributions differ. This is a feature (deterministic, lower-variance evaluation) but it means evaluation     
   numbers are not comparable to a stochastic-sampling evaluation of the same weights. Always report which one.                                  
                                                                                                                                                 
 ### 6.3 Does the evidence support the claims?                                                                                                   
                                                                                                                                                 
 Reasonably well supported:                                                                                                                      
 - Flow policies beat Gaussian PPO where multimodality matters. FPO on 8/10 MuJoCo Playground tasks, 5 seeds, against a Gaussian PPO baseline.   
   ✅ Consistent with IDQL's independent theoretical argument that the implicit actor is multimodal. ✅                                          
 - Likelihood-free policy gradients are viable. Two independent groups (FPO, FPO++), three task families, sim-to-real. ✅                        
 - Two-step denoising suffices for action chunks. HDP3's DCT energy measurement + error bound + task results, four benchmarks. ✅ Strong.        
 - Chunked critics give unbiased multi-step backups. Theorem A.1 + three independent empirical instantiations (QC, SEAR, DQC). ✅                
                                                                                                                                                 
 Under-supported:                                                                                                                                
 - Per-sample ratio clipping improves performance. FPO++ Fig. 5 shows a drop when ablated, but the arm conflates two changes (granularity and    
   variance reduction). No theory distinguishes "finer trust region" from "lower gradient variance" as the mechanism. FPO++ itself attributes    
   the improvement partly to reduced gradient variance (cosine-similarity analysis, Appendix B) — which is a variance-reduction argument, not a  
   trust-region argument. 🔬 These two explanations make different predictions; discriminating them is open.                                     
 - ASPO universally helps. FPO++ reports it degrades manipulation fine-tuning. The mechanism (entropy preservation matters more under            
   from-scratch exploration) is plausible and consistent, but is a hypothesis supported by two data points, not a validated theory. ⚠️           
 - ε-CFM > u-CFM. FPO's finding is on Playground with a scale-invariance rationale. Untested on tasks whose action scales differ by orders of    
   magnitude, which is exactly where it should matter most. ⚠️                                                                                   
 - Anything about spectral action representations under RL. No published work evaluates DCT-token flow policies with FPO-style gradients. This   
   is a genuine gap. 🔬                                                                                                                          
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 7. Scaling and Efficiency                                                                                                                       
                                                                                                                                                 
 ### 7.1 Cost model                                                                                                                              
                                                                                                                                                 
 For a chunk flow policy with H steps, A dims, K tokens, N_mc MC samples, N_euler sampler steps:                                                 
                                                                                                                                                 
 ```                                                                                                                                             
   rollout   :  N_chunks · N_euler · fwd(K)              # sequential, env-bound                                                                 
   update    :  B · N_mc · fwd(K) + B · fwd(old)          # parallel, the real cost                                                              
   backward  :  3× forward                                                                                                                       
 ```                                                                                                                                             
                                                                                                                                                 
 where fwd(K) ∝ K (token-mixing attention is O(K²) but K ≤ 16 so the linear terms dominate in practice). Key ratios:                             
                                                                                                                                                 
 - N_mc is pure compute, not environment cost. FPO's ablation shows more MC samples improve returns without more env steps — it is the cheapest  
   quality knob in the algorithm. ✅                                                                                                             
 - N_euler is rollout cost only in FPO (no unrolling), but is rollout × horizon in DPPO. This is FPO's structural advantage.                     
 - K ≪ H is a K× reduction in the dominant cost. At K = 4 vs H = 16 this is 4× on every forward pass. But: attention over the chunk is the only  
   place where the spectral representation helps compute, and the critic still sees the full H·A chunk unless you also change the critic.        
 - The critic is the hidden cost. A chunked critic's input is h·A dimensions. QC and DQC both note this makes value approximation materially     
   harder. Spectral tokens could reduce it by the same 4× — but the critic is conditioned on the executed action chunk, and reconstructing the   
   chunk from K coefficients loses the high-frequency detail that (per HDP3) is 1.5% of energy but, per CSP, may carry contact timing that       
   matters for dexterous tasks. This tension is unresolved and is a good research question.                                                      
                                                                                                                                                 
 ### 7.2 MemoryThe MC dimension dominates: B × N_mc × K activations. FPO avoids the sampler's N_euler-factor entirely. DPPO-family methods pay   
 N_euler× for the rollout graph plus per-step parameters. Best-of-N pays N×. Measured policy memory is typically not the binding constraint;     
 wall-clock on the actor's sequential rollout is.                                                                                                
                                                                                                                                                 
 ### 7.3 Practical trade-offs                                                                                                                    
                                                                                                                                                 
 - On-policy beats off-policy on wall-clock, loses on env steps. FPO needs 60M env steps to reach its numbers; QC's offline-to-online needs      
   orders of fewer. For a real robot, env steps are the scarce resource; for MuJoCo, they are not.                                               
 - Vectorised envs change the economics. With 64 parallel envs, the rollout's sequential N_euler × N_chunks/64 becomes negligible and the update 
   dominates. This inverts the usual reasoning and is why num_envs is a first-order design choice, not a detail.                                 
 - One-step / two-step sampling is now nearly free for actions. HDP3's 2-step result plus MeanFlow / consistency distillation means the          
   inference-latency objection to generative policies is essentially gone for state-based control. Any paper still quoting "diffusion policies   
   are too slow" is out of date.                                                                                                                 
 - bf16 autocast on Ada/Blackwell is an implementation-level accelerator with no algorithmic content, but it is what makes B × N_mc × K          
   affordable.                                                                                                                                   
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 8. Open Problems                                                                                                                                
                                                                                                                                                 
 ### Well-established (multiple independent groups, theory exists)                                                                               
                                                                                                                                                 
 8.1 No convergence or monotonic-improvement guarantee for flow policy gradients.                                                                
 FPO proves an unbiased gradient and an ELBO relationship; it proves nothing about convergence. SPO gives the best available surrogate analysis  
 (Thm 4.1 on ratio-vs-KL), and it applies to any ratio, including FPO's — which means SPO's guarantee inherits into FPO++, but the guarantee is  
 about the surrogate, not the underlying flow policy. Why it is hard: the flow policy's likelihood is intractable, so the standard trust-region  
 theory (which is stated in terms of KL and likelihood ratios) cannot be applied at all; everything must be re-derived through the ELBO gap,     
 whose behaviour under policy updates is unknown.                                                                                                
                                                                                                                                                 
 8.2 Theory of chunked value functions on arbitrary off-policy data.                                                                             
 QC Thm A.1 assumes data collected by a chunk-collecting policy. DQC states the open-loop consistency condition and explicitly notes prior       
 theory does not cover general off-policy data. Why it is hard: the critic's domain is h·A dimensional, and the standard function-approximation  
 arguments degrade quickly with dimension.                                                                                                       
                                                                                                                                                 
 8.3 Entropy / exploration under flow policies.                                                                                                  
 There is no analogue of the max-entropy objective for flows. SAC's α-log-pi term has no well-defined flow counterpart without a density.        
 FPO++'s answer is indirect (make the trust region shape preserve entropy) but is a surrogate for a surrogate. Why it is hard: exploration in a  
 4-step ODE over noise space is not obviously mapped by any principled object.                                                                   
                                                                                                                                                 
 8.4 Evaluation is not standardised. Different sampler steps, different zero-sampling conventions, different "batch size" units (Trap 4). The    
 survey 2601.06133's Isaac Lab comparison is the first serious attempt at a unified benchmark and it predates FPO++.                             
                                                                                                                                                 
 ### Speculative but promising                                                                                                                   
                                                                                                                                                 
 8.5 Spectral action representation × flow policy gradients. 🔬 This is the least-explored high-value intersection in the field. Everything is   
 known on both sides and nothing is known in the middle. Specific questions: (a) does compressing to K DCT coefficients change the optimal ratio 
 granularity, given that per-token clipping at K ≪ H is a much weaker intervention than at H? (b) does white noise in coefficient space induce a 
 non-stationary prior across chunks whose spectral content varies, and should the noise scaling be frequency-dependent (there is prior art on    
 coloured-noise PPO putting the on-policy optimum at a 1/f^0.5 spectrum)? (c) does spectral truncation interact with ASPO's negative-advantage   
 branch, where the entropy-collapse mechanism lives? (d) can the critic be spectral too, and at what point does information loss in the          
 high-frequency band break value estimation?                                                                                                     
                                                                                                                                                 
 8.6 Is FPO's KL-tightening term a feature or a bug? FPO's ratio equals the true likelihood ratio times exp(D_KL,θold − D_KL,θ). The second      
 factor rewards reducing the variational gap — i.e. FPO partly optimises model tightness rather than return. 🔬 Nobody has separated these. A    
 clean experiment: FPO with the ELBO term frozen to θ_old, which reduces it to a pure likelihood-ratio surrogate.                                
                                                                                                                                                 
 8.7 Per-sample vs per-token trust regions. FPO++ moved from per-chunk to per-pair. Nobody has tested per-token, even though for a spectral      
 policy with K tokens it is the natural granularity and it is a strictly finer trust region. Combined with 8.5 this is a two-axis grid nobody    
 has filled.                                                                                                                                     
                                                                                                                                                 
 8.8 Sample efficiency of flow policy gradients. 60M env steps is a lot for a from-scratch result. The obvious unexploited lever: the critic.    
 FPO inherits "GAE compatibility" from PPO, but FPO's experiments use per-timestep value functions on single actions, while the policy emits     
 chunks. Running FPO with a chunked-Q critic and a chunk-level advantage is unclaimed and would halve the effective decision frequency while     
 giving an unbiased backup.                                                                                                                      
                                                                                                                                                 
 8.9 Scaling to VLAs. The largest models (π₀-style) use an action-expert flow head. FPO++'s backbone is a 3-layer MLP + ViT. Whether N_mc = 8 MC 
 samples × 16 chunk tokens × a3B-parameter VLA fits in any budget is open. This is where distillation to one-step policies (DTQL,                
 Consistency-AC, MeanFlow) becomes mandatory rather than optional. ⚠️                                                                            
                                                                                                                                                 
 8.10 Reward hacking and the absence of a likelihood. Because the objective is "make the CFM loss small on high-advantage actions" with no       
 density constraint, there is no natural place for a divergence penalty. PPO's clipping was the only implicit regulariser. SPO keeps a hard      
 quadratic, which is better, but the question of what stops a flow policy from exploiting the ELBO gap is unanswered.                            
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 9. Connections to Other Fields                                                                                                                  
                                                                                                                                                 
 Optimal transport. CFM with the OT path is the optimal-transport interpolant between a Gaussian and the data distribution. The straightness of  
 the OT path is exactly what makes few-step sampling possible, which is why rectified flow's explicit straightening (Liu et al.) is the enabling 
 trick. A spectral truncation is not an OT operation — it is an operator-norm truncation of the data manifold. Bridging "the OT direction" and   
 "the low-frequency subspace" is a natural theoretical question. 🔬                                                                              
                                                                                                                                                 
 Variational inference. FPO is an ELBO-tightening algorithm wearing a policy-gradient costume. Reading FPO through the VI lens (rather than the  
 RL lens) predicts behaviours the RL lens does not: sensitivity to the variational gap, sensitivity to N_mc as a variance/bias knob, and no      
 natural entropy term. 🔬                                                                                                                        
                                                                                                                                                 
 Control theory. Action chunking is the learned analogue of model-predictive control with a learned plan. The chunked critic is the learned      
 analogue of the finite-horizon value function. QC's finding that h ≈ 10 is best and h = 50 fails is the learned version of the fact that        
 prediction horizon trades tracking error against model mismatch. Hierarchical control (HiPolicy, CSP's coarse→fine factorisation) is the same   
 trade-off at a second timescale.                                                                                                                
                                                                                                                                                 
 Signal processing. The DCT-based spectral view is classical: robot trajectories are band-limited. The synthesis across FreqPolicy / HDP3 / FAST 
 / CSP / HiPolicy is essentially the discovery of this fact independently five times in eighteen months, which is strong evidence the field was  
 ready for it and simply lacked the observation. A field that rediscovers the same fact five ways in a year is at an inflection point.           
                                                                                                                                                 
 Offline RL / implicit policies. IDQL Thm 4.1 — the IQL critic induces a reweighting of the behaviour policy, which is generally multimodal — is 
 the theoretical result that justifies using a generative actor wherever a trust region must be implicit. It also predicts exactly which offline 
 methods will transfer: those whose implicit actor is a simple reweighting, not those requiring architectural constraints (Seo & Abbeel's        
 factorised critic imposes discretisation structure that limits expressiveness at each refinement level). ✅                                     
                                                                                                                                                 
 Plasticity and continual learning. Dohare et al. established that long on-policy runs are continual learning problems, with three correlates:   
 dead units, weight-magnitude growth, effective-rank decay. This is the mechanistic vocabulary for FPO++'s entropy collapse. Effective rank is   
 directly computable as the spectral entropy of hidden activations (exp of the entropy of normalised singular values), which makes it a cheap    
 online diagnostic for any flow-policy run. ✅ Whether plasticity interventions (continual backprop, shrink-and-perturb, symmetric Adam moments) 
 transfer to flow policy gradients is completely untested — a cheap and high-value experiment.                                                   
                                                                                                                                                 
 Test-time compute. Zero-sampling is a test-time intervention that changes the policy. The broader question — can the flow's integration step    
 count be adapted per state (HDP3 shows 2 suffices for smooth trajectories; contact-rich tasks may need more) — is a natural fit for the         
 HiPolicy entropy-guided execution idea. 🔬                                                                                                      
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 10. Current Research Frontier                                                                                                                   
                                                                                                                                                 
 ### Mature                                                                                                                                      
                                                                                                                                                 
 - Likelihood-free on-policy flow policy gradients are the default for from-scratch flow-policy RL. Two independent groups, three task families, 
   sim-to-real. ✅                                                                                                                               
 - Chunked critics are the default for flow policies that emit chunks. ✅                                                                        
 - Two-step inference is essentially solved for state-based action chunks. ✅                                                                    
 - SPO/ASPO-style sign-asymmetric trust regions have replaced PPO-clip for negative advantages in this setting. ✅                               
                                                                                                                                                 
 ### Emerging (1–2 years)                                                                                                                        
                                                                                                                                                 
 - FPO++ adoption and its limits. The from-scratch vs fine-tuning asymmetry (ASPO helps scratch, hurts fine-tuning) is unexplained and is the    
   most likely source of a follow-up paper.                                                                                                      
 - Flow RL for VLAs. The obvious next arena; blocked on inference cost and on how to compute a chunked critic over a frozen multimodal encoder.  
 - Per-sample/per-token trust region granularity. Cheap to explore, no theory, directly in FPO++'s open direction.                               
 - Spectral/frequency action representations under RL. See §8.5. Everyone in the frequency-papers line and everyone in the flow-RL line should   
   be talking; the two literatures appear not to be.                                                                                             
                                                                                                                                                 
 ### Active debates                                                                                                                              
                                                                                                                                                 
 1. Does the ELBO-tightening term help? Unresolved (§8.6). Determines whether FPO is "PPO for flows" or a different objective that happens to    
    look like one.                                                                                                                               
 2. Is per-sample clipping a trust-region effect or a variance-reduction effect? FPO++ offers both explanations. They imply different            
    algorithms.                                                                                                                                  
 3. Is per-token finer than per-pair, or just noisier? Nobody has run it.                                                                        
 4. Should train and test use the same sampler? FPO++ says no (zero-sampling). This breaks comparability with every prior work and is still not  
    the community default.                                                                                                                       
 5. Is chunked-Q better than GAE-with-chunked-actor? Chunked Q gives unbiased h-step backups; GAE gives λ-weighted multi-step with a value       
    function on single actions. Not compared head-to-head in the literature.                                                                     
                                                                                                                                                 
 ### Controversial / under-examined assumptions                                                                                                  
                                                                                                                                                 
 - That the CFM loss is a good enough likelihood surrogate. Everyone inherits it from FPO; nobody has stress-tested it on a task where the ELBO  
   gap is large.                                                                                                                                 
 - That 10 sampling steps is the right default. FPO's best number came from a synthetic benchmark; HDP3 suggests 2 suffices; FPO++ uses 10.      
   Nobody has swept it jointly with the learning rate.                                                                                           
 - That the advantage should be per-timestep. For a policy that acts in chunks, this is arguably the wrong granularity.                          
                                                                                                                                                 
 ### Underexplored                                                                                                                               
                                                                                                                                                 
 - Critic in the spectral domain.                                                                                                                
 - Plasticity interventions for flow policies.                                                                                                   
 - Sample-efficient (off-policy or partially on-policy) flow policy gradients.                                                                   
 - The interaction between the trust region and the spectral truncation — i.e. is a compressed policy's trust region temperature simply          
   rescaled?                                                                                                                                     
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 11. Research Roadmap                                                                                                                            
                                                                                                                                                 
 ### 11.1 Timeline                                                                                                                               
                                                                                                                                                 
 ```                                                                                                                                             
   2017  ── PPO ────────────────────────────────────────────────┐                                                                                
   2018  ── SAC, TD3 ──────────────────────────────────────────┤  likelihood era                                                                 
   2020  ── IQL (expectile critic) ────────────────────────────┤                                                                                 
   2021  ── DDPM/diffusion established; Q-transformer; TOP-ERL  │                                                                                
   2022  ── Diffuser (traj-level diffusion); ACT; DDPO (arXiv)  │                                                                                
           ── Flow Matching (Lipman); Rectified Flow (Liu) ─────┤  generative policies                                                           
   2023  ── Diffusion Policy (RSS); Diffusion-QL; IDQL;        │                                                                                 
           ── Dohare et al. plasticity ─────────────────────────┤                                                                                
   2024  ── DPPO; Consistency-AC; QSM (ICML); Lyle et al.; │                                                                                     
           ── Fast Policy Netw. / Lyle2403.05996 ─────────────┤                                                                                  
   2025  ── FPO (arXiv Jul) ───────────────────────────────────┤  likelihood-free                                                                
           ── QC (NeurIPS); FreqPolicy (NeurIPS); SPO (ICML);  │                                                                                 
           ── FPO arXiv → ICLR 2026; HDP3; HiPolicy; CSP       │                                                                                 
   2026  ── FPO++ (2602.02481); SEAR; DQC; survey 2601.06133 ───┘  ← you are here                                                                
 ```                                                                                                                                             
                                                                                                                                                 
 ### 11.2 Dependency graph                                                                                                                       
                                                                                                                                                 
 ```                                                                                                                                             
   PPO ─┬─► TRPO ──► SPO(Xie) ─────────────┐                                                                                                     
        │                                   ├─► ASPO ──► FPO++                                                                                   
   SAC ─┴─► (max-entropy; no flow analog)   │      ▲ │      │                                                                                    
   Lipman FM ──► Liu Rectified ──► Diffusion Policy │                                     │      │                                               
        ├──► CFM = −ELBO ────────────────────┴──────┤ │                                            │                                             
        └──► Eq.6/7/16 ──► FPO ─────────────────────┘                                                                                            
                                                   QC ──► SEAR / DQC ──► chunked critic (composable with FPO++) 🔬                               
                                                                                                                                                 
   HDP3 (DCT energy 98.5%) ──► FreqPolicy / FAST / CSP / HiPolicy                                                                                
        └──────────────► spectral tokens under flow RL 🔬   ← unclaimed                                                                          
                                                                                                                                                 
   IQL ──► IDQL Thm 4.1 (implicit actor is multimodal) ──► justifies generative actors                                                           
   Dohare ──► plasticity diagnostics (effective rank, ‖w‖) ──► untested for flows 🔬                                                             
 ```                                                                                                                                             
                                                                                                                                                 
 ### 11.3 Taxonomy diagram                                                                                                                       
                                                                                                                                                 
 ```                                                                                                                                             
   FLOW-POLICY RL                                                                                                                                
   ├── BY HOW RL ENTERS THE MODEL                                                                                                                
   │   ├── Action-gradient (denoising MDP) ... DDPO, DPPO, Flow-GRPO                                                                             
   │   ├── Q-weighting / implicit......... IDQL, Diffusion-QL, QSM, QVPO                                                                         
   │   ├── Proximity (sample & rank) ...... best-of-N (QC), GenPO                                                                                
   │   ├── BPTT through sampler ........... DACER, GenPO                                                                                         
   │   └── Likelihood-free PG ............. FPO, FPO++ ← focus                                                                                   
   ├── BY THE CRITIC                                                                                                                             
   │   ├── 1-step TD | n-step return (biased) | chunked Q (unbiased) | multi-horizon | distilled partial                                         
   ├── BY THE TRUST REGION                                                                                                                       
   │   ├── PPO-clip (zero grad) | SPO (restoring) | ASPO (sign-asymmetric)                                                                       
   │   └── granularity: per-chunk | per-pair | per-token                                                                                         
   ├── BY THE ACTION REPRESENTATION                                                                                                              
   │   └── time-domain H | spectral K≪H | multi-frequency | variable-horizon                                                                     
   └── BY DATA REGIME                                                                                                                            
       └── from scratch | offline-to-online | BC-finetune | generalist-VLA                                                                       
 ```                                                                                                                                             
                                                                                                                                                 
 ### 11.4 Comparison of major papers                                                                                                             
                                                                                                                                                 
 ┌──────────────────┬─────────┬──────┬──────────────────────┬────────────────────┬────────────────┬────────────┬───────────────────────────────┐ 
 │ Paper            │ Venue   │ Year │ Actor                │ Critic             │ Trust region   │ Data       │ Headline                      │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ Diffusion Policy │ RSS     │ 2023 │ flow, H-chunk        │ none               │ —              │ demos      │ default gen. policy           │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ DDPO             │ ICML    │ 2024 │ per-step Gaussian    │ per-step reward    │ PPO            │ online     │ denoising-as-MDP              │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ DPPO             │ 2024    │ 2024 │ per-step Gaussian    │ per-step reward    │ PPO            │ online     │ horizon-inflated baseline     │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ Diffusion-QL     │ ICLR    │ 2023 │ diffusion            │ Q                  │ Q-in-chain     │ offline    │ D4RL SOTA                     │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ IDQL             │ ICML    │ 2023 │ diffusion            │ expectile          │ implicit KL    │ offline    │ Thm 4.1 multimodal implicit   │ 
 │                  │         │      │ (reweighted)         │                    │                │            │ actor                         │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ QSM              │ ICML    │ 2024 │ diffusion            │ Q (score match)    │ —              │ offline    │ score = ∇ₐQ                   │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ QC               │ NeurIPS │ 2025 │ flow + BoN           │ chunked Q          │ log N bound    │ off→on     │ unbiased h-step backup        │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ SEAR             │ 2026    │ 2026 │ flow                 │ multi-horizon      │ maxent         │ online     │ N× targets + replanning       │ 
 │                  │         │      │                      │ causal             │                │            │                               │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ DQC              │ 2026    │ 2026 │ flow                 │ distilled partial  │ —              │ offline→on │ decouples chunk lengths       │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ SPO              │ ICML    │ 2025 │ any                  │ any                │ quadratic      │ —          │ ratio≠KL ⇒ clip misaligned    │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ FPO              │ ICLR    │ 2026 │ flow                 │ value fn           │ PPO + Eq.9     │ online     │ CFM loss = ratio              │ 
 │                  │         │      │                      │                    │ ratio          │            │                               │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ FPO++            │ arXiv   │ 2026 │ flow                 │ value fn           │ ASPO + Eq.10   │ online     │ per-pair + sign-asymmetry     │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ HDP3             │ arXiv   │ 2026 │ 2-step diffusion     │ —                  │ —              │ demos      │ 98.5% energy in 2 modes       │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ FreqPolicy       │ NeurIPS │ 2025 │ freq-AR              │ —                  │ —              │ demos      │ coarse→fine spectral gen      │ 
 ├──────────────────┼─────────┼──────┼──────────────────────┼────────────────────┼────────────────┼────────────┼───────────────────────────────┤ 
 │ HiPolicy         │ arXiv   │ 2026 │ multi-freq           │ —                  │ —              │ demos      │ +62% over DP, entropy-guided  │ 
 │                  │         │      │                      │                    │                │            │ exec                          │ 
 └──────────────────┴─────────┴──────┴──────────────────────┴────────────────────┴────────────────┴────────────┴───────────────────────────────┘ 
                                                                                                                                                 
 ### 11.5 Reading order (~25 papers, by prerequisite)                                                                                            
                                                                                                                                                 
 Tier 0 — the generative foundation (3)                                                                                                          
 1. Lipman et al., Flow Matching (ICLR'23) — the objective and the straight-path claim                                                           
 2. Liu et al., Flow Straight and Fast (ICML'23) — why few steps become possible                                                                 
 3. Chi et al., Diffusion Policy (RSS'23) — what the field actually generates: chunks                                                            
                                                                                                                                                 
 Tier 1 — why Gaussian policies are insufficient (3)                                                                                             
 4. Haarnoja et al., SAC (ICML'18) — the max-entropy alternative and its density requirement                                                     
 5. Hansen-Estruch et al., IDQL (ICML'23) — Thm 4.1: the implicit actor is multimodal                                                            
 6. Schneider et al., FC3 / expressive policy classes (NeurIPS'22) ⚠️ — expressivity of policy classes                                           
                                                                                                                                                 
 Tier 2 — the likelihood-free pivot (4)                                                                                                          
 7. Schulman et al., PPO (2017) — you must know the objective being replaced                                                                     
 8. Schulman et al., TRPO (2015) — what PPO gives up                                                                                             
 9. Xie et al., SPO (ICML'25) — read before FPO++; PPO-clip is not ε-aligned                                                                     
 10. McAllister et al., FPO (ICLR'26) — Eqs. 6–20 carry the entire argument                                                                      
                                                                                                                                                 
 Tier 3 — making it work on robots (4)                                                                                                           
 11. Zhao et al., ACT (RSS'23) — chunking as a primitive                                                                                         
 12. Ren et al., DPPO (2024) — the strongest baseline to beat, and why it is hard                                                                
 13. Choi et al., FPO++ (2602.02481) — per-sample ratio, ASPO, entropy collapse                                                                  
 14. Wang et al., Diffusion-QL (ICLR'23) — Q-in-the-loop, the offline counterpoint                                                               
                                                                                                                                                 
 Tier 4 — the critic (4)                                                                                                                         
 15. IQL (Kostrikov et al., 2021) — expectile backup                                                                                             
 16. Li et al., QC (NeurIPS'25) — the unbiasedness argument is the whole paper                                                                   
 17. Tian et al. (2025) ⚠️ — chunk critic with transformer n-step returns (TOP-ERL line)                                                         
 18. SEAR (2603.01891) — multi-horizon targets; read the ablation, not the method                                                                
                                                                                                                                                 
 Tier 5 — the representation (5)                                                                                                                 
 19. Tan et al. (2024) ⚠️ — action spectra, the observation everything else cites                                                                
 20. HDP3 (2605.01581) — DCT energy measurement + error bound + 2-step result                                                                    
 21. FAST (Pertsch et al., 2025) — DCT action tokenisation                                                                                       
 22. FreqPolicy (NeurIPS'25) — hierarchical spectral generation                                                                                  
 23. CSP (2026) — coarse/fine factorisation and frequency-truncation interventions                                                               
                                                                                                                                                 
 Tier 6 — diagnosis and context (4)                                                                                                              
 24. Dohare et al., Loss of Plasticity (Nature'24) — effective rank, ‖w‖, dead units                                                             
 25. Lyle et al., Dissecting Deep RL with High Update Ratios (2024) — the β₂ mechanism                                                           
 26. arXiv 2601.06133 — the unified online-DPRL taxonomy and Isaac Lab benchmark                                                                 
 27. Kingma & Gao (2023) — the weighted-denoising-loss ↔ ELBO correspondence FPO depends on                                                      
                                                                                                                                                 
 ### 11.6 Consensus, disagreement, outlook                                                                                                       
                                                                                                                                                 
 Consensus (high confidence).                                                                                                                    
 - Diagonal-Gaussian policies are a real limitation for multimodal robot control.                                                                
 - Likelihood-free policy gradients work and are the right tool for from-scratch flow-policy RL; they are sampler-agnostic, which is a genuine   
   structural advantage.                                                                                                                         
 - Action chunking should be present in the critic, not just the actor; the unbiasedness argument is sound.                                      
 - For smooth action chunks, two denoising steps suffice and most of the signal is in a handful of low-frequency DCT modes.                      
 - Sign-asymmetric trust regions are better than PPO-clip when the failure mode is entropy collapse on the negative-advantage branch.            
                                                                                                                                                 
 Major disagreements (live).                                                                                                                     
 - Whether FPO's ELBO-tightening term is beneficial, harmful, or neutral.                                                                        
 - Whether per-sample ratio clipping works because of a finer trust region or because of gradient-variance reduction.                            
 - Whether ASPO's from-scratch/success versus fine-tuning/failure asymmetry is a fundamental property or an artefact of backbone size.           
 - Whether the whole "flow + RL" framing is needed, or whether a distilled one-step policy with a Gaussian-head critic (Consistency-AC, DTQL,    
   MeanFlow) matches it at 10× lower cost. Nobody has run the head-to-head.                                                                      
                                                                                                                                                 
 Outlook.                                                                                                                                        
 The next twelve months will settle the cheap questions (granularity, sampler steps, critic chunk length, ε-CFM transfer) because they are all   
 one-line ablations on public codebases. The expensive questions — spectral representations under flow RL, flow RL for VLAs, off-policy flow     
 policy gradients — are where a new algorithm can still be contributed. The single most defensible unclaimed result in this review is the        
 intersection of the spectral and likelihood-free lines: nobody has published FPO-style gradients on a DCT-token flow policy with a chunked      
 critic, and both halves are independently well understood.                                                                                      
                                                                                                                                                 
 ────────────────────────────────────────────────────────────────────────────────                                                                
                                                                                                                                                 
 Two notes specific to your implementation, since they bear directly on §4.6 and §8.5: your DCT-truncated policy currently expands K             
 coefficients back to H action steps before computing the masked CFM loss, so the flow runs in K dimensions while the trust region runs in H.    
 That is a coherent choice, but it means num_kept_tokens changes the actor's compute and the ratio's granularity independently — worth stating   
 explicitly in any paper you write. And your cfm_mask_unexecuted flag only gates the DCT-on path; with the DCT off the mask applies regardless   
 (now documented in the docstring). Both are cheap to test and neither is documented in the literature.                                          
