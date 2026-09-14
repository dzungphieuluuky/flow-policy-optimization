# SHINY.md — core innovations in SJFPO

What is genuinely new here versus borrowed machinery. Ordered by novelty.

## 1. Flow matching in truncated DCT action space

The policy does not diffuse raw actions. It flows in orthonormal DCT-II
coefficient space, keeping only the lowest `Ka = round(top_p · H)` frequency
rows and zero-padding the rest at decode (`predict_action_chunk` → `sample`
→ pad → `IDCT` → clamp).

Two consequences:

- **Structural smoothness prior.** The model cannot output high-frequency
  jitter it never represents. Most diffusion/flow policies work in raw action
  space and hope for smoothness; here it is enforced by the representation.
- **Exact likelihood bookkeeping.** Orthonormality (`MᵀM = I`, Parseval) makes
  the transform measure-preserving (log-det = 0), so a Gaussian in
  coefficient space is a Gaussian in action space and the CFM losses transfer
  without Jacobian corrections.

## 2. Chunk-level semi-MDP GAE

GAE rewritten for action-chunk decision epochs (`gae.compute_advantages`,
numpy fp64). Per chunk `k` with executed length `L_k`:

- `R_k = Σᵢ γⁱ·r_{k,i}` over executed steps only (intradiscounted),
- `δ_k = R_k + γ^{L_k}·V_{k+1}·(1−terminated_k) − V_k`,
- `A_k = δ_k + (γλ)^{L_k}·(1−terminated_k)·A_{k+1}`.

Terminated chunks cut bootstrap and recursion; truncated (TimeLimit) chunks
bootstrap `δ` from the pre-reset value but cut the λ-recursion into the
post-reset episode. Variable executed lengths (mid-chunk death, step-budget
cutoff) are handled by exponentiating with *that chunk's* length — the point
most chunked-action treatments hand-wave.

## 3. ASPO trust region on estimated likelihoods

Per-sample MC flow losses `ℓ` become ratios `ρ = exp(ℓ_old − ℓ_new)` under one
shared `(τ, ε)` draw (else `ρ` is pure MC noise), and the objective is
**mean-of-ψ over samples** (paper Eq.13) — never exp-of-mean. The trust
region itself is asymmetric and piecewise (`fpo_core.compute_aspo_surrogate`):

- `Â ≥ 0`: PPO-clip with split bounds `clip(ρ, 1−ε_low, 1+ε_high)`,
- `Â < 0`: SPO quadratic penalty `ρÂ − (|Â|/(2ε_low))·(ρ−1)²`,
- plus an STE log-ratio cap (clamped forward, gradient flows) and an
  `ℓ_new ← min(ℓ_new, ℓ_old + cfm_clamp)` clamp for negative advantages.

This is a trust region designed for *noisy estimated* likelihoods rather than
PPO's exact Gaussian densities — a different problem, hence different math.

## 4. Parameter-free spectral state conditioning

The observation window is DCT-II transformed over time and the `K` lowest raw
bins become conditioning tokens (`SpectralEncoder`: no weights, no projector,
no normalization). Pooling into the DiT conditioning vector is order-aware
(`SpectralAttentionPool`, queried by the flow timestep) instead of a mean.
Frequency-domain memory with near-zero parameters — no RNN, no longer
Transformer context.

## 5. Chained `next.action` SARSA bootstrapping

Every `Transition` carries the chunk proposed at the *next* replan, so the
`V(s,a)` critic bootstraps `V(s′,a′)` from behavior proposals, SARSA-style at
chunk scale (`bootstrap_action_lengths` masks to the next chunk's executed
length; truncated/tail proposals stay full). It reuses computation the
rollout already did — no extra policy queries for targets.

## Borrowed machinery (not claimed as new)

CFM/flow-matching, DiT + adaLN-Zero, GAE's core idea, the PPO outer loop
(behavior snapshot → multi-epoch clipped update → clipped value loss),
running observation normalization. The contribution is the composition —
frequency-space flow + chunk-MDP math + ratio machinery for estimated
likelihoods — more than any single trick.
