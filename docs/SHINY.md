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

## 4. Fused spectral state conditioning

The observation window and its DCT-II bins over time fuse externally in
`StateConditioner`: state tokens query spectral bins (Flamingo-style
cross-attn, residual, mean-pool) into one `[B,D]` vector. The flow head only
sees `(x_t, t, condition)`; state, spectral, and time sum into a single adaLN
vector `c = time + condition` (pure-DiT: the transformer sees action tokens
only, no prefix tokens). Bin order rides sinusoidal codes, so attention
weights index frequency. Frequency-domain memory with no RNN, no longer
Transformer context.

## 5. Chained `next.action` bootstrapping

Every `Transition` carries the chunk proposed at the *next* replan, so the
`V(s)` critic bootstraps `V(s′)` on the stored next window. It reuses
computation the rollout already did — no extra policy queries for targets.

## Borrowed machinery (not claimed as new)

CFM/flow-matching, DiT + adaLN-Zero, GAE's core idea, the PPO outer loop
(behavior snapshot → multi-epoch clipped update → clipped value loss),
running observation normalization. The contribution is the composition —
frequency-space flow + chunk-MDP math + ratio machinery for estimated
likelihoods — more than any single trick.
