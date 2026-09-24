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

## 2. Step-level GAE for action chunks

GAE runs over every executed environment transition
(`gae.compute_step_advantages_torch`). The policy still predicts a full
action chunk, so the first executed step's GAE target becomes that chunk's
policy target.

For each executed transition `t`:

- `δ_t = r_t + γ·(1−terminated_t)·V(s_{t+1}) − V(s_t)`,
- `A_t = δ_t + γλ·(1−terminated_t)·(1−truncated_t)·A_{t+1}`.

Terminated transitions mask the bootstrap and cut recursion. Truncated
transitions keep the next-state bootstrap but cut recursion into the new
episode. Normal action-chunk boundaries recurse through the shared
step-level stream. This handles variable executed prefixes directly, without
a second chunk-level GAE implementation.

## 3. ASPO trust region on estimated likelihoods

Per-sample MC flow losses `ℓ` become ratios `ρ = exp(ℓ_old − ℓ_new)` under one
shared `(τ, ε)` draw per collection round (else `ρ` is pure MC noise), and the
objective is **mean-of-ψ over samples** (paper Eq.13) — never exp-of-mean.
The trust region follows the official repo (`amazon-far/fpo-control`):
single `ε_clip = 0.05` for both branches, OFF `sqrt` CFM reduction
(`sum/sqrt(D)`), symmetric loss clamp `20`, negative-advantage clamp `20`,
and STE log-ratio cap `10`:

- `Â > 0`: PPO-clip with single bound `clip(ρ, 1−ε, 1+ε)`,
- `Â ≤ 0`: SPO quadratic penalty `ρÂ − (|Â|/(2ε))·(ρ−1)²`.

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

## 5. Next-window V(s) bootstrapping

Every `Transition` carries the window after execution, so the `V(s)` critic
bootstraps `V(s′)` with no extra policy queries for targets.

## Borrowed machinery (not claimed as new)

CFM/flow-matching, DiT + adaLN-Zero, GAE's core idea, the PPO outer loop
(behavior snapshot → multi-epoch clipped update → clipped value loss),
running observation normalization. The contribution is the composition —
frequency-space flow + chunk-MDP math + ratio machinery for estimated
likelihoods — more than any single trick.
