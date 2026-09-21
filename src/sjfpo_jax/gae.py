"""Chunk-level semi-MDP GAE (pure JAX; same math as ``sjfpo.gae``).

Backward recursion runs as ``jax.lax.scan`` over the reversed chunk stream
(jit-friendly); intra-chunk discounted rewards use padded segment sums so
shapes stay static. fp32 throughout (matches training precision); population
std (``jnp.std``, ddof=0) matches numpy up to fp32 rounding.

Performance notes
-----------------
* ``exp(x · log γ)`` replaces the general ``pow`` in both the intra-chunk
  discounting and the scan body (SFU op vs. general pow).
* Per-chunk ``γ^L`` and ``(γλ)^L`` are precomputed vectorized before the scan,
  so the body does no transcendental math.
* ``chunk_ids`` is built via ``jnp.repeat`` (O(N)) rather than
  ``jnp.searchsorted`` (O(N log N)).
* ``done = terminated | truncated`` is precomputed once and passed to the scan.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = ["compute_advantages", "validate_stream"]


def compute_advantages(
    rewards_1d: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    lengths: jax.Array,
    terminated: jax.Array,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
    truncated: jax.Array | None = None,
    clamp: float | None = 100.0,
) -> tuple[jax.Array, jax.Array]:
    """Semi-MDP chunk GAE -> (advantages [N], returns [N]), fp32 compute."""
    rewards_1d = rewards_1d.reshape(-1).astype(jnp.float32)
    values = values.reshape(-1).astype(jnp.float32)
    next_values = next_values.reshape(-1).astype(jnp.float32)
    lengths = lengths.reshape(-1).astype(jnp.int32)
    terminated = terminated.reshape(-1).astype(bool)

    n = int(values.shape[0])
    if n == 0:
        raise ValueError("need a non-empty stream of chunks")

    truncated_t = (
        jnp.zeros((n,), dtype=bool)
        if truncated is None
        else truncated.reshape(-1).astype(bool)
    )

    # --- Precompute scalar logs (compile-time constants under jit) ---
    log_gamma = math.log(gamma)              # log γ
    log_gamma_lam = math.log(gamma * lam)    # log(γλ)

    len_f = lengths.astype(jnp.float32)
    gamma_len = jnp.exp(len_f * log_gamma)          # γ^{L_k}
    gamma_lam_len = jnp.exp(len_f * log_gamma_lam)  # (γλ)^{L_k}

    # --- Intra-chunk discounted rewards (static-shape segment sums) ---
    # chunk_ids[j] = index of the chunk that step j belongs to.
    # searchsorted (not repeat): repeat needs static repeat counts, which
    # traced lengths cannot provide under jit.
    offsets = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(lengths)]
    )
    step_ids = jnp.arange(rewards_1d.shape[0], dtype=jnp.int32)
    chunk_ids = jnp.searchsorted(offsets[1:], step_ids, side="right").astype(jnp.int32)
    pos = (step_ids - offsets[chunk_ids]).astype(jnp.float32)
    # γ^{pos} = exp(pos · log γ); pos is a small non-negative integer.
    r_chunk = jax.ops.segment_sum(
        rewards_1d * jnp.exp(pos * log_gamma),
        chunk_ids,
        num_segments=n,
    )

    # --- Backward GAE recursion over chunks (scan, reversed) ---
    done = terminated | truncated_t

    def body(next_adv, k_inputs):
        r_k, v_k, nv_k, gL_k, gLlam_k, done_k = k_inputs
        # δ_k = r_k − v_k + [¬done_k] · γ^{L_k} · V_{k+1}
        delta = r_k - v_k + jnp.where(done_k, 0.0, gL_k * nv_k)
        # Continuation coefficient: 0 at terminal or truncated chunks.
        continuation = jnp.where(done_k, 0.0, gLlam_k)
        adv = delta + continuation * next_adv
        return adv, adv

    stacked = (
        r_chunk[::-1],
        values[::-1],
        next_values[::-1],
        gamma_len[::-1],
        gamma_lam_len[::-1],
        done[::-1],
    )
    _, advantages = jax.lax.scan(body, jnp.float32(0.0), stacked)
    advantages = advantages[::-1]
    returns = advantages + values

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if clamp is not None:
            advantages = jnp.clip(advantages, -clamp, clamp)
    return advantages, returns


def validate_stream(
    rewards_1d: jax.Array, values: jax.Array, lengths: jax.Array
) -> None:
    """Host-side stream checks (call at batch boundaries, never under jit)."""
    import numpy as np

    r = np.asarray(rewards_1d).reshape(-1)
    v = np.asarray(values).reshape(-1)
    l = np.asarray(lengths).reshape(-1).astype(np.int64)
    if v.shape[0] == 0 or bool((l <= 0).any()):
        raise ValueError("need a non-empty stream with every chunk >= 1 step")
    if int(l.sum()) != int(r.shape[0]):
        raise ValueError("rewards length != sum(lengths)")