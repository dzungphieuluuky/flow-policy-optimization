"""Chunk-level semi-MDP GAE (pure JAX; same math as ``sjfpo.gae``).

Backward recursion runs as ``jax.lax.scan`` over the reversed chunk stream
(jit-friendly); intra-chunk discounted rewards use padded segment sums so
shapes stay static. Population std (``jnp.std``, ddof=0) matches numpy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["compute_advantages", "validate_stream"]


def compute_advantages(
    rewards_1d: jnp.ndarray,
    values: jnp.ndarray,
    next_values: jnp.ndarray,
    lengths: jnp.ndarray,
    terminated: jnp.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
    truncated: jnp.ndarray | None = None,
    clamp: float | None = 100.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Semi-MDP chunk GAE -> (advantages [N], returns [N]), fp64 compute."""
    rewards_1d = rewards_1d.reshape(-1).astype(jnp.float64)
    values = values.reshape(-1).astype(jnp.float64)
    next_values = next_values.reshape(-1).astype(jnp.float64)
    lengths = lengths.reshape(-1).astype(jnp.int32)
    terminated = terminated.reshape(-1).astype(bool)
    # Static shape check only (safe under jit); data-dependent validation
    # lives in validate_stream (host-side, call at batch boundaries).
    n = int(values.shape[0])
    if n == 0:
        raise ValueError("need a non-empty stream of chunks")
    if truncated is None:
        truncated_t = jnp.zeros((n,), dtype=bool)
    else:
        truncated_t = truncated.reshape(-1).astype(bool)

    # ------- intra-chunk discounted rewards (static-shape segment sums) -------
    offsets = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(lengths)]
    )
    step_ids = jnp.arange(rewards_1d.shape[0])
    chunk_ids = jnp.searchsorted(offsets[1:], step_ids, side="right").astype(jnp.int32)
    pos = (step_ids - offsets[chunk_ids]).astype(jnp.float64)
    r_chunk = jax.ops.segment_sum(
        rewards_1d * jnp.power(gamma, pos),
        chunk_ids,
        num_segments=n,
    )

    # ------- backward GAE recursion over chunks (scan, reversed) -------
    def body(next_adv, k_inputs):
        r_k, v_k, nv_k, len_k, term_k, trunc_k = k_inputs
        len_f = len_k.astype(jnp.float64)
        delta = jnp.where(
            term_k,
            r_k - v_k,
            r_k + jnp.power(gamma, len_f) * nv_k - v_k,
        )
        continuation = jnp.where(
            term_k | trunc_k, 0.0, jnp.power(gamma * lam, len_f)
        )
        adv = delta + continuation * next_adv
        return adv, adv

    stacked = (
        r_chunk[::-1],
        values[::-1],
        next_values[::-1],
        lengths[::-1],
        terminated[::-1],
        truncated_t[::-1],
    )
    _, advantages = jax.lax.scan(body, jnp.float64(0.0), stacked)
    advantages = advantages[::-1]
    returns = advantages + values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if clamp is not None:
            advantages = jnp.clip(advantages, -clamp, clamp)
    return advantages, returns


def validate_stream(
    rewards_1d: jnp.ndarray, values: jnp.ndarray, lengths: jnp.ndarray
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
