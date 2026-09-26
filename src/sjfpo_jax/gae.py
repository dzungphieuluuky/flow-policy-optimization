"""Step-level GAE over chunk-level rollout (JAX)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


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
    """Chunk-level semi-MDP GAE (vectorized segment sum + ``lax.scan``)."""
    dtype = jnp.float32
    rewards = jnp.asarray(rewards_1d, dtype=dtype).reshape(-1)
    values = jnp.asarray(values, dtype=dtype).reshape(-1)
    next_values = jnp.asarray(next_values, dtype=dtype).reshape(-1)
    lengths = jnp.asarray(lengths, dtype=jnp.int32).reshape(-1)
    terminated = jnp.asarray(terminated, dtype=bool).reshape(-1)

    n = values.shape[0]
    if n == 0:
        empty = jnp.zeros((0,), dtype=dtype)
        return empty, empty

    if truncated is None:
        truncated = jnp.zeros(n, dtype=bool)
    else:
        truncated = jnp.asarray(truncated, dtype=bool).reshape(-1)

    # Validation (host-side; only on first trace).
    total_steps = int(jnp.sum(lengths))
    if total_steps != rewards.shape[0]:
        raise ValueError(
            f"rewards length {rewards.shape[0]} != sum(lengths) {total_steps}"
        )
    if bool(jnp.any(lengths <= 0)):
        raise ValueError("every chunk must have length >= 1")

    chunk_ids = jnp.repeat(jnp.arange(n, dtype=jnp.int32), lengths)
    offsets = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(lengths)])
    pos = (jnp.arange(rewards.shape[0], dtype=jnp.int32) - offsets[chunk_ids]).astype(
        dtype
    )
    log_gamma = math.log(gamma)
    r_chunk = jnp.zeros(n, dtype=dtype)
    r_chunk = r_chunk.at[chunk_ids].add(rewards * jnp.exp(pos * log_gamma))

    len_f = lengths.astype(dtype)
    gamma_len = jnp.exp(len_f * log_gamma)
    gamma_lam_log = math.log(gamma * lam)
    gamma_lam_len = jnp.exp(len_f * gamma_lam_log)
    done = terminated | truncated

    r_rev = r_chunk[::-1]
    v_rev = values[::-1]
    nv_rev = next_values[::-1]
    gl_rev = gamma_len[::-1]
    gll_rev = gamma_lam_len[::-1]
    done_rev = done[::-1]

    def body(next_adv, i):
        delta = (
            r_rev[i]
            - v_rev[i]
            + jnp.where(done_rev[i], 0.0, gl_rev[i] * nv_rev[i])
        )
        cont = jnp.where(done_rev[i], 0.0, gll_rev[i])
        adv = delta + cont * next_adv
        return adv, adv

    _, adv_rev = jax.lax.scan(body, jnp.zeros((), dtype=dtype), jnp.arange(n))
    advantages = adv_rev[::-1]
    returns = advantages + values

    if normalize:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )
        if clamp is not None:
            advantages = jnp.clip(advantages, -clamp, clamp)

    return advantages, returns