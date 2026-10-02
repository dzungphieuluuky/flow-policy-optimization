"""Chunk-level GAE in JAX: one chunk is one macro-transition.

Mirror of ``spectral_joint_policy/gae.py``. Same contract, same docstring
mathematics; the only difference is that the backward recursion over
``n_chunks`` is a ``lax.scan`` instead of a Python loop.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def compute_chunk_advantages(
    step_rewards: jax.Array,
    chunk_value: jax.Array,
    chunk_next_value: jax.Array,
    chunk_starts: jax.Array,
    chunk_lengths: jax.Array,
    *,
    chunk_terminated: jax.Array,
    chunk_truncated: jax.Array,
    chunk_env_id: jax.Array,
    chunk_size: int,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    """Chunk-level GAE over a rollout's executed chunks.

    All chunk tensors must be the same length. Returns per-chunk advantages and
    ``sorted_values + advantages``, the value target. ``ValueError`` if
    ``gamma`` or ``lam`` leaves ``[0, 1]``, or lengths disagree.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    rewards = step_rewards.astype(jnp.float32).reshape(-1)
    chunk_value = chunk_value.astype(jnp.float32).reshape(-1)
    chunk_next_value = chunk_next_value.astype(jnp.float32).reshape(-1)
    starts = chunk_starts.astype(jnp.int32).reshape(-1)
    lengths = chunk_lengths.astype(jnp.int32).reshape(-1)
    terminated = chunk_terminated.astype(jnp.bool_).reshape(-1)
    truncated = chunk_truncated.astype(jnp.bool_).reshape(-1)
    env_id = chunk_env_id.astype(jnp.int32).reshape(-1)

    n_chunks = starts.shape[0]
    for name, tensor in (
        ("chunk_lengths", lengths),
        ("chunk_terminated", terminated),
        ("chunk_truncated", truncated),
        ("chunk_env_id", env_id),
        ("chunk_value", chunk_value),
        ("chunk_next_value", chunk_next_value),
    ):
        if tensor.shape[0] != n_chunks:
            raise ValueError(f"{name} must have {n_chunks} entries, got {tensor.shape[0]}")

    if n_chunks == 0:
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return empty, empty

    # One shot gather, then discount by gamma^k. A cumsum diff would need
    # gamma**-s_i, which loses all fp32 precision once s_i hits a few thousand.
    offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    window = starts[:, None] + offsets                              # [B, chunk_size]
    gathered = rewards[jnp.clip(window, max=rewards.shape[0] - 1)]  # [B, chunk_size]
    valid = offsets[None, :] < lengths[:, None]                     # [B, chunk_size]
    # offsets as fp32, not int: a Python float raised to an integer power takes
    # the default float type, which is float64 now that x64 is on.
    gamma_k = gamma ** offsets.astype(jnp.float32)                   # [chunk_size]
    chunk_rewards = jnp.sum(gathered * valid * gamma_k, axis=1)     # [B]

    # Wavefront order puts one env's chunks n_envs apart, never adjacent, so an
    # adjacency mask finds no links. A stable sort groups each env and keeps
    # its temporal order.
    order = jnp.argsort(env_id, stable=True)
    unsort = jnp.argsort(order)
    lengths, env_id = lengths[order], env_id[order]
    terminated, truncated = terminated[order], truncated[order]
    chunk_rewards = chunk_rewards[order]
    sorted_values, sorted_next_values = chunk_value[order], chunk_next_value[order]
    same_env_sorted = env_id[1:] == env_id[:-1]

    gamma_h = gamma ** lengths.astype(jnp.float32)                   # [B]
    # Only real termination kills the bootstrap. A timeout keeps it.
    delta = chunk_rewards + gamma_h * (~terminated) * sorted_next_values - sorted_values

    continuation_factor = jnp.zeros(n_chunks, dtype=jnp.float32)
    if n_chunks > 1:
        continues = same_env_sorted & ~(terminated[:-1] | truncated[:-1])
        continuation_factor = continuation_factor.at[:-1].set(
            gamma_h[:-1] * lam * continues.astype(jnp.float32)
        )

    # c[i] is the coefficient on A_{i+1}; c[-1] stays 0, which kills the tail
    # exactly like continuation_factor[-1] in the torch loop. reverse=True
    # reverses xs and flips ys back, so the output is already in buffer order.
    def step(next_advantage, delta_c):
        carry = delta_c[0] + delta_c[1] * next_advantage
        return carry, carry

    _, advantages = lax.scan(
        step,
        jnp.float32(0.0),
        (delta, continuation_factor),
        reverse=True,
    )

    advantages = advantages[unsort]
    return advantages, advantages + sorted_values[unsort]
