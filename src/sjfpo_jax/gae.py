"""Chunk-level GAE in JAX: one chunk is one macro-transition.

Mirror of ``spectral_joint_policy/gae.py``. Same contract, same docstring
mathematics; the only difference is that the backward recursion over
``n_chunks`` is a ``lax.scan`` instead of a Python loop.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

# The batch payload's diagnostic block, and its order. Mirrors
# ``modeling_sjfpo._GAE_DIAGNOSTIC_KEYS``; the trainer stacks the dict's values
# in this order, so it is an interface, not a convenience.
GAE_DIAGNOSTIC_KEYS = (
    "linked_frac",
    "advantage_std",
    "return_rms",
    "advantage_std_fraction",
    "budget_truncated_chunk_frac",
)


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


def rms(x):
    """Root mean square over every element, one reduction."""
    return jnp.linalg.vector_norm(x) / math.sqrt(x.size)


def compute_gae_diagnostics(
    env_id: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    advantages: jax.Array,
    raw_returns: jax.Array,
    budget_truncated: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Trace-liveness and scale probes for the chunk backup, as 0-dim arrays.

    ``linked_frac`` is the one that matters: the share of chunk pairs the
    recursion joined. Chunks arrive in wavefront order, so raw adjacency joins
    nothing and the ``env_id`` sort is what makes the chain live.

    0-dim arrays rather than floats, like :func:`policy.compute_info`: the
    trainer does one ``device_get`` over the whole dict, and inside ``jit`` there
    is no ``tolist()``.
    """
    n = env_id.shape[0]
    zeros = jnp.zeros((), dtype=jnp.float64)
    if n == 0:
        return {key: zeros for key in GAE_DIAGNOSTIC_KEYS}

    order = jnp.argsort(env_id.astype(jnp.int64), stable=True)
    env_sorted = env_id.astype(jnp.int64)[order]
    ends = (terminated.astype(jnp.bool_) | truncated.astype(jnp.bool_))[order][:-1]

    ret = raw_returns.astype(jnp.float32)
    adv = advantages.astype(jnp.float32)
    # Population std, matching the advantage normaliser and returns_var. The
    # unbiased default is NaN at a single chunk; correction=0 is defined at
    # n == 1, where clamp_min gives the same 0.0 a guard would.
    adv_std = jnp.std(adv, ddof=0)
    ret_std = jnp.clip(jnp.std(ret, ddof=0), min=1e-8)
    linked = jnp.sum((env_sorted[1:] == env_sorted[:-1]) & ~ends)
    fields = [
        # Each field keeps the fp32 arithmetic torch gave it and is widened only
        # here, so the stack is one dtype without changing any reduction.
        linked.astype(jnp.float64) / max(n - 1, 1),
        adv_std.astype(jnp.float64),
        rms(ret).astype(jnp.float64),
        # A = R - V, so Var(A) <= Var(R), equal only when V is 0. 1.0 means the
        # critic is inert; it falls toward 1 - R^2 as the critic improves.
        (adv_std / ret_std).astype(jnp.float64),
        (
            jnp.mean(budget_truncated.astype(jnp.float32)).astype(jnp.float64)
            if budget_truncated is not None
            else zeros
        ),
    ]
    return dict(zip(GAE_DIAGNOSTIC_KEYS, fields))
