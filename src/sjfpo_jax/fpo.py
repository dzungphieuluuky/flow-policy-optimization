"""FPO++ core math in pure JAX (mirrors ``sjfpo.fpo_core``).

RNG is explicit (``jax.random`` keys in/out); defaults mirror
``SJFPOConfig`` values as literals so this module never imports torch.

Performance notes
-----------------
* All ``**2`` replaced with ``jnp.square`` (better XLA lowering, cleaner grad).
* Common subexpressions hoisted in ``compute_aspo_surrogate`` and
  ``compute_cfm_loss``.
* ``sample_action`` scans over ``(t, dt)`` pairs so the body has no
  dynamic indexing into ``grid``.
* ``compute_cfm_loss`` sums over trailing axes directly rather than
  reshaping to a flat ``[B, M, D]`` view first.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "sample_timesteps",
    "sample_noise",
    "run_mc_simulation",
    "compute_cfm_loss",
    "compute_aspo_surrogate",
    "compute_policy_ratio",
    "compute_policy_loss",
    "sample_action",
]


def sample_timesteps(
    key: jax.Array,
    num_samples: int,
    lo: float = 0.005,
    hi: float = 0.995,
    mode: str = "uniform",
) -> tuple[jax.Array, jax.Array]:
    """Flow times τ [num_samples] in [lo, hi]; returns (new_key, tau)."""
    key, sub = jax.random.split(key)
    if mode == "uniform":
        tau = jax.random.uniform(sub, (num_samples,)) * (hi - lo) + lo
    elif mode == "normal":
        mid, std = 0.5 * (lo + hi), 0.25 * (hi - lo)
        tau = jnp.clip(
            jax.random.normal(sub, (num_samples,)) * std + mid, lo, hi
        )
    else:
        raise ValueError(f"mode={mode!r} unknown; expected 'uniform' or 'normal'")
    return key, tau


def sample_noise(
    key: jax.Array, shape: tuple[int, ...], mode: str = "normal"
) -> tuple[jax.Array, jax.Array]:
    """MC noise ε of ``shape``; returns (new_key, eps)."""
    key, sub = jax.random.split(key)
    if mode == "normal":
        eps = jax.random.normal(sub, shape)
    elif mode == "uniform":
        s = math.sqrt(3.0)
        eps = (jax.random.uniform(sub, shape) * 2.0 - 1.0) * s
    else:
        raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")
    return key, eps


def run_mc_simulation(
    a: jax.Array, eps: jax.Array, tau: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[int, ...], int]:
    """Broadcast (a, eps, tau) onto the shared [B*num_mc_samples] MC axis.

    Returns ``(t, eps_flat, a_flat, flow_shape, num_mc)``.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    t = jnp.asarray(tau)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, (b, num_mc))
    elif t.ndim == 1:
        if t.shape[0] != num_mc:
            raise ValueError(f"tau vector must have num_mc_samples={num_mc} entries")
        t = jnp.broadcast_to(t, (b, num_mc))
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError("tau must be scalar, [num_mc_samples] or [B, num_mc_samples]")

    d = a.size // b
    # Explicit shapes: broadcast first, reshape once at the end.
    a_flat = jnp.broadcast_to(a.reshape(b, 1, d), (b, num_mc, d)).reshape(b * num_mc, d)
    eps_flat = eps.reshape(b * num_mc, d)
    return t.reshape(b * num_mc), eps_flat, a_flat, flow_shape, num_mc


def compute_cfm_loss(
    v_hat: jax.Array,
    v_star: jax.Array,
    reduce_mode: str = "scalar",
    clamp: float | None = None,
    cfm_clamp: float = 20.0,
) -> jax.Array:
    """CFM squared-velocity loss; per-draw losses are ``sum / sqrt(D)``."""
    if v_hat.shape != v_star.shape:
        raise ValueError(f"v_hat shape {v_hat.shape} != v_star shape {v_star.shape}")

    sq = jnp.square(v_hat - v_star)

    if reduce_mode == "scalar":
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample'/'bin' only")
        return sq.mean()

    if reduce_mode == "sample":
        # Sum over all trailing axes (action × feature), normalize by sqrt(total).
        tail = 1
        for s in sq.shape[2:]:
            tail *= s
        reduced = sq.sum(axis=tuple(range(2, sq.ndim))) / math.sqrt(max(tail, 1))
    elif reduce_mode == "bin":
        reduced = sq.sum(axis=-1) / math.sqrt(sq.shape[-1])
    else:
        raise ValueError(
            f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
        )

    return jnp.clip(reduced, max=cfm_clamp if clamp is None else clamp)


def compute_aspo_surrogate(
    rho: jax.Array,
    advantage: jax.Array,
    epsilon_clip: float = 0.05,
    symmetrical_ppo: bool = False,
) -> jax.Array:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12) or standard PPO clip.

    Hoists ``rho * advantage`` and the clipped ratio so neither is computed
    twice along the branchy path.
    """
    rho_adv = rho * advantage
    rho_clipped = jnp.clip(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
    pos = jnp.minimum(rho_adv, rho_clipped * advantage)

    if symmetrical_ppo:
        return pos

    neg = rho_adv - (jnp.abs(advantage) / (2.0 * epsilon_clip)) * jnp.square(rho - 1.0)
    return jnp.where(advantage > 0, pos, neg)


def compute_policy_ratio(
    l_old: jax.Array,
    l_new: jax.Array,
    cfm_clamp: float = 20.0,
    cap: float = 10.0,
) -> jax.Array:
    """ρ = exp(ℓ_old − ℓ_new) with OFF clamps; STE cap on the log-ratio."""
    delta = jnp.clip(l_old, max=cfm_clamp) - jnp.clip(l_new, max=cfm_clamp)
    # Forward: clipped to cap. Backward: identity (straight-through).
    delta_st = delta + jax.lax.stop_gradient(jnp.clip(delta, max=cap) - delta)
    return jnp.exp(delta_st)


def compute_policy_loss(
    l_old: jax.Array,
    l_new: jax.Array,
    advantage: jax.Array,
    epsilon_clip: float = 0.05,
    cfm_clamp: float = 20.0,
    cap: float = 10.0,
) -> jax.Array:
    """Scalar FPO++ policy objective: MEAN of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=cap)
    return compute_aspo_surrogate(rho, advantage, epsilon_clip).mean()


def sample_action(
    predict_velocity: Callable[[jax.Array, jax.Array, Any], jax.Array],
    shape: tuple[int, ...],
    condition: jax.Array,
    num_inference_steps: int,
    eps: jax.Array | None = None,
    key: jax.Array | None = None,
    mode: str = "euler",
) -> jax.Array:
    """Integrate the learned flow t=0 (noise) -> t=1 (data); Euler via scan.

    The scan carries ``x`` and iterates over ``(t, dt)`` pairs, so the body
    contains no dynamic indexing into the time grid.
    """
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")
    if eps is None:
        if key is None:
            raise ValueError("sample_action needs eps= or key= for the initial draw")
        _, eps = sample_noise(key, shape)

    grid = jnp.linspace(0.0, 1.0, num_inference_steps + 1)
    ts = grid[:-1]                # t_i
    dts = jnp.diff(grid)          # t_{i+1} - t_i

    def body(x, td):
        t, dt = td
        v_hat = predict_velocity(x, t, condition)
        return x + dt * v_hat, None

    x, _ = jax.lax.scan(body, eps, (ts, dts))
    return x