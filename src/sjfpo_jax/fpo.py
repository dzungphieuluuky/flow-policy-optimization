"""FPO++ core math in pure JAX (mirrors ``sjfpo.fpo_core``).

RNG is explicit (``jax.random`` keys in/out); defaults mirror
``SJFPOConfig`` values as literals so this module never imports torch.
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
) -> tuple[jax.Array, jnp.ndarray]:
    """Flow times τ [num_samples] in [lo, hi]; returns (new_key, tau)."""
    if mode == "uniform":
        key, sub = jax.random.split(key)
        return key, jax.random.uniform(sub, (num_samples,)) * (hi - lo) + lo
    if mode == "normal":
        key, sub = jax.random.split(key)
        mid, std = 0.5 * (lo + hi), 0.25 * (hi - lo)
        return key, jnp.clip(jax.random.normal(sub, (num_samples,)) * std + mid, lo, hi)
    raise ValueError(f"mode={mode!r} unknown; expected 'uniform' or 'normal'")


def sample_noise(
    key: jax.Array, shape: tuple[int, ...], mode: str = "normal"
) -> tuple[jax.Array, jnp.ndarray]:
    """MC noise ε of ``shape``; returns (new_key, eps)."""
    if mode == "normal":
        key, sub = jax.random.split(key)
        return key, jax.random.normal(sub, shape)
    if mode == "uniform":
        key, sub = jax.random.split(key)
        s = math.sqrt(3.0)
        return key, (jax.random.uniform(sub, shape) * 2.0 - 1.0) * s
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def run_mc_simulation(
    a: jnp.ndarray, eps: jnp.ndarray, tau: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, tuple[int, ...], int]:
    """Broadcast (a, eps, tau) onto the shared [B*num_mc_samples] MC axis."""
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
    return (
        t.reshape(b * num_mc),
        eps.reshape(b * num_mc, d),
        jnp.broadcast_to(a.reshape(b, d)[:, None, :], (b, num_mc, d)).reshape(
            b * num_mc, d
        ),
        flow_shape,
        num_mc,
    )


def compute_cfm_loss(
    v_hat: jnp.ndarray,
    v_star: jnp.ndarray,
    reduce_mode: str = "scalar",
    clamp: float | None = None,
    cfm_clamp: float = 20.0,
) -> jnp.ndarray:
    """CFM squared-velocity loss; per-draw losses are ``sum / sqrt(D)``."""
    if v_hat.shape != v_star.shape:
        raise ValueError(f"v_hat shape {v_hat.shape} != v_star shape {v_star.shape}")
    squared_errors = (v_hat - v_star) ** 2
    if reduce_mode == "scalar":
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample'/'bin' only")
        return squared_errors.mean()
    elif reduce_mode == "sample":
        flat = squared_errors.reshape(*squared_errors.shape[:2], -1)
        reduced = flat.sum(axis=-1) / math.sqrt(flat.shape[-1])
    elif reduce_mode == "bin":
        reduced = squared_errors.sum(axis=-1) / math.sqrt(squared_errors.shape[-1])
    else:
        raise ValueError(
            f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
        )
    return jnp.clip(reduced, max=cfm_clamp if clamp is None else clamp)


def compute_aspo_surrogate(
    rho: jnp.ndarray,
    advantage: jnp.ndarray,
    epsilon_clip: float = 0.05,
    symmetrical_ppo: bool = False,
) -> jnp.ndarray:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12) or standard PPO clip."""
    if symmetrical_ppo:
        return jnp.minimum(
            rho * advantage,
            jnp.clip(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantage,
        )
    pos = jnp.minimum(
        rho * advantage,
        jnp.clip(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantage,
    )
    neg = rho * advantage - (jnp.abs(advantage) / (2.0 * epsilon_clip)) * (rho - 1.0) ** 2
    return jnp.where(advantage > 0, pos, neg)


def compute_policy_ratio(
    l_old: jnp.ndarray,
    l_new: jnp.ndarray,
    cfm_clamp: float = 20.0,
    cap: float = 10.0,
) -> jnp.ndarray:
    """ρ = exp(ℓ_old − ℓ_new) with OFF clamps; STE cap on the log-ratio."""
    l_old_c = jnp.clip(l_old, max=cfm_clamp)
    l_new_c = jnp.clip(l_new, max=cfm_clamp)
    delta = l_old_c - l_new_c
    # Straight-through: forward clamped, gradient flows through raw delta.
    delta_st = delta + jax.lax.stop_gradient(jnp.clip(delta, max=cap) - delta)
    return jnp.exp(delta_st)


def compute_policy_loss(
    l_old: jnp.ndarray,
    l_new: jnp.ndarray,
    advantage: jnp.ndarray,
    epsilon_clip: float = 0.05,
    cfm_clamp: float = 20.0,
    cap: float = 10.0,
) -> jnp.ndarray:
    """Scalar FPO++ policy objective: MEAN of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=cap)
    return compute_aspo_surrogate(rho, advantage, epsilon_clip).mean()


def sample_action(
    predict_velocity: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
    shape: tuple[int, ...],
    condition: jnp.ndarray,
    num_inference_steps: int,
    eps: jnp.ndarray | None = None,
    key: jax.Array | None = None,
    mode: str = "euler",
) -> jnp.ndarray:
    """Integrate the learned flow t=0 (noise) -> t=1 (data); Euler via scan."""
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")
    if eps is None:
        if key is None:
            raise ValueError("sample_action needs eps= or key= for the initial draw")
        _, eps = sample_noise(key, shape)
    grid = jnp.linspace(0.0, 1.0, num_inference_steps + 1)

    def body(x, i):
        v_hat = predict_velocity(x, grid[i], condition)
        return x + (grid[i + 1] - grid[i]) * v_hat, None

    x, _ = jax.lax.scan(body, eps, jnp.arange(num_inference_steps))
    return x
