"""FPO++ core math (JAX): CFM losses, per-sample ratios, ASPO, Euler sampler."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from sjfpo_jax.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()

_DEFAULT_CFG = None


def _defaults():
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        from sjfpo_jax.configuration_sjfpo import SJFPOConfig
        _DEFAULT_CFG = SJFPOConfig()
    return _DEFAULT_CFG


def sample_timesteps(
    num_samples: int, tau_range=None, *, key=None, mode="uniform",
    alpha=None, beta=None,
) -> jnp.ndarray:
    """Flow times τ ∈ [lo, hi]."""
    cfg = _defaults()
    lo, hi = (cfg.min_timestep, cfg.max_timestep) if tau_range is None else tau_range
    if key is None:
        key = jax.random.PRNGKey(0)
    if mode == "uniform":
        return jax.random.uniform(key, (num_samples,), minval=lo, maxval=hi)
    if mode == "normal":
        mid, std = 0.5 * (lo + hi), 0.25 * (hi - lo)
        z = jax.random.normal(key, (num_samples,))
        return jnp.clip(z * std + mid, lo, hi)
    if mode == "beta":
        if alpha is None or beta is None:
            raise ValueError("mode='beta' needs alpha= and beta= kwargs")
        u = jax.random.beta(key, alpha, beta, (num_samples,))
        return u * (hi - lo) + lo
    raise ValueError(f"mode={mode!r} unknown; expected 'uniform', 'normal' or 'beta'")


def sample_noise(shape, *, key=None, mode="normal", dtype=jnp.float32) -> jnp.ndarray:
    if key is None:
        key = jax.random.PRNGKey(0)
    if mode == "normal":
        return jax.random.normal(key, shape, dtype=dtype)
    if mode == "uniform":
        s = math.sqrt(3.0)
        return (jax.random.uniform(key, shape, dtype=dtype) * 2.0 - 1.0) * s
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def run_mc_simulation(a, eps, tau):
    """Broadcast ``(a, eps, tau)`` onto the shared ``[B*M]`` MC axis."""
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    if hasattr(tau, "ndim"):
        t = jnp.asarray(tau, dtype=STORAGE_DTYPE)
    else:
        t = jnp.asarray(tau, dtype=STORAGE_DTYPE)

    if t.ndim == 0:
        t = jnp.broadcast_to(t, (b, num_mc))
    elif t.ndim == 1:
        if t.shape[0] != num_mc:
            raise ValueError(
                f"tau vector must have num_mc_samples={num_mc} entries, got {t.shape[0]}"
            )
        t = jnp.broadcast_to(t[None, :], (b, num_mc))
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be scalar, [num_mc] or [B, num_mc]"
        )

    d = a.size // b
    a_flat = jnp.broadcast_to(a.reshape(b, 1, d), (b, num_mc, d)).reshape(b * num_mc, d)
    return t.reshape(b * num_mc), eps.reshape(b * num_mc, d), a_flat, flow_shape, num_mc


def compute_cfm_loss(v_hat, v_star, reduce_mode: str = "scalar", clamp=None):
    """CFM squared-velocity loss (paper Eq.7-8)."""
    if v_hat.shape != v_star.shape:
        raise ValueError(f"v_hat shape {v_hat.shape} != v_star shape {v_star.shape}")
    sq = (v_hat - v_star) ** 2

    if reduce_mode == "scalar":
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample'/'bin' only")
        return sq.mean()

    if reduce_mode == "sample":
        flat = sq.reshape(sq.shape[0], sq.shape[1], -1)
        reduced = flat.sum(-1) / math.sqrt(flat.shape[-1])
    elif reduce_mode == "bin":
        reduced = sq.sum(-1) / math.sqrt(sq.shape[-1])
    else:
        raise ValueError(
            f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
        )

    cfg = _defaults()
    clamp = cfg.cfm_clamp if clamp is None else clamp
    return jnp.minimum(reduced, clamp)


def compute_aspo_surrogate(rho, advantage, epsilon_clip=None, symmetrical_ppo=None):
    """ASPO trust region (Eq.12) or PPO clip."""
    cfg = _defaults()
    if epsilon_clip is None:
        epsilon_clip = cfg.epsilon_clip
    if symmetrical_ppo is None:
        symmetrical_ppo = not cfg.use_aspo

    rho_adv = rho * advantage
    rho_clipped = jnp.clip(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
    pos = jnp.minimum(rho_adv, rho_clipped * advantage)

    if symmetrical_ppo:
        return pos
    neg = rho_adv - (jnp.abs(advantage) / (2.0 * epsilon_clip)) * (rho - 1.0) ** 2
    return jnp.where(advantage > 0, pos, neg)


def _ste_clamp(x, lo, hi):
    """Straight-through clamp: forward clamped, gradient identity.

    Forward = ``clip(x, lo, hi)``; backward = ``dx`` (grad flows through the
    unclamped input). Mirrors ``safe + (x - safe).detach()``.
    """
    return jax.lax.stop_gradient(jnp.clip(x, lo, hi) - x) + x


def compute_policy_ratio(l_old, l_new, cfm_clamp=None, cap=None):
    """Per-sample importance ratios ρ = exp(ℓ_old − ℓ_new) with OFF clamps."""
    cfg = _defaults()
    cfm_clamp = cfg.cfm_clamp if cfm_clamp is None else cfm_clamp
    cap = cfg.log_ratio_cap if cap is None else cap

    delta = jnp.minimum(l_old, cfm_clamp) - jnp.minimum(l_new, cfm_clamp)
    delta_st = _ste_clamp(delta, -cap, cap)
    return jnp.exp(delta_st)


def compute_policy_loss(
    l_old, l_new, advantage, epsilon_clip=None, cfm_clamp=None, cap=None
):
    """Scalar FPO++ policy objective: mean over i of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=cap)
    return compute_aspo_surrogate(rho, advantage, epsilon_clip).mean()


def sample_action(
    predict_velocity: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    shape,
    condition,
    num_inference_steps: int,
    eps: jnp.ndarray | None = None,
    key=None,
    mode: str = "euler",
) -> jnp.ndarray:
    """Euler flow t=0 (noise) → t=1 (data)."""
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")
    if eps is None:
        if key is None:
            key = jax.random.PRNGKey(0)
        eps = jax.random.normal(key, shape, dtype=jnp.float32)

    grid = jnp.linspace(0.0, 1.0, num_inference_steps + 1, dtype=jnp.float32)
    dts = grid[1:] - grid[:-1]

    def body(x, i):
        v = predict_velocity(x, grid[i], condition)
        return x + dts[i] * v, None

    x, _ = jax.lax.scan(body, eps, jnp.arange(num_inference_steps))
    return x