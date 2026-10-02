"""FPO++ core math in JAX: CFM losses, ratios, ASPO trust region, Euler sampler.

Function-for-function mirror of ``spectral_joint_policy/fpo_core.py``. ``cfg``
is duck-typed -- any object carrying the fields each function reads -- so the
parity tests hand the torch ``SJFPOConfig`` to both sides.

Two deliberate differences from the torch version:

* device and generator arguments are gone. JAX has one device; randomness is a
  typed ``jax.random.key`` in the argument list.
* ``run_monte_carlo_sampling`` returns three arrays, not five. ``flow_shape``
  and ``num_mc`` come off ``a.shape`` and ``eps.shape``, and a jitted function
  cannot return Python ints.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax


def sample_timesteps(shape: tuple[int, ...], cfg, key: jax.Array) -> jax.Array:
    """Draw flow timesteps in ``[min_timestep, max_timestep]``.

    Mode comes from ``cfg.timestep_sampling_mode``. Returns ``shape``.
    ``ValueError`` on an unknown mode.
    """
    lo, hi = cfg.min_timestep, cfg.max_timestep
    mode = cfg.timestep_sampling_mode
    if mode == "uniform":
        return jax.random.uniform(key, shape, dtype=jnp.float32) * (hi - lo) + lo
    if mode == "normal":
        mid = 0.5 * (lo + hi)
        std = 0.25 * (hi - lo)
        z = jax.random.normal(key, shape, dtype=jnp.float32)
        return jnp.clip(z * std + mid, lo, hi)
    if mode == "fpo_beta":
        u = jax.random.uniform(key, shape, dtype=jnp.float32)
        beta = cfg.timestep_inverse_cdf_beta
        return lo + (hi - lo) * (1.0 - jnp.power(1.0 - u, 1.0 / beta))
    if mode == "beta":
        # torch delegates to lerobot's torch.distributions.Beta sampler, which has
        # no JAX counterpart. Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b)):
        # same law, different draws, so this one mode is tested on statistics.
        alpha = jax.random.gamma(key, cfg.timestep_beta_alpha, shape, dtype=jnp.float32)
        beta = jax.random.gamma(
            jax.random.fold_in(key, 1), cfg.timestep_beta_beta, shape, dtype=jnp.float32
        )
        return alpha / (alpha + beta) * (hi - lo) + lo
    raise ValueError(
        f"mode={mode!r} unknown; expected 'uniform', 'normal', 'beta', or 'fpo_beta'"
    )


def compute_freq_weighted_std(
    num_modes: int, horizon: int, beta: float = 1.0
) -> jax.Array:
    """Per-mode std of 1/f^beta noise over DCT-II modes.

    NEED (Eberhard et al. 2023): beta=1 (pink) beats white and OU off-policy.
    Colored-noise PPO (Rückstieß et al., AAAI 2024, 20480 runs) finds beta=0.5
    best *on-policy*, with the optimum drifting to larger beta as the update
    batch grows -- which is this policy's regime.

    beta=0 gives std 1.0, i.e. exactly noise_scaling_mode="white"; beta=1 is
    the original hardcoded pink. The exponent is -beta/2 because the std of a
    1/f^beta *variance* is 1/f^(beta/2), and it keeps both endpoints exact.
    """
    k = jnp.arange(num_modes, dtype=jnp.float32)
    return (1.0 + (k * (jnp.pi / horizon)) ** 2) ** (-0.5 * beta)


def sample_noise(shape: tuple[int, ...], cfg, key: jax.Array) -> jax.Array:
    """Draw Monte-Carlo noise of ``shape``.

    Marginal from ``cfg.noise_sampling_mode``; per-mode std over the token axis
    from ``cfg.noise_scaling_mode``. ``ValueError`` on an unknown mode.
    """
    mode = cfg.noise_sampling_mode
    if mode == "normal":
        eps = jax.random.normal(key, shape, dtype=jnp.float32)
    elif mode == "uniform":
        eps = (
            jax.random.uniform(key, shape, dtype=jnp.float32) * 2.0 - 1.0
        ) * math.sqrt(3.0)
    else:
        raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")

    if cfg.noise_scaling_mode == "white":
        return eps
    if cfg.noise_scaling_mode == "freq_weighted":
        # Axis -2 holds the DCT coeffs. Broadcast the per-mode std over the
        # leading batch/MC axes and the trailing action dim.
        std = compute_freq_weighted_std(
            shape[-2], cfg.action_horizon, cfg.noise_scaling_beta
        )
        return eps * std.reshape(*(1,) * (len(shape) - 2), -1, 1)
    raise ValueError(
        f"mode={cfg.noise_scaling_mode!r} unknown; expected 'white' or 'freq_weighted'"
    )


def run_monte_carlo_sampling(
    a: jax.Array, eps: jax.Array, tau: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Broadcast action, noise, and per-token timesteps over MC.

    Only the tokenwise form the trainer passes: ``tau`` is ``[B, M, K, 1]`` and
    comes back as ``[B*M, K, 1]``, one time per token shared across its action
    dims. The torch version also accepts scalar, ``[M]`` and ``[B, M]`` legacy
    forms; nothing in the tree builds them, and shapes are static in JAX, so
    they are not ported.

    Returns the flattened timestep, noise and broadcast action.
    ``ValueError`` if shapes disagree.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = a.shape[1:]
    if eps.shape != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {eps.shape} != {(b, num_mc, *flow_shape)}")
    expected_tau = (b, num_mc, flow_shape[0], 1)
    if tau.shape != expected_tau:
        raise ValueError(f"tau shape {tau.shape} != {expected_tau}")

    d = a.size // b
    return (
        tau.reshape(b * num_mc, flow_shape[0], 1),
        eps.reshape(b * num_mc, d),
        jnp.broadcast_to(a.reshape(b, 1, d), (b, num_mc, d)).reshape(b * num_mc, d),
    )


def compute_cfm_loss(
    v_hat: jax.Array,
    v_star: jax.Array,
    cfg,
    reduce_mode: str,
    action_is_pad: jax.Array | None = None,
):
    """CFM squared-velocity loss (paper Eq.7-8).

    ``sample`` is the active path: FPO++'s reduction over the flattened chunk,
    ``mean``, ``sum`` or ``sqrt`` from ``cfg.cfm_loss_reduction``. ``token_mean``
    is diagnostic only: mean over the token axis of the operand's space, divided
    by ``action_dim``, so masking cannot move the value. ``both`` returns
    ``(sample, token_mean)`` off one masked square.

    ``action_is_pad``: optional ``[batch, horizon]``, True where padded. Padded
    rows dropped and each sample renormalised by its executed count, so a short
    chunk keeps a full chunk's magnitude.

    Scalar for ``scalar``, a per-sample array for ``sample`` / ``token_mean``, a
    pair for ``both``. ``ValueError`` if velocity shapes disagree or
    ``reduce_mode`` is unknown.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(f"v_hat shape {v_hat.shape} != v_star shape {v_star.shape}")
    sq = jnp.square(v_hat - v_star)
    n_valid = None
    if action_is_pad is not None:
        # action_is_pad marks what to drop, so flip it: keep is 1 on executed
        # positions, 0 on padded ones.
        keep = (~action_is_pad).astype(sq.dtype)[:, None, :, None]
        # [n, 1]: valid tokens per sample, scaled by action_dim so the divisor
        # counts scalar elements, not tokens.
        n_valid = (~action_is_pad).sum(axis=-1, keepdims=True).astype(sq.dtype)
        n_valid = jnp.clip(n_valid * sq.shape[-1], min=1.0)
        sq = sq * keep

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / n_valid.sum()

    if reduce_mode in ("sample", "both"):
        flat = sq.reshape(sq.shape[0], sq.shape[1], -1)
        if cfg.cfm_loss_reduction == "mean":
            sample = flat.mean(axis=-1) if n_valid is None else flat.sum(axis=-1) / n_valid
        elif cfg.cfm_loss_reduction == "sum":
            sample = flat.sum(axis=-1)
        else:
            total = flat.sum(axis=-1)
            sample = (
                total / math.sqrt(flat.shape[-1])
                if n_valid is None
                else total / jnp.sqrt(n_valid)
            )
        if reduce_mode == "sample":
            return sample
        return sample, sq.sum(axis=-1) / sq.shape[-1]
    if reduce_mode == "token_mean":
        # Per-token diagnostic. A token aggregates only action_dim, so dividing
        # by that keeps the value independent of token count. The mask already
        # zeroed the padded tokens.
        return sq.sum(axis=-1) / sq.shape[-1]
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', 'token_mean', "
        "or 'both'"
    )


def compute_aspo_surrogate(
    ratio: jax.Array, advantage: jax.Array, cfg
) -> jax.Array:
    """FPO++ trust region: PPO clipping, SPO, or asymmetric ASPO.

    ASPO = PPO on positive advantages, SPO on negative. ``ratio`` is the FPO++
    likelihood ratio (rho in the paper). Returns the elementwise surrogate.
    """
    ratio_adv = ratio * advantage
    ratio_clipped = jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = jnp.minimum(ratio_adv, ratio_clipped * advantage)
    spo = ratio_adv - (jnp.abs(advantage) / (2.0 * cfg.clip_param)) * jnp.square(
        ratio - 1.0
    )

    if cfg.trust_region_mode == "ppo":
        return ppo
    if cfg.trust_region_mode == "spo":
        return spo
    return jnp.where(advantage > 0, ppo, spo)


def compute_policy_ratio(
    l_old: jax.Array,
    l_new: jax.Array,
    cfg,
    advantage: jax.Array | None = None,
) -> jax.Array:
    """Official FPO++ ratio exp(old CFM loss - new CFM loss).

    Clamps both CFM losses from above, optionally clamps the new loss on
    negative-advantage samples, then applies a one-sided STE cap to the
    log-ratio: forward uses the capped delta, backward flows through the
    original. ``advantage`` is required when
    ``cfm_loss_clamp_negative_advantages`` is set. Returns rho, standing in for
    a Gaussian policy's ``actions_log_prob`` ratio.
    """
    old = jnp.clip(l_old, max=cfg.cfm_clamp)
    new = jnp.clip(l_new, max=cfg.cfm_clamp)
    if cfg.cfm_loss_clamp_negative_advantages and advantage is not None:
        new = jnp.where(
            advantage < 0,
            jnp.clip(new, max=cfg.cfm_loss_clamp_negative_advantages_max),
            new,
        )
    delta = old - new
    safe_delta = jnp.clip(delta, max=cfg.log_ratio_cap)
    delta_st = lax.stop_gradient(safe_delta) + delta - lax.stop_gradient(delta)
    return jnp.exp(delta_st)


def sample_action(
    predict_velocity: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    eps: jax.Array,
    condition: jax.Array,
    num_inference_steps: int,
) -> jax.Array:
    """Euler-sample the learned flow t=0 (noise) -> t=1 (data).

    Step sizes are computed once up front and the 16 sequential steps become one
    ``lax.scan``, so the whole integration is a single compiled program.
    ``eps`` is the flow's start point and the caller draws it, so
    ``noise_sampling_mode`` and ``noise_scaling_mode`` both apply; a
    zero-sampling caller passes zeros. Returns the integrated sample.
    """
    grid = jnp.linspace(0.0, 1.0, num_inference_steps + 1, dtype=jnp.float32)

    def step(x, t_dt):
        t_current, dt = t_dt
        v_hat = predict_velocity(x, t_current.reshape(1), condition)
        return x + v_hat * dt, None

    dts = grid[1:] - grid[:-1]
    x, _ = lax.scan(step, eps, (grid[:-1], dts))
    return x
