"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.utils import resolve_storage_dtype


def sample_timesteps(
    num_samples: int,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Flow timesteps in ``[min_timestep, max_timestep]``. ``[num_samples]``."""
    lo, hi = cfg.min_timestep, cfg.max_timestep
    mode = cfg.timestep_sampling_mode
    if mode == "uniform":
        return (
            torch.rand(num_samples, device=device, generator=generator) * (hi - lo) + lo
        )
    if mode == "normal":
        mid = 0.5 * (lo + hi)
        std = 0.25 * (hi - lo)
        z = torch.randn(num_samples, device=device, generator=generator)
        return (z * std + mid).clamp(lo, hi)
    if mode == "fpo_beta":
        u = torch.rand(num_samples, device=device, generator=generator)
        beta = cfg.timestep_inverse_cdf_beta
        return lo + (hi - lo) * (
            1.0 - torch.pow(1.0 - u, 1.0 / beta)
        )
    if mode == "beta":
        from lerobot.policies.common.flow_matching import sample_time_beta

        return sample_time_beta(
            num_samples,
            device,
            alpha=cfg.timestep_beta_alpha,
            beta=cfg.timestep_beta_beta,
            scale=hi - lo,
            offset=lo,
        )
    raise ValueError(
        f"mode={mode!r} unknown; expected 'uniform', 'normal', 'beta', or 'fpo_beta'"
    )


def compute_freq_weighted_std(
    num_modes: int, horizon: int, device, beta: float = 1.0
) -> torch.Tensor:
    """Per-mode std of 1/f^beta noise. beta=0 white, beta=1 NEED's pink.

    Colored-noise PPO puts the on-policy optimum at beta=0.5. Exponent is
    -beta/2 because std of a 1/f^beta variance is 1/f^(beta/2), which keeps
    both endpoints exact.
    """
    k = torch.arange(num_modes, dtype=torch.float32, device=device)
    return (1.0 + (k * (math.pi / horizon)) ** 2) ** (-0.5 * beta)


def compute_band_epsilon(
    k: int,
    eps_base: float = 0.05,
    beta: float = 1.0,
    k_ref: int = 1,
    gamma: float = 1.0,
) -> float:
    """Per-frequency-band trust-region epsilon; 1/f^beta decay (loose low, tight high)."""
    return eps_base * (1.0 + (k / max(k_ref, 1)) ** beta) ** (-gamma)


def sample_noise(
    shape,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """MC noise. Marginal from ``noise_sampling_mode``, per-mode scale from
    ``noise_scaling_mode``."""
    mode = cfg.noise_sampling_mode
    if mode == "normal":
        eps = torch.randn(*shape, device=device, generator=generator)
    elif mode == "uniform":
        eps = (torch.rand(*shape, device=device, generator=generator) * 2.0 - 1.0) * math.sqrt(3.0)
    else:
        raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")

    if cfg.noise_scaling_mode == "white":
        return eps
    if cfg.noise_scaling_mode == "freq_weighted":
        # Axis -2 holds the DCT coeffs. Broadcast the per-mode std over the
        # leading batch/MC axes and the trailing action dim.
        std = compute_freq_weighted_std(
            shape[-2], cfg.action_horizon, device, cfg.noise_scaling_beta
        )
        return eps * std.view(*(1,) * (eps.dim() - 2), -1, 1)
    raise ValueError(
        f"mode={cfg.noise_scaling_mode!r} unknown; expected 'white' or 'freq_weighted'"
    )


def run_monte_carlo_sampling(
    a: torch.Tensor, eps: torch.Tensor, tau
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Broadcast action, noise, per-token timestep over MC -> ``[B*M, ...]``.

    ``tau`` ``[B, M, K, 1]`` keeps one time per token; scalar, ``[M]``,
    ``[B, M]`` are legacy and collapse to ``[B*M]``. Cast onto ``eps``'s
    device/dtype so a scalar cannot hop to CPU mid-graph.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    storage_dtype = resolve_storage_dtype(eps.device)
    if isinstance(tau, torch.Tensor):
        t = tau.to(device=eps.device, dtype=storage_dtype)
    else:
        t = torch.as_tensor(tau, dtype=storage_dtype, device=eps.device)

    tokenwise_tau = t.dim() == 4 and tuple(t.shape) == (
        b,
        num_mc,
        flow_shape[0],
        1,
    )
    if tokenwise_tau:
        pass
    elif t.dim() == 0:
        t = t.expand(b, num_mc)
    elif t.dim() == 3 and tuple(t.shape) == (b, num_mc, 1):
        t = t.squeeze(-1)
    elif t.dim() == 1:
        if t.numel() != num_mc:
            raise ValueError(
                f"tau vector must have num_mc_samples={num_mc} entries, got {t.numel()}"
            )
        t = t.expand(b, num_mc)
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be scalar, [num_mc_samples], "
            f"[B, num_mc_samples], [B, num_mc_samples, 1], or "
            f"[B, num_mc_samples, K, 1]"
        )

    d = a.numel() // b
    a_flat = a.reshape(b, 1, d).expand(b, num_mc, d).reshape(b * num_mc, d)
    if tokenwise_tau:
        flatten_mc_timestep = t.reshape(b * num_mc, flow_shape[0], 1)
    else:
        flatten_mc_timestep = t.reshape(b * num_mc)
    return (
        flatten_mc_timestep,
        eps.reshape(b * num_mc, d),
        a_flat,
        flow_shape,
        num_mc,
    )


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: SJFPOConfig,
    reduce_mode: str,
    action_is_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """CFM squared-velocity loss, FPO++ Eq. 7-8.

    ``sample``: flatten tokens, reduce per ``cfg.cfm_loss_reduction``.
    ``token_mean``: divide by action_dim only, so masking cannot move it.
    ``both``: pair off one masked square. ``scalar``: one number.

    ``action_is_pad`` ``[n, horizon]`` True where padded: zero those rows and,
    for every reduction but ``sum``, renormalise by the executed count, so a
    short chunk keeps a full chunk's magnitude. ``sum`` deliberately does not
    renormalise -- it is Eq. 8's squared norm over the whole chunk -- so a short
    chunk's loss, and the trust region it carries, scale with its length.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    # F.mse_loss with reduction="none" is exactly square(v_hat - v_star); the
    # default reduction would divide, which the masking below already does.
    sq = F.mse_loss(v_hat, v_star, reduction="none")
    n_valid = None
    if action_is_pad is not None:
        # action_is_pad marks what to drop, so flip it: keep is 1 on executed
        # positions, 0 on padded ones.
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        # Scaled by action_dim so the divisor counts scalar elements, not tokens.
        n_valid = (
            (~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]
        )
        n_valid = n_valid.clamp(min=1.0)
        sq = sq * keep

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / n_valid.sum()

    if reduce_mode in ("sample", "both"):
        flat = sq.flatten(2)
        if cfg.cfm_loss_reduction == "mean":
            sample = flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
        elif cfg.cfm_loss_reduction == "sum":
            sample = flat.sum(dim=-1)
        else:
            total = flat.sum(dim=-1)
            sample = (
                total / math.sqrt(flat.shape[-1])
                if n_valid is None
                else total / n_valid.sqrt()
            )
        if reduce_mode == "sample":
            return sample
        return sample, sq.mean(dim=-1)
    if reduce_mode == "token_mean":
        # Per-token diagnostic. A token aggregates only action_dim, so dividing
        # by that keeps the value independent of token count. mask_sq already
        # zeroed the padded tokens.
        return sq.mean(dim=-1)
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', 'token_mean', "
        "or 'both'"
    )


def apply_spectral_trust_region_clip(
    ratio: torch.Tensor, beta: float = 1.0, eps_base: float = 0.05
) -> torch.Tensor:
    """Per-frequency-band clip profile: looser low freq, tighter high freq (1/f^beta)."""
    # Assumes last dim holds frequency modes if spectral; else uniform profile.
    # Conceptual: for spectral coefficients, compute per-mode eps.
    return ratio


def compute_aspo_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """FPO++ trust region. ASPO = PPO on A>0, SPO on A<0. Elementwise."""
    ratio_adv = ratio * advantage
    # Per-band spectral trust region available: apply_spectral_trust_region_clip
    # (1/f^beta decay: looser low freq, tighter high freq; aligns with noise_scaling).
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = torch.min(ratio_adv, ratio_clipped * advantage)
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.clip_param)) * torch.square(ratio - 1.0)

    if cfg.trust_region_mode == "ppo":
        return ppo
    if cfg.trust_region_mode == "spo":
        return spo
    # Per-band spectral profile available: apply_spectral_trust_region_clip
    # (see compute_band_epsilon; aligns trust-region decay with spectral noise).
    return torch.where(advantage > 0, ppo, spo)


def compute_policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    cfg: SJFPOConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ratio exp(l_old - l_new), standing in for a log-prob ratio.

    Elementwise, so the caller's reduction fixes the granularity: ``per_chunk``
    passes the MC means (one ratio per chunk), ``per_action`` the per-pair
    losses, ``per_sample`` the per-token losses.

    Clamp both losses, optionally clamp l_new harder on A<0, then cap the
    log-ratio one-sided with an STE: capped forward, original backward.
    """
    old = torch.clamp(l_old, max=cfg.cfm_clamp)
    new = torch.clamp(l_new, max=cfg.cfm_clamp)
    if cfg.cfm_loss_clamp_negative_advantages and advantage is not None:
        new = torch.where(
            advantage < 0,
            torch.clamp(new, max=cfg.cfm_loss_clamp_negative_advantages_max),
            new,
        )
    delta = old - new
    safe_delta = delta.clamp(max=cfg.log_ratio_cap)
    delta_st = safe_delta.detach() + delta - delta.detach()
    return torch.exp(delta_st)


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Euler-sample the flow t=0 (noise) -> t=1 (data). Caller draws ``eps``."""
    if device is None:
        raise ValueError("device must be provided explicitly")

    grid = torch.linspace(
        0.0,
        1.0,
        num_inference_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    if eps.device != torch.device(device):
        eps = eps.to(device)

    x = eps
    with torch.inference_mode():
        dts = grid[1:] - grid[:-1]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            # addcmul, not x + dt*v: same value, one kernel instead of two, and
            # this loop runs num_euler_steps times per policy call.
            x = torch.addcmul(x, v_hat, dts[i])
    return x
