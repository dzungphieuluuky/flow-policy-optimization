"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.utils import resolve_storage_dtype


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_timesteps(
    num_samples: int,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Flow times using the explicit configuration and sample shape."""
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


def sample_noise(
    shape,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """MC noise ε of shape ``[*shape]`` using the explicit configuration."""
    mode = cfg.noise_sampling_mode
    if mode == "normal":
        return torch.randn(*shape, device=device, generator=generator)
    if mode == "uniform":
        s = math.sqrt(3.0)
        return (torch.rand(*shape, device=device, generator=generator) * 2.0 - 1.0) * s
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def run_mc_simulation(
    a: torch.Tensor, eps: torch.Tensor, tau
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Broadcast action, noise, and per-coordinate timesteps over MC.

    Legacy scalar/[M]/[B,M] tau inputs return ``flatten_mc_timestep`` with shape
    ``[B*M]``. The training path uses per-token tau with shape
    ``[B, M, K, 1]`` and receives ``[B*M, K, 1]``; each token timestep is shared
    across that token's action dimensions.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    # Coerce tau to a device-matched tensor without a silent CPU hop.
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


# ---------------------------------------------------------------------------
# CFM loss
# ---------------------------------------------------------------------------


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: SJFPOConfig,
    reduce_mode: str,
) -> torch.Tensor:
    """CFM squared-velocity loss (paper Eq.7-8).

    The active ``sample`` path follows FPO++'s configurable reduction over the
    flattened action chunk (``mean``, ``sum``, or ``sqrt``; default ``sqrt``).
    ``bin`` remains a legacy diagnostic reduction.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    sq = torch.square(v_hat - v_star)

    if reduce_mode == "scalar":
        return sq.mean()

    if reduce_mode == "sample":
        flat = sq.flatten(2)
        if cfg.cfm_loss_reduction == "mean":
            return flat.mean(dim=-1)
        if cfg.cfm_loss_reduction == "sum":
            return flat.sum(dim=-1)
        return flat.sum(dim=-1) / math.sqrt(flat.shape[-1])
    if reduce_mode == "bin":
        # Legacy diagnostic reduction; the official FPO policy uses "sample".
        return sq.sum(dim=-1) / math.sqrt(sq.shape[-1])
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
    )


# ---------------------------------------------------------------------------
# ASPO surrogate / policy ratio / policy loss
# ---------------------------------------------------------------------------


def compute_aspo_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
    epsilon_clip: float | torch.Tensor,
) -> torch.Tensor:
    """FPO++ trust region: PPO clipping, SPO, or asymmetric ASPO.

    ASPO uses PPO for positive advantages and SPO for negative advantages.
    """
    rho_adv = rho * advantage
    rho_clipped = torch.clamp(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
    ppo = torch.min(rho_adv, rho_clipped * advantage)
    spo = rho_adv - (advantage.abs() / (2.0 * epsilon_clip)) * torch.square(rho - 1.0)

    if cfg.trust_region_mode == "ppo":
        return ppo
    if cfg.trust_region_mode == "spo":
        return spo
    return torch.where(advantage > 0, ppo, spo)


def compute_policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    cfg: SJFPOConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Official FPO++ ratio exp(old CFM loss - new CFM loss).

    The official implementation clamps both CFM losses from above, optionally
    clamps the new loss for negative-advantage samples, then applies a
    one-sided STE cap to the log-ratio.
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
    # STE: forward uses the capped delta, backward flows through the original.
    delta_st = safe_delta.detach() + delta - delta.detach()
    return torch.exp(delta_st)


def compute_policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfg, advantage=advantage)
    return compute_aspo_surrogate(rho, advantage, cfg, cfg.epsilon_clip).mean()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor | None,
    generator: torch.Generator | None,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample the learned flow t=0 (noise) → t=1 (data) with Euler steps."""
    if device is None:
        raise ValueError("device must be provided explicitly")

    grid = torch.linspace(
        0.0,
        1.0,
        num_inference_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    if eps is None:
        eps = torch.randn(*shape, device=device, generator=generator)
    elif device is not None and eps.device != torch.device(device):
        eps = eps.to(device)

    x = eps
    with torch.inference_mode():
        # Precompute dt once; XLA/torch compile friendly.
        dts = grid[1:] - grid[:-1]  # [steps]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            x = x + dts[i] * v_hat
    return x
