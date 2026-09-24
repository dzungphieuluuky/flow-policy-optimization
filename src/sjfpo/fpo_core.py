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
    """Flow times τ [num_samples] using the explicit configuration."""
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
    raise ValueError(f"mode={mode!r} unknown; expected 'uniform', 'normal' or 'beta'")


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
    """Broadcast (a, eps, tau) onto the shared ``[B*M]`` MC axis.

    Returns ``(tau_bn, eps_flat, a_rep, flow_shape, num_mc)``.
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

    if t.dim() == 0:
        t = t.expand(b, num_mc)
    elif t.dim() == 1:
        if t.numel() != num_mc:
            raise ValueError(
                f"tau vector must have num_mc_samples={num_mc} entries, got {t.numel()}"
            )
        t = t.expand(b, num_mc)
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be scalar, [num_mc_samples] or [B, num_mc_samples]"
        )

    d = a.numel() // b
    a_flat = a.reshape(b, 1, d).expand(b, num_mc, d).reshape(b * num_mc, d)
    return (
        t.reshape(b * num_mc),
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

    ``sample``/``bin`` use ``sum / sqrt(D)`` — preserves variance across
    action dims instead of shrinking log-ratios like a mean.
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
        reduced = flat.sum(dim=-1) / math.sqrt(flat.shape[-1])
    elif reduce_mode == "bin":
        reduced = sq.sum(dim=-1) / math.sqrt(sq.shape[-1])
    else:
        raise ValueError(
            f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
        )

    return torch.clamp(reduced, max=cfg.cfm_clamp)


# ---------------------------------------------------------------------------
# ASPO surrogate / policy ratio / policy loss
# ---------------------------------------------------------------------------


def compute_aspo_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
    epsilon_clip: float | torch.Tensor,
) -> torch.Tensor:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12) or PPO clip.

    Hoists ``rho * advantage`` and the clipped ratio so neither is computed
    twice on the non-symmetric branch.
    """
    symmetrical_ppo = not cfg.use_aspo
    rho_adv = rho * advantage
    rho_clipped = torch.clamp(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip)
    pos = torch.min(rho_adv, rho_clipped * advantage)

    if symmetrical_ppo:
        return pos

    neg = rho_adv - (advantage.abs() / (2.0 * epsilon_clip)) * torch.square(rho - 1.0)
    return torch.where(advantage > 0, pos, neg)


def compute_policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """Per-sample importance ratios ρ_i = exp(ℓ_old,i − ℓ_new,i) with OFF clamps.

    Symmetric max-clamp on both losses (``cfm_loss_clamp``) and a symmetric
    STE cap on the log-ratio (``cfm_diff_clamp_max``) — forward clamped,
    gradient flows.
    """
    delta = torch.clamp(l_old, max=cfg.cfm_clamp) - torch.clamp(
        l_new, max=cfg.cfm_clamp
    )
    safe_delta = delta.clamp(-cfg.log_ratio_cap, cfg.log_ratio_cap)
    # STE: forward uses clamped delta, backward flows through the unclamped delta.
    delta_st = safe_delta.detach() + delta - delta.detach()
    return torch.exp(delta_st)


def compute_policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfg)
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
