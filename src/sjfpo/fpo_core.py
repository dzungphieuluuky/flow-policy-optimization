"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.utils import resolve_storage_dtype


def sample_timesteps(
    num_samples: int,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Draw flow timesteps in ``[min_timestep, max_timestep]``.

    The distribution is selected by ``cfg.timestep_sampling_mode``. Returns a
    tensor of shape ``[num_samples]``. Raises ``ValueError`` if the mode is not a
    known mode.
    """
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
    """Draw Monte-Carlo noise of shape ``shape``.

    The distribution is selected by ``cfg.noise_sampling_mode``. Raises
    ``ValueError`` if the mode is not a known mode.
    """
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
    ``[B*M]``. The training path uses per-token tau with shape ``[B, M, K, 1]`` and
    receives ``[B*M, K, 1]``; each token timestep is shared across that token's
    action dimensions. ``tau`` is always coerced onto ``eps``'s device and storage
    dtype so a scalar never triggers a silent CPU hop part-way through the graph.

    Returns the flattened timestep, the flattened noise, the broadcast action, the
    per-sample flow shape, and the Monte-Carlo sample count. Raises
    ``ValueError`` if ``eps`` disagrees with the action shape, or if ``tau``
    matches none of the accepted shapes.
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
    """CFM squared-velocity loss (paper Eq.7-8).

    The active ``sample`` path follows FPO++'s configurable reduction over the
    flattened action chunk (``mean``, ``sum``, or ``sqrt``; default ``mean``).
    ``bin`` is a legacy diagnostic reduction kept for comparison, since the
    official FPO policy only ever uses ``sample``.

    ``action_is_pad`` is an optional ``[batch, horizon]`` mask of never-executed
    action positions, True where padded. When given, padded positions are dropped
    and each sample is renormalized by its own executed count, so a chunk cut
    short contributes the same loss magnitude as a full one.

    Returns a scalar for ``scalar``, otherwise a per-sample loss tensor. Raises
    ``ValueError`` if the velocity shapes disagree, or if ``reduce_mode`` is
    unknown.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    sq = torch.square(v_hat - v_star)
    n_valid = None
    if action_is_pad is not None:
        # action_is_pad marks what to discard, so flip it before multiplying:
        # keep is 1 on executed positions and 0 on padded ones.
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        # [n, 1]: valid tokens per sample, scaled by action_dim so the divisor
        # counts scalar elements rather than tokens.
        n_valid = (
            (~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]
        )
        n_valid = n_valid.clamp(min=1.0)
        sq = sq * keep

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / n_valid.sum()

    if reduce_mode == "sample":
        flat = sq.flatten(2)
        if cfg.cfm_loss_reduction == "mean":
            return flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
        if cfg.cfm_loss_reduction == "sum":
            return flat.sum(dim=-1)
        total = flat.sum(dim=-1)
        if n_valid is None:
            return total / math.sqrt(flat.shape[-1])
        return total / n_valid.sqrt()
    if reduce_mode == "bin":
        # Per-token diagnostic; masking sq already zeroes padded tokens.
        return sq.sum(dim=-1) / math.sqrt(sq.shape[-1])
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
    )


def compute_aspo_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
    epsilon_clip: float | torch.Tensor,
) -> torch.Tensor:
    """FPO++ trust region: PPO clipping, SPO, or asymmetric ASPO.

    ASPO uses PPO for positive advantages and SPO for negative advantages.
    Returns the elementwise surrogate objective.
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
    clamps the new loss for negative-advantage samples, then applies a one-sided
    STE cap to the log-ratio: the forward pass uses the capped delta while the
    backward pass flows through the original. ``advantage`` is required when
    ``cfm_loss_clamp_negative_advantages`` is set.
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


def compute_policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of psi per chunk (Eq.13).
    """
    rho = compute_policy_ratio(l_old, l_new, cfg, advantage=advantage)
    return compute_aspo_surrogate(rho, advantage, cfg, cfg.epsilon_clip).mean()


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor | None,
    generator: torch.Generator | None,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample the learned flow t=0 (noise) -> t=1 (data) with Euler steps.

    The step sizes are computed once up front, which keeps the integration loop
    free of tensor arithmetic and friendly to XLA. Returns an integrated sample of
    shape ``shape``; raises ``ValueError`` if ``device`` is not provided.
    """
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
        dts = grid[1:] - grid[:-1]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            x = x + dts[i] * v_hat
    return x
