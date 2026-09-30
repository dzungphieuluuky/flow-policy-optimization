"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.utils import resolve_storage_dtype


def sample_timesteps(
    num_samples: int,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Draw flow timesteps in ``[min_timestep, max_timestep]``.

    Mode comes from ``cfg.timestep_sampling_mode``. Returns ``[num_samples]``.
    ``ValueError`` on an unknown mode.
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


def _freq_weighted_std(num_modes: int, horizon: int, device) -> torch.Tensor:
    """Per-mode std of pink (1/f) noise over DCT-II modes.

    NEED (Eberhard et al. 2023): beta=1 coloring beats white and OU for
    continuous control. Chunks are band-limited, so base variance belongs in
    the low modes, not spread evenly over the coefficient axis.
    """
    k = torch.arange(num_modes, dtype=torch.float32, device=device)
    return torch.rsqrt(1.0 + (k * (math.pi / horizon)) ** 2)


def sample_noise(
    shape,
    cfg: SJFPOConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Draw Monte-Carlo noise of ``shape``.

    Marginal from ``cfg.noise_sampling_mode``; per-mode std over the token axis
    from ``cfg.noise_scaling_mode``. ``ValueError`` on an unknown mode.
    """
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
        std = _freq_weighted_std(shape[-2], cfg.action_horizon, device)
        return eps * std.view(*(1,) * (eps.dim() - 2), -1, 1)
    raise ValueError(
        f"mode={cfg.noise_scaling_mode!r} unknown; expected 'white' or 'freq_weighted'"
    )


def run_monte_carlo_sampling(
    a: torch.Tensor, eps: torch.Tensor, tau
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Broadcast action, noise, and per-token timesteps over MC.

    Training passes ``tau`` as ``[B, M, K, 1]`` and gets ``[B*M, K, 1]``. One
    time per token, shared across its action dims. Scalar, ``[M]``, ``[B, M]``
    forms are legacy and return ``[B*M]``. ``tau`` is moved onto ``eps``'s
    device and dtype, so a scalar cannot cause a silent CPU hop mid-graph.

    Returns flattened timestep, noise, broadcast action, flow shape, MC count.
    ``ValueError`` if shapes disagree.
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

    ``sample`` is the active path:

    - FPO++'s reduction over the flattened chunk
    - ``mean``, ``sum``, or ``sqrt``; default ``mean``

    ``token_mean`` is diagnostic only:

    - mean over the token axis of the operand's space
    - freqs in coefficient space, action steps on the step grid
    - divisor is ``action_dim``, the only axis a token aggregates
    - so masking cannot move the value

    ``action_is_pad``: optional ``[batch, horizon]``, True where padded.

    - padded rows dropped
    - each sample renormalised by its executed count
    - so a short chunk keeps a full chunk's magnitude

    Scalar for ``scalar``, else a per-sample loss tensor. ``ValueError`` if
    velocity shapes disagree or ``reduce_mode`` is unknown.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    sq = torch.square(v_hat - v_star)
    n_valid = None
    if action_is_pad is not None:
        # action_is_pad marks what to drop, so flip it: keep is 1 on executed
        # positions, 0 on padded ones.
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        # [n, 1]: valid tokens per sample, scaled by action_dim so the divisor
        # counts scalar elements, not tokens.
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
    if reduce_mode == "token_mean":
        # Per-token diagnostic. A token aggregates only action_dim, so dividing
        # by that keeps the value independent of token count. mask_sq already
        # zeroed the padded tokens.
        return sq.sum(dim=-1) / sq.shape[-1]
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'token_mean'"
    )


def compute_aspo_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    cfg: SJFPOConfig,
) -> torch.Tensor:
    """FPO++ trust region: PPO clipping, SPO, or asymmetric ASPO.

    ASPO = PPO on positive advantages, SPO on negative. ``ratio`` is the FPO++
    likelihood ratio (rho in the paper). Returns the elementwise surrogate.
    """
    ratio_adv = ratio * advantage
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = torch.min(ratio_adv, ratio_clipped * advantage)
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.clip_param)) * torch.square(ratio - 1.0)

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

    Clamps both CFM losses from above, optionally clamps the new loss on
    negative-advantage samples, then applies a one-sided STE cap to the
    log-ratio: forward uses the capped delta, backward flows through the
    original. ``advantage`` is required when
    ``cfm_loss_clamp_negative_advantages`` is set. Returns rho, standing in for
    a Gaussian policy's ``actions_log_prob`` ratio.
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
    """Euler-sample the learned flow t=0 (noise) -> t=1 (data).

    Step sizes are computed once up front, so the integration loop holds no
    tensor arithmetic and stays XLA friendly. ``eps`` is the flow's start point
    and the caller draws it, so ``noise_sampling_mode`` and
    ``noise_scaling_mode`` both apply; a zero-sampling caller passes zeros.
    Returns a sample of ``shape``. ``ValueError`` if no ``device``.
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

    if eps.device != torch.device(device):
        eps = eps.to(device)

    x = eps
    with torch.inference_mode():
        dts = grid[1:] - grid[:-1]
        for i in range(num_inference_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            x = x + dts[i] * v_hat
    return x
