"""FPO++ core maths. The module holds the CFM loss, the ratio, the ASPO trust
region, the timestep and noise laws, and the flow sampler.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from myfpo.configuration import MyfpoConfig
from myfpo.utils import resolve_storage_dtype


def sample_timesteps(
    num_samples: int,
    cfg: MyfpoConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """Training timesteps in ``[cfg.min_sampled_timestep, cfg.max_sampled_timestep]``.``[num_samples]``."""
    lo, hi = cfg.min_sampled_timestep, cfg.max_sampled_timestep
    mode = cfg.timestep_sampling_mode
    if mode == "uniform":
        return (
            torch.rand(num_samples, device=device, generator=generator) * (hi - lo) + lo
        )
    if mode == "fpo_beta":
        u = torch.rand(num_samples, device=device, generator=generator)
        beta = cfg.timestep_inverse_cdf_beta
        return lo + (hi - lo) * (
            1.0 - torch.pow(1.0 - u, 1.0 / beta)
        )
    raise ValueError(
        f"mode={mode!r} unknown; expected 'uniform' or 'fpo_beta'"
    )


def sample_noise(
    shape,
    cfg: MyfpoConfig,
    device: torch.device | str,
    generator: torch.Generator | None,
):
    """MC noise. The marginal comes from ``noise_sampling_mode``."""
    mode = cfg.noise_sampling_mode
    if mode == "normal":
        return torch.randn(*shape, device=device, generator=generator)
    if mode == "uniform":
        return (torch.rand(*shape, device=device, generator=generator) * 2.0 - 1.0) * math.sqrt(3.0)
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def broadcast_over_mc_axis(
    action: torch.Tensor, eps: torch.Tensor, t
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Broadcast action, noise, and timestep over the MC axis. Returns ``[B*M, ...]``."""
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(action.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")

    storage_dtype = resolve_storage_dtype(eps.device)
    if isinstance(t, torch.Tensor):
        t = t.to(device=eps.device, dtype=storage_dtype)
    else:
        t = torch.as_tensor(t, dtype=storage_dtype, device=eps.device)

    if t.dim() != 4 or tuple(t.shape) != (b, num_mc, flow_shape[0], 1):
        raise ValueError(
            f"t shape {tuple(t.shape)} must be [B, num_mc_samples, K, 1] "
            f"with B={b}, num_mc_samples={num_mc}, K={flow_shape[0]}"
        )

    d = action.numel() // b
    action_flat = action.reshape(b, 1, d).expand(b, num_mc, d).reshape(b * num_mc, d)
    flatten_mc_timestep = t.reshape(b * num_mc, flow_shape[0], 1)
    return (
        flatten_mc_timestep,
        eps.reshape(b * num_mc, d),
        action_flat,
        flow_shape,
        num_mc,
    )


def clip_velocity(velocity: torch.Tensor, bound: float | None) -> torch.Tensor:
    """Bound the actor's raw velocity output (FPO++ ``transported_clip_value``). ``None`` does nothing."""
    if bound is None:
        return velocity
    return velocity.clamp(-bound, bound)


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: MyfpoConfig,
    reduce_mode: str,
    action_is_pad: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CFM squared-velocity loss (FPO++ Eq. 7-8). ``reduce_mode`` picks the reduction; ``action_is_pad`` masks padded rows."""
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    diff = v_hat - v_star
    if cfg.cfm_loss_use_huber:
        delta = cfg.cfm_loss_huber_delta
        abs_diff = diff.abs()
        sq = torch.where(
            abs_diff <= delta,
            diff.square(),
            2.0 * delta * abs_diff - delta * delta,
        )
    else:
        sq = diff.square()
    n_valid = None
    if action_is_pad is not None:
        keep = (~action_is_pad).to(sq.dtype)[:, None, :, None]
        n_valid = (
            (~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]
        )
        n_valid = n_valid.clamp(min=1.0)
        sq = sq * keep

    if weight is not None:
        sq = sq * weight

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / n_valid.sum()

    if reduce_mode in ("sample", "sample_and_token"):
        flat = sq.flatten(2)
        if cfg.cfm_loss_reduction == "sum":
            sample = flat.sum(dim=-1)
        else:
            sample = flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
        if reduce_mode == "sample":
            return sample
        return sample, sq.mean(dim=-1)
    raise ValueError(
        f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'sample_and_token'"
    )


def group_cfm_loss(loss: torch.Tensor, group_size: int) -> torch.Tensor:
    """Average CFM loss over MC groups. ``-1`` = one group over all samples (FPO Eq. 9)."""
    if group_size == 1:
        return loss
    n, num_mc = loss.shape[0], loss.shape[1]
    size = num_mc if group_size == -1 else group_size
    if num_mc % size != 0:
        raise ValueError(
            f"num_mc_samples={num_mc} must divide evenly by group size {size}"
        )
    return loss.reshape(n, num_mc // size, size, *loss.shape[2:]).mean(dim=2)


def _sigreg_sketch(z: torch.Tensor, sketch_dim: int) -> torch.Tensor:
    """Project rows to at most ``sketch_dim`` columns with a random map."""
    n, c = z.shape
    s = min(sketch_dim, n, c)
    if c <= s:
        return z
    mat = torch.randn(s, c, device=z.device, dtype=z.dtype) / (c**0.5)
    return z @ mat.T



# A covariance needs two samples to divide by (n - 1). Below that the loss is
# undefined, so it returns zero instead.
_MIN_SIGREG_SAMPLES = 2
def sigreg_weak_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Weak SIGReg. Minimizes ||cov - eye||_F (distance to identity), not a match."""
    z = z.float()
    if z.shape[0] < _MIN_SIGREG_SAMPLES:
        return torch.zeros((), device=z.device)
    z = _sigreg_sketch(z, sketch_dim)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    eye = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    return torch.norm(cov - eye, p="fro")


def sigreg_decorr_loss(z: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    """Decorrelation-only SIGReg. Standardises the columns, then penalises the
    off-diagonal correlation. No match to the identity is enforced.
    """
    z = z.float()
    if z.shape[0] < _MIN_SIGREG_SAMPLES:
        return torch.zeros((), device=z.device)
    z = _sigreg_sketch(z, sketch_dim)
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = ((z / std).T @ (z / std)) / (z.shape[0] - 1)
    off = corr - torch.diag(torch.diag(corr))
    return torch.norm(off, p="fro")


def cfm_loss_weight(t: torch.Tensor, cfg: MyfpoConfig) -> torch.Tensor:
    """Per-token CFM loss weight (broadcasts over action dims). Weight rises linearly to 1.0 at t=1 (0.01 + 0.99*t); no verified 'reversed vs reference' claim."""
    mode = cfg.cfm_loss_weight_from_t
    if mode == "constant":
        return torch.ones_like(t)
    if mode == "linear_0.1_to_1":
        return 0.1 + 0.9 * t
    return 0.01 + 0.99 * t


def velocity_to_target_param(
    v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, cfg: MyfpoConfig
) -> torch.Tensor:
    """Map velocity into the target space of ``cfg.cfm_loss_target``. Only the loss compares; network predicts velocity."""
    mode = cfg.cfm_loss_target
    if mode == "velocity":
        return v
    if mode == "action":
        return x_t + (1.0 - t) * v
    return x_t - t * v


def cfm_target(
    a: torch.Tensor, eps: torch.Tensor, cfg: MyfpoConfig
) -> torch.Tensor:
    """The regression target for ``cfg.cfm_loss_target``. ``[B, *flow]``."""
    mode = cfg.cfm_loss_target
    if mode == "velocity":
        return a.unsqueeze(1) - eps
    if mode == "action":
        return a.unsqueeze(1).expand_as(eps)
    return eps


def compute_aspo_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    cfg: MyfpoConfig,
) -> torch.Tensor:
    """FPO++ trust region. ASPO is PPO for A>0 and SPO for A<0. It runs on each
    ratio element independently, so both branches are always computed.
    """
    ratio_adv = ratio * advantage
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = torch.min(ratio_adv, ratio_clipped * advantage)
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.spo_clip_param)) * torch.square(
        ratio - 1.0
    )

    if cfg.trust_region_mode == "ppo":
        return ppo
    if cfg.trust_region_mode == "spo":
        return spo
    return torch.where(advantage > 0, ppo, spo)


def clamp_saturation(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: MyfpoConfig,
) -> torch.Tensor:
    """Share of elements where both old and new losses exceed ``cfm_clamp`` (ratio pinned to 1, zero gradient)."""
    if cfm_loss_old.shape != cfm_loss_new.shape:
        raise ValueError(
            f"cfm_loss_old shape {tuple(cfm_loss_old.shape)} != "
            f"{tuple(cfm_loss_new.shape)}"
        )
    bound = cfg.cfm_clamp
    both = (cfm_loss_old > bound) & (cfm_loss_new > bound)
    return both.to(torch.float32).mean()


# NOTE (publishability): Approximate ratio rho_hat = exp(L_old - L_new)
# is a biased proxy for log pi_new/pi_old. See FPO (arXiv:2507.21053),
# FPO++ (arXiv:2602.02481), and literature_review.md (Sec 4.2).
def compute_policy_ratio(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: MyfpoConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``exp(cfm_loss_old - cfm_loss_new)``, with clamping and straight-through cap on delta."""
    old = torch.clamp(cfm_loss_old, max=cfg.cfm_clamp)
    new = torch.clamp(cfm_loss_new, max=cfg.cfm_clamp)
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
    num_steps: int,
    eps: torch.Tensor,
    device: torch.device | str | None,
):
    """Euler-sample flow from t=0 (noise) to t=1 (data). ``device`` is required explicitly (no default)."""
    if device is None:
        raise ValueError("device must be provided explicitly")
    if tuple(eps.shape) != tuple(shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != requested {tuple(shape)}")

    grid = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=device,
        dtype=torch.float32,
    )

    if eps.device != torch.device(device):
        eps = eps.to(device)

    x = eps
    with torch.inference_mode():
        dts = grid[1:] - grid[:-1]
        for i in range(num_steps):
            t_current = grid[i].reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            x = torch.addcmul(x, v_hat, dts[i])
    return x
