"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()


# Lazily materialized config: keeps import cheap and lets callers override
# before the first use via `_defaults()`.
_DEFAULT_CFG: SJFPOConfig | None = None


def _defaults() -> SJFPOConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = SJFPOConfig()
    return _DEFAULT_CFG


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_timesteps(
    num_samples: int,
    tau_range=None,
    device=None,
    mode="uniform",
    *,
    alpha=None,
    beta=None,
    generator: torch.Generator | None = None,
):
    """Flow times τ [num_samples] in [lo, hi].

    ``mode``: ``"uniform"`` | ``"normal"`` (mid-weighted) | ``"beta"``
    (lerobot openpi Beta(alpha, beta) scaled to [lo, hi]).
    """
    cfg = _defaults()
    lo, hi = (cfg.min_timestep, cfg.max_timestep) if tau_range is None else tau_range
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
        if alpha is None or beta is None:
            raise ValueError("mode='beta' needs alpha= and beta= kwargs")
        from lerobot.policies.common.flow_matching import sample_time_beta

        return sample_time_beta(
            num_samples,
            device,
            alpha=alpha,
            beta=beta,
            scale=hi - lo,
            offset=lo,
        )
    raise ValueError(f"mode={mode!r} unknown; expected 'uniform', 'normal' or 'beta'")


def sample_noise(
    shape,
    device=None,
    mode="normal",
    generator: torch.Generator | None = None,
):
    """MC noise ε of shape ``[*shape]``.

    ``mode``: ``"normal"`` = N(0, I); ``"uniform"`` = U[-√3, √3] (unit variance).
    """
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
    if isinstance(tau, torch.Tensor):
        t = tau.to(device=eps.device, dtype=STORAGE_DTYPE)
    else:
        t = torch.as_tensor(tau, dtype=STORAGE_DTYPE, device=eps.device)

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
    reduce_mode: str = "scalar",
    clamp: float | None = None,
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
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample'/'bin' only")
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

    cfg = _defaults()
    clamp = cfg.cfm_clamp if clamp is None else clamp
    return torch.clamp(reduced, max=clamp)


# ---------------------------------------------------------------------------
# ASPO surrogate / policy ratio / policy loss
# ---------------------------------------------------------------------------


def compute_aspo_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    epsilon_clip: float | None = None,
    symmetrical_ppo: bool | None = None,
) -> torch.Tensor:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12) or PPO clip.

    Hoists ``rho * advantage`` and the clipped ratio so neither is computed
    twice on the non-symmetric branch.
    """
    cfg = _defaults()
    if epsilon_clip is None:
        epsilon_clip = cfg.epsilon_clip
    if symmetrical_ppo is None:
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
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Per-sample importance ratios ρ_i = exp(ℓ_old,i − ℓ_new,i) with OFF clamps.

    Symmetric max-clamp on both losses (``cfm_loss_clamp``) and a symmetric
    STE cap on the log-ratio (``cfm_diff_clamp_max``) — forward clamped,
    gradient flows.
    """
    cfg = _defaults()
    cfm_clamp = cfg.cfm_clamp if cfm_clamp is None else cfm_clamp
    cap = cfg.log_ratio_cap if cap is None else cap

    delta = torch.clamp(l_old, max=cfm_clamp) - torch.clamp(l_new, max=cfm_clamp)
    safe_delta = delta.clamp(-cap, cap)
    # STE: forward uses clamped delta, backward flows through the unclamped delta.
    delta_st = safe_delta + (delta - safe_delta).detach()
    return torch.exp(delta_st)


def compute_policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    advantage: torch.Tensor,
    epsilon_clip: float | None = None,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of ψ per chunk (Eq.13)."""
    rho = compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=cap)
    return compute_aspo_surrogate(rho, advantage, epsilon_clip).mean()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_inference_steps: int,
    eps: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device=None,
    mode: str = "euler",
) -> torch.Tensor:
    """Sample the learned flow t=0 (noise) → t=1 (data); Euler by default.

    The grid and the initial noise are placed on ``device`` (defaults to
    ``condition.device``), so no per-step CPU↔GPU syncs occur.
    """
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")

    # Resolve device from the condition when the caller didn't specify one.
    if device is None:
        device = condition.device if isinstance(condition, torch.Tensor) else None

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
