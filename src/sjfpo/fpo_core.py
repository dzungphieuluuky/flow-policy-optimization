"""FPO++ core math: CFM losses, per-sample ratios, ASPO trust region, Euler sampler (arXiv 2602.02481 Eq.3-13)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import math
import torch
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.utils import resolve_storage_dtype
STORAGE_DTYPE = resolve_storage_dtype()  # cached once


# Single source for ``None``-fallback defaults.
_DEFAULTS = SJFPOConfig()


def sample_timesteps(
    num_samples: int,
    tau_range=None,
    device=None,
    mode="uniform",
    *,
    alpha=None,
    beta=None,
):
    """Flow times τ for the shared MC draw: [num_samples] in [lo, hi].

    ``mode``: "uniform" = U[lo, hi] (default); "normal" = N(mid, range/4)
    clamped, oversampling mid-times where the field curves most;
    "beta" = lerobot openpi Beta(alpha, beta) scaled to [lo, hi]
    (needs alpha=/beta= kwargs).
    """
    lo, hi = (
        (_DEFAULTS.min_timestep, _DEFAULTS.max_timestep)
        if tau_range is None
        else tau_range
    )
    if mode == "uniform":
        return torch.rand(num_samples, device=device) * (hi - lo) + lo
    if mode == "normal":
        mid = 0.5 * (lo + hi)
        std = 0.25 * (hi - lo)
        return (torch.randn(num_samples, device=device) * std + mid).clamp(lo, hi)
    if mode == "beta":
        if alpha is None or beta is None:
            raise ValueError("mode='beta' needs alpha= and beta= kwargs")
        from lerobot.policies.common.flow_matching import sample_time_beta

        return sample_time_beta(
            num_samples, device, alpha=alpha, beta=beta, scale=hi - lo, offset=lo
        )
    raise ValueError(f"mode={mode!r} unknown; expected 'uniform', 'normal' or 'beta'")


def sample_noise(shape, device=None, mode="normal"):
    """MC noise ε of shape [*shape].

    ``mode``: "normal" = N(0, I) (CFM-correct, v* assumes Gaussian endpoints);
    "uniform" = U[-√3, √3] (unit variance, same velocity scale; ablations).
    """
    if mode == "normal":
        return torch.randn(*shape, device=device)
    if mode == "uniform":
        s = math.sqrt(3.0)
        return (torch.rand(*shape, device=device) * 2.0 - 1.0) * s
    raise ValueError(f"mode={mode!r} unknown; expected 'normal' or 'uniform'")


def run_mc_simulation(a: torch.Tensor, eps: torch.Tensor, tau):
    """Broadcast (a, eps, tau) onto the shared [B*num_mc_samples] MC axis.

    tau: scalar, [num_mc_samples], or [B, num_mc_samples].
    Returns (tau_bn, eps_flat, a_rep, flow_shape, num_mc); raises ValueError
    on shape mismatches.
    """
    b, num_mc = eps.shape[0], eps.shape[1]
    flow_shape = tuple(a.shape[1:])
    if tuple(eps.shape) != (b, num_mc, *flow_shape):
        raise ValueError(f"eps shape {tuple(eps.shape)} != {(b, num_mc, *flow_shape)}")
    t = torch.as_tensor(tau, dtype=STORAGE_DTYPE)
    if t.dim() == 0:
        t = einx.id(" -> b m", t, b=b, m=num_mc)
    elif t.dim() == 1:
        if t.numel() != num_mc:
            raise ValueError(
                f"tau vector must have num_mc_samples={num_mc} entries, got {t.numel()}"
            )
        t = einx.id("m -> b m", t, b=b)
    elif tuple(t.shape) != (b, num_mc):
        raise ValueError(
            f"tau shape {tuple(t.shape)} must be scalar, [num_mc_samples] or [B, num_mc_samples]"
        )
    d = a.numel() // b
    a_flat = a.reshape(b, d)
    return (
        einx.id("b m -> (b m)", t),
        eps.reshape(b * num_mc, d),
        einx.id("b d -> (b m) d", a_flat, m=num_mc),
        flow_shape,
        num_mc,
    )


def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    reduce_mode: str = "scalar",
    clamp: float | None = None,
) -> torch.Tensor:
    """CFM squared-velocity loss with selectable reduction (paper Eq.7-8).

    OFF ``sqrt`` convention (``rl_cfg.cfm_loss_reduction`` default): per-draw
    losses are ``sum / sqrt(D)`` over flow dims, which preserves variance
    across action dims instead of shrinking log-ratios like a mean. ``bin``
    applies the same scaling per frequency bin (diagnostic path only).
    Symmetric max-clamp (OFF ``cfm_loss_clamp``) applies to ``sample``/``bin``.
    """
    if v_hat.shape != v_star.shape:
        raise ValueError(
            f"v_hat shape {tuple(v_hat.shape)} != v_star shape {tuple(v_star.shape)}"
        )
    squared_errors = (v_hat - v_star).pow(2)
    if reduce_mode == "scalar":
        if clamp is not None:
            raise ValueError("clamp applies to reduce_mode='sample'/'bin' only")
        return squared_errors.mean()
    elif reduce_mode == "sample":
        flat = squared_errors.flatten(2)
        reduced = flat.sum(dim=-1) / math.sqrt(flat.shape[-1])
    elif reduce_mode == "bin":
        reduced = squared_errors.sum(dim=-1) / math.sqrt(squared_errors.shape[-1])
    else:
        raise ValueError(
            f"reduce_mode={reduce_mode!r} must be 'scalar', 'sample', or 'bin'"
        )

    clamp = _DEFAULTS.cfm_clamp if clamp is None else clamp
    return torch.clamp(reduced, max=clamp)


def compute_aspo_surrogate(
    rho: torch.Tensor,
    advantage: torch.Tensor,
    epsilon_clip: float | None = None,
) -> torch.Tensor:
    """ASPO piecewise trust region ψ(ρ, Â) (paper Eq.12).

    Single clipping parameter for both branches: Â > 0 PPO-clip,
    Â ≤ 0 SPO quadratic. Elementwise; scalars and tensors both supported.
    """
    epsilon_clip = _DEFAULTS.epsilon_clip if epsilon_clip is None else epsilon_clip
    pos = torch.min(
        rho * advantage,
        torch.clamp(rho, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantage,
    )
    neg = rho * advantage - (advantage.abs() / (2.0 * epsilon_clip)) * (rho - 1.0) ** 2
    return torch.where(advantage > 0, pos, neg)


def compute_policy_ratio(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Per-sample importance ratios (paper Eq.10 + OFF clamps).

    ρ_i = exp(ℓ_old,i − ℓ_new,i) with OFF's stabilizers: symmetric max-clamp
    on both losses (``cfm_loss_clamp``) and an STE cap on the log-ratio
    (forward clamped, gradient flows; ``cfm_diff_clamp_max``). OFF's separate
    negative-advantage max coincides with ``cfm_clamp`` here (single knob),
    so the symmetric clamp already covers it.
    """
    cfm_clamp = _DEFAULTS.cfm_clamp if cfm_clamp is None else cfm_clamp
    l_old_c = torch.clamp(l_old, max=cfm_clamp)
    l_new_c = torch.clamp(l_new, max=cfm_clamp)
    cap = _DEFAULTS.log_ratio_cap if cap is None else cap
    delta = l_old_c - l_new_c
    delta_st = delta + (torch.clamp(delta, max=cap) - delta).detach()
    return torch.exp(delta_st)


def compute_policy_loss(
    l_old: torch.Tensor,
    l_new: torch.Tensor,
    advantage: torch.Tensor,
    epsilon_clip: float | None = None,
    cfm_clamp: float | None = None,
    cap: float | None = None,
) -> torch.Tensor:
    """Scalar FPO++ policy objective: MEAN over i of ψ per chunk (Eq.13).

    Per-sample ratios enter the mean directly — never an exp-of-mean estimator.
    """
    rho = compute_policy_ratio(l_old, l_new, cfm_clamp=cfm_clamp, cap=cap)
    return compute_aspo_surrogate(rho, advantage, epsilon_clip).mean()


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
    """Sample the learned flow from t=0 (noise) to t=1 (data).

    ``predict_velocity`` maps (x, t, condition) -> v̂, where condition is the
    pre-fused state vector; the loop only integrates (Euler or other modes)
    and never touches network weights.

    Euler: ``x ← x + δ·v̂`` on the ascending linspace(0, 1, steps+1) grid,
    δ = 1/steps at the left-endpoint t. ``eps=None`` draws N(0, I);
    explicit ``eps`` (e.g. zeros) bypasses randomness. Inference-only.
    """
    if mode != "euler":
        raise ValueError(f"mode={mode!r} unknown; expected 'euler'")
    grid = torch.linspace(0.0, 1.0, num_inference_steps + 1)
    if eps is None:
        eps = torch.randn(*shape, generator=generator)
        if device is not None:
            eps = eps.to(device)
    x = eps
    with torch.inference_mode():
        # ------- integrate velocity field from noise to data -------
        for i in range(num_inference_steps):
            v_hat = predict_velocity(x, grid[i], condition)
            x = x + (grid[i + 1] - grid[i]) * v_hat
    return x
