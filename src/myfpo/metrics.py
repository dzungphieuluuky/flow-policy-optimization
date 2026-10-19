"""The metric and sampling functions that compute a training step."""

from __future__ import annotations

import einx
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import (
    broadcast_over_mc_axis,
    sample_noise,
    sample_timesteps,
)
from myfpo.utils import rms

__all__ = [
    "broadcast_condition",
    "compute_gae_diagnostics",
    "compute_mc_xt",
    "compute_ratio_metrics",
    "compute_value_loss",
    "compute_value_metrics",
    "compute_cfm_metrics",
    "compute_policy_info",
    "mc_draws",
    "sample_eps",
    "sample_mc_timesteps",
    "subsampled_min",
]


def compute_value_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Squared error against the return. ``target`` broadcasts over the members."""
    return torch.square(pred - target.expand_as(pred))


def subsampled_min(q: torch.Tensor, size: int = 2) -> torch.Tensor:
    """Min over ``size`` randomly drawn members of ``[num_critics, N]``, per column.

    REDQ (Chen et al. 2021) and TD-MPC2 (arXiv:2310.16828). A full min over every
    member is more pessimistic than either.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be [num_critics, N], got {tuple(q.shape)}")
    members = q.shape[0]
    if members <= size:
        return q.amin(dim=0)
    idx = torch.rand(members, q.shape[1], device=q.device).argsort(dim=0)[:size]
    return q.gather(0, idx).amin(dim=0)


_GAE_DIAGNOSTIC_KEYS = (
    "linked_frac",
    "advantage_std",
    "return_rms",
    "advantage_std_over_return_std",
)


def compute_gae_diagnostics(
    env_id: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    advantages: torch.Tensor,
    raw_returns: torch.Tensor,
) -> dict[str, float]:
    """Trace-liveness and scale probes for the chunk backup."""
    n = env_id.numel()
    if n == 0:
        return dict.fromkeys(_GAE_DIAGNOSTIC_KEYS, float("nan"))
    order = torch.argsort(env_id.to(torch.long), stable=True)
    e = env_id.to(torch.long)[order]
    ends = terminated.to(torch.bool)[order][:-1] | truncated.to(torch.bool)[order][:-1]
    ret = raw_returns.float()
    adv_std = advantages.float().std(correction=0)
    ret_std = ret.std(correction=0).clamp_min(1e-8)
    fields = [
        ((e[1:] == e[:-1]) & ~ends).sum().double() / max(n - 1, 1),
        adv_std,
        rms(ret).double(),
        adv_std / ret_std,
    ]
    return dict(zip(_GAE_DIAGNOSTIC_KEYS, torch.stack(fields).tolist()))


def sample_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Antithetic MC noise: pair eps with -eps. ``[n, 2*half, tokens, action_dim]``."""
    half = max(1, cfg.num_mc_samples // 2)
    shape_half = (n, half, cfg.action_horizon, action_dim)
    eps_half = sample_noise(shape_half, cfg, device, None)
    return torch.cat([eps_half, -eps_half], dim=1)  # [n, 2*half, ...]

def compute_mc_xt(
    flatten_mc_timestep: torch.Tensor,
    flatten_mc_noise: torch.Tensor,
    flatten_mc_action: torch.Tensor,
    flow_shape: tuple[int, ...],
) -> torch.Tensor:
    """Interpolate noise to action. Returns ``[B * M, *flow_shape]``."""
    t = flatten_mc_timestep.expand(-1, -1, flow_shape[-1])
    noise = flatten_mc_noise.reshape(-1, *flow_shape)
    action = flatten_mc_action.reshape(-1, *flow_shape)
    return torch.lerp(noise, action, t)


def mc_draws(
    actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """The shared MC prologue of ``compute_behavior_cfm_losses`` and ``forward``."""
    flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
        broadcast_over_mc_axis(actions, eps, t)
    )
    x_t = compute_mc_xt(flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape)
    return x_t, flatten_timestep.squeeze(-1), flow_shape, num_mc


def broadcast_condition(condition: torch.Tensor, num_mc: int) -> torch.Tensor:
    """Repeat each row ``m`` times, row-major: ``flat[i * m + j] = condition[i]``."""
    return einx.id("n m d -> (n m) d", einx.id("n d -> n m d", condition, m=num_mc))


def sample_mc_timesteps(cfg: SJFPOConfig, n: int, device: torch.device) -> torch.Tensor:
    """MC timesteps, one trailing axis: ``[n, M, horizon, 1]``."""
    return sample_timesteps(
        (n, cfg.num_mc_samples, cfg.action_horizon), cfg, device, None
    ).unsqueeze(-1)


def compute_ratio_metrics(
    *,
    ratio: torch.Tensor,
    ratio_token: torch.Tensor,
    delta: torch.Tensor,
    action_is_pad: torch.Tensor,
    cfg: SJFPOConfig,
) -> dict[str, torch.Tensor]:
    """Trust-region diagnostics, with a per-token axis split."""
    log_ratio_abs = delta.abs()
    log_ratio_clip_frac = (delta > cfg.log_ratio_cap).float().mean()

    ratio_token_mean_per_step = ratio_token.mean(dim=(0, 1))
    num_tokens = ratio_token_mean_per_step.numel()
    split = max(1, num_tokens // 2)
    def outside(t: torch.Tensor) -> torch.Tensor:
        return ((t < 1.0 - cfg.clip_param) | (t > 1.0 + cfg.clip_param)).float().mean()

    early_outside = outside(ratio_token[..., :split])
    late_outside = (
        outside(ratio_token[..., split:]) if num_tokens > 1 else early_outside
    )

    early_mean = ratio_token_mean_per_step[:split].mean()
    late_mean = (
        ratio_token_mean_per_step[split:].mean()
        if num_tokens > 1
        else ratio_token_mean_per_step.mean()
    )

    return {
        "mean_log_ratio": delta.mean(),
        "log_ratio_max": log_ratio_abs.max(),
        "log_ratio_clip_frac": log_ratio_clip_frac,
        "clipfrac": outside(ratio),
        "flow_ratio_mean": ratio.mean(),
        "flow_ratio_std": ratio.std(correction=0),
        "flow_ratio_token_range": (
            ratio_token_mean_per_step.max() - ratio_token_mean_per_step.min()
        ),
        "flow_ratio_token_early_half": early_mean,
        "flow_ratio_token_late_half": late_mean,
        "flow_ratio_token_early_half_outside_frac": early_outside,
        "flow_ratio_token_late_half_outside_frac": late_outside,
        "flow_ratio_token_outside_frac": outside(ratio_token),
        "truncated_chunk_frac": action_is_pad.any(dim=1).float().mean(),
        "executed_action_step_frac": (~action_is_pad).float().mean(),
    }


def compute_value_metrics(
    *,
    raw_value_pred: torch.Tensor,
    raw_returns_flat: torch.Tensor,
    policy_per_chunk: torch.Tensor,
    value_loss_per_chunk: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Critic diagnostics."""
    mean_value_pred = raw_value_pred.mean(dim=0)
    min_value_pred = raw_value_pred.amin(dim=0)
    returns_var = raw_returns_flat.var(correction=0).clamp_min(1e-8)
    # Every member regresses onto the same scalar target, so a gap trending to
    # zero means the ensemble has collapsed onto one function. Logged, not
    # gated: there is no baseline to pick a threshold against.
    member_gap = raw_value_pred.amax(dim=0) - raw_value_pred.amin(dim=0)
    return {
        "policy_loss": policy_per_chunk.mean(),
        "value_loss": value_loss_per_chunk.mean(),
        "explained_var": 1.0
        - (raw_returns_flat - mean_value_pred).var(correction=0) / returns_var,
        "explained_var_min": 1.0
        - (raw_returns_flat - min_value_pred).var(correction=0) / returns_var,
        "value_abs_err_max": mean_value_pred.sub(raw_returns_flat).abs().max(),
        "value_pred_max": mean_value_pred.abs().max(),
        "value_target_std": raw_returns_flat.std(correction=0),
        "value_pred_std": raw_value_pred.std(correction=0),
        "value_pred_std_member_mean": raw_value_pred.std(dim=1, correction=0).mean(),
        "value_pred_member_gap": member_gap.mean(),
        "value_pred_member_gap_p95": torch.quantile(member_gap.float(), 0.95),
    }


def compute_cfm_metrics(
    *,
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    v_star_f: torch.Tensor,
    above_clamp_frac: torch.Tensor,
    both_clamped_frac: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Flow-matching loss diagnostics."""
    cfm_loss_old_std = cfm_loss_old.std(correction=0)
    cfm_loss_new_std = cfm_loss_new.std(correction=0)
    corr_denom = cfm_loss_old_std * cfm_loss_new_std + 1e-8
    out = {
        "flow_v_star_rms": rms(v_star_f),
        "l_old_mean": cfm_loss_old.mean(),
        "l_new_mean": cfm_loss_new.mean(),
        "cfm_loss_old_std": cfm_loss_old_std,
        "cfm_loss_new_std": cfm_loss_new_std,
        "cfm_above_clamp_frac": above_clamp_frac,
        "cfm_both_clamped_frac": both_clamped_frac,
        "l_old_new_corr": ((cfm_loss_old - cfm_loss_old.mean()) * (cfm_loss_new - cfm_loss_new.mean())).mean()
        / corr_denom,
    }
    return out


@torch.inference_mode()
def compute_policy_info(
    *,
    cfm_loss_old,
    cfm_loss_new,
    ratio,
    ratio_token,
    delta,
    action_is_pad,
    v_star_f,
    raw_value_pred,
    raw_returns_flat,
    policy_per_chunk,
    value_loss_per_chunk,
    cond_stats,
    above_clamp_frac,
    both_clamped_frac,
    cfg: SJFPOConfig,
) -> dict[str, float]:
    """Merge the metric groups into one device transfer."""
    scalars = {
        **compute_ratio_metrics(
            ratio=ratio,
            ratio_token=ratio_token,
            delta=delta,
            action_is_pad=action_is_pad,
            cfg=cfg,
        ),
        **compute_value_metrics(
            raw_value_pred=raw_value_pred,
            raw_returns_flat=raw_returns_flat,
            policy_per_chunk=policy_per_chunk,
            value_loss_per_chunk=value_loss_per_chunk,
        ),
        **compute_cfm_metrics(
            cfm_loss_old=cfm_loss_old,
            cfm_loss_new=cfm_loss_new,
            v_star_f=v_star_f,
            above_clamp_frac=above_clamp_frac,
            both_clamped_frac=both_clamped_frac,
        ),
    }
    if cond_stats:
        scalars.update(
            {
                f"cond_{k}": torch.as_tensor(cond_stats[k], device=ratio.device)
                for k in cond_stats
            }
        )
    packed = [
        s.detach().reshape(())
        if isinstance(s, torch.Tensor)
        else torch.as_tensor(float(s))
        for s in scalars.values()
    ]
    return dict(zip(scalars.keys(), torch.stack(packed).float().cpu().tolist()))
