"""The metric and sampling functions that compute a training step.

Every function here is free. It takes tensors, a config, or both, and returns
tensors or a dict of tensors. None of them reads or writes instance state, so
none of them can be a method on the policy.

``SJFPOPolicy.forward`` assembles the loss. This module reports on it. Keeping
the two apart means the loss path and the reporting path can be read
separately, and it means a change to a log key cannot reach the gradients.

``build_optimizer`` stays in ``modeling_sjfpo.py``. It needs the policy class,
and importing that class here would make a cycle.
"""

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
    "mc_draws",
    "sample_eps",
    "sample_mc_timesteps",
]


def compute_value_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Squared error against the return. The function broadcasts ``target`` over
    the members.

    The function uses plain MSE, which is what both FPO++ variants use
    (``fpo.py:497``, and ``clip_vloss`` off in the manipulation config). It has
    no clipped variant. The release reports that value clipping adds instability
    with the tighter policy band of FPO (``rl_cfg.py:181-185``).
    """
    return torch.square(pred - target.expand_as(pred))


# Diagnostics dict pack order. The trainer reads by key.
# budget_truncated_chunk_frac is always NaN in training, because no rollout
# path supplies budget_truncated.
_GAE_DIAGNOSTIC_KEYS = (
    "linked_frac",
    "advantage_std",
    "return_rms",
    "advantage_std_over_return_std",
    "budget_truncated_chunk_frac",
)


def compute_gae_diagnostics(
    env_id: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    advantages: torch.Tensor,
    raw_returns: torch.Tensor,
    budget_truncated: torch.Tensor | None = None,
) -> dict[str, float]:
    """Trace-liveness and scale probes for the chunk backup.

    The probes use adjacent same-env chunk pairs, not split by episode end. A
    value near 0 means a dead trace. Read the probes only when gae_lambda > 0,
    which is the default.

    ``budget_truncated`` is NaN when unsupplied, which is normal. The value is
    NaN, not 0.0, because zero claims that nothing was cut.
    """
    n = env_id.numel()
    if n == 0:
        return dict.fromkeys(_GAE_DIAGNOSTIC_KEYS, float("nan"))
    order = torch.argsort(env_id.to(torch.long), stable=True)
    e = env_id.to(torch.long)[order]
    ends = terminated.to(torch.bool)[order][:-1] | truncated.to(torch.bool)[order][:-1]
    ret = raw_returns.float()
    # Population std. The unbiased form is NaN at n == 1.
    adv_std = advantages.float().std(correction=0)
    ret_std = ret.std(correction=0).clamp_min(1e-8)
    fields = [
        ((e[1:] == e[:-1]) & ~ends).sum().double() / max(n - 1, 1),
        adv_std,
        rms(ret).double(),
        # adv_std / ret_std. A value of 1.0 reads as an inert critic.
        # The value is not bounded by 1, because Var(A) > Var(R) is possible.
        # The key names the denominator. "fraction" alone said only that the
        # value was a quotient, and not of what.
        adv_std / ret_std,
        # Budget cut. Not chunk_truncated, not truncated_chunk_frac.
        budget_truncated.to(torch.bool).float().mean().double()
        if budget_truncated is not None
        else ret.double().new_tensor(float("nan")),
    ]
    return dict(zip(_GAE_DIAGNOSTIC_KEYS, torch.stack(fields).tolist()))


def sample_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """MC noise: ``[n, M, tokens, action_dim]``."""
    shape = (n, cfg.num_mc_samples, cfg.action_horizon, action_dim)
    return sample_noise(shape, cfg, device, None)

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
    # The code uses lerp. It gives the same value in one kernel, not four.
    return torch.lerp(noise, action, t)


def mc_draws(
    actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """The shared MC prologue of ``compute_behavior_cfm_losses`` and ``forward``.
    The function interpolates the draws, with one t per token."""
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
    """Trust-region diagnostics, with a per-token axis split.

    ``log_ratio_*`` reads raw delta, never the clamped one.
    ``log_ratio_clip_frac`` counts the upper side only. That is the side
    ``compute_policy_ratio`` clamps.
    ``clipfrac`` compares against ``clip_param``, the PPO width. Under ASPO
    the negative-advantage branch uses ``spo_clip_param``, so ``clipfrac`` is
    not the whole trust-region binding fraction.

    The ``_early_half`` and ``_late_half`` keys split the token axis at
    ``split = num_tokens // 2``. The split is an INDEX cut along the action-step
    axis, not a frequency cut. The old names ended in ``_pos_low`` and
    ``_pos_high``, which read as a position in the frequency spectrum. "pos"
    also names the write cursor in the collector, so the same three letters
    meant two things.

    ``mean_log_ratio``, ``log_ratio_max``, ``log_ratio_clip_frac``,
    ``clipfrac``, and ``flow_ratio_*`` average over every element of ``delta``
    and ``ratio``. Their axis follows the ratio granularity: the token axis
    under the default flags. ``flow_ratio_token_*`` is always per token.

    The token axis is the action steps of the chunk. ``_early_half`` covers the
    first half of them and ``_late_half`` the second half.
    """
    log_ratio_abs = delta.abs()
    log_ratio_clip_frac = (delta > cfg.log_ratio_cap).float().mean()

    ratio_token_mean_per_step = ratio_token.mean(dim=(0, 1))
    num_tokens = ratio_token_mean_per_step.numel()
    split = max(1, num_tokens // 2)
    # One epsilon across the token axis, as the surrogate.
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
        # The key counts the chunks with an unexecuted row. The mask drops the
        # row.
        "truncated_chunk_frac": action_is_pad.any(dim=1).float().mean(),
        # Executed rows. That is the CFM loss that survives the mask.
        "executed_action_step_frac": (~action_is_pad).float().mean(),
    }


def compute_value_metrics(
    *,
    raw_value_pred: torch.Tensor,
    raw_returns_flat: torch.Tensor,
    policy_per_chunk: torch.Tensor,
    value_loss_per_chunk: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Critic diagnostics.

    ``explained_var`` scores the member mean, which the loss trains.
    The backup uses ``amin``. ``explained_var_min`` is that same reading for
    the pessimistic member. The two are not ordered: a pointwise-smaller
    predictor need not lower the residual variance.
    """
    mean_value_pred = raw_value_pred.mean(dim=0)
    min_value_pred = raw_value_pred.amin(dim=0)
    returns_var = raw_returns_flat.var(correction=0).clamp_min(1e-8)
    return {
        "policy_loss": policy_per_chunk.mean(),
        "value_loss": value_loss_per_chunk.mean(),
        "explained_var": 1.0
        - (raw_returns_flat - mean_value_pred).var(correction=0) / returns_var,
        # The pessimism the backup actually uses.
        "explained_var_min": 1.0
        - (raw_returns_flat - min_value_pred).var(correction=0) / returns_var,
        "value_abs_err_max": mean_value_pred.sub(raw_returns_flat).abs().max(),
        "value_pred_max": mean_value_pred.abs().max(),
        "value_target_std": raw_returns_flat.std(correction=0),
        # Pooled over [num_critics, n]. That is the within- and between-member
        # spread.
        "value_pred_std": raw_value_pred.std(correction=0),
        "value_pred_std_member_mean": raw_value_pred.std(dim=1, correction=0).mean(),
        "value_pred_member_gap": (
            raw_value_pred.amax(dim=0) - raw_value_pred.amin(dim=0)
        ).mean(),
    }


def compute_cfm_metrics(
    *,
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    v_star_f: torch.Tensor,
    above_clamp_frac: torch.Tensor,
    both_clamped_frac: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Flow-matching loss diagnostics.

    ``cfm_above_clamp_frac``. A clamped sample gives no gradient, so a stuck run
    trains the critic only.

    ``cfm_both_clamped_frac``. The share whose two losses both sit above the
    clamp. The ratio there is exactly 1.0, so the policy receives no gradient at
    all. This key is the direct cause. The key ``cfm_above_clamp_frac`` can read
    high while the ratio still carries a gradient.

    Both arguments come from the caller. The caller knows which pair of losses
    the ratio clamps. Neither key can be recomputed here, because ``cfm_loss_new``
    is the per-sample loss while the ratio clamps the per-token one. Reading it
    gave 1.0 while 73% of the clamped elements sat above the bound at
    ``cfm_clamp=0.5``.
    """
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
