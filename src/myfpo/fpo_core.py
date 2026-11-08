"""Owns: pure math / flow-matching helpers used by policy + rollout. Key: compute_policy_ratio, cfm_target, mc_samples.
FPO++ core maths. The module holds the CFM loss, the ratio, the trust-region surrogates (ppo, spo, aspo), the timestep and noise laws, and the flow sampler.
"""

from __future__ import annotations
import math
from collections.abc import Callable
from typing import Any
import torch
from myfpo.configuration import MyfpoConfig
from myfpo.layers import resolve_storage_dtype
import numpy as np
from myfpo.layers import rms

__all__ = ["compute_cfm_metrics", "compute_eval_summary_metrics", "compute_gae_diagnostics", "compute_policy_info", "compute_ratio_metrics", "compute_value_loss", "compute_value_metrics", "sample_eps", "sample_mc_timesteps", "subsampled_min"]

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
    """Bound the actor's raw velocity output (FPO++ ``velocity_clip_value``). ``None`` does nothing."""
    if bound is None:
        return velocity
    return velocity.clamp(-bound, bound)


def expand_mc_samples(
    actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """The shared MC prologue of the behavior loss and the training loss.

    Returns the flow point ``x_t``, the per-token time, the flow shape, and
    the number of MC samples. The flow point is the noise moved ``t`` of the
    way to the action.
    """
    flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
        broadcast_over_mc_axis(actions, eps, t)
    )
    time = flatten_timestep.expand(-1, -1, flow_shape[-1])
    noise = flatten_mc_noise.reshape(-1, *flow_shape)
    action = flatten_mc_action.reshape(-1, *flow_shape)
    # torch.lerp wants one dtype for all three operands: the weight has to
    # match the flow tensors, so a mixed fp32/bf16 triple raises instead of
    # interpolating. The common type is the promotion of the three.
    dtype = torch.promote_types(
        torch.promote_types(noise.dtype, action.dtype), time.dtype
    )
    x_t = torch.lerp(noise.to(dtype), action.to(dtype), time.to(dtype))
    return x_t, flatten_timestep.squeeze(-1), flow_shape, num_mc


def broadcast_condition(condition: torch.Tensor, num_mc: int) -> torch.Tensor:
    """Repeat each row ``m`` times, row-major: ``flat[i * m + j] = condition[i]``."""
    n, d = condition.shape
    return condition.unsqueeze(1).expand(n, num_mc, d).reshape(n * num_mc, d)


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
    if mode == "eps":
        return x_t - t * v
    raise ValueError(f"cfm_loss_target={mode!r} unknown; expected 'velocity', 'action', or 'eps'")


def cfm_target(
    a: torch.Tensor, eps: torch.Tensor, cfg: MyfpoConfig
) -> torch.Tensor:
    """The regression target for ``cfg.cfm_loss_target``. ``[B, *flow]``."""
    mode = cfg.cfm_loss_target
    if mode == "velocity":
        return a.unsqueeze(1) - eps
    if mode == "action":
        return a.unsqueeze(1).expand_as(eps)
    if mode == "eps":
        return eps
    raise ValueError(f"cfm_loss_target={mode!r} unknown; expected 'velocity', 'action', or 'eps'")


def compute_trust_region_surrogate(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    cfg: MyfpoConfig,
) -> torch.Tensor:
    """Trust-region surrogate, selected by ``cfg.trust_region_mode``.

    - ``"ppo"``: PPO clipped surrogate on both signs.
    - ``"spo"``: SPO quadratic surrogate on both signs.
    - ``"aspo"``: PPO for ``A > 0``, SPO for ``A < 0`` (Wang et al. 2025 / FPO++).

    ``cfg.spo_clip_param`` sets the quadratic penalty width; larger values are
    a looser trust region for the negative-advantage branch.
    """
    ratio_adv = ratio * advantage
    ratio_clipped = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
    ppo = torch.min(ratio_adv, ratio_clipped * advantage)
    mode = cfg.trust_region_mode
    if mode == "ppo":
        return ppo
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.spo_clip_param)) * torch.square(
        ratio - 1.0
    )
    if mode == "spo":
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


# NOTE (publishability): every mode is a proxy for log pi_new - log pi_old.
# The exact ratio needs the flow's divergence. See FPO (arXiv:2507.21053) and
# FPO++ (arXiv:2602.02481).
# POLICY RATIO

def compute_cfm_log_ratio(cfm_loss_old, cfm_loss_new, cfg, advantage):
    """Return the log-ratio delta between the losses, in the mode ``ratio_mode`` selects."""
    mode = cfg.ratio_mode
    if mode in ("cfm", "cfm_raw"):
        if mode == "cfm":
            old = torch.clamp(cfm_loss_old, max=cfg.cfm_clamp)
            new = torch.clamp(cfm_loss_new, max=cfg.cfm_clamp)
            if cfg.cfm_loss_clamp_negative_advantages and advantage is not None:
                new = torch.where(advantage < 0, torch.clamp(new, max=cfg.cfm_loss_clamp_negative_advantages_max), new)
        else:
            old, new = cfm_loss_old, cfm_loss_new
        return old - new
    raise ValueError(f"ratio_mode={mode!r}")


def compute_policy_ratio(
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    cfg: MyfpoConfig,
    advantage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``exp(log_delta)`` with a straight-through cap on ``log_delta``.

    ``cfg.ratio_mode`` picks the log-delta:

    * ``"cfm"`` (default): ``clamp(L_old) - clamp(L_new)``, the FPO++ form.
      ``cfm_clamp`` and ``cfm_loss_clamp_negative_advantages`` apply.
    * ``"cfm_raw"``: ``L_old - L_new`` with no clamp. Ablation only.

    The MC axis is grouped by :func:`group_cfm_loss` before this call;
    the function operates on already-grouped tensors.
    """
    delta = compute_cfm_log_ratio(cfm_loss_old, cfm_loss_new, cfg, advantage)
    safe_delta = delta.clamp(min=-cfg.log_ratio_cap, max=cfg.log_ratio_cap)  # cap=2.0; ST lets raw delta backprop
    delta_st = safe_delta.detach() + delta - delta.detach()
    return torch.exp(delta_st)


def sample_action(
    predict_velocity: Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor],
    shape,
    condition,
    num_steps: int,
    eps: torch.Tensor,
    device: torch.device | str | None,
    *,
    t_min: float,
    t_max: float,
):
    """Euler-sample flow from t=0 (noise) to t=1 (data). ``device`` is required explicitly (no default).

    The velocity is read at each grid time clamped into ``[t_min, t_max]``, the
    band the flow was trained on. The step sizes keep the grid, so the path still
    ends at t=1.
    """
    if device is None:
        raise ValueError("device must be provided explicitly")
    if not 0.0 <= t_min <= t_max <= 1.0:
        raise ValueError(
            "t_min and t_max must satisfy 0 <= t_min <= t_max <= 1, "
            f"got ({t_min}, {t_max})"
        )
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
            t_current = grid[i].clamp(t_min, t_max).reshape(1)
            v_hat = predict_velocity(x, t_current, condition)
            x = torch.addcmul(x, v_hat, dts[i])
    return x


# ---- from metrics.py ----
def compute_value_loss(
    pred: torch.Tensor, target: torch.Tensor, cfg, beta: float | None = None
) -> torch.Tensor:
    """Huber loss against the return. ``target`` broadcasts over the members.

    beta defaults to ``cfg.value_loss_beta`` (1.0 restores the pre-release
    squared-error knee where |error| > beta is linear instead of quadratic).
    """
    if beta is None:
        beta = cfg.value_loss_beta
    return torch.nn.functional.smooth_l1_loss(
        pred, target.expand_as(pred), reduction="none", beta=beta
    )


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


GAE_DIAGNOSTIC_KEYS = (
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
        return dict.fromkeys(GAE_DIAGNOSTIC_KEYS, float("nan"))
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
    return dict(zip(GAE_DIAGNOSTIC_KEYS, torch.stack(fields).tolist()))


def sample_eps(
    cfg: MyfpoConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Antithetic MC noise: pair eps with -eps. ``[n, 2*half, tokens, action_dim]``."""
    half = max(1, cfg.num_mc_samples // 2)
    shape_half = (n, half, cfg.action_horizon, action_dim)
    eps_half = sample_noise(shape_half, cfg, device, None)
    return torch.cat([eps_half, -eps_half], dim=1)  # [n, 2*half, ...]


def sample_mc_timesteps(cfg: MyfpoConfig, n: int, device: torch.device) -> torch.Tensor:
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
    chunk_truncated: torch.Tensor,
    cfg: MyfpoConfig,
) -> dict[str, torch.Tensor]:
    """Trust-region diagnostics, with a per-token axis split."""
    if chunk_truncated.shape != action_is_pad.shape[:1]:
        raise ValueError(
            f"chunk_truncated shape {tuple(chunk_truncated.shape)} != "
            f"one flag per chunk {tuple(action_is_pad.shape[:1])}"
        )
    log_ratio_abs = delta.abs()
    log_ratio_clip_frac = (log_ratio_abs > cfg.log_ratio_cap).float().mean()

    ratio_token_mean_per_step = ratio_token.mean(dim=(0, 1))
    num_tokens = ratio_token_mean_per_step.numel()
    split = max(1, num_tokens // 2)
    def outside(t: torch.Tensor) -> torch.Tensor:
        """Return the share of ratios outside the ``clip_param`` band."""
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
        "truncated_chunk_frac": chunk_truncated.float().mean(),
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
    # The correlation over the batch axis compares the members' value
    # functions, not their values on one state. A value near one means the
    # members agree on every state and the ensemble adds nothing. A single
    # member has no pair, so the value is NaN. A member that is constant over
    # the batch has no correlation to measure, so the value is NaN too. NaN
    # is how the repo says "unmeasured".
    if raw_value_pred.shape[0] < 2:
        # Same device as the metric tensors, or the stack in
        # compute_policy_info mixes devices on CUDA.
        member_corr = torch.tensor(
            float("nan"), dtype=torch.float32, device=raw_value_pred.device
        )
    else:
        centred = raw_value_pred - raw_value_pred.mean(dim=1, keepdim=True)
        norms = centred.norm(dim=1, keepdim=True)
        if bool((norms < 1e-12).any()):
            member_corr = torch.tensor(
                float("nan"), dtype=torch.float32, device=raw_value_pred.device
            )
        else:
            unit = centred / norms
            corr = unit @ unit.t()
            n_members = raw_value_pred.shape[0]
            off_diagonal = corr[
                ~torch.eye(n_members, dtype=torch.bool, device=corr.device)
            ]
            member_corr = off_diagonal.mean()
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
        "value_pred_member_corr": member_corr,
    }


def compute_eval_summary_metrics(
    mean_return: float,
    eval_results: list[dict],
    all_eval_returns: list[float],
    *,
    r_rand: float | None = None,
    best_return: float | None = None,
    steps_to_best: int | None = None,
) -> dict:
    """Return the eval scalars for one eval window."""
    returns = [r["return"] for r in eval_results]
    lengths = [r["length"] for r in eval_results]
    successes = [r["success"] for r in eval_results if r.get("success") is not None]
    jerks = [r["jerk"] for r in eval_results if r.get("jerk") is not None]

    def stat(values, fn):
        """Apply ``fn`` to the values. An empty list becomes NaN."""
        return float(fn(values)) if values else float("nan")

    data = {
        "eval/return_mean": float(mean_return),
        "eval/return_min": stat(returns, np.min),
        "eval/return_max": stat(returns, np.max),
        "eval/return_std": stat(returns, np.std),
    }
    if best_return is not None:
        data["eval/return_best"] = float(best_return)
    if steps_to_best is not None:
        data["eval/steps_to_best"] = int(steps_to_best)
    if r_rand is not None:
        data["eval/random_policy_return"] = float(r_rand)
    if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
        prev = all_eval_returns[-2]
        data["eval/return_improvement"] = float(mean_return - prev)
        data["eval/return_improvement_frac"] = float(
            (mean_return - prev) / max(abs(prev), 1e-8)
        )
    data.update(
        {
            "eval/mean_length": stat(lengths, np.mean),
            "eval/max_length": stat(lengths, np.max),
            "eval/std_length": stat(lengths, np.std),
            "eval/episode_count": len(returns),
        }
    )
    if successes:
        data["eval/success_rate"] = float(np.mean(successes))
    if jerks:
        data["eval/mean_jerk"] = float(np.mean(jerks))
        data["eval/max_jerk"] = float(np.max(jerks))
    return data


def compute_cfm_metrics(
    *,
    cfm_loss_old: torch.Tensor,
    cfm_loss_new: torch.Tensor,
    target_velocity: torch.Tensor,
    above_clamp_frac: torch.Tensor,
    both_clamped_frac: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Flow-matching loss diagnostics."""
    cfm_loss_old_std = cfm_loss_old.std(correction=0)
    cfm_loss_new_std = cfm_loss_new.std(correction=0)
    corr_denom = cfm_loss_old_std * cfm_loss_new_std + 1e-8
    return {
        "flow_v_star_rms": rms(target_velocity),
        "l_old_mean": cfm_loss_old.mean(),
        "l_new_mean": cfm_loss_new.mean(),
        "cfm_loss_old_std": cfm_loss_old_std,
        "cfm_loss_new_std": cfm_loss_new_std,
        "cfm_above_clamp_frac": above_clamp_frac,
        "cfm_both_clamped_frac": both_clamped_frac,
        "l_old_new_corr": ((cfm_loss_old - cfm_loss_old.mean()) * (cfm_loss_new - cfm_loss_new.mean())).mean()
        / corr_denom,
    }


@torch.inference_mode()
def compute_policy_info(
    *,
    cfm_loss_old,
    cfm_loss_new,
    ratio,
    ratio_token,
    delta,
    action_is_pad,
    chunk_truncated,
    target_velocity,
    raw_value_pred,
    raw_returns_flat,
    policy_per_chunk,
    value_loss_per_chunk,
    cond_stats,
    above_clamp_frac,
    both_clamped_frac,
    cfg: MyfpoConfig,
) -> dict[str, float]:
    """Merge the metric groups into one device transfer."""
    scalars = {
        **compute_ratio_metrics(
            ratio=ratio,
            ratio_token=ratio_token,
            delta=delta,
            action_is_pad=action_is_pad,
            chunk_truncated=chunk_truncated,
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
            target_velocity=target_velocity,
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
    # A python float builds on the default device. The metric row is one
    # stack on the batch device, so every scalar must sit there first.
    device = packed[0].device
    if any(t.device != device for t in packed):
        packed = [t.to(device) for t in packed]
    return dict(zip(scalars.keys(), torch.stack(packed).float().cpu().tolist()))
