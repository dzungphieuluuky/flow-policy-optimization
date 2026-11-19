"""flow_policy_optimization.objective: CFM, ratio and chunk-GAE math."""

from __future__ import annotations

import math

import torch

from flow_policy_optimization.config import FlowPolicyOptimizationConfig

def expand_mc_samples(actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int]:
    """Interpolate ``x_t = lerp(eps, actions, t)`` over the MC axis -> ``[B*M, ...]``."""
    b, m = eps.shape[0], eps.shape[1]
    flow_shape = tuple(actions.shape[1:])

    flat_t = t.reshape(b * m, flow_shape[0], 1)
    flat_act = actions.unsqueeze(1).expand(b, m, *flow_shape).reshape(b * m, *flow_shape)
    flat_eps = eps.reshape(b * m, *flow_shape)
    x_t = torch.lerp(flat_eps.float(), flat_act.float(), flat_t.expand_as(flat_act).float())
    return x_t, flat_t.squeeze(-1), flow_shape, m
def compute_cfm_loss(
    v_hat: torch.Tensor,
    v_star: torch.Tensor,
    cfg: FlowPolicyOptimizationConfig,
    reduce_mode: str,
    action_is_pad: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    diff = v_hat - v_star
    if cfg.cfm_loss_use_huber:
        delta = cfg.cfm_loss_huber_delta
        abs_d = diff.abs()
        sq = torch.where(abs_d <= delta, diff.square(), 2.0 * delta * abs_d - delta * delta)
    else:
        sq = diff.square()

    n_valid = None
    if action_is_pad is not None:
        sq = sq * (~action_is_pad).to(sq.dtype)[:, None, :, None]
        n_valid = ((~action_is_pad).sum(dim=-1, keepdim=True).to(sq.dtype) * sq.shape[-1]).clamp(min=1.0)

    if reduce_mode == "scalar":
        return sq.mean() if n_valid is None else sq.sum() / (n_valid.sum() * sq.shape[1])

    flat = sq.flatten(2)
    if cfg.cfm_loss_reduction == "sum":
        sample = flat.sum(dim=-1)
    elif cfg.cfm_loss_reduction == "sqrt":
        sample = flat.sum(dim=-1) / (n_valid.sqrt() if n_valid is not None else math.sqrt(flat.shape[-1]))
    else:
        sample = flat.mean(dim=-1) if n_valid is None else flat.sum(dim=-1) / n_valid
    return sample, sq.mean(dim=-1)
def compute_policy_ratio(cfm_loss_old: torch.Tensor, cfm_loss_new: torch.Tensor, cfg: FlowPolicyOptimizationConfig) -> torch.Tensor:
    delta = cfm_loss_old.clamp(max=cfg.cfm_clamp) - cfm_loss_new.clamp(max=cfg.cfm_clamp)
    # Symmetric cap. The straight-through term keeps a bounded gradient past the
    # cap instead of a dead zone. An uncapped delta gives exp(10) weights.
    safe_delta = delta.clamp(cfg.cfm_diff_clamp_min, cfg.cfm_diff_clamp_max)
    return torch.exp(safe_delta.detach() + delta - delta.detach())
def compute_trust_region_surrogate(ratio: torch.Tensor, advantage: torch.Tensor, cfg: FlowPolicyOptimizationConfig) -> torch.Tensor:
    ratio_adv = ratio * advantage
    ppo = torch.min(ratio_adv, ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantage)
    if cfg.trust_region_mode == "ppo":
        return ppo
    # ASPO penalises a ratio spike quadratically and pulls the ratio back
    # toward 1 - eps instead of zeroing the gradient past the clip.
    spo = ratio_adv - (advantage.abs() / (2.0 * cfg.spo_clip_param)) * (ratio - 1.0).square()
    if cfg.trust_region_mode == "spo":
        return spo
    return torch.where(advantage > 0, ppo, spo)
def sample_action(predict_velocity, shape, condition, num_steps: int, eps: torch.Tensor, device, *, t_min: float, t_max: float) -> torch.Tensor:
    if device is None or not 0.0 <= t_min <= t_max <= 1.0 or tuple(eps.shape) != tuple(shape):
        raise ValueError("invalid sample_action arguments")
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
    x = eps.to(device)
    # The step size comes from the unclamped grid; the velocity is evaluated
    # at the clamped t. Off-grid sampling times still get consistent increments.
    with torch.inference_mode():
        for i in range(num_steps):
            t_cur = grid[i].clamp(t_min, t_max).reshape(1)
            x = torch.addcmul(x, predict_velocity(x, t_cur, condition), grid[i + 1] - grid[i])
    return x


# ---------------------------------------------------------------------------
# Policy
def token_cfm_loss(
    v: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor, mask: torch.Tensor, cfg: FlowPolicyOptimizationConfig, *, n: int, num_mc: int, flow_shape
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_cfm_loss(v.reshape(n, num_mc, *flow_shape).float(), (actions.unsqueeze(1) - eps).float(), cfg, "sample_and_token", mask)
def to_env_action(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + (x + 1.0) * 0.5 * (high - low)
def bound_for_storage(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1.0, 1.0)
def to_critic_action(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return x if mask is None else x * mask


@torch.no_grad()
def compute_chunk_advantages(
    step_rewards: torch.Tensor,
    v_current: torch.Tensor,
    v_next: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_lengths: torch.Tensor,
    *,
    chunk_terminated: torch.Tensor,
    chunk_size: int,
    gamma: float,
    chunk_truncated: torch.Tensor | None = None,
    chunk_prev: torch.Tensor | None = None,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V-anchored chunk GAE returning ``(advantages, lambda_returns)``."""
    device = v_current.device
    rewards = step_rewards.to(device=device, dtype=torch.float32).reshape(-1)
    v_cur = v_current.to(device=device, dtype=torch.float32).reshape(-1)
    v_nxt = v_next.to(device=device, dtype=torch.float32).reshape(-1)
    starts = chunk_starts.to(device=device, dtype=torch.long).reshape(-1)
    lengths = chunk_lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated = chunk_terminated.to(device=device, dtype=torch.bool).reshape(-1)
    n = starts.numel()
    if n == 0:
        empty = torch.empty(0, device=device, dtype=torch.float32)
        return empty, empty

    truncated = torch.zeros(n, device=device, dtype=torch.bool) if chunk_truncated is None else chunk_truncated.to(device=device, dtype=torch.bool).reshape(-1)
    if gae_lambda > 0.0 and (chunk_truncated is None or chunk_prev is None):
        raise ValueError("chunk_truncated and chunk_prev are required when gae_lambda > 0")

    offsets = torch.arange(chunk_size, device=device)
    gathered = rewards[(starts.unsqueeze(1) + offsets).clamp(max=rewards.numel() - 1)]
    valid = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    chunk_rew = (gathered * valid * (gamma ** offsets)).sum(dim=1)

    gamma_h = gamma ** lengths.float()
    done = terminated | truncated
    chunk_rew = chunk_rew + gamma_h * (truncated & ~terminated) * v_nxt
    delta = chunk_rew + gamma_h * (~done) * v_nxt - v_cur
    if gae_lambda == 0.0:
        return delta, delta + v_cur

    prev = chunk_prev.to(device=device, dtype=torch.long).reshape(-1)
    linked = prev >= 0
    successor = torch.full((n,), -1, dtype=torch.long, device=prev.device)
    successor[prev[linked]] = torch.arange(n, device=prev.device)[linked]
    adv = gae_trace(delta, gamma_h * (~done) * gae_lambda, successor.cpu())
    return adv, adv + v_cur
def gae_trace(delta: torch.Tensor, trace: torch.Tensor, successor: torch.Tensor) -> torch.Tensor:
    """Traced chunk advantages; the sequential recursion runs on the host."""
    d = delta.detach().cpu()
    t = trace.detach().cpu()
    adv = d.clone()
    for i in range(d.numel() - 1, -1, -1):
        j = int(successor[i])
        if j >= 0:
            adv[i] = d[i] + t[i] * adv[j]
    return adv.to(device=delta.device, dtype=delta.dtype)
