"""flow_policy_optimization.update: optimizer and minibatch update."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.objective import (
    compute_policy_ratio,
    compute_trust_region_surrogate,
    expand_mc_samples,
    to_critic_action,
    token_cfm_loss,
)
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy

def make_optimizer(policy: FlowPolicyOptimizationPolicy) -> torch.optim.Optimizer:
    cfg = policy.config
    decay, no_decay = [], []
    for module in (policy.actor, policy.encoder):
        for name, p in module.named_parameters():
            if p.ndim >= 2 and not (module is policy.actor and name == "out.weight"):
                decay.append(p)
            else:
                no_decay.append(p)
    fused = next(policy.parameters()).device.type == "cuda"
    return torch.optim.Adam(
        [
            {"params": decay, "lr": cfg.policy_lr, "weight_decay": cfg.optimizer_weight_decay},
            {"params": no_decay, "lr": cfg.policy_lr, "weight_decay": 0.0},
            {"params": list(policy.critic.parameters()), "lr": cfg.critic_lr, "weight_decay": 0.0},
        ],
        betas=tuple(cfg.adam_betas),
        eps=1e-8,
        fused=fused,
    )
def compute_loss(policy: FlowPolicyOptimizationPolicy, batch: dict[str, torch.Tensor], device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = policy.config
    states = policy.obs_normalizer(torch.as_tensor(batch["observation.state"], dtype=torch.float32, device=device))
    raw_actions = torch.as_tensor(batch["action"], dtype=torch.float32, device=device)
    flow_actions = policy._to_flow_space(raw_actions)
    action_is_pad = torch.as_tensor(batch["action_is_pad"], dtype=torch.bool, device=device)
    advantages = torch.as_tensor(batch["normalized_advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["raw_returns"], dtype=torch.float32, device=device)
    t = torch.as_tensor(batch["mc_t"], dtype=torch.float32, device=device)
    eps = torch.as_tensor(batch["mc_eps"], dtype=torch.float32, device=device)
    cfm_old_tok = torch.as_tensor(batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device)

    x_t, flat_t, flow_shape, num_mc = expand_mc_samples(flow_actions, eps, t)
    cond = policy.encoder(states)
    cond_flat = cond.unsqueeze(1).expand(states.shape[0], num_mc, cond.shape[1]).reshape(-1, cond.shape[1])
    v_new = policy.compute_predicted_velocity(x_t, flat_t, cond_flat).float()

    cfm_loss_new, cfm_loss_new_tok = token_cfm_loss(
        v_new, flow_actions, eps, action_is_pad, cfg, n=states.shape[0], num_mc=num_mc, flow_shape=flow_shape
    )
    ratio = compute_policy_ratio(cfm_old_tok, cfm_loss_new_tok, cfg)
    surrogate = compute_trust_region_surrogate(ratio, advantages[:, None, None], cfg)
    keep = (~action_is_pad).to(surrogate.dtype)
    policy_loss = -((surrogate * keep.unsqueeze(1)).sum(dim=(1, 2)) / (keep.sum(dim=-1).clamp_min(1.0) * surrogate.shape[1])).mean()

    critic_actions = to_critic_action(raw_actions, mask=keep.unsqueeze(-1))
    exec_frac = keep.sum(dim=-1) / float(cfg.action_horizon)
    value_pred = policy.critic(states, critic_actions, exec_frac).float()
    value_loss = F.smooth_l1_loss(value_pred, returns.reshape(-1), reduction="mean", beta=cfg.value_loss_beta)
    v_pred = policy.critic.forward_v(states).float()
    v_loss = F.smooth_l1_loss(v_pred, returns.reshape(-1), reduction="mean", beta=cfg.value_loss_beta)
    loss = policy_loss + cfg.value_loss_coef * (value_loss + v_loss)

    with torch.no_grad():
        r_var = returns.var(unbiased=False).clamp_min(1e-8)
        ratio_fp = ratio.detach()
        metrics = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "v_loss": v_loss.detach(),
            "total_loss": loss.detach(),
            "cfm_loss_new": cfm_loss_new.mean().detach(),
            "ratio_mean": ratio_fp.mean(),
            "ratio_std": ratio_fp.std(correction=0),
            "ratio_max": ratio_fp.max(),
            "clipfrac": (ratio_fp - 1.0).abs().gt(cfg.clip_param).to(ratio_fp.dtype).mean(),
            "cfm_above_clamp_frac": (cfm_loss_new_tok > cfg.cfm_clamp).to(ratio_fp.dtype).mean(),
            "cfm_both_clamped_frac": ((cfm_old_tok > cfg.cfm_clamp) & (cfm_loss_new_tok > cfg.cfm_clamp)).to(ratio_fp.dtype).mean(),
            "explained_var": 1.0 - (returns - value_pred).var(unbiased=False) / r_var,
        }
    return loss, metrics
def _param_groups_for_clip(policy: FlowPolicyOptimizationPolicy) -> dict[str, list[nn.Parameter]]:
    """Split trainable params into actor (actor + encoder) and critic."""
    actor_ids = {id(p) for p in (*policy.actor.parameters(), *policy.encoder.parameters())}
    actor_params = [p for p in policy.parameters() if p.requires_grad and id(p) in actor_ids]
    critic_params = [p for p in policy.parameters() if p.requires_grad and id(p) not in actor_ids]
    return {"actor": actor_params, "critic": critic_params}
def update(policy: FlowPolicyOptimizationPolicy, optimizer, batch: dict[str, Any], cfg: FlowPolicyOptimizationConfig) -> tuple[dict[str, float], float, float]:
    policy.train()
    device = next(policy.actor.parameters()).device
    n = int(batch["raw_returns"].shape[0])
    if n == 0:
        return {}, 0.0, 0.0

    clip_groups = _param_groups_for_clip(policy)
    actor_params, critic_params = clip_groups["actor"], clip_groups["critic"]

    # Stage training tensors on device once per rollout instead of once per minibatch.
    dev_batch = {
        "observation.state": torch.as_tensor(batch["observation.state"], dtype=torch.float32, device=device),
        "action": torch.as_tensor(batch["action"], dtype=torch.float32, device=device),
        "action_is_pad": torch.as_tensor(batch["action_is_pad"], dtype=torch.bool, device=device),
        "normalized_advantages": torch.as_tensor(batch["normalized_advantages"], dtype=torch.float32, device=device),
        "raw_returns": torch.as_tensor(batch["raw_returns"], dtype=torch.float32, device=device),
        "mc_t": torch.as_tensor(batch["mc_t"], dtype=torch.float32, device=device),
        "mc_eps": torch.as_tensor(batch["mc_eps"], dtype=torch.float32, device=device),
        "behavior_cfm_loss_token": torch.as_tensor(batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device),
    }

    totals: dict[str, torch.Tensor] = {}
    counts = 0
    grad_norm_actor = grad_norm_critic = 0.0
    for _ in range(cfg.num_learning_epochs):
        inds = torch.randperm(n, device=device)
        for start in range(0, n, cfg.chunks_per_minibatch):
            mb = inds[start : start + cfg.chunks_per_minibatch]
            loss, metrics = compute_loss(policy, {k: v[mb] for k, v in dev_batch.items()}, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_actor = float(torch.nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm))
            grad_norm_critic = float(torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm))
            optimizer.step()
            with torch.no_grad():
                for pt, p in zip(policy.critic_target.parameters(), policy.critic.parameters()):
                    pt.copy_(cfg.target_tau * p + (1.0 - cfg.target_tau) * pt)
            for k, v in metrics.items():
                totals[k] = totals[k] + v if k in totals else v
            counts += 1

    # Critic-target lag on the rollout states, one extra forward per update.
    with torch.no_grad():
        lag_states = policy.obs_normalizer(dev_batch["observation.state"])
        lag_keep = (~dev_batch["action_is_pad"]).to(dev_batch["action"].dtype)
        lag_frac = lag_keep.sum(dim=-1) / float(cfg.action_horizon)
        lag_actions = to_critic_action(dev_batch["action"], mask=lag_keep.unsqueeze(-1))
        lag = float((policy.critic(lag_states, lag_actions, lag_frac) - policy.critic_target(lag_states, lag_actions, lag_frac)).abs().float().mean().item())
    return ({**{k: float((v / counts).item()) for k, v in totals.items()}, "critic_target_lag": lag}, grad_norm_actor, grad_norm_critic)


# ---------------------------------------------------------------------------
# Evaluation & ManiSkill3 wrapper
