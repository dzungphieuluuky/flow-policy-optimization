"""The target critic moves every optimizer step; the online critic never copies back.

One Polyak step per rollout left the target nearly frozen: a rollout is many
gradient steps, and the bootstrap mixed two value functions an order of
magnitude apart. The target must track the online critic per gradient step.
"""

from __future__ import annotations

import numpy as np
import torch

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, FlowPolicyOptimizationPolicy, make_optimizer, update

N = 2


def _cfg(**kw) -> FlowPolicyOptimizationConfig:
    base = dict(
        hidden_dim=32,
        num_obs_steps=2,
        action_horizon=2,
        num_action_steps=2,
        num_mc_samples=2,
        num_learning_epochs=1,
        chunks_per_update=2,
        chunks_per_minibatch=2,
    )
    base.update(kw)
    return FlowPolicyOptimizationConfig(**base)


def _policy(cfg) -> FlowPolicyOptimizationPolicy:
    torch.manual_seed(0)
    low = np.full(3, -1.0, dtype=np.float32)
    high = np.full(3, 1.0, dtype=np.float32)
    return FlowPolicyOptimizationPolicy(cfg, 5, 3, low, high)


def _batch(cfg) -> dict:
    return {
        "observation.state": torch.zeros(N, cfg.num_obs_steps, 5),
        "action": torch.tanh(torch.randn(N, cfg.action_horizon, 3)),
        "action_is_pad": torch.zeros(N, cfg.action_horizon, dtype=torch.bool),
        "chunk_truncated": torch.zeros(N, dtype=torch.bool),
        "normalized_advantages": torch.tensor([1.0, -1.0]),
        "raw_returns": torch.ones(N),
        "mc_t": torch.ones(N, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(N, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(N, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(N, cfg.num_mc_samples, cfg.action_horizon),
    }


def test_each_optimizer_step_moves_the_target_toward_the_online():
    cfg = _cfg()
    policy = _policy(cfg)
    optimizer = make_optimizer(policy)
    before = [p.detach().clone() for p in policy.critic_target.parameters()]
    update(policy, optimizer, _batch(cfg), cfg)
    moved = False
    for old, new in zip(before, policy.critic_target.parameters()):
        if not torch.equal(old, new):
            moved = True
            break
    assert moved, "the target critic did not move on an optimizer step"


def test_the_target_tracks_the_online_critic_to_within_five_percent():
    cfg = _cfg(target_tau=0.2)
    policy = _policy(cfg)
    optimizer = make_optimizer(policy)
    batch = _batch(cfg)
    for _ in range(200):
        update(policy, optimizer, batch, cfg)

    import flow_policy_optimization.flow_policy_optimization as m

    with torch.no_grad():
        states = policy.obs_normalizer(batch["observation.state"])
        keep = (~batch["action_is_pad"]).to(torch.float32)
        exec_frac = keep.sum(dim=-1) / float(cfg.action_horizon)
        critic_actions = m.to_critic_action(batch["action"], mask=keep.unsqueeze(-1))
        q_online = policy.critic(states, critic_actions, exec_frac)
        q_target = policy.critic_target(states, critic_actions, exec_frac)
    diff = (q_online - q_target).abs().mean()
    scale = q_online.abs().mean().clamp_min(1e-8)
    assert float(diff / scale) < 0.05


def test_the_update_reports_the_critic_target_lag():
    cfg = _cfg()
    policy = _policy(cfg)
    optimizer = make_optimizer(policy)
    result = update(policy, optimizer, _batch(cfg), cfg)
    assert "critic_target_lag" in result.metrics
    assert np.isfinite(result.metrics["critic_target_lag"])
    assert result.metrics["critic_target_lag"] >= 0.0
