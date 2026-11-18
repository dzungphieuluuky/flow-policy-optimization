"""The actor predicts a raw velocity, not a tanh-squashed one.

``predict_action_chunk`` documents the chunk as unbounded raw flow output, and
the squash is applied once on the way to the env. The actor's forward pass used
to return ``torch.tanh(out)``, which silently capped every velocity at 1 and
made the CFM target ``a - eps`` unreachable for ``|a - eps| > 1``.
"""

from __future__ import annotations

import numpy as np
import torch

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, FlowPolicyOptimizationPolicy

OBS, ACT, HORIZON = 3, 2, 4


def _policy(**kw):
    cfg = FlowPolicyOptimizationConfig(
        hidden_dim=32,
        num_obs_steps=4,
        action_horizon=HORIZON,
        num_action_steps=HORIZON,
        **kw,
    )
    torch.manual_seed(0)
    low = np.full(ACT, -1.0, dtype=np.float32)
    high = np.full(ACT, 1.0, dtype=np.float32)
    return FlowPolicyOptimizationPolicy(cfg, OBS, ACT, low, high)


def _velocity(policy):
    x_t = torch.zeros(2, HORIZON, ACT)
    t = torch.full((2, HORIZON, 1), 0.5)
    condition = torch.zeros(2, policy.actor.hidden_dim)
    return policy.compute_predicted_velocity(x_t, t, condition)


def test_the_velocity_can_exceed_one():
    policy = _policy()
    with torch.no_grad():
        policy.actor.out.weight.zero_()
        policy.actor.out.bias.fill_(3.0)

    velocity = _velocity(policy)
    assert torch.allclose(velocity, torch.full_like(velocity, 3.0))
    assert float(velocity.detach().abs().max()) > 1.0


def test_the_opt_in_clip_is_the_only_bound():
    policy = _policy(velocity_clip_value=0.5)
    with torch.no_grad():
        policy.actor.out.weight.zero_()
        policy.actor.out.bias.fill_(3.0)

    velocity = _velocity(policy)
    assert torch.allclose(velocity, torch.full_like(velocity, 0.5))
