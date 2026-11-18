"""The rollout updates the normalizer before the behaviour loss uses it.

The first rollout trains on raw obs because the normalizer was still
``uninitialized`` while the critics and the behaviour loss were computed.
Collection still sees the previous statistics, which cannot be fixed, but the
loss and the update must read the current ones.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest
import torch

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, FlowPolicyOptimizationPolicy, make_rollout_state, make_train_env, run_rollout


@pytest.fixture
def config():
    return FlowPolicyOptimizationConfig(
        hidden_dim=32,
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=4,
        chunks_per_update=8,
        chunks_per_minibatch=4,
        num_learning_epochs=1,
    )


def test_the_first_rollout_initializes_the_normalizer(config):
    env = gym.make("Pendulum-v1")
    train_env = make_train_env(env, "Pendulum-v1", 1)
    policy = FlowPolicyOptimizationPolicy(
        config, 3, 1,
        np.full(1, -2.0, dtype=np.float32), np.full(1, 2.0, dtype=np.float32),
    )
    policy.train()
    state = make_rollout_state(3, 1, config, chunks_hint=8)
    try:
        batch = run_rollout(
            policy, train_env, 2,
            action_low=np.full(1, -2.0, dtype=np.float32),
            action_high=np.full(1, 2.0, dtype=np.float32),
            state=state,
        )
    finally:
        env.close()
    assert not policy.obs_normalizer.uninitialized
    states = batch["observation.state"]
    assert not torch.equal(policy.obs_normalizer(states), states)
