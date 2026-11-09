"""The bootstrap reads the critic in eval mode and hands the critic back as it was.

``bootstrap_values`` puts both critics in eval mode for the target read. It used
to record ``policy.critic.training`` *after* that call, so the flag was always
False and the training loop lost the critic's dropout for the next update.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import pytest

from myfpo.configuration import MyfpoConfig
from myfpo.rollout import RolloutBuffers, run_rollout
from myfpo.training import make_policy


@pytest.fixture
def config():
    return dataclasses.replace(
        MyfpoConfig(),
        obs_dim=3,
        action_dim=1,
        hidden_dim=32,
        dit_depth=1,
        dit_heads=2,
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=4,
        chunks_per_update=8,
        chunks_per_minibatch=4,
        num_learning_epochs=1,
    )


def _rollout(config):
    env = gym.make("Pendulum-v1")
    policy = make_policy(
        3, 1, config, device="cpu", action_low=[-2.0], action_high=[2.0]
    )
    policy.train()
    try:
        batch = run_rollout(
            policy, RolloutBuffers(), env, 2, action_low=-2.0, action_high=2.0
        )
    finally:
        env.close()
    return policy, batch


def test_a_rollout_leaves_the_critic_in_the_mode_it_found(config):
    policy, _ = _rollout(config)
    assert policy.critic.training is True


def test_train_keeps_the_target_critic_in_eval(config):
    """``policy.train()`` must not switch dropout on in the bootstrap target."""
    policy = make_policy(
        3, 1, config, device="cpu", action_low=[-2.0], action_high=[2.0]
    )
    policy.train()
    assert policy.critic.training is True
    assert policy.critic_target.training is False


def test_a_rollout_leaves_the_critic_target_in_eval(config):
    policy, _ = _rollout(config)
    assert policy.critic_target.training is False


def test_the_batch_carries_no_dead_v_estimate(config):
    """The removed block read an undefined ``chunk_env_id`` and wrote a key
    nothing consumed."""
    _, batch = _rollout(config)
    assert "v_estimate" not in batch


def test_the_batch_carries_no_behavior_velocity(config):
    """The behavior velocity is used inside the loss, never by the training step."""
    _, batch = _rollout(config)
    assert "behavior_velocity_token" not in batch
