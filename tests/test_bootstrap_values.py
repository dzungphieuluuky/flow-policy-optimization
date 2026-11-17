"""The bootstrap reads the critic in eval mode and leaves it as it found it.

The critic and the target critic are handled separately: the rollout
bootstrap reads ``critic_target`` for the next state and ``critic`` for the
taken chunk; the target critic never trains. These tests pin those contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from myfpo.myfpo import MyfpoConfig, MyfpoPolicy, make_rollout_state, make_train_env, run_rollout

import gymnasium as gym
import torch


@pytest.fixture
def config():
    return MyfpoConfig(
        hidden_dim=32,
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=4,
        chunks_per_update=8,
        chunks_per_minibatch=4,
        num_learning_epochs=1,
    )


def _policy(cfg):
    low = np.full(1, -2.0, dtype=np.float32)
    high = np.full(1, 2.0, dtype=np.float32)
    torch.manual_seed(0)
    return MyfpoPolicy(cfg, 3, 1, low, high)


def _rollout(config):
    env = gym.make("Pendulum-v1")
    train_env = make_train_env(env, "Pendulum-v1", 1)
    policy = _policy(config)
    policy.train()
    state = make_rollout_state(3, 1, config, chunks_hint=8)
    try:
        batch = run_rollout(
            policy,
            train_env,
            2,
            action_low=np.full(1, -2.0, dtype=np.float32),
            action_high=np.full(1, 2.0, dtype=np.float32),
            state=state,
        )
    finally:
        env.close()
    return policy, batch


def test_a_rollout_leaves_the_critic_in_the_mode_it_found(config):
    policy, _ = _rollout(config)
    assert policy.critic.training is True


def test_train_keeps_the_target_critic_in_eval(config):
    """``policy.train()`` must keep the bootstrap target critic in eval mode."""
    policy = _policy(config)
    policy.train()
    assert policy.critic_target.training is False


def test_a_rollout_leaves_the_critic_target_in_eval(config):
    policy, _ = _rollout(config)
    assert policy.critic_target.training is False


def test_the_batch_carries_no_dead_v_estimate(config):
    _, batch = _rollout(config)
    assert "v_estimate" not in batch


def test_the_batch_carries_no_behavior_velocity(config):
    """The behavior velocity is used inside the loss, never by the training step."""
    _, batch = _rollout(config)
    assert "behavior_velocity_token" not in batch


def test_the_bootstrap_does_not_sample_actor_proposals(config):
    """The baseline is the V-head: one forward per end, zero actor chunks sampled."""
    env = gym.make("Pendulum-v1")
    train_env = make_train_env(env, "Pendulum-v1", 1)
    policy = _policy(config)
    policy.train()
    state = make_rollout_state(3, 1, config, chunks_hint=8)
    calls = []
    original = policy.sample_chunk

    def counting(cond):
        calls.append(cond)
        return original(cond)

    policy.sample_chunk = counting
    # The collection generates behaviour chunks; isolate the bootstrap by stubbing it.
    policy.predict_action_chunk = lambda states: torch.zeros(
        states.shape[0], config.action_horizon, 1
    )
    try:
        run_rollout(
            policy,
            train_env,
            2,
            action_low=np.full(1, -2.0, dtype=np.float32),
            action_high=np.full(1, 2.0, dtype=np.float32),
            state=state,
        )
    finally:
        env.close()
    assert calls == [], f"the rollout sampled {len(calls)} actor chunks for the baseline"


def test_the_bootstrap_reads_the_v_head_directly(config):
    """forward_v is an encode + one linear; no actor samples and no critic Q."""
    cfg = MyfpoConfig(
        hidden_dim=32,
        num_obs_steps=config.num_obs_steps,
        action_horizon=config.action_horizon,
        num_action_steps=config.num_action_steps,
    )
    policy = _policy(cfg)
    torch.manual_seed(3)
    states = torch.zeros(2, cfg.num_obs_steps, 3)
    v_direct = policy.critic.forward_v(states)
    assert v_direct.shape == (2,)
    assert torch.isfinite(v_direct).all()
