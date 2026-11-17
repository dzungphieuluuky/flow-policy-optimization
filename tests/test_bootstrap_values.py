"""The bootstrap reads the critic in eval mode and leaves it as it found it.

The critic and the target critic are handled separately: the rollout
bootstrap reads ``critic_target`` for the next state and ``critic`` for the
taken chunk; the target critic never trains. These tests pin those contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from myfpo.myfpo import MyfpoConfig, MyfpoPolicy, _estimate_state_values, make_rollout_state, make_train_env, run_rollout

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
    state = make_rollout_state(3, 1, config, total_steps_hint=8)
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


def test_estimate_state_values_averages_fresh_policy_samples(config):
    """The advantage baseline is E_a[Q(s, a)] over fresh chunks, not Q(s, a_taken)."""
    cfg = MyfpoConfig(
        hidden_dim=32,
        num_obs_steps=config.num_obs_steps,
        action_horizon=config.action_horizon,
        num_action_steps=config.num_action_steps,
        num_baseline_samples=3,
    )
    policy = _policy(cfg)
    n, horizon = 2, cfg.action_horizon
    proposals = [
        torch.full((n, horizon, 1), 0.2),
        torch.full((n, horizon, 1), 0.6),
        torch.full((n, horizon, 1), 1.0),
    ]
    remaining = iter(proposals)
    policy.encode_states = lambda states: torch.zeros(states.shape[0], cfg.hidden_dim)
    policy.sample_chunk = lambda _: next(remaining)

    class StateIndependentCritic(torch.nn.Module):
        def forward(self, states, actions, exec_length_frac=None):
            del states, exec_length_frac
            return actions.mean(dim=(1, 2))

    states = torch.zeros(n, cfg.num_obs_steps, 3)
    v = _estimate_state_values(
        policy, states, states,
        exec_frac=torch.ones(n), mask=torch.ones(n, horizon, 1),
        critic=StateIndependentCritic(),
    )
    expected = torch.stack([p.mean(dim=(1, 2)) for p in proposals]).mean(dim=0)
    torch.testing.assert_close(v, expected)
    # A baseline that fell back to one proposal (or the taken chunk) would not
    # equal the three-proposal mean, and its sign would differ here.
    assert float(v.mean()) > 0.0


def test_the_baseline_encodes_the_window_once(config):
    """The window is the same for every sample, so the encoder runs once."""
    cfg = MyfpoConfig(
        hidden_dim=32,
        num_obs_steps=config.num_obs_steps,
        action_horizon=config.action_horizon,
        num_action_steps=config.num_action_steps,
        num_baseline_samples=4,
    )
    policy = _policy(cfg)
    calls = []
    original = policy.encode_states

    def counting(states):
        calls.append(states)
        return original(states)

    policy.encode_states = counting
    _estimate_state_values(
        policy,
        torch.zeros(2, cfg.num_obs_steps, 3),
        torch.zeros(2, cfg.num_obs_steps, 3),
        exec_frac=torch.ones(2),
        mask=torch.ones(2, cfg.action_horizon, 1),
        critic=policy.critic,
    )
    assert len(calls) == 1, f"the encoder ran {len(calls)} times for one window"
