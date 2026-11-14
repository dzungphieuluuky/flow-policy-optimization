"""The bootstrap reads the critic in eval mode and hands the critic back as it was.

``bootstrap_values`` puts both critics in eval mode for the target read. It used
to record ``policy.critic.training`` *after* that call, so the flag was always
False and the training loop lost the critic's eval-mode invariant for the next update.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import pytest

from myfpo.config import MyfpoConfig
from myfpo.collect import RolloutBuffers, run_rollout
from myfpo.train import make_policy


@pytest.fixture
def config():
    return dataclasses.replace(
        MyfpoConfig(),
        obs_dim=3,
        action_dim=1,
        hidden_dim=32,
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
    """``policy.train()`` must keep the bootstrap target critic in eval mode."""
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


def test_estimate_state_values_averages_fresh_policy_samples(config):
    """The advantage baseline is E_a[Q(s, a)] over fresh chunks, not Q(s, a_taken).

    With a calibrated critic, Q(s, a_taken) is zero-mean as an advantage given
    (s, a), so the actor would train on critic error. The helper must draw
    ``num_baseline_samples`` fresh proposals and average their critic scores.
    """
    import torch

    from myfpo.actions import squash_to_env_bounds
    from myfpo.collect import estimate_state_values

    cfg = dataclasses.replace(config, num_baseline_samples=3, action_processing_mode="tanh_squash")
    policy = make_policy(
        3, 1, cfg, device="cpu", action_low=[-2.0], action_high=[2.0]
    )
    n, horizon = 2, cfg.action_horizon
    proposals = [
        torch.full((n, horizon, 1), 0.2),
        torch.full((n, horizon, 1), 0.6),
        torch.full((n, horizon, 1), 1.0),
    ]
    remaining = iter(proposals)
    policy.encode_states = lambda states: torch.zeros(states.shape[0], cfg.hidden_dim)
    policy.sample_chunk = lambda _condition, **k: next(remaining)

    class StateIndependentCritic(torch.nn.Module):
        def forward(self, states, actions, exec_length_frac=None):
            del states, exec_length_frac
            member = actions.mean(dim=(1, 2))
            return torch.stack([member, member])  # min of identical members is the member

    states = torch.zeros(n, cfg.num_obs_steps, 3)
    mask = torch.ones(n, horizon, 1)
    frac = torch.ones(n)
    v = estimate_state_values(
        policy, states, states,
        exec_length_frac=frac, mask=mask,
        critic=StateIndependentCritic(),
    )
    squashed = [
        squash_to_env_bounds(p, policy.action_low, policy.action_high) for p in proposals
    ]
    expected = torch.stack([a.mean(dim=(1, 2)) for a in squashed]).mean(dim=0)
    torch.testing.assert_close(v, expected)
    # A baseline that silently fell back to one proposal (or the taken chunk)
    # would not equal the three-proposal mean, and its sign would differ here.
    assert float(v.mean()) > 0.0


def test_the_baseline_encodes_the_window_once(config):
    """The window is the same for every sample, so the encoder runs once.

    The encoder is a read of identical data per sample otherwise: the same
    window goes through it ``num_baseline_samples`` times.
    """
    import torch

    from myfpo.collect import estimate_state_values

    cfg = dataclasses.replace(config, num_baseline_samples=4)
    policy = make_policy(3, 1, cfg, device="cpu", action_low=[-2.0], action_high=[2.0])
    calls = []
    original = policy.encode_states

    def counting(states):
        calls.append(states)
        return original(states)

    policy.encode_states = counting
    n = 2
    estimate_state_values(
        policy,
        torch.zeros(n, cfg.num_obs_steps, 3),
        torch.zeros(n, cfg.num_obs_steps, 3),
        exec_length_frac=torch.ones(n),
        mask=torch.ones(n, cfg.action_horizon, 1),
        critic=policy.critic,
    )
    assert len(calls) == 1, f"the encoder ran {len(calls)} times for one window"
