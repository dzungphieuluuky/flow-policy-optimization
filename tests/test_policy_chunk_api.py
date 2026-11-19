"""The policy's chunk API: encode once, sample many.

``predict_action_chunk`` encodes the window and samples a chunk in one call.
It is ``encode_states`` + ``sample_chunk``, so a caller that samples many
chunks from one window encodes it once.

Two invariants hold here:

- The two entry points agree. The composed call and the split call draw the
  same numbers.
- The condition the baseline samples from is the window the critic reads. The
  rollout and the training pass must normalize the same window, or the
  behaviour loss is not the loss the training pass recomputes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy
from flow_policy_optimization.rollout import make_rollout_state, run_rollout
from flow_policy_optimization.train import make_train_env


def _config() -> FlowPolicyOptimizationConfig:
    return FlowPolicyOptimizationConfig(
        hidden_dim=32,
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=4,
        chunks_per_update=8,
        chunks_per_minibatch=4,
        num_learning_epochs=1,
    )


def _policy(cfg) -> FlowPolicyOptimizationPolicy:
    low = np.full(1, -2.0, dtype=np.float32)
    high = np.full(1, 2.0, dtype=np.float32)
    torch.manual_seed(0)
    return FlowPolicyOptimizationPolicy(cfg, 3, 1, low, high)


def test_the_composed_and_the_split_call_agree():
    """Same seed, same numbers."""
    cfg = _config()
    policy = _policy(cfg)
    policy.train()
    states = torch.randn(3, cfg.num_obs_steps, 3)

    torch.manual_seed(11)
    composed = policy.predict_action_chunk(states)
    torch.manual_seed(11)
    split = policy.sample_chunk(policy.encode_states(states))
    assert torch.equal(composed, split)


def test_the_baseline_condition_is_the_window_the_critic_reads():
    """The rollout and the training pass normalize the same window.

    The critic reads ``normalized_states``, which the rollout builds by
    normalizing the stored window. The baseline samples from a condition built
    the same way. A different window here would make the baseline a read of a
    state the critic never saw.
    """
    cfg = _config()
    env = gym.make("Pendulum-v1")
    policy = _policy(cfg)
    policy.train()
    train_env = make_train_env(env, "Pendulum-v1", 1)
    state = make_rollout_state(3, 1, cfg, chunks_hint=8)
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

    states = batch["observation.state"]
    condition = policy.encode_states(states)
    normalized = policy.obs_normalizer(states.to(torch.float32))
    torch.testing.assert_close(condition, policy.encoder(normalized))
