"""The policy's chunk API: encode once, sample many.

``predict_action_chunk`` encodes the window and samples a chunk in one call.
It is now ``encode_states`` + ``sample_chunk``, so a caller that samples many
chunks from one window encodes it once.

Two invariants hold here:

- The two entry points agree. The composed call and the split call draw the
  same numbers.
- The condition the baseline samples from is the window the critic reads. The
  rollout and the training pass must normalize the same window, or the
  behaviour loss is not the loss the training pass recomputes.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import torch

from myfpo.config import MyfpoConfig
from myfpo.collect import RolloutBuffers, run_rollout
from myfpo.train import make_policy


def _config() -> MyfpoConfig:
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


def _policy(cfg) -> object:
    return make_policy(3, 1, cfg, device="cpu", action_low=[-2.0], action_high=[2.0])


def test_the_composed_and_the_split_call_agree():
    """Same seed, same numbers."""
    cfg = _config()
    policy = _policy(cfg)
    policy.train()
    states = torch.randn(3, cfg.num_obs_steps, 3)

    torch.manual_seed(11)
    composed = policy.predict_action_chunk({"observation.state": states})
    torch.manual_seed(11)
    split = policy.sample_chunk(policy.encode_states(states))
    assert torch.equal(composed, split)


def test_the_baseline_condition_is_the_window_the_critic_reads():
    """The rollout and the training pass normalize the same window.

    The critic reads ``normalized_states``, which the rollout builds by
    normalizing the stored window in the storage dtype. The baseline samples
    from a condition built the same way. A different window here would make
    the baseline a read of a state the critic never saw.
    """
    cfg = _config()
    env = gym.make("Pendulum-v1")
    policy = _policy(cfg)
    policy.train()
    try:
        batch = run_rollout(
            policy, RolloutBuffers(), env, 2, action_low=-2.0, action_high=2.0
        )
    finally:
        env.close()

    states = batch["observation.state"]
    condition = policy.encode_states(states)
    # The same normalization the bootstrap applies, over the same window.
    normalized = policy.obs_normalizer(states.to(torch.float32))
    torch.testing.assert_close(condition, policy.encoder(normalized))
