"""The rollout batch: the diagnostic-only keys and the advantage dtype.

Two contracts:

- The bootstrap values exist only for the drift diagnostic. They cost one flow
  integration and one critic read over every chunk, so the rollout only builds
  them when the caller asks for them.
- ``raw_advantages`` stays fp32. The per-minibatch restandardization reads
  its mean and standard deviation, and a bf16 standard deviation is only good
  to about 0.4%.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import pytest
import torch

from myfpo.configuration import MyfpoConfig
from myfpo.rollout import RolloutBuffers, run_rollout
from myfpo.training import make_policy, measure_value_drift


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


def _rollout(config, *, need_bootstrap: bool = False):
    env = gym.make("Pendulum-v1")
    policy = make_policy(
        3, 1, config, device="cpu", action_low=[-2.0], action_high=[2.0]
    )
    policy.train()
    try:
        batch = run_rollout(
            policy, RolloutBuffers(), env, 2, action_low=-2.0, action_high=2.0,
            need_bootstrap=need_bootstrap,
        )
    finally:
        env.close()
    return policy, batch


def test_the_batch_carries_no_bootstrap_values_without_the_diagnostic(config):
    """The drift diagnostic is off by default, so its inputs are not built."""
    _, batch = _rollout(config)
    assert "boot_values_start" not in batch
    assert "boot_actions" not in batch


def test_the_batch_carries_the_bootstrap_values_on_request(config):
    _, batch = _rollout(config, need_bootstrap=True)
    assert "boot_values_start" in batch
    assert "boot_actions" in batch
    assert batch["boot_values_start"].shape[0] == batch["raw_returns"].shape[0]


def test_the_drift_diagnostic_says_why_it_cannot_run(config):
    """A missing bootstrap value must not read as a number."""
    policy, batch = _rollout(config)
    with pytest.raises(ValueError, match="no bootstrap values"):
        measure_value_drift(policy, batch, "cpu")


def test_the_advantages_stay_fp32(config):
    """The per-minibatch restandardization reads their mean and std."""
    _, batch = _rollout(config)
    assert batch["raw_advantages"].dtype == torch.float32
    assert batch["raw_returns"].dtype == torch.float32
