"""The action the flow is trained on is the action the flow produces.

The flow samples a raw action. The env receives the squash of it. The buffer
must keep the raw value, because the CFM regression target is built from it:
if the buffer kept the squash, the flow would be trained on the squash of its
own output, and the two would never be the same quantity.

These tests pin that one invariant, and they are the regression guard for the
whole action-space change.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import OBS_STATE

from myfpo.configuration import MyfpoConfig
from myfpo.processor import squash_to_env
from myfpo.training import build_policy
from test_collector_batch_equivalence import FakeVecEnv

OBS, ACT = 11, 3
LOW = [-2.0] * ACT
HIGH = [2.0] * ACT
TOTAL_STEPS = 256


def _policy():
    cfg = dataclasses.replace(
        MyfpoConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
    )
    return build_policy(
        OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH
    )


def _raw_chunk(horizon: int, value: float = 3.0) -> torch.Tensor:
    """A chunk whose values are outside the env's range, so the squash shows."""
    return torch.full((1, horizon, ACT), value, dtype=torch.float32)


def _fake_predict(horizon: int, value: float = 3.0):
    """A policy that always plans the same raw chunk, whatever the batch size."""

    def fake(batch, **kwargs):
        n = batch[OBS_STATE].shape[0]
        return torch.full((n, horizon, ACT), value, dtype=torch.float32)

    return fake


def _expected_executed(raw: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(LOW)
    high = torch.tensor(HIGH)
    return squash_to_env(raw, (high - low) / 2, (high + low) / 2)


def test_predict_action_chunk_returns_the_raw_chunk():
    """The policy's output is unbounded: it is what the flow was fit to."""
    policy = _policy()
    raw = _raw_chunk(policy.config.action_horizon)
    policy.predict_action_chunk = lambda *a, **k: raw
    out = policy.predict_action_chunk({"observation.state": torch.zeros(1, OBS)})
    assert torch.equal(out, raw)
    assert float(out.abs().max()) > float(torch.tensor(HIGH).abs().max()), (
        "the raw chunk was squashed inside the policy"
    )


def test_select_action_returns_the_bounded_env_action():
    """The action handed to the env is the squash of the raw chunk."""
    policy = _policy()
    raw = _raw_chunk(policy.config.action_horizon)
    policy.predict_action_chunk = lambda *a, **k: raw
    policy.reset()
    action = policy.select_action({"observation.state": torch.zeros(OBS)})
    assert float(action.abs().max()) <= float(torch.tensor(HIGH).abs().max()) + 1e-6
    assert torch.allclose(
        action, _expected_executed(raw)[0, 0], atol=1e-6
    ), "the env-facing action is not the squash of the raw chunk"


def test_the_squash_is_applied_exactly_once():
    """Squash(raw) != raw, so a second squash would be visible."""
    policy = _policy()
    raw = _raw_chunk(policy.config.action_horizon)
    policy.predict_action_chunk = lambda *a, **k: raw
    policy.reset()
    action = policy.select_action({"observation.state": torch.zeros(OBS)})
    squashed = _expected_executed(raw)[0, 0]
    assert not torch.allclose(squashed, raw[0, 0])
    assert torch.allclose(action, squashed, atol=1e-6), (
        "the raw chunk was squashed twice, or not at all"
    )


def test_the_stored_chunk_is_the_flow_sample():
    """The buffer keeps the raw chunk, not the squash of it."""
    policy = _policy()
    horizon = policy.config.action_horizon
    policy.predict_action_chunk = _fake_predict(horizon)

    env = FakeVecEnv(n=4, obs_dim=OBS, act_dim=ACT, episode=1000)
    batch = policy.generate_rollout(env, TOTAL_STEPS, action_low=LOW, action_high=HIGH)

    stored = batch["action"]
    assert torch.allclose(stored, torch.full_like(stored, 3.0)), (
        "the stored chunk is not the flow's own output"
    )
    assert float(stored.abs().max()) > float(torch.tensor(HIGH).abs().max()), (
        "the stored chunk is squashed, so the flow would be trained on the "
        "squash of its own output"
    )


def test_the_env_receives_the_squash_of_the_stored_chunk():
    """The env's action and the stored chunk are one squash apart."""
    policy = _policy()
    horizon = policy.config.action_horizon
    policy.predict_action_chunk = _fake_predict(horizon)

    env = FakeVecEnv(n=4, obs_dim=OBS, act_dim=ACT, episode=1000)
    batch = policy.generate_rollout(env, TOTAL_STEPS, action_low=LOW, action_high=HIGH)

    seen = torch.from_numpy(np.stack(env.actions_seen))
    expected = _expected_executed(batch["action"])
    # The first chunk of the first env runs one step per token, so the env's
    # first `horizon` actions are that chunk's tokens, squashed.
    assert torch.allclose(seen[:horizon, 0, :], expected[0, :horizon, :], atol=1e-5), (
        "the env did not receive the squash of the stored chunk"
    )


def test_boot_actions_live_in_the_stored_actions_space():
    """The backup's action is in the same space as the chunks it backs up."""
    policy = _policy()
    horizon = policy.config.action_horizon
    policy.predict_action_chunk = _fake_predict(horizon)

    env = FakeVecEnv(n=4, obs_dim=OBS, act_dim=ACT, episode=1000)
    batch = policy.generate_rollout(env, TOTAL_STEPS, action_low=LOW, action_high=HIGH)

    assert batch["boot_actions"].shape == batch["action"].shape
    # A live row's backup is a fresh proposal; a terminated row's is its own
    # chunk. Both are raw.
    assert float(batch["boot_actions"].abs().max()) > float(
        torch.tensor(HIGH).abs().max()
    ), "the backup is in the env's bounds, so it is in a different space"
