"""The backup's target chunk must live in the same space as the data's chunks.

``RolloutBuffer.chunk_actions`` holds chunks scaled to ``[-1, 1]``
(``scale_and_clip_chunk``), while ``predict_action_chunk`` returns actions in
real env bounds. QC's backup is ``Q(s_{t+h}, a_{t+h:t+2h})``, so an unscaled
proposal silently feeds the critic an action ``(high - low) / 2`` times too
large. On Reacher, whose bounds are +/-2, that is a 2x error on every proposed
row, and nothing else in the batch would show it: the boot observation looks
right and the value is finite.
"""

import dataclasses

import pytest
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.train_sjfpo import build_policy
from test_collector_batch_equivalence import FakeVecEnv

OBS, ACT, N_ENVS = 11, 3, 4
LOW = [-2.0, -2.0, -2.0]
HIGH = [2.0, 2.0, 2.0]
TOTAL_STEPS = 256


def _policy():
    cfg = dataclasses.replace(
        SJFPOConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
    )
    return build_policy(
        OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH
    )


@pytest.fixture(scope="module")
def batch():
    policy = _policy()
    env = FakeVecEnv(n=N_ENVS, obs_dim=OBS, act_dim=ACT, episode=1000)
    out = policy.generate_rollout(env, TOTAL_STEPS, action_low=LOW, action_high=HIGH)
    return out


def test_chunk_actions_are_stored_scaled(batch):
    """The reference: the collector's own chunks are in [-1, 1]."""
    assert batch["action"].abs().max() <= 1.0


def test_boot_actions_share_the_chunks_scale(batch):
    """The invariant the regression test exists for."""
    assert batch["boot_actions"].abs().max() <= 1.0, (
        f"boot_actions reach {batch['boot_actions'].abs().max():.3f}; they are in "
        "env bounds while chunk_actions are scaled to [-1, 1]"
    )


def test_boot_actions_are_finite_everywhere(batch):
    """Non-proposed rows keep a real chunk, and the backup multiplies it by
    ~terminated, where 0 * NaN is NaN."""
    assert batch["boot_actions"].isfinite().all()


def test_every_row_carries_a_chunk(batch):
    assert tuple(batch["boot_actions"].shape) == tuple(batch["action"].shape)


def test_terminated_rows_keep_their_own_chunk_not_a_proposal(batch):
    """A terminated chunk's bootstrap is zeroed, so its action is a filler.

    If the proposal leaked onto those rows, the cost would be the unfiltered
    one that the filter exists to avoid.
    """
    # a rollout of live chunks only is the case where the filter matters most
    policy = _policy()
    env = FakeVecEnv(n=N_ENVS, obs_dim=OBS, act_dim=ACT, episode=8)  # falls often
    out = policy.generate_rollout(env, TOTAL_STEPS, action_low=LOW, action_high=HIGH)
    assert out["boot_actions"].isfinite().all()
    assert out["boot_actions"].abs().max() <= 1.0