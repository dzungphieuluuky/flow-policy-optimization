"""A done step must hand the chunk the true terminal frame.

The collector stores the done step's observation as the chunk's ``next_state``,
and the truncation bootstrap reads it. An env that resets inside the step (the
gymnasium SAME_STEP contract, and ManiSkill's hardcoded behaviour) returns the
reset frame in ``obs`` and the real ``o_T`` in an info key. The collector used
to keep the reset frame, so the bootstrap started from a foreign episode.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flow_policy_optimization.flow_policy_optimization import RolloutBuffer, collect_chunks_vec, take_final_observations

OBS, ACT, H = 3, 2, 4
FINAL, RESET = 7.0, -7.0


class ResetFrameEnv:
    """A SAME_STEP-style env: ``step`` returns the reset frame, info the final one."""

    autoreset_mode = type("M", (), {"name": "SAME_STEP"})()

    def __init__(self, n: int = 2):
        self.num_envs = n
        self.observation_space = type("S", (), {"shape": (OBS,)})()
        self.single_observation_space = self.observation_space
        self.single_action_space = type("S", (), {"shape": (ACT,)})()
        self.t = 0

    def reset(self, seed=None):  # noqa: ARG002 - gym calls reset(seed=...)
        self.t = 0
        return np.zeros((self.num_envs, OBS), dtype=np.float32), {}

    def step(self, actions):  # noqa: ARG002 - actions do not matter here
        self.t += 1
        done = np.ones(self.num_envs, dtype=bool)
        obs = np.full((self.num_envs, OBS), RESET, dtype=np.float32)
        final = np.full((self.num_envs, OBS), FINAL, dtype=np.float32)
        info = {"final_obs": np.full(self.num_envs, None, dtype=object)}
        info["final_obs"][:] = list(final)
        return obs, np.zeros(self.num_envs, np.float32), done, np.zeros_like(done), info


def test_a_done_step_stores_the_final_frame_not_the_reset_frame():
    env = ResetFrameEnv(n=2)
    buffer = RolloutBuffer(
        max_chunks=4,
        max_steps=4,
        obs_steps=H,
        obs_dim=OBS,
        action_horizon=H,
        action_dim=ACT,
    )

    def policy(states):
        return torch.zeros(states.shape[0], H, ACT)

    collect_chunks_vec(
        env,
        policy,
        1,
        -torch.ones(ACT),
        torch.ones(ACT),
        num_obs_steps=H,
        num_action_steps=H,
        buffer=buffer,
        seed=0,
    )

    active = buffer.active_prefix()
    assert buffer.n_chunks == 2
    last = active["chunk_next_state"][:, -1]
    assert torch.equal(last, torch.full((2, OBS), FINAL))
    # The terminal frame the buffer records is the same frame.
    lengths = active["chunk_lengths"].to(torch.long)
    starts = active["chunk_starts"].to(torch.long)
    for i in range(buffer.n_chunks):
        end = starts[i] + lengths[i] - 1
        assert torch.equal(active["normalizer_observations"][end], torch.full((OBS,), FINAL))


def test_the_gymnasium_key_and_the_maniskill_key_both_work():
    reference = torch.full((2, OBS), RESET)
    final = torch.full((2, OBS), FINAL)
    done = np.array([True, False])

    gym = take_final_observations(
        {"final_obs": np.asarray([final[0].numpy(), None], dtype=object)},
        done,
        reference,
        strict=False,
    )
    assert torch.equal(gym[0], final[0])
    assert torch.equal(gym[1], reference[1])

    mani = take_final_observations(
        {"final_observation": final}, done, reference, strict=False
    )
    assert torch.equal(mani[0], final[0])
    assert torch.equal(mani[1], reference[1])


def test_no_info_key_leaves_the_observation_alone():
    """Under NEXT_STEP autoreset the step's own frame is already final."""
    reference = torch.full((2, OBS), RESET)
    out = take_final_observations({}, np.array([True, False]), reference, strict=False)
    assert torch.equal(out, reference)


def test_a_final_frame_with_one_row_per_done_row_is_read():
    """The env may report only the done rows, not one row per env."""
    reference = torch.full((3, OBS), RESET)
    final = torch.full((2, OBS), FINAL)
    done = np.array([True, False, True])

    out = take_final_observations(
        {"final_observation": final}, done, reference, strict=False
    )
    assert torch.equal(out[0], final[0])
    assert torch.equal(out[1], reference[1])
    assert torch.equal(out[2], final[1])


def test_a_final_frame_with_an_unusable_shape_raises():
    """A shape the collector cannot map onto the done rows is an error."""
    reference = torch.full((3, OBS), RESET)
    final = torch.full((5, OBS), FINAL)
    with pytest.raises(ValueError, match="rows"):
        take_final_observations(
            {"final_observation": final},
            np.array([True, False, True]),
            reference,
            strict=False,
        )


def test_take_final_observations_strict_mode_requires_the_final_frame():
    reference = torch.full((2, OBS), RESET)
    final = torch.full((2, OBS), FINAL)
    done = np.array([True, False])
    for info in ({"final_obs": final}, {"final_observation": final}):
        out = take_final_observations(info, done, reference, strict=True)
        assert torch.equal(out[0], final[0])
        assert torch.equal(out[1], torch.full((OBS,), RESET))
    with pytest.raises(ValueError, match="done rows"):
        take_final_observations({}, done, reference, strict=True)
    with pytest.raises(ValueError, match="done row 0"):
        take_final_observations(
            {"final_observation": np.array([None, final[1].numpy()], dtype=object)},
            done,
            reference,
            strict=True,
        )
    # Lenient mode keeps the old tolerant behavior for NEXT_STEP/undeclared envs.
    out = take_final_observations({}, done, reference, strict=False)
    assert torch.equal(out, reference)
