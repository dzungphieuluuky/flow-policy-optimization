"""The collector's autoreset state machine must match the env's reset contract.

A SAME_STEP env resets inside the done ``step()`` and hands back the new
episode's o₀. It never parks a row waiting for a reset. A NEXT_STEP env does
park one, and the parked row takes the reset frame from the step after the
done one.

The collector used to run the parked-row branch for both modes. Under
SAME_STEP the frame it would copy is o_T, the previous episode's terminal
frame, so the new episode would start from the old episode's last
observation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from myfpo.collect import (
    EnvCarry,
    RolloutBuffer,
    close_chunks,
    collect_chunks_vec,
    make_state,
)

OBS, ACT, H = 3, 2, 4
FINAL, RESET = 7.0, -7.0


class SameStepEnv:
    """A SAME_STEP env: the done step returns the reset frame."""

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
        info = {"final_observation": final}
        return obs, np.zeros(self.num_envs, np.float32), done, np.zeros_like(done), info


def _buffer(n_chunks: int = 8) -> RolloutBuffer:
    return RolloutBuffer(
        max_chunks=n_chunks,
        max_steps=n_chunks,
        obs_steps=H,
        obs_dim=OBS,
        action_horizon=H,
        action_dim=ACT,
    )


def test_a_stale_carry_must_not_reach_a_same_step_collector():
    """A carry from a NEXT_STEP run cannot be loaded by a SAME_STEP env."""
    env = SameStepEnv(n=2)
    carry = EnvCarry(
        windows=torch.zeros(2, H, OBS),
        awaiting=[True, False],
    )
    with pytest.raises(RuntimeError, match="awaiting rows"):
        collect_chunks_vec(
            env,
            lambda states: torch.zeros(states.shape[0], H, ACT),
            2,
            -torch.ones(ACT),
            torch.ones(ACT),
            num_obs_steps=H,
            num_action_steps=H,
            buffer=_buffer(),
            seed=0,
            carry=carry,
        )


def test_a_parked_row_under_same_step_raises_instead_of_storing_o_t():
    """The parked-row branch would copy o_T into the new episode's window."""
    state = make_state(
        2, H, OBS, H, ACT,
        torch.zeros(2, H, OBS),
        remaining=8,
        autoreset_mode="SAME_STEP",
    )
    state.awaiting[:] = [True, False]
    # o_T in the done row, the reset frame nowhere.
    obs_t = torch.full((2, OBS), FINAL)
    with pytest.raises(RuntimeError, match="awaiting rows"):
        close_chunks(
            state,
            obs_t,
            np.array([True, False]),
            np.array([False, False]),
            np.array([True, False]),
            _buffer(),
            H,
            reset_frames=torch.full((2, OBS), RESET),
            autoreset_mode="SAME_STEP",
        )


def test_the_next_step_parked_row_still_takes_the_reset_frame():
    """The NEXT_STEP path keeps its behaviour: the parked row copies obs_t."""
    state = make_state(
        2, H, OBS, H, ACT,
        torch.zeros(2, H, OBS),
        remaining=8,
        autoreset_mode="NEXT_STEP",
    )
    state.awaiting[:] = [True, False]
    obs_t = torch.full((2, OBS), RESET)
    close_chunks(
        state,
        obs_t,
        np.array([False, False]),
        np.array([False, False]),
        np.array([False, False]),
        _buffer(),
        H,
        autoreset_mode="NEXT_STEP",
    )
    assert not state.awaiting.any()
    assert torch.equal(state.windows[0], torch.full((H, OBS), RESET))
