"""The collector speaks SameStep only.

Two guards:

- An env that declares ``NEXT_STEP`` autoreset is rejected at collection time,
  loudly, before a single chunk is planned. A silent NEXT_STEP run would
  archive the reset frames of the previous episodes as their ``next_state``.
- An env that declares SAME_STEP (or nothing) but whose done step hands back
  a reset frame without a final observation on a done row raises, instead of
  storing the previous episode's terminal frame as the next episode's o_T.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flow_policy_optimization.flow_policy_optimization import RolloutBuffer, collect_chunks_vec

OBS, ACT, H = 3, 2, 4
FINAL, RESET = 7.0, -7.0


class NextStepEnv:
    autoreset_mode = type("M", (), {"name": "NEXT_STEP"})()

    def __init__(self, n: int = 2):
        self.num_envs = n
        self.observation_space = type("S", (), {"shape": (OBS,)})()
        self.single_observation_space = self.observation_space
        self.single_action_space = type("S", (), {"shape": (ACT,)})()


class SilentSameStepEnv:
    """SAME_STEP in name, but the done step carries no final observation."""

    autoreset_mode = type("M", (), {"name": "SAME_STEP"})()

    def __init__(self, n: int = 2):
        self.num_envs = n
        self.observation_space = type("S", (), {"shape": (OBS,)})()
        self.single_observation_space = self.observation_space
        self.single_action_space = type("S", (), {"shape": (ACT,)})()

    def reset(self, seed=None):  # noqa: ARG002 - gym calls reset(seed=...)
        return np.zeros((self.num_envs, OBS), dtype=np.float32), {}

    def step(self, actions):  # noqa: ARG002 - actions do not matter here
        done = np.ones(self.num_envs, dtype=bool)
        obs = np.full((self.num_envs, OBS), RESET, dtype=np.float32)
        info = {}  # the final observation is missing
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


def _policy(states):
    return torch.zeros(states.shape[0], H, ACT)


def test_a_next_step_env_is_rejected():
    env = NextStepEnv(n=2)
    with pytest.raises(ValueError, match="SameStep"):
        collect_chunks_vec(
            env,
            _policy,
            2,
            -torch.ones(ACT),
            torch.ones(ACT),
            num_obs_steps=H,
            num_action_steps=H,
            buffer=_buffer(),
            seed=0,
        )


def test_a_done_row_without_a_final_observation_fails_loudly():
    env = SilentSameStepEnv(n=2)
    with pytest.raises(ValueError, match="final"):
        collect_chunks_vec(
            env,
            _policy,
            1,
            -torch.ones(ACT),
            torch.ones(ACT),
            num_obs_steps=H,
            num_action_steps=H,
            buffer=_buffer(),
            seed=0,
        )
