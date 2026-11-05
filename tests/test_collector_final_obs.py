"""A done step must hand the chunk the true terminal frame.

The collector stores the done step's observation as the chunk's ``next_state``,
and the truncation bootstrap reads it. An env that resets inside the step (the
gymnasium SAME_STEP contract, and ManiSkill's hardcoded behaviour) returns the
reset frame in ``obs`` and the real ``o_T`` in an info key. The collector used
to keep the reset frame, so the bootstrap started from a foreign episode.
"""

from __future__ import annotations

import numpy as np
import torch

from myfpo.maniskill_gpu import NextStepShim
from myfpo.rollout import RolloutBuffer, collect_chunks_vec, take_final_observations

OBS, ACT, H = 3, 2, 4
FINAL, RESET = 7.0, -7.0


class ResetFrameEnv:
    """A SAME_STEP-style env: ``step`` returns the reset frame, info the final one."""

    # No declared mode: the collector must not assume NEXT_STEP semantics.
    autoreset_mode = None

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
        2,
        -torch.ones(ACT),
        torch.ones(ACT),
        n_obs_steps=H,
        n_action_steps=H,
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
        assert torch.equal(active["step_frames"][end], torch.full((OBS,), FINAL))


def test_the_gymnasium_key_and_the_maniskill_key_both_work():
    reference = torch.full((2, OBS), RESET)
    final = torch.full((2, OBS), FINAL)
    done = np.array([True, False])

    gym = take_final_observations(
        {"final_obs": np.asarray([final[0].numpy(), None], dtype=object)},
        done,
        reference,
        torch.device("cpu"),
    )
    assert torch.equal(gym[0], final[0])
    assert torch.equal(gym[1], reference[1])

    mani = take_final_observations(
        {"final_observation": final}, done, reference, torch.device("cpu")
    )
    assert torch.equal(mani[0], final[0])
    assert torch.equal(mani[1], reference[1])


def test_no_info_key_leaves_the_observation_alone():
    """Under NEXT_STEP autoreset the step's own frame is already final."""
    reference = torch.full((2, OBS), RESET)
    out = take_final_observations({}, np.array([True, False]), reference, torch.device("cpu"))
    assert torch.equal(out, reference)


class _TorchVec:
    """A SAME_STEP torch env whose info dict is fixed by the test."""

    num_envs = 2
    device = "cpu"
    single_observation_space = None
    single_action_space = None

    def __init__(self, info):
        self.info = info

    def step(self, actions):  # noqa: ARG002 - the frame does not depend on it
        obs = torch.full((2, OBS), RESET)
        reward = torch.zeros(2)
        terminated = torch.tensor([True, False])
        truncated = torch.zeros(2, dtype=torch.bool)
        return obs, reward, terminated, truncated, self.info


def test_the_shim_uses_whichever_final_key_the_env_supplies():
    final = torch.full((2, OBS), FINAL)
    for info in ({"final_observation": final}, {"final_obs": final}):
        obs, *_ = NextStepShim(_TorchVec(info)).step_torch(torch.zeros(2, ACT))
        assert torch.equal(obs[0], final[0])
        assert torch.equal(obs[1], torch.full((OBS,), RESET))


def test_the_shim_passes_a_done_step_through_without_a_final_key():
    """A NEXT_STEP env already returns the final frame, so the key is absent."""
    obs, *_ = NextStepShim(_TorchVec({})).step_torch(torch.zeros(2, ACT))
    assert torch.equal(obs, torch.full((2, OBS), RESET))
