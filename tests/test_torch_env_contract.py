"""The collector on a torch env: the done step must report the true ``o_{t+1}``.

ManiSkill resets a finished env inside the done ``step()`` and hands the real
next observation through ``info["final_observation"]``. The collector
consumes that SAME_STEP contract natively, without a forced reset step.
These tests pin it without SAPIEN: the fake below has ManiSkill's autoreset
behavior.
"""

import numpy as np
import pytest
import torch

from myfpo.myfpo import RolloutBuffer, collect_chunks_vec

OBS_STEPS, OBS_DIM, HORIZON, ACT_DIM = 2, 2, 2, 2
NUM_ENVS = 2
EPISODE_LEN = 3
RESET_MARKER = -100.0


class FakeSameStepVec:
    """A SAME_STEP torch vector env with ManiSkill's step contract."""

    def __init__(self, n: int = NUM_ENVS, obs_dim: int = OBS_DIM, episode_len: int = EPISODE_LEN):
        gym = pytest.importorskip("gymnasium")
        self.num_envs = n
        self.obs_dim = obs_dim
        self.episode_len = episode_len
        self.device = torch.device("cpu")
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.steps = torch.zeros(n, dtype=torch.long)
        self.state = torch.zeros(n, obs_dim)
        self.resets = 0
        self.actions: list[torch.Tensor] = []
        self.autoreset_mode = type("M", (), {"name": "SAME_STEP"})()

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.state = torch.full((self.num_envs, self.obs_dim), RESET_MARKER)
        return self.state.clone(), {}

    def step(self, actions):
        action = torch.as_tensor(actions, dtype=torch.float32)
        self.steps += 1
        self.actions.append(action.clone())
        self.state = self.state + action
        obs = self.state.clone()
        final = obs.clone()
        done = self.steps >= self.episode_len
        if bool(done.any()):
            # ManiSkill: the done rows are reset here, and the real next
            # observation survives only in final_observation.
            self.state[done] = RESET_MARKER
            self.steps[done] = 0
            obs[done] = RESET_MARKER
            self.resets += 1
        zero = torch.zeros(self.num_envs)
        info = {"final_observation": final}
        return obs, zero, done.clone(), torch.zeros(self.num_envs, dtype=torch.bool), info

    def close(self):
        return None



def test_collector_stores_the_true_terminal_frame():
    """A chunk closed by a termination stores o_{t+1}, not the reset frame."""
    env = FakeSameStepVec()
    buffer = RolloutBuffer(
        max_chunks=64,
        max_steps=64,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    collect_chunks_vec(
        env,
        lambda states: torch.ones(states.shape[0], HORIZON, ACT_DIM),
        total_steps=12,
        num_action_steps=HORIZON,
        action_low=-1.0,
        action_high=1.0,
        num_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    n = buffer.n_chunks
    terminated = buffer.chunk_terminated[:n]
    assert bool(terminated.any()), "the fake terminates every three steps"
    assert env.resets > 0, "the fake never reset, so nothing was tested"
    # The defect this guards: a SAME_STEP env puts the reset frame in the buffer.
    # The fake marks it with RESET_MARKER, so any leak is exact, not approximate.
    # A window may legitimately *start* on a reset frame, so only the step
    # frames (recorded observations) and the bootstrap frame are checked.
    assert not (buffer.normalizer_observations[: buffer.n_steps] == RESET_MARKER).any()
    for row in terminated.nonzero().flatten().tolist():
        start = int(buffer.chunk_starts[row])
        length = int(buffer.chunk_lengths[row])
        step_frame = buffer.normalizer_observations[start + length - 1]
        next_frame = buffer.chunk_next_state[row, -1]
        assert torch.equal(next_frame, step_frame)
        assert not torch.equal(next_frame, torch.full_like(next_frame, RESET_MARKER))


def test_a_numpy_env_still_takes_the_numpy_path():
    """The numpy (mujoco) path runs through the same SameStep vector env."""
    gym = pytest.importorskip("gymnasium")

    def make_one():
        env = gym.make("Pendulum-v1")
        env.reset(seed=0)
        env.action_space.seed(0)
        return env

    envs = gym.vector.SyncVectorEnv([make_one for _ in range(NUM_ENVS)], autoreset_mode="SameStep")
    buffer = RolloutBuffer(
        max_chunks=64,
        max_steps=64,
        obs_steps=OBS_STEPS,
        obs_dim=3,
        action_horizon=HORIZON,
        action_dim=1,
    )
    collect_chunks_vec(
        envs,
        lambda states: torch.zeros(states.shape[0], HORIZON, 1),
        total_steps=8,
        num_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        num_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    envs.close()
    assert buffer.n_chunks > 0
    assert buffer.normalizer_observations[: buffer.n_steps].shape == (buffer.n_steps, 3)


def test_same_step_env_never_gets_a_forced_reset_step():
    """A done step resets the env inside the call; no zero action is forced."""
    env = FakeSameStepVec()
    buffer = RolloutBuffer(
        max_chunks=64,
        max_steps=64,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    collect_chunks_vec(
        env,
        lambda states: torch.ones(states.shape[0], HORIZON, ACT_DIM),
        total_steps=12,
        num_action_steps=HORIZON,
        action_low=-1.0,
        action_high=1.0,
        num_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    assert buffer.n_steps == 12
    # Both rows step together and hit their target on the same call, so no
    # row is ever parked while it still owes steps. Every executed action is
    # the policy's nonzero ones; an all-zero row means a forced reset step
    # leaked back in.
    for call in env.actions:
        assert not bool((call == 0).all(dim=-1).any())


class NoFinalSameStepVec(FakeSameStepVec):
    """SAME_STEP env whose info carries no final observation."""

    def step(self, actions):
        obs, rew, term, trunc, _info = super().step(actions)
        return obs, rew, term, trunc, {}


def test_same_step_env_without_final_observation_raises():
    """A missing final frame must not be patched over with the reset frame."""
    env = NoFinalSameStepVec(n=1)
    buffer = RolloutBuffer(
        max_chunks=4,
        max_steps=4,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    with pytest.raises(ValueError, match="done rows"):
        collect_chunks_vec(
            env,
            lambda states: torch.ones(states.shape[0], HORIZON, ACT_DIM),
            total_steps=4,
            num_action_steps=HORIZON,
            action_low=-1.0,
            action_high=1.0,
            num_obs_steps=OBS_STEPS,
            buffer=buffer,
            seed=0,
        )


def test_an_env_without_a_same_step_declaration_is_rejected():
    from types import SimpleNamespace

    def policy(states):
        return torch.zeros(states.shape[0], HORIZON, ACT_DIM)

    env = SimpleNamespace(
        num_envs=2,
        observation_space=SimpleNamespace(shape=(OBS_DIM,)),
        single_observation_space=SimpleNamespace(shape=(OBS_DIM,)),
        single_action_space=SimpleNamespace(shape=(ACT_DIM,)),
        autoreset_mode=SimpleNamespace(name="DISABLED"),
    )
    with pytest.raises(ValueError, match="SameStep"):
        collect_chunks_vec(
            env,
            policy,
            1,
            -torch.ones(ACT_DIM),
            torch.ones(ACT_DIM),
            num_obs_steps=OBS_STEPS,
            num_action_steps=HORIZON,
            buffer=RolloutBuffer(
                max_chunks=4,
                max_steps=4,
                obs_steps=OBS_STEPS,
                obs_dim=OBS_DIM,
                action_horizon=HORIZON,
                action_dim=ACT_DIM,
            ),
            seed=0,
        )
