"""The collector on a torch env: the done step must report the true ``o_{t+1}``.

ManiSkill resets a finished env inside the done ``step()`` and hands the real
next observation through ``info["final_observation"]``. These tests pin the shim
that swaps them, and pin the collector storing the swapped frame, without
needing SAPIEN: the fake below has ManiSkill's autoreset behavior.
"""

import numpy as np
import pytest
import torch

from myfpo.rollout import RolloutBuffer, collect_chunks_vec
from myfpo.rollout import close_chunks as original_close

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

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.state = torch.full((self.num_envs, self.obs_dim), RESET_MARKER)
        return self.state.clone(), {}

    def step(self, actions):
        action = torch.as_tensor(actions, dtype=torch.float32)
        self.steps += 1
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


def _shim(vec):
    from myfpo.maniskill_gpu import NextStepShim

    return NextStepShim(vec)


def test_done_step_reports_the_true_next_observation():
    """The one frame the collector stores must not be the reset frame."""
    vec = FakeSameStepVec(n=1, obs_dim=2, episode_len=2)
    env = _shim(vec)
    env.reset(seed=[0])
    action = torch.ones(1, 2)
    env.step_torch(action)
    obs, _, terminated, _, info = env.step_torch(action)
    assert bool(terminated[0])
    assert torch.equal(obs, info["final_observation"])
    # Two accumulated actions from the marker, not the marker itself.
    assert torch.allclose(obs, torch.full((1, 2), RESET_MARKER + 2.0))


def test_the_shim_declares_the_contract_the_collector_checks():
    """`validate_autoreset_mode` reads this attribute; ManiSkill has none."""
    env = _shim(FakeSameStepVec())
    assert env.autoreset_mode.name == "NEXT_STEP"


def test_collector_stores_the_true_terminal_frame():
    """A chunk closed by a termination stores o_{t+1}, not the reset frame."""
    env = _shim(FakeSameStepVec())
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
        n_action_steps=HORIZON,
        action_low=-1.0,
        action_high=1.0,
        n_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    n = buffer.n_chunks
    terminated = buffer.chunk_terminated[:n]
    assert bool(terminated.any()), "the fake terminates every three steps"
    assert env.vec.resets > 0, "the fake never reset, so nothing was tested"
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


FIELD_NAMES = (
    "normalizer_observations",
    "step_rewards",
    "chunk_state",
    "chunk_next_state",
    "chunk_actions",
    "chunk_starts",
    "chunk_lengths",
    "chunk_terminated",
    "chunk_truncated",
    "chunk_env_id",
)


def _collect_once(monkeypatch, *, force_legacy: bool, total_steps: int = 12):
    """One collector run on the fake env, with the writer forced either way.

    ``original_close`` is bound at module import, not read off the module here:
    a function-local import would pick up the previous call's patch, and the
    inner ``batched=True`` would win, comparing the batched writer to itself.
    """
    import myfpo.rollout as R

    forced = not force_legacy
    monkeypatch.setattr(
        R,
        "close_chunks",
        lambda *a, **kw: original_close(*a, **{**kw, "batched": forced}),
    )
    env = _shim(FakeSameStepVec())
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
        n_action_steps=HORIZON,
        action_low=-1.0,
        action_high=1.0,
        n_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    return buffer
    collect_chunks_vec(
        env,
        lambda states: torch.ones(states.shape[0], HORIZON, ACT_DIM),
        total_steps=total_steps,
        n_action_steps=HORIZON,
        action_low=-1.0,
        action_high=1.0,
        n_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    return buffer


def test_the_batched_writer_matches_the_per_chunk_writer(monkeypatch):
    """The fast close must write the same bytes as the loop it replaces.

    This is the whole safety argument for the gather: same rows, same offsets,
    same packing, so a buffer written either way is interchangeable.
    """
    fast = _collect_once(monkeypatch, force_legacy=False)
    slow = _collect_once(monkeypatch, force_legacy=True)
    assert fast.n_chunks == slow.n_chunks > 0
    assert fast.n_steps == slow.n_steps > 0
    for name in FIELD_NAMES:
        got = getattr(fast, name)[: fast.max_chunks]
        want = getattr(slow, name)[: slow.max_chunks]
        assert torch.equal(got, want), f"{name} differs between the two writers"
    assert fast.carry.awaiting == slow.carry.awaiting
    assert torch.equal(fast.carry.windows, slow.carry.windows)


def test_the_two_writers_agree_when_a_row_goes_idle(monkeypatch):
    """13 steps over 2 envs: one row hits its target first and keeps stepping.

    The batched writer must still put that row's late frames into the carry,
    or the next rollout starts from a state the env has left.
    """
    fast = _collect_once(monkeypatch, force_legacy=False, total_steps=13)
    slow = _collect_once(monkeypatch, force_legacy=True, total_steps=13)
    assert fast.carry.awaiting == slow.carry.awaiting
    assert torch.equal(fast.carry.windows, slow.carry.windows)


def test_the_legacy_writer_is_the_one_the_numpy_path_gets(monkeypatch):
    """A numpy env must never reach the gather, whatever its buffer device."""
    import myfpo.rollout as R
    from myfpo.rollout import RolloutBuffer

    gym = pytest.importorskip("gymnasium")
    calls = {"batched": 0, "chunks": 0}
    real_close = R.close_chunks
    real_add = RolloutBuffer.add_chunk

    def spy_close(*a, **kw):
        calls["batched"] += bool(kw.get("batched"))
        return real_close(*a, **kw)

    def spy_add(self, **kw):
        calls["chunks"] += 1
        return real_add(self, **kw)

    monkeypatch.setattr(R, "close_chunks", spy_close)
    monkeypatch.setattr(RolloutBuffer, "add_chunk", spy_add)

    def make_one():
        env = gym.make("Pendulum-v1")
        env.reset(seed=0)
        env.action_space.seed(0)
        return env

    envs = gym.vector.SyncVectorEnv([make_one for _ in range(NUM_ENVS)])
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
        n_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    envs.close()
    assert calls["batched"] == 0, "a numpy env reached the batched writer"
    assert calls["chunks"] == buffer.n_chunks > 0


def test_a_numpy_env_still_takes_the_numpy_path():
    """The MuJoCo path is not rewritten by the torch path: same buffer, same steps."""
    gym = pytest.importorskip("gymnasium")

    def make_one():
        env = gym.make("Pendulum-v1")
        env.reset(seed=0)
        env.action_space.seed(0)
        return env

    envs = gym.vector.SyncVectorEnv([make_one for _ in range(NUM_ENVS)])
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
        n_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    envs.close()
    assert buffer.n_chunks > 0
    assert buffer.normalizer_observations[: buffer.n_steps].shape == (buffer.n_steps, 3)
