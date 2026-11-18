"""The buffer stores one obs window per chunk plus one frame per step."""

import inspect

import pytest
import torch

from flow_policy_optimization.flow_policy_optimization import RolloutBuffer, collect_chunks_vec

OBS_STEPS, OBS_DIM, HORIZON, ACT_DIM = 4, 3, 4, 1
NUM_ENVS = 3


def _pendulum_vec(n: int = NUM_ENVS, limit: int | None = None):
    gym = pytest.importorskip("gymnasium")

    def make(seed: int):
        def fn():
            kwargs = {} if limit is None else {"max_episode_steps": limit}
            env = gym.make("Pendulum-v1", **kwargs)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env

        return fn

    return gym.vector.SyncVectorEnv([make(i) for i in range(n)], autoreset_mode="SameStep")


def _buffer() -> RolloutBuffer:
    """A buffer big enough for any rollout in this file."""
    return RolloutBuffer(
        max_chunks=1000,
        max_steps=4000,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )


def _collect(chunks_per_env: int = 20, n: int = NUM_ENVS):
    envs = _pendulum_vec(n)
    buffer = _buffer()
    collect_chunks_vec(
        envs,
        lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
        chunks_per_env=chunks_per_env,
        num_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        num_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    envs.close()
    return buffer


def test_shapes_match_their_consumers():
    """One row per chunk for the windows, one per step for the frames."""
    buf = _collect()
    n = buf.n_chunks
    assert buf.chunk_state[:n].shape == (n, OBS_STEPS, OBS_DIM)
    assert buf.chunk_next_state[:n].shape == (n, OBS_STEPS, OBS_DIM)
    assert buf.normalizer_observations[: buf.n_steps].shape == (buf.n_steps, OBS_DIM)


def test_step_frames_are_the_per_step_observations():
    """Every executed step contributed exactly one frame, and no padding."""
    buf = _collect()
    assert buf.n_steps == int(buf.chunk_lengths[: buf.n_chunks].sum())
    # Frames are raw observations, so they are unbounded but finite and not all identical.
    assert torch.isfinite(buf.normalizer_observations).all()
    assert buf.normalizer_observations.std(dim=0).max() > 0


def test_chunk_state_is_the_window_the_policy_saw():
    """A recording policy captures what it was handed. The buffer must match."""
    pytest.importorskip("gymnasium")
    envs = _pendulum_vec()
    buffer = _buffer()
    seen: list[torch.Tensor] = []

    def recording_policy(states: torch.Tensor) -> torch.Tensor:
        seen.append(states.detach().clone())
        return torch.zeros(states.shape[0], HORIZON, ACT_DIM)

    collect_chunks_vec(
        envs,
        recording_policy,
        chunks_per_env=10,
        num_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        num_obs_steps=OBS_STEPS,
        buffer=buffer,
        seed=0,
    )
    envs.close()

    # The policy is called once per wavefront, in the order of `need`.
    flat = torch.cat(seen, dim=0)
    assert flat.shape[0] == buffer.n_chunks
    torch.testing.assert_close(buffer.chunk_state[: buffer.n_chunks], flat)


def test_chunk_next_state_is_the_final_window():
    """chunk_next_state[c] is where chunk c ended, not where it started."""
    buf = _collect()
    starts = buf.chunk_starts[: buf.n_chunks]
    lengths = buf.chunk_lengths[: buf.n_chunks]
    # Last frame of the next-state window is the raw obs after the final step.
    last_step = (starts + lengths - 1).to(torch.long)
    torch.testing.assert_close(
        buf.chunk_next_state[: buf.n_chunks, -1, :],
        buf.normalizer_observations.index_select(0, last_step),
    )


def test_next_state_frames_agree_across_chunks():
    """The invariant that made the shift look global."""
    buf = _collect()
    env_id = buf.chunk_env_id[: buf.n_chunks].to(torch.long)

    checked = 0
    for c in range(buf.n_chunks):
        successor = [
            d
            for d in range(c + 1, buf.n_chunks)
            if int(env_id[d]) == int(env_id[c])
        ]
        if not successor:
            continue
        d = successor[0]
        # The window the next chunk of the same env started from is the window.
        torch.testing.assert_close(
            buf.chunk_state[d][-1, :],
            buf.chunk_next_state[c][-1, :],
            msg=f"chunk {c} -> {d} (env {int(env_id[c])})",
        )
        checked += 1
    assert checked > 0, "no same-env successor pairs; the test proved nothing"


def test_chunk_prev_links_each_chunk_to_its_envs_previous_chunk():
    """The buffer records GAE's adjacency at write time, not at the backup."""
    buf = _collect()
    env_id = buf.chunk_env_id[: buf.n_chunks].to(torch.long)
    prev = buf.chunk_prev[: buf.n_chunks].to(torch.long)
    last: dict[int, int] = {}
    for c in range(buf.n_chunks):
        e = int(env_id[c])
        assert int(prev[c]) == last.get(e, -1), f"chunk {c} (env {e})"
        last[e] = c
    assert len(last) > 1, "single env; the test proved nothing"


def test_single_step_chunk():
    """n_exec == 1: chunk_next_state[c] is the one and only step window."""
    buf = _buffer()
    window = torch.full((OBS_STEPS, OBS_DIM), 0.25)
    next_window = torch.full((OBS_STEPS, OBS_DIM), 0.75)
    frames = torch.zeros(1, HORIZON, OBS_DIM)
    frames[0, 0] = next_window[-1]
    rewards = torch.zeros(1, HORIZON)
    rewards[0, 0] = 1.0
    buf.write_chunks(
        start_window=window.unsqueeze(0),
        next_window=next_window.unsqueeze(0),
        actions=torch.zeros(1, HORIZON, ACT_DIM),
        lengths=torch.tensor([1], dtype=torch.int64),
        frames=frames,
        rewards=rewards,
        terminated=torch.tensor([True], dtype=torch.bool),
        truncated=torch.tensor([False], dtype=torch.bool),
        env_ids=torch.tensor([0], dtype=torch.int64),
        n_steps=1,
    )
    assert buf.n_steps == 1
    torch.testing.assert_close(buf.chunk_next_state[0], next_window)
    torch.testing.assert_close(buf.normalizer_observations[0], next_window[-1, :])
    torch.testing.assert_close(buf.chunk_state[0], window)


def test_reset_keeps_storage_and_clears_cursors():
    """The buffer is reused across rollouts. Storage must survive."""
    buf = _collect(chunks_per_env=7)
    before = buf.chunk_state.clone()
    buf.reset()
    assert buf.n_chunks == 0 and buf.n_steps == 0
    assert buf.chunk_state.shape[0] == before.shape[0]
    assert torch.equal(buf.chunk_state, before), "reset() cleared storage"


def test_the_buffer_is_the_only_source_of_the_row_width():
    """The collector must not be able to disagree with the buffer."""
    import inspect

    params = inspect.signature(collect_chunks_vec).parameters
    assert "chunk_size" not in params, (
        "collect_chunks_vec regrew a second source for the row width"
    )
    buf = RolloutBuffer(
        max_chunks=4, max_steps=8, obs_steps=2, obs_dim=OBS_DIM,
        action_horizon=HORIZON + 3, action_dim=ACT_DIM,
    )
    assert buf.action_horizon == HORIZON + 3


def test_chunks_span_rollouts_when_carry_is_passed():
    """The window one call returns is the window the next call starts from.

    No reset sits between two collected rollouts, so the first chunk of the
    second call starts from the returned carry window. The equality is exact.
    """
    pytest.importorskip("gymnasium")
    envs = _pendulum_vec(1)
    try:
        kwargs = dict(
            num_action_steps=HORIZON,
            action_low=-2.0,
            action_high=2.0,
            num_obs_steps=OBS_STEPS,
            seed=0,
        )
        buf1 = _buffer()
        buf1 = _buffer()
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            chunks_per_env=2,
            buffer=buf1,
            **kwargs,
        )
        carry = buf1.carry
        buf2 = _buffer()
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            chunks_per_env=1,
            buffer=buf2,
            carry=carry,
            **kwargs,
        )
    finally:
        envs.close()
    assert torch.equal(buf2.chunk_state[0], carry.windows[0]), (
        "the second rollout did not start from the carried window"
    )


def _collect_limited(
    n: int, chunks: int, limit: int, max_steps_per_env: int | None = None,
) -> RolloutBuffer:
    """Collect from n envs whose episodes end every ``limit`` steps."""
    envs = _pendulum_vec(n=n, limit=limit)
    steps = max_steps_per_env * n if max_steps_per_env is not None else chunks * HORIZON
    buffer = RolloutBuffer(
        max_chunks=chunks * n,
        max_steps=steps,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    try:
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            chunks,
            num_action_steps=HORIZON,
            action_low=-2.0,
            action_high=2.0,
            num_obs_steps=OBS_STEPS,
            buffer=buffer,
            seed=0,
            max_steps_per_env=max_steps_per_env,
        )
    finally:
        envs.close()
    return buffer


def test_the_chunk_rows_cover_an_episode_that_ends_every_step():
    """One chunk per step is the worst case.

    ``RolloutBuffer`` sizes ``max_chunks`` from the planned chunk count for
    this reason. A cap of ``chunks // action_horizon`` would overflow here.
    """
    buffer = _collect_limited(n=4, chunks=16, limit=1)
    assert buffer.n_chunks == 64
    assert buffer.n_steps == 64


def test_an_episode_longer_than_the_chunk_truncates_at_the_step_cap():
    """Episodes that end mid-chunk are cut by the cap, not the horizon.

    32 steps per env in 4-step chunks with a 3-step episode end at
    [3 * 10 + 2] steps. The row count does not depend on the horizon.
    """
    buffer = _collect_limited(n=4, chunks=64, limit=3, max_steps_per_env=32)
    assert buffer.n_chunks == 44
    assert buffer.n_steps == 128



def _write_one(buf: RolloutBuffer, *, env_id: int = 0) -> None:
    """One chunk's worth of arguments for ``write_chunks`` with k=1."""
    window = torch.zeros(OBS_STEPS, OBS_DIM)
    buf.write_chunks(
        start_window=window.unsqueeze(0),
        next_window=window.unsqueeze(0),
        actions=torch.zeros(1, HORIZON, ACT_DIM),
        lengths=torch.tensor([1], dtype=torch.int64),
        frames=torch.zeros(1, HORIZON, OBS_DIM),
        rewards=torch.zeros(1, HORIZON),
        terminated=torch.tensor([False], dtype=torch.bool),
        truncated=torch.tensor([False], dtype=torch.bool),
        env_ids=torch.tensor([env_id], dtype=torch.int64),
        n_steps=1,
    )


def test_the_guard_and_the_row_count_are_one_capacity():
    """A second capacity knob let the guard pass and the row write IndexError.

    ``collect_chunks_vec`` flushes on RuntimeError only, so an IndexError from
    the row write skipped the flush. max_chunks now owns both, and there is no
    second parameter to split them again.
    """
    assert "chunk_capacity" not in inspect.signature(RolloutBuffer).parameters
    buf = RolloutBuffer(
        max_chunks=2,
        max_steps=8,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    assert buf.chunk_state.shape[0] == buf.max_chunks
    assert buf.chunk_env_id.shape[0] == buf.max_chunks
    _write_one(buf)
    _write_one(buf)
    with pytest.raises(RuntimeError, match="chunk capacity exceeded"):
        _write_one(buf)
    assert buf.n_chunks == 2 and buf.n_steps == 2



