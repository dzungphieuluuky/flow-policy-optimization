"""The buffer stores one obs window per chunk plus one frame per step."""

import inspect

import numpy as np
import pytest
import torch

from myfpo.rollout import RolloutBuffer, collect_chunks_vec

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

    return gym.vector.SyncVectorEnv([make(i) for i in range(n)])


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


def _collect(total_steps: int = 240, n: int = NUM_ENVS):
    envs = _pendulum_vec(n)
    buffer = _buffer()
    collect_chunks_vec(
        envs,
        lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
        total_steps=total_steps,
        n_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_obs_steps=OBS_STEPS,
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
        total_steps=120,
        n_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_obs_steps=OBS_STEPS,
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


def test_single_step_chunk():
    """n_exec == 1: chunk_next_state[c] is the one and only step window."""
    buf = _buffer()
    window = torch.full((OBS_STEPS, OBS_DIM), 0.25)
    next_window = torch.full((OBS_STEPS, OBS_DIM), 0.75)
    buf.add_chunk(
        state=window,
        next_state=next_window,
        action=torch.zeros(HORIZON, ACT_DIM),
        rewards=np.array([1.0], dtype=np.float32),
        normalizer_observations=next_window[-1:],
        terminated=True,
        truncated=False,
        env_id=0,
    )
    assert buf.n_steps == 1
    torch.testing.assert_close(buf.chunk_next_state[0], next_window)
    torch.testing.assert_close(buf.normalizer_observations[0], next_window[-1, :])
    torch.testing.assert_close(buf.chunk_state[0], window)


def test_reset_keeps_storage_and_clears_cursors():
    """The buffer is reused across rollouts. Storage must survive."""
    buf = _collect(total_steps=80)
    before = buf.chunk_state.clone()
    buf.reset()
    assert buf.n_chunks == 0 and buf.n_steps == 0
    assert buf.chunk_state.shape[0] == before.shape[0]
    assert torch.equal(buf.chunk_state, before), "reset() cleared storage"


def test_copy_to_matches_the_active_prefix():
    """The H2D transfer must carry every field, at the right length."""
    buf = _collect(total_steps=120)
    gpu = _buffer()
    buf.copy_to(gpu)
    active, mirrored = buf.active_prefix(), gpu.active_prefix()
    assert set(active) == set(mirrored)
    for name in active:
        assert torch.equal(active[name], mirrored[name]), name


def test_the_buffer_is_the_only_source_of_the_row_width():
    """The collector must not be able to disagree with the buffer.

    ``collect_chunks_vec`` used to take ``chunk_size`` and size its plan by it,
    while ``RolloutBuffer`` sized storage by ``action_horizon``. The two agreed
    only because the single caller passed ``cfg.action_horizon``. A second
    caller, or a swept ``action_horizon``, could have split them. One source
    removes the question.
    """
    import inspect

    from myfpo.rollout import check_chunk_dims

    params = inspect.signature(collect_chunks_vec).parameters
    assert "chunk_size" not in params, (
        "collect_chunks_vec regrew a second source for the row width"
    )

    # The horizon is required, so there is no silent fallback to the config.
    horizon = inspect.signature(check_chunk_dims).parameters["action_horizon"]
    assert horizon.default is inspect.Parameter.empty, (
        "action_horizon became optional again, which restores the second source"
    )

    # And it is the buffer's own value, not a fresh read of the config.
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
            n_action_steps=HORIZON,
            action_low=-2.0,
            action_high=2.0,
            n_obs_steps=OBS_STEPS,
            seed=0,
        )
        buf1 = _buffer()
        buf1 = _buffer()
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            total_steps=2 * HORIZON,
            buffer=buf1,
            **kwargs,
        )
        carry = buf1.carry
        buf2 = _buffer()
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            total_steps=HORIZON,
            buffer=buf2,
            carry=carry,
            **kwargs,
        )
    finally:
        envs.close()
    assert torch.equal(buf2.chunk_state[0], carry.windows[0]), (
        "the second rollout did not start from the carried window"
    )


def _collect_limited(n: int, total: int, limit: int) -> RolloutBuffer:
    """Collect from n envs whose episodes end every ``limit`` steps."""
    envs = _pendulum_vec(n=n, limit=limit)
    buffer = RolloutBuffer(
        max_chunks=total,
        max_steps=total,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    try:
        collect_chunks_vec(
            envs,
            lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
            total_steps=total,
            n_action_steps=HORIZON,
            action_low=-2.0,
            action_high=2.0,
            n_obs_steps=OBS_STEPS,
            buffer=buffer,
            seed=0,
        )
    finally:
        envs.close()
    return buffer


def test_the_chunk_rows_cover_an_episode_that_ends_every_step():
    """One chunk per step is the worst case.

    ``collect_chunks`` sizes ``max_chunks`` from ``total_steps`` for this
    reason. A cap of ``total_steps // action_horizon`` would overflow here.
    """
    buffer = _collect_limited(n=4, total=64, limit=1)
    assert buffer.n_chunks == 64
    assert buffer.n_steps == 64


def test_an_off_boundary_episode_writes_more_chunks_than_the_planned_count():
    """A 3-step episode in a 4-step chunk adds a row per episode.

    32 planned chunks over 128 steps become 44. The rows are sized from
    ``total_steps``, not from ``total_steps // action_horizon``.
    """
    buffer = _collect_limited(n=4, total=128, limit=3)
    assert buffer.n_chunks == 44
    assert buffer.n_steps == 128


def _row() -> dict:
    """One chunk's worth of arguments for ``add_chunk``."""
    window = torch.zeros(OBS_STEPS, OBS_DIM)
    return dict(
        state=window,
        next_state=window,
        action=torch.zeros(HORIZON, ACT_DIM),
        rewards=np.zeros(1, dtype=np.float32),
        normalizer_observations=torch.zeros(1, OBS_DIM),
        terminated=False,
        truncated=False,
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
    buf.add_chunk(**_row())
    buf.add_chunk(**_row())
    with pytest.raises(RuntimeError, match="chunk capacity exceeded"):
        buf.add_chunk(**_row())
    assert buf.n_chunks == 2 and buf.n_steps == 2


def test_a_failed_copy_leaves_the_destination_cursors_alone():
    """copy_to must not claim rows it failed to write.

    The counters used to be set before the transfer, so a destination too
    small reported rows of earlier data.
    """
    src = _buffer()
    src.add_chunk(**_row())
    src.add_chunk(**_row())
    small = RolloutBuffer(
        max_chunks=1,
        max_steps=1,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    with pytest.raises(RuntimeError):
        src.copy_to(small)
    assert small.n_chunks == 0 and small.n_steps == 0

