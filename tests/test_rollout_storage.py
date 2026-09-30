"""The buffer stores one obs window per chunk plus one frame per step.

Three shapes, three consumers, and nothing else may be stored:

    chunk_state      [C, obs_steps, obs_dim]   the window the policy saw
    chunk_next_state [C, obs_steps, obs_dim]   the window the chunk ran to
    step_frames      [T, obs_dim]              raw obs after each executed step

The old layout kept ``[T, obs_steps, obs_dim]`` for the first two, so 15/16 of
every row and 15/16 of all rows went unread. These tests pin the invariants that
make the split safe, and they pin them against the real collector, not a
synthetic buffer, because the layout is chunk-major with envs interleaved.
"""

import numpy as np
import pytest
import torch

from spectral_joint_policy.collector import RolloutBuffer, collect_chunks_vec

OBS_STEPS, OBS_DIM, HORIZON, ACT_DIM = 4, 3, 4, 1
NUM_ENVS = 3


def _pendulum_vec(n: int = NUM_ENVS):
    gym = pytest.importorskip("gymnasium")

    def make(seed: int):
        def fn():
            env = gym.make("Pendulum-v1")
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env

        return fn

    return gym.vector.SyncVectorEnv([make(i) for i in range(n)])


def _collect(total_steps: int = 240, n: int = NUM_ENVS):
    envs = _pendulum_vec(n)
    buffer = RolloutBuffer(
        max_chunks=1000,
        max_steps=4000,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    collect_chunks_vec(
        envs,
        lambda s: torch.zeros(s.shape[0], HORIZON, ACT_DIM),
        total_steps=total_steps,
        n_action_steps=HORIZON,
        action_low=-2.0,
        action_high=2.0,
        n_obs_steps=OBS_STEPS,
        chunk_size=HORIZON,
        buffer=buffer,
        seed=0,
    )
    envs.close()
    return buffer


def test_shapes_match_their_consumers():
    """One row per chunk for the windows, one per step for the frames.

    The chunk tensors are allocated at chunk_capacity, not at n_chunks, because
    the chunk count is only known once the rollout ends. So assert the written
    prefix, not the allocation.
    """
    buf = _collect()
    n = buf.n_chunks
    assert buf.chunk_state[:n].shape == (n, OBS_STEPS, OBS_DIM)
    assert buf.chunk_next_state[:n].shape == (n, OBS_STEPS, OBS_DIM)
    assert buf.step_frames[: buf.n_steps].shape == (buf.n_steps, OBS_DIM)


def test_step_frames_are_the_per_step_observations():
    """Every executed step contributed exactly one frame, and no padding."""
    buf = _collect()
    assert buf.n_steps == int(buf.chunk_lengths[: buf.n_chunks].sum())
    # Frames are raw observations, so they are unbounded but finite and not
    # all identical. A zero-filled tail would show up as a constant block.
    assert torch.isfinite(buf.step_frames).all()
    assert buf.step_frames.std(dim=0).max() > 0


def test_chunk_state_is_the_window_the_policy_saw():
    """A recording policy captures what it was handed; the buffer must match."""
    gym = pytest.importorskip("gymnasium")
    envs = _pendulum_vec()
    buffer = RolloutBuffer(
        max_chunks=1000,
        max_steps=4000,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
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
        chunk_size=HORIZON,
        buffer=buffer,
        seed=0,
    )
    envs.close()

    # The policy is called once per wavefront, in the order of `need`. Flattening
    # those calls in call order must line up with the chunks in buffer order.
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
        buf.step_frames.index_select(0, last_step),
    )


def test_next_state_frames_agree_across_chunks():
    """The invariant that made the shift look global. Wavefront order means
    adjacent chunks belong to different envs, so a naive ``t + 1`` index over the
    flat step axis is wrong at every chunk boundary. This is the test that
    catches a regression to exactly that mistake.
    """
    buf = _collect()
    starts = buf.chunk_starts[: buf.n_chunks]
    lengths = buf.chunk_lengths[: buf.n_chunks]
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
        # The window the next chunk of the same env started from is the window
        # the previous chunk of that env ran to.
        torch.testing.assert_close(
            buf.chunk_state[d][-1, :],
            buf.chunk_next_state[c][-1, :],
            msg=f"chunk {c} -> {d} (env {int(env_id[c])})",
        )
        checked += 1
    assert checked > 0, "no same-env successor pairs; the test proved nothing"


def test_single_step_chunk():
    """n_exec == 1: chunk_next_state[c] is the one and only step window.

    tests/test_gae.py has a chunk_starts=zeros(1) case, but that is n_chunks=1,
    not n_exec=1. This path is the one that a ``[:, -1, :]`` slice or an
    off-by-one in add_chunk would break silently.
    """
    buf = RolloutBuffer(
        max_chunks=4,
        max_steps=8,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    window = torch.full((OBS_STEPS, OBS_DIM), 0.25)
    next_window = torch.full((OBS_STEPS, OBS_DIM), 0.75)
    buf.add_chunk(
        state=window,
        next_state=next_window,
        action=torch.zeros(HORIZON, ACT_DIM),
        rewards=np.array([1.0], dtype=np.float32),
        step_frames=next_window[-1:],
        terminated=True,
        truncated=False,
        clip_count=0,
        env_id=0,
    )
    assert buf.n_steps == 1
    torch.testing.assert_close(buf.chunk_next_state[0], next_window)
    torch.testing.assert_close(buf.step_frames[0], next_window[-1, :])
    torch.testing.assert_close(buf.chunk_state[0], window)


def test_reset_keeps_storage_and_clears_cursors():
    """The buffer is reused across rollouts; storage must survive."""
    buf = _collect(total_steps=80)
    before = buf.chunk_state.clone()
    buf.reset()
    assert buf.n_chunks == 0 and buf.n_steps == 0
    assert buf.chunk_state.shape[0] == before.shape[0]
    assert torch.equal(buf.chunk_state, before), "reset() cleared storage"


def test_written_rows_survive_capacity_growth():
    """Growth reallocates every chunk tensor, so an old row must still be there.

    A chunk shorter than the horizon is what forces the growth in the first
    place, and it arrives after the rows already written.
    """
    buf = RolloutBuffer(
        max_chunks=64,
        max_steps=256,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
        chunk_capacity=4,
    )
    before_capacity = buf.chunk_capacity
    rows = []
    for c in range(before_capacity + 5):
        state = torch.full((OBS_STEPS, OBS_DIM), float(c))
        nxt = torch.full((OBS_STEPS, OBS_DIM), float(c) + 0.5)
        buf.add_chunk(
            state=state,
            next_state=nxt,
            action=torch.zeros(HORIZON, ACT_DIM),
            rewards=np.full(HORIZON, float(c), dtype=np.float32),
            step_frames=nxt[-1:].expand(HORIZON, OBS_DIM).contiguous(),
            terminated=False,
            truncated=False,
            clip_count=0,
            env_id=0,
        )
        rows.append((state, nxt))
    assert buf.chunk_capacity > before_capacity, "test did not force a growth"
    for c, (state, nxt) in enumerate(rows):
        torch.testing.assert_close(buf.chunk_state[c], state)
        torch.testing.assert_close(buf.chunk_next_state[c], nxt)


def test_copy_to_grows_the_destination():
    """copy_to must not truncate a source that outgrew the destination."""
    src = RolloutBuffer(
        max_chunks=64,
        max_steps=256,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    dst = RolloutBuffer(
        max_chunks=64,
        max_steps=256,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
        chunk_capacity=1,
    )
    nxt = torch.full((OBS_STEPS, OBS_DIM), 0.5)
    for c in range(10):
        src.add_chunk(
            state=torch.full((OBS_STEPS, OBS_DIM), float(c)),
            next_state=nxt,
            action=torch.zeros(HORIZON, ACT_DIM),
            rewards=np.ones(HORIZON, dtype=np.float32),
            step_frames=nxt[-1:].expand(HORIZON, OBS_DIM).contiguous(),
            terminated=False,
            truncated=False,
            clip_count=0,
            env_id=0,
        )
    src.copy_to(dst)
    assert dst.chunk_capacity >= src.n_chunks
    torch.testing.assert_close(dst.chunk_state[: src.n_chunks], src.chunk_state[: src.n_chunks])


def test_copy_to_matches_the_active_prefix():
    """The H2D transfer must carry every field, at the right length."""
    buf = _collect(total_steps=120)
    gpu = RolloutBuffer(
        max_chunks=buf.max_chunks,
        max_steps=buf.max_steps,
        obs_steps=OBS_STEPS,
        obs_dim=OBS_DIM,
        action_horizon=HORIZON,
        action_dim=ACT_DIM,
    )
    buf.copy_to(gpu)
    active, mirrored = buf.active(), gpu.active()
    assert set(active) == set(mirrored)
    for name in active:
        assert torch.equal(active[name], mirrored[name]), name
