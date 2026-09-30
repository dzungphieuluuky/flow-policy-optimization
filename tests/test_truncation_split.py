"""The rollout budget and the env's time limit must stay separate signals.

They used to share chunk_truncated. GAE reads that flag to decide whether a
chunk hands carried credit to the next chunk of the same episode, so a budget cut
was read as an episode end and the chain was cut at every rollout boundary. In
collect_chunks it was worse: done_kind == "continue" set the flag, so every chunk
of a live episode looked truncated and linked_frac was pinned at zero.
"""

import numpy as np
import torch

from spectral_joint_policy.collector import RolloutBuffer as ChunkBuffer, collect_chunks


def _buffer(n_chunks: int = 1) -> ChunkBuffer:
    return ChunkBuffer(
        max_chunks=n_chunks, max_steps=n_chunks * 4, obs_steps=1, obs_dim=2, action_dim=1, action_horizon=4
    )


def _add(buf: ChunkBuffer, **kw) -> int:
    return buf.add_chunk(
        state=torch.zeros(1, 2),
        next_state=torch.zeros(1, 2),
        action=torch.zeros(4, 1),
        rewards=torch.ones(4),
        step_frames=torch.zeros(4, 2),
        **kw,
    )


def test_mark_incomplete_does_not_flag_truncation():
    buf = _buffer()
    idx = _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)
    buf.mark_incomplete(idx)
    assert bool(buf.chunk_budget_truncated[idx]) is True
    assert bool(buf.chunk_truncated[idx]) is False
    assert bool(buf.chunk_terminated[idx]) is False


def test_env_truncation_still_flags_truncation():
    buf = _buffer()
    idx = _add(buf, terminated=False, truncated=True, clip_count=0, env_id=0)
    assert bool(buf.chunk_truncated[idx]) is True
    assert bool(buf.chunk_budget_truncated[idx]) is False


def test_survives_the_host_copy():
    """The new field must travel with the buffer, or the flag is lost on flush."""
    buf, other = _buffer(), _buffer()
    idx = _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)
    buf.mark_incomplete(idx)
    buf.copy_to(other)
    assert bool(other.chunk_budget_truncated[idx]) is True


def test_as_dict_exposes_the_both_causes():
    buf = _buffer()
    idx = _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)
    buf.mark_incomplete(idx)
    d = buf.active()
    assert "chunk_budget_truncated" in d
    assert d["chunk_budget_truncated"].shape == (1,)


# --- reuse across rollouts -------------------------------------------------
#
# reset() zeroes the cursors, not the storage, and the training path reuses one
# buffer. So every flag add_chunk writes must be written unconditionally, or the
# previous rollout's value survives on the same chunk index.


def test_add_chunk_clears_a_stale_budget_flag():
    buf = _buffer(n_chunks=2)
    first = _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)
    buf.mark_incomplete(first)

    buf.reset()
    second = _add(buf, terminated=True, truncated=False, clip_count=0, env_id=0)

    assert second == first, "same buffer, same slot; the leak needs an index match"
    assert bool(buf.chunk_budget_truncated[second]) is False


def test_a_budget_flag_does_not_survive_into_the_metric():
    """The reported frac is per-rollout, so it must fall back to zero."""
    buf = _buffer(n_chunks=1)
    buf.mark_incomplete(_add(buf, terminated=False, truncated=False, clip_count=0, env_id=0))
    assert float(buf.active()["chunk_budget_truncated"].float().mean()) == 1.0

    buf.reset()
    _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)
    assert float(buf.active()["chunk_budget_truncated"].float().mean()) == 0.0


def test_the_clear_survives_the_host_copy():
    """copy_to copies the prefix, so a stale flag there is just as wrong."""
    buf, other = _buffer(n_chunks=1), _buffer(n_chunks=1)
    buf.mark_incomplete(_add(buf, terminated=False, truncated=False, clip_count=0, env_id=0))
    buf.reset()
    _add(buf, terminated=False, truncated=False, clip_count=0, env_id=0)

    buf.copy_to(other)
    assert bool(other.chunk_budget_truncated[0]) is False


# --- the collector must not flag a termination as a budget cut -------------


class _TerminatingEnv:
    """Terminates on a fixed step, so a chunk always ends early."""

    def __init__(self, term_at: int = 3):
        self.term_at = term_at
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.full(3, float(self.t), dtype=np.float32)
        return obs, 1.0, self.t >= self.term_at, False, {}


def test_a_terminated_chunk_is_not_marked_incomplete():
    """done_kind must gate the flag: a short chunk is also an early termination."""
    buf = ChunkBuffer(
        max_chunks=8, max_steps=16, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    # 3 chunks of 3, so total_steps lands exactly and the budget never cuts one.
    collect_chunks(
        _TerminatingEnv(term_at=3),
        lambda state: torch.zeros(4, 2),
        total_steps=9,
        action_low=-1.0,
        action_high=1.0,
        n_obs_steps=2,
        n_action_steps=4,
        chunk_size=4,
        buffer=buf,
    )

    assert buf.chunk_terminated[: buf.n_chunks].all(), "every chunk ended terminated"
    assert bool(buf.chunk_budget_truncated[: buf.n_chunks].any()) is False


def test_a_budget_cut_is_still_marked_incomplete():
    """The fix must not disable the flag it was guarding."""
    buf = ChunkBuffer(
        max_chunks=8, max_steps=16, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    # total_steps=7 with n_action_steps=4: 4 + 3, so the last chunk is cut.
    collect_chunks(
        _TerminatingEnv(term_at=1000),
        lambda state: torch.zeros(4, 2),
        total_steps=7,
        action_low=-1.0,
        action_high=1.0,
        n_obs_steps=2,
        n_action_steps=4,
        chunk_size=4,
        buffer=buf,
    )

    last = buf.n_chunks - 1
    assert int(buf.chunk_lengths[last]) == 3
    assert bool(buf.chunk_budget_truncated[last]) is True
