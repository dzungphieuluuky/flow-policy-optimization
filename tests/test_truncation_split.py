"""The rollout budget and the env's time limit must stay separate signals.

They used to share chunk_truncated. GAE reads that flag to decide whether a
chunk hands carried credit to the next chunk of the same episode, so a budget cut
was read as an episode end and the chain was cut at every rollout boundary. In
collect_chunks it was worse: done_kind == "continue" set the flag, so every chunk
of a live episode looked truncated and linked_frac was pinned at zero.
"""

import torch

from sjfpo.collector import RolloutBuffer as ChunkBuffer


def _buffer(n_chunks: int = 1) -> ChunkBuffer:
    return ChunkBuffer(
        max_chunks=n_chunks, max_steps=n_chunks * 4, obs_steps=1, obs_dim=2, action_dim=1, action_horizon=4
    )


def _add(buf: ChunkBuffer, **kw) -> int:
    return buf.add_chunk(
        state=torch.zeros(1, 2),
        action=torch.zeros(4, 1),
        rewards=torch.ones(4),
        step_next_states=torch.zeros(4, 1, 2),
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
