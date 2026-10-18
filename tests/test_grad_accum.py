"""Gradient accumulation groups must tile the microbatch list exactly."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def _accumulation_group_sizes(n_microbatches, accumulation_steps):
    full, remainder = divmod(n_microbatches, accumulation_steps)
    return [accumulation_steps] * full + ([remainder] if remainder else [])

CASES = [
    # (n_microbatches, accumulation_steps).
    (1, 1),
    (8, 1),  # no accumulation.
    (8, 8),  # exactly one group.
    (8, 4),  # divides evenly.
    (8, 3),  # remainder: last group is short.
    (8, 16),  # a group wider than the whole list.
    (1, 4),  # a single short group.
    (5, 2),
    (13, 5),
]


@pytest.mark.parametrize("n_microbatches,accumulation_steps", CASES)
def test_groups_tile_the_microbatch_list_exactly(n_microbatches, accumulation_steps):
    sizes = _accumulation_group_sizes(n_microbatches, accumulation_steps)
    assert sum(sizes) == n_microbatches, sizes
    assert all(s > 0 for s in sizes), sizes
    # Every group but the last is full, and there is never more than one.
    assert all(s == accumulation_steps for s in sizes[:-1]), sizes
    assert sizes[-1] <= accumulation_steps, sizes


@pytest.mark.parametrize("n_microbatches,accumulation_steps", CASES)
def test_group_boundaries_are_where_the_loop_flushes(n_microbatches, accumulation_steps):
    """A boundary falls at every cumulative sum, and always at the last one."""
    sizes = _accumulation_group_sizes(n_microbatches, accumulation_steps)
    ends, running = [], 0
    for s in sizes:
        running += s
        ends.append(running)
    assert ends[-1] == n_microbatches
    # Mirrors `is_boundary` in the update loop: the group must end on a multiple.
    for end in ends:
        assert end % accumulation_steps == 0 or end == n_microbatches, (ends, end)


@pytest.mark.parametrize("n_microbatches,accumulation_steps", [(0, 1), (4, 0), (0, 0), (-1, 2)])
def test_degenerate_counts_are_rejected(n_microbatches, accumulation_steps):
    with pytest.raises(ValueError, match="must both be >= 1"):
        _accumulation_group_sizes(n_microbatches, accumulation_steps)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))