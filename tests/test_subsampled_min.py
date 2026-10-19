"""The value target is a min of two random draws, not a min over the ensemble.

With five members a full min is strictly more pessimistic than REDQ's min of
two, so the reduction itself is the thing worth pinning.
"""

from __future__ import annotations

import pytest
import torch

from myfpo.metrics import subsampled_min


def _ensemble() -> torch.Tensor:
    return torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]).repeat(1, 64)


def test_the_shape_collapses_the_member_axis():
    assert subsampled_min(_ensemble(), 2).shape == (64,)


def test_the_draw_lies_between_the_min_and_the_second_largest():
    q = _ensemble()
    got = subsampled_min(q, 2)
    ordered = q.sort(dim=0).values
    assert bool((got >= ordered[0]).all())
    assert bool((got <= ordered[-2]).all())


def test_a_full_min_is_strictly_more_pessimistic():
    q = _ensemble()
    assert subsampled_min(q, 2).mean().item() > q.amin(dim=0).mean().item()


@pytest.mark.parametrize("size", [5, 6, 9])
def test_an_ensemble_no_wider_than_the_draw_falls_back_to_a_full_min(size):
    q = _ensemble()
    assert torch.equal(subsampled_min(q, size), q.amin(dim=0))


def test_the_draw_is_independent_per_column():
    q = _ensemble()
    got = subsampled_min(q, 2)
    # Independent draws over 64 columns must not all land on the same member.
    assert got.unique().numel() > 1


def test_a_bad_rank_is_rejected():
    with pytest.raises(ValueError, match=r"q must be \[num_critics, N\]"):
        subsampled_min(torch.randn(4), 2)