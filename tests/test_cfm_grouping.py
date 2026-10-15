"""``group_cfm_loss`` is the MC-sample half of FPO++'s granularity knobs.

``cfm_average_group_size`` folds the Monte-Carlo axis before the exponent
(``finetune_online_rl.py:803-806``): ``-1`` is one group over every sample, which
is FPO Eq. 9's "mean the MC losses, then exponentiate". ``1`` is the identity, so
per-sample ratios cost nothing.

Averaging the loss is the same as averaging the difference, since
``exp(mean(old) - mean(new)) == exp(mean(old - new))``. The test below pins that
identity rather than assuming it.
"""

import pytest
import torch

from myfpo.fpo_core import group_cfm_loss


def test_group_size_one_is_the_identity():
    loss = torch.randn(4, 8, 3)
    assert group_cfm_loss(loss, 1) is loss


def test_minus_one_puts_every_sample_in_one_group():
    loss = torch.randn(4, 8, 3)
    got = group_cfm_loss(loss, -1)
    assert got.shape == (4, 1, 3)
    assert torch.allclose(got[:, 0], loss.mean(dim=1), atol=1e-6)


@pytest.mark.parametrize("size,groups", [(2, 4), (4, 2), (8, 1)])
def test_partial_grouping_splits_the_sample_axis(size, groups):
    loss = torch.randn(3, 8)
    got = group_cfm_loss(loss, size)
    assert got.shape == (3, groups)
    for g in range(groups):
        assert torch.allclose(
            got[:, g], loss[:, g * size : (g + 1) * size].mean(dim=1), atol=1e-6
        )


def test_grouping_the_loss_equals_grouping_the_difference():
    """Why the reference can average the losses instead of the log-ratio."""
    old = torch.randn(2, 8)
    new = torch.randn(2, 8)
    grouped = group_cfm_loss(old, -1) - group_cfm_loss(new, -1)
    direct = group_cfm_loss(old - new, -1)
    assert torch.allclose(grouped, direct, atol=1e-6)


def test_a_group_size_that_does_not_divide_is_rejected():
    """The reference asserts divisibility; so does this, with the numbers."""
    with pytest.raises(ValueError, match="divide evenly"):
        group_cfm_loss(torch.randn(2, 6), 4)


def test_a_token_axis_survives_grouping():
    """Per-token ratios keep the token axis: ``[n, groups, H]``."""
    loss = torch.randn(2, 8, 5)
    assert group_cfm_loss(loss, -1).shape == (2, 1, 5)
    assert group_cfm_loss(loss, 4).shape == (2, 2, 5)