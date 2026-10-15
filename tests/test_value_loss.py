"""The value loss is plain MSE, which is what FPO++ trains.

Both released variants use the squared error: ``fpo.py:497``, and
``clip_vloss=False`` in the manipulation config. This repo used to run a Huber
whose knee tracked the critic's residual RMS (``value_huber_delta_scale``); that
was a deviation, not a port, and it is gone.

Kept here: the target broadcasts over the critic's member axis, and the loss is
reduced per chunk so the value loss lands on the right scale.
"""

import torch

from myfpo.modeling_sjfpo import compute_value_loss


def test_it_is_the_squared_error():
    pred = torch.tensor([1.0, 3.0])
    target = torch.tensor([2.0, 1.0])
    assert torch.equal(compute_value_loss(pred, target), torch.tensor([1.0, 4.0]))


def test_a_perfect_prediction_costs_nothing():
    target = torch.tensor([1.5, -2.0, 0.0])
    assert torch.equal(compute_value_loss(target, target), torch.zeros(3))


def test_the_loss_grows_quadratically_with_the_error():
    """The property the Huber knee used to blunt."""
    target = torch.zeros(1)
    small = compute_value_loss(torch.tensor([1.0]), target)
    big = compute_value_loss(torch.tensor([2.0]), target)
    assert torch.allclose(big, 4.0 * small)


def test_target_broadcasts_over_the_member_axis():
    """One return per chunk, one prediction per ensemble member."""
    pred = torch.zeros(3, 4)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = compute_value_loss(pred, target)
    assert out.shape == (3, 4)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_there_is_no_clipped_value_branch():
    """Deliberate: the release reports value clipping adds instability here.

    ``clip_vloss`` is False in FPO++ (``finetune_online_rl.py:122``) and
    ``use_clipped_value_loss`` defaults False (``rl_cfg.py:183``), because with
    FPO's tighter policy band the clipped bound inflates the penalty more often
    than it flatters it. This repo trains the plain squared error and does not
    carry the branch, so there is nothing to test beyond its absence.
    """
    import inspect

    import myfpo.modeling_sjfpo as m

    src = inspect.getsource(m.compute_value_loss)
    assert "clamp" not in src, "a clipped branch crept back into the value loss"
    assert not hasattr(m, "compute_value_huber_loss")
