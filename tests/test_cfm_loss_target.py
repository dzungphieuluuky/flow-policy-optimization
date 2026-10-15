"""``cfm_loss_target`` picks which endpoint the CFM loss regresses.

FPO++'s ``cfm_loss_mode`` (``flow_model.py:592-631``) has three modes: velocity
(``u``), the action endpoint (its ``x0``) and the noise endpoint (its ``eps``).
The names differ because the reference runs time the other way -- ``t=0`` is the
action there, ``t=1`` is the action here -- so a field literally called ``x0``
would mean the opposite endpoint in this repo.

The network keeps predicting velocity in every mode, so the only thing that moves
is the comparison. That buys an exact invariant: on the straight path
``x_t = (1 - t) * eps + t * a`` and ``v = a - eps``, so feeding the true velocity
to any mode must give a target-exact prediction and therefore a zero CFM loss.
"""

import dataclasses

import pytest
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import cfm_prediction, cfm_target, compute_cfm_loss

MODES = ("velocity", "action", "noise")


def _cfg(**kw):
    return dataclasses.replace(SJFPOConfig(), cfm_loss_use_huber=False, **kw)


def _path(n=3, m=4, k=5, d=2, seed=0):
    """A straight-path draw: ``x_t``, the exact velocity, the action, the noise."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, k, d, generator=g)
    eps = torch.randn(n, m, k, d, generator=g)
    t = torch.rand(n, m, k, 1, generator=g)
    x_t = (1.0 - t) * eps + t * a.unsqueeze(1)
    return a, eps, t, x_t, a.unsqueeze(1) - eps


@pytest.mark.parametrize("mode", MODES)
def test_the_true_velocity_is_a_perfect_prediction_in_every_mode(mode):
    """``x_t + (1 - t) * v == a`` and ``x_t - t * v == eps``.

    ``velocity`` compares the prediction with the target expression itself, so
    it is bit-exact. The endpoint modes reconstruct through ``(1 - t) * eps +
    t * a``, which fp32 only carries to rounding -- 1e-15 here -- so the bar is
    fp32 noise, not zero.
    """
    a, eps, t, x_t, v = _path()
    cfg = _cfg(cfm_loss_target=mode)

    pred = cfm_prediction(v, x_t, t, cfg)
    loss = compute_cfm_loss(pred, cfm_target(a, eps, cfg), cfg, "scalar")

    bar = 0.0 if mode == "velocity" else 1e-6
    assert float(loss) <= bar, f"{mode}: residual {float(loss)} at the true velocity"


@pytest.mark.parametrize("mode", MODES)
def test_a_wrong_velocity_is_caught_in_every_mode(mode):
    """The zero case must not come from a target that ignores the prediction."""
    a, eps, t, x_t, v = _path()
    cfg = _cfg(cfm_loss_target=mode)

    pred = cfm_prediction(v + 0.5, x_t, t, cfg)
    loss = compute_cfm_loss(pred, cfm_target(a, eps, cfg), cfg, "scalar")

    assert float(loss) > 0.0, f"{mode}: a wrong velocity scored zero"


def test_velocity_is_the_default_and_is_the_identity_map():
    """Parity default: the mode the release runs, and the pre-existing path."""
    assert SJFPOConfig().cfm_loss_target == "velocity"
    a, eps, t, x_t, v = _path()
    cfg = _cfg()
    assert torch.equal(cfm_prediction(v, x_t, t, cfg), v)
    assert torch.equal(cfm_target(a, eps, cfg), a.unsqueeze(1) - eps)


def test_an_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="cfm_loss_target"):
        _cfg(cfm_loss_target="x0").validate()


def test_the_target_shapes_stay_broadcastable():
    """Every mode returns ``[n, m, k, d]``, which the reductions index."""
    n, m, k, d = 2, 3, 4, 5
    a = torch.randn(n, k, d)
    eps = torch.randn(n, m, k, d)
    for mode in MODES:
        target = cfm_target(a, eps, _cfg(cfm_loss_target=mode))
        assert target.shape == (n, m, k, d), mode