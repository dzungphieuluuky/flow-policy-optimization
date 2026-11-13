"""``cfm_loss_target`` picks which endpoint the CFM loss regresses."""

import dataclasses

import pytest
import torch

from myfpo.config import MyfpoConfig
from myfpo.flow import velocity_to_target_param, cfm_target, compute_cfm_loss

MODES = ("velocity", "action", "eps")


def _cfg(**kw):
    return dataclasses.replace(MyfpoConfig(), cfm_loss_use_huber=False, **kw)


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
    """``x_t + (1 - t) * v == a`` and ``x_t - t * v == eps``."""
    a, eps, t, x_t, v = _path()
    cfg = _cfg(cfm_loss_target=mode)

    pred = velocity_to_target_param(v, x_t, t, cfg)
    loss = compute_cfm_loss(pred, cfm_target(a, eps, cfg), cfg, "scalar")

    bar = 0.0 if mode == "velocity" else 1e-6
    assert float(loss) <= bar, f"{mode}: residual {float(loss)} at the true velocity"


@pytest.mark.parametrize("mode", MODES)
def test_a_wrong_velocity_is_caught_in_every_mode(mode):
    """The zero case must not come from a target that ignores the prediction."""
    a, eps, t, x_t, v = _path()
    cfg = _cfg(cfm_loss_target=mode)

    pred = velocity_to_target_param(v + 0.5, x_t, t, cfg)
    loss = compute_cfm_loss(pred, cfm_target(a, eps, cfg), cfg, "scalar")

    assert float(loss) > 0.0, f"{mode}: a wrong velocity scored zero"


def test_velocity_is_the_default_and_is_the_identity_map():
    """Parity default: the mode the release runs, and the pre-existing path."""
    assert MyfpoConfig().cfm_loss_target == "velocity"
    a, eps, t, x_t, v = _path()
    cfg = _cfg()
    assert torch.equal(velocity_to_target_param(v, x_t, t, cfg), v)
    assert torch.equal(cfm_target(a, eps, cfg), a.unsqueeze(1) - eps)


def test_an_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="cfm_loss_target"):
        _cfg(cfm_loss_target="x0").validate()


def test_the_map_names_the_flow_network_parameter_it_stands_in_for():
    """The name must not claim the function predicts.

    ``velocity_to_target_param`` maps an output the network already produced.
    FPO++ reaches the same three endpoints by moving ``u``, its
    ``flow_network_output_param``. The new name says which parameter.
    """
    from myfpo import flow

    assert hasattr(flow, "velocity_to_target_param")
    assert not hasattr(flow, "cfm_prediction"), "the old name is still exported"

    # The map is not a prediction: it is invertible on the straight path, so the
    # same velocity always lands on the same parameter value.
    _, _, t, x_t, v = _path()
    cfg = _cfg()
    first = velocity_to_target_param(v, x_t, t, cfg)
    second = velocity_to_target_param(v.clone(), x_t.clone(), t.clone(), cfg)
    assert torch.equal(first, second)


def test_the_target_shapes_stay_broadcastable():
    """Every mode returns ``[n, m, k, d]``, which the reductions index."""
    n, m, k, d = 2, 3, 4, 5
    a = torch.randn(n, k, d)
    eps = torch.randn(n, m, k, d)
    for mode in MODES:
        target = cfm_target(a, eps, _cfg(cfm_loss_target=mode))
        assert target.shape == (n, m, k, d), mode