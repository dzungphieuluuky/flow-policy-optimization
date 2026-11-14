"""The CFM target: velocity is the only surviving regression mode."""

import torch

from myfpo.myfpo import MyfpoConfig
from myfpo.myfpo import cfm_target, compute_cfm_loss

MODES = ("velocity",)


def _cfg(**kw):
    return MyfpoConfig(cfm_loss_use_huber=False, **kw)


def _path(n=3, m=4, k=5, d=2, seed=0):
    """A straight-path draw: ``x_t``, the exact velocity, the action, the noise."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, k, d, generator=g)
    eps = torch.randn(n, m, k, d, generator=g)
    t = torch.rand(n, m, k, 1, generator=g)
    x_t = (1.0 - t) * eps + t * a.unsqueeze(1)
    return a, eps, t, x_t, a.unsqueeze(1) - eps


def test_the_true_velocity_is_a_perfect_prediction():
    a, eps, _t, _x_t, v = _path()
    cfg = _cfg()

    pred = v
    loss = compute_cfm_loss(pred, cfm_target(a, eps), cfg, "scalar")

    assert float(loss) == 0.0


def test_a_wrong_velocity_is_caught():
    a, eps, _t, _x_t, v = _path()
    cfg = _cfg()

    pred = v + 0.5
    loss = compute_cfm_loss(pred, cfm_target(a, eps), cfg, "scalar")

    assert float(loss) > 0.0


def test_cfm_target_is_the_regression_target():
    a, eps, _t, _x_t, _v = _path()
    assert torch.equal(cfm_target(a, eps), a.unsqueeze(1) - eps)


def test_the_target_shape_is_broadcastable():
    """``cfm_target`` returns ``[n, m, k, d]``, which the reductions index."""
    n, m, k, d = 2, 3, 4, 5
    a = torch.randn(n, k, d)
    eps = torch.randn(n, m, k, d)
    assert cfm_target(a, eps).shape == (n, m, k, d)
