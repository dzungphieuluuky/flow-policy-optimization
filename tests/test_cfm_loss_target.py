"""The CFM target: velocity is the only surviving regression mode."""

import math

import pytest
import torch

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.objective import compute_cfm_loss

MODES = ("velocity",)


def cfm_target(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Velocity is the only regression mode; the policy inlines this expression."""
    return a.unsqueeze(1) - eps


def _cfg(**kw):
    return FlowPolicyOptimizationConfig(cfm_loss_use_huber=False, **kw)


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


def test_a_masked_mean_divides_by_the_valid_element_count():
    """The default reduction divides by valid elements, not by token count or sqrt."""
    v_hat = torch.full((1, 1, 4, 1), 2.0)
    v_star = torch.zeros(1, 1, 4, 1)
    pad = torch.tensor([[False, False, True, True]])  # 2 of 4 token positions valid

    cfg = _cfg()  # default cfm_loss_reduction is "mean"
    loss, token = compute_cfm_loss(v_hat, v_star, cfg, "sample_and_token", pad)

    # sq = 4 on the two valid positions, 0 elsewhere: sum = 8,
    # n_valid = 2 valid tokens * 1 action_dim = 2, so the loss is 4.0.
    assert float(loss[0, 0]) == pytest.approx(8.0 / 2.0)
    assert token.tolist() == [[[4.0, 4.0, 0.0, 0.0]]]


def test_the_true_sqrt_reduction_divides_by_the_square_root_of_valid_elements():
    v_hat = torch.full((1, 1, 4, 1), 2.0)
    v_star = torch.zeros(1, 1, 4, 1)
    pad = torch.tensor([[False, False, True, True]])

    cfg = _cfg(cfm_loss_reduction="sqrt")
    loss, _ = compute_cfm_loss(v_hat, v_star, cfg, "sample_and_token", pad)

    assert float(loss[0, 0]) == pytest.approx(8.0 / math.sqrt(2.0))
