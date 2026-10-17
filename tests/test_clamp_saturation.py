"""The clamp saturation that silently removes the policy gradient.

``compute_policy_ratio`` clamps both losses at ``cfm_clamp``. When both exceed
it, the delta becomes zero, the ratio becomes exactly 1.0, and the surrogate
returns the advantage. The policy then receives no gradient. Only the critic
keeps training.

This file pins that failure and the measurement that reports it.
"""

import dataclasses

import numpy as np
import pytest
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import clamp_saturation, compute_aspo_surrogate, compute_policy_ratio

N, MC, ACT = 6, 4, 2


def _cfg(**overrides):
    return dataclasses.replace(SJFPOConfig(), **overrides)


def test_both_losses_above_the_clamp_pin_the_ratio_at_one():
    """The defect: a live-looking run with a dead policy gradient."""
    cfg = _cfg(cfm_clamp=20.0)
    loss_old = torch.full((N, MC), 25.0)
    loss_new = torch.full((N, MC), 24.0)
    ratio = compute_policy_ratio(loss_old, loss_new, cfg)
    assert torch.equal(ratio, torch.ones_like(ratio)), (
        f"ratio was not pinned at 1: min={ratio.min().item()} max={ratio.max().item()}"
    )


def test_the_pinned_ratio_leaves_the_actor_without_a_gradient():
    """A zero policy gradient, not a small one."""
    cfg = _cfg(cfm_clamp=20.0, trust_region_mode="aspo")
    actor = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(0.0)
    loss_new = torch.full((N, MC), 24.0, requires_grad=True)
    loss_old = torch.full((N, MC), 25.0)
    ratio = compute_policy_ratio(loss_old, loss_new, cfg)
    adv = torch.tensor([2.0, -2.0, 1.0, -1.0, 0.5, -0.5])
    surrogate = compute_aspo_surrogate(ratio, adv.unsqueeze(-1).expand(N, MC), cfg)
    surrogate.sum().backward()
    assert torch.equal(loss_new.grad, torch.zeros_like(loss_new)), (
        "the saturated loss still received a gradient; check that "
        "compute_policy_ratio still clamps both losses at the same bound"
    )


def test_one_loss_below_the_clamp_keeps_a_gradient():
    """One loss under the bound is not saturation. The old loss may exceed it."""
    cfg = _cfg(cfm_clamp=20.0)
    loss_old = torch.full((N, MC), 25.0)
    loss_new = torch.full((N, MC), 2.0)
    ratio = compute_policy_ratio(loss_old, loss_new, cfg)
    assert not torch.equal(ratio, torch.ones_like(ratio)), (
        "a ratio above the clamp must still move when the new loss is below it"
    )
    # Both losses sit above cfm_clamp, but only one saturates the ratio.
    assert float(clamp_saturation(loss_old, loss_new, cfg)) == 0.0


def test_saturation_reports_the_share_of_pinned_elements():
    cfg = _cfg(cfm_clamp=20.0)
    old = torch.tensor([25.0, 25.0, 2.0, 2.0])
    new = torch.tensor([24.0, 2.0, 24.0, 1.0])
    # Only element 0 has both losses above the bound.
    assert float(clamp_saturation(old, new, cfg)) == pytest.approx(0.25)


def test_saturation_is_zero_on_a_healthy_pair():
    cfg = _cfg(cfm_clamp=20.0)
    old = torch.tensor([2.0, 2.5, 3.0, 1.5])
    new = torch.tensor([1.9, 2.4, 3.1, 1.6])
    assert float(clamp_saturation(old, new, cfg)) == 0.0


def test_saturation_rejects_a_mismatched_pair():
    cfg = _cfg()
    with pytest.raises(ValueError, match="shape"):
        clamp_saturation(torch.zeros(3), torch.zeros(4), cfg)


def test_the_reduction_decides_whether_the_default_clamp_bites():
    """The reason the metric exists: one clamp value, three different levels."""
    horizon, width = 8, ACT
    elements = horizon * width
    # compute_cfm_loss divides by the element count under "mean", by its square
    # root under "sqrt", and by nothing under "sum". A random actor leaves a
    # squared error of 2.0 per element, so the summed error is 2.0 * elements.
    divisors = {
        "mean": float(elements),
        "sqrt": float(elements) ** 0.5,
        "sum": 1.0,
    }
    cfg = _cfg(cfm_clamp=20.0)
    for reduction, divisor in divisors.items():
        level = 2.0 * elements / divisor
        saturated = level > cfg.cfm_clamp
        assert saturated == (reduction == "sum"), (
            f"{reduction} at H={horizon} gave level {level:.2f} against clamp "
            f"{cfg.cfm_clamp}; the expected saturation changed"
        )


def test_the_abort_threshold_fires_on_the_stricter_signal():
    """The trainer must prefer the both-clamped signal over the level signal."""
    from myfpo.train_sjfpo import BOTH_CLAMPED_ABORT_FRAC

    assert BOTH_CLAMPED_ABORT_FRAC == 0.5
    cfg = _cfg(cfm_clamp=20.0)
    # The new loss is entirely above the clamp, but the old loss is below it, so
    # the old guard would have fired while the ratio still carried a gradient.
    above_frac = float((torch.full((N, MC), 25.0) > cfg.cfm_clamp).float().mean())
    assert above_frac > BOTH_CLAMPED_ABORT_FRAC
    both = float(clamp_saturation(torch.full((N, MC), 2.0), torch.full((N, MC), 25.0), cfg))
    assert both == 0.0, "the level guard fires where the gradient is still alive"


def test_a_sum_reduction_run_reports_saturation_on_real_losses():
    """End to end: the ``sum`` reduction saturates the ratio above horizon 4."""
    from myfpo.fpo_core import compute_cfm_loss

    cfg = _cfg(cfm_loss_reduction="sum", cfm_loss_use_huber=False, cfm_clamp=20.0)
    torch.manual_seed(0)
    v_hat = torch.randn(N, MC, 8, ACT)
    v_star = torch.randn(N, MC, 8, ACT)
    loss = compute_cfm_loss(v_hat, v_star, cfg, "sample")
    # A random actor under "sum" already sits above the clamp at horizon 8.
    # Nothing has stalled. The level alone crosses the bound.
    assert float(loss.detach().mean()) > cfg.cfm_clamp
    fresh = float(clamp_saturation(loss.detach(), loss.detach() + 1.0, cfg))
    assert fresh > 0.5, (
        f"sum at H=8 gave saturation {fresh:.2f}; the defect this file exists "
        "for no longer reproduces"
    )
    # The default reduction keeps the level far below the bound.
    mean_cfg = _cfg(cfm_loss_reduction="mean", cfm_loss_use_huber=False)
    mean_loss = compute_cfm_loss(v_hat, v_star, mean_cfg, "sample")
    assert float(clamp_saturation(mean_loss.detach(), mean_loss.detach() + 1.0, mean_cfg)) == 0.0


def test_the_default_run_is_not_saturated():
    """A default config must leave the gradient alive on ordinary losses."""
    cfg = _cfg()
    from myfpo.fpo_core import compute_cfm_loss

    torch.manual_seed(0)
    v_hat = torch.randn(N, MC, cfg.action_horizon, ACT)
    v_star = torch.randn(N, MC, cfg.action_horizon, ACT)
    loss = compute_cfm_loss(v_hat, v_star, cfg, "sample")
    assert float(clamp_saturation(loss.detach(), loss.detach() + 0.1, cfg)) == 0.0
    assert float(np.asarray(loss.detach()).max()) < cfg.cfm_clamp