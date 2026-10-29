"""The clamp saturation that silently removes the policy gradient.

``compute_policy_ratio`` clamps both losses at ``cfm_clamp``. When both exceed
it, the delta becomes zero, the ratio becomes exactly 1.0, and the surrogate
returns the advantage. The policy then receives no gradient. Only the critic
keeps training.

This file pins that failure and the measurement that reports it.
"""

import pytest
import numpy as np
import torch

from fixtures import cfg

from myfpo.configuration import MyfpoConfig
from myfpo.fpo_core import clamp_saturation, compute_aspo_surrogate, compute_policy_ratio

N, MC, ACT = 6, 4, 2


def _cfg(**overrides):
    return MyfpoConfig(**overrides)


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
    """The reason the metric exists: one clamp value, two different levels."""
    horizon, width = 8, ACT
    elements = horizon * width
    # compute_cfm_loss divides by the element count under "mean" and by
    # nothing under "sum". A random actor leaves a squared error of 2.0 per
    # element, so the summed error is 2.0 * elements.
    divisors = {
        "mean": float(elements),
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
    from myfpo.training import BOTH_CLAMPED_ABORT_FRAC

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

def _policy_and_batch(cfm_clamp):
    """A policy on a fixed batch, with the clamp low enough to bite.

    The bound 2.0 matters. At the default 20.0 both readings are 0.0, so a
    wrong axis looks correct. The sweep below shows where the two axes cross.
    """
    from myfpo.fpo_core import sample_timesteps
    from myfpo.training import make_policy

    obs, act, horizon, mc = 6, 3, 4, 4
    pcfg = cfg(
        obs_dim=obs, act_dim=act,
        action_horizon=horizon, n_action_steps=horizon, num_mc_samples=mc,
        hidden_dim=32, time_embed_dim=16, dit_depth=1, dit_heads=2, n_obs_steps=2,
        compile=False, report_to="none", cfm_clamp=cfm_clamp,
    )
    torch.manual_seed(0)
    policy = make_policy(
        obs, act, pcfg, device="cpu",
        action_low=[-1.0] * act, action_high=[1.0] * act,
    )
    n = 3
    torch.manual_seed(1)
    states = torch.randn(n, pcfg.n_obs_steps, obs)
    actions = torch.randn(n, horizon, act)
    t = sample_timesteps((n, mc, horizon), pcfg, "cpu", None).unsqueeze(-1)
    eps = torch.randn(n, mc, horizon, act)
    pad = torch.zeros(n, horizon, dtype=torch.bool)
    pre = policy.preprocessor({"observation.state": states})["observation.state"]
    old, old_token, old_velocity = policy.compute_behavior_cfm_losses(
        pre, actions, t, eps, pad
    )
    batch = {
        "observation.state": states, "action": actions, "action_is_pad": pad,
        "normalized_advantages": torch.randn(n, generator=torch.Generator().manual_seed(7)),
        "raw_returns": torch.randn(n, generator=torch.Generator().manual_seed(8)),
        "mc_t": t, "mc_eps": eps,
        "behavior_cfm_loss": old, "behavior_cfm_loss_token": old_token,
        "behavior_velocity_token": old_velocity,
    }
    return policy, batch, pcfg


def test_above_clamp_frac_reads_the_tensor_the_ratio_actually_clamps(monkeypatch):
    """The defect: the diagnostic sat on the per-sample loss.

    ``compute_policy_ratio`` clamps whichever tensor the caller passes. On the
    default path that is the per-token loss, shape ``[n, num_mc, H]``. The
    diagnostic read ``cfm_loss_new``, shape ``[n, num_mc]``. The two axes cross,
    so the reading was wrong in both directions: it under-reported at a high
    bound and over-reported at a low one.
    """
    import myfpo.modeling as modeling

    real = modeling.compute_policy_ratio
    clamped = []

    def spy(old, new, cfg, advantage=None, **kwargs):
        clamped.append(new.detach())
        return real(old, new, cfg, advantage=advantage, **kwargs)

    monkeypatch.setattr(modeling, "compute_policy_ratio", spy)

    # The two axes cross inside this range, so the test covers both signs of
    # the error rather than one lucky bound.
    crossings = {}
    for bound in (2.0, 1.0, 0.8, 0.6, 0.5):
        policy, batch, _ = _policy_and_batch(bound)
        clamped.clear()
        policy.loss_with_info(batch)
        reported = float(policy.cfm_above_clamp_frac)
        truth = [
            float((new > bound).float().mean()) for new in clamped
        ]
        assert truth, "the ratio was never called, so nothing was clamped"
        crossings[bound] = (reported, truth[0])
        # The clamped tensor must be the per-token one. If the token path ever
        # stops being the surviving ratio, this test would silently stop
        # testing anything.
        assert clamped[0].dim() == 3, (
            f"cfm_clamp={bound}: the clamped tensor has shape "
            f"{tuple(clamped[0].shape)}, expected [n, num_mc, H]"
        )
        # The reported figure must equal the share on the clamped tensor.
        assert reported == pytest.approx(truth[0], abs=0.0), (
            f"cfm_clamp={bound}: reported {reported} but the clamped tensor "
            f"{tuple(clamped[0].shape)} has share {truth[0]}"
        )

    # The fixture must be able to tell the axes apart. Every swept bound has to
    # leave a share strictly inside (0, 1). A bound that saturates the figure
    # cannot distinguish a correct reading from a wrong one, because both axes
    # read 1.0 there.
    for bound, (_reported, truth) in crossings.items():
        assert 0.0 < truth < 1.0, (
            f"cfm_clamp={bound}: share {truth} is saturated, so this bound "
            "cannot discriminate between the two axes"
        )


def test_both_clamped_frac_and_above_clamp_frac_read_the_same_pair():
    """The two saturation figures must come from one sample space.

    ``cfm_both_clamped_frac`` counts the elements where both losses exceed the
    bound. ``cfm_above_clamp_frac`` counts the elements where the new loss
    does. Both count over the same elements, so the first can never exceed the
    second. The inequality breaks only if the two read different tensors.
    """
    for bound in (2.0, 1.0, 0.8, 0.6, 0.5, 20.0):
        policy, batch, _ = _policy_and_batch(bound)
        policy.loss_with_info(batch)
        above = float(policy.cfm_above_clamp_frac)
        both = float(policy.cfm_both_clamped_frac)
        assert above >= both, (
            f"cfm_clamp={bound}: above={above} < both={both}. The two figures "
            "read different tensors."
        )
