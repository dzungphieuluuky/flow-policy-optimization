"""The ratio modes of ``compute_policy_ratio``.

The velocity modes are checked against the CFM mode on the same flow point.
Where the algebra says "equal", the check is ``atol=1e-12`` on float64.
"""

from __future__ import annotations

import dataclasses

import torch

from myfpo.configuration import MyfpoConfig
from myfpo.fpo_core import cfm_loss_weight, compute_cfm_loss, compute_policy_ratio


def _cfg(**kw):
    return dataclasses.replace(MyfpoConfig(), cfm_loss_use_huber=False, **kw)


def _flow(seed: int = 0):
    """A flow point in float64, with a small velocity step."""
    g = torch.Generator().manual_seed(seed)
    v_old = torch.randn(2, 4, 3, 5, generator=g, dtype=torch.float64)
    target = torch.randn(2, 4, 3, 5, generator=g, dtype=torch.float64)
    v_new = v_old + 0.01 * torch.randn(2, 4, 3, 5, generator=g, dtype=torch.float64)
    t = torch.rand(2, 4, 3, 1, generator=g, dtype=torch.float64)
    return v_old, v_new, target, t


def test_cfm_is_the_default_mode():
    assert MyfpoConfig().ratio_mode == "cfm"


def test_cfm_mode_keeps_the_published_formula():
    cfg = _cfg()
    old = torch.tensor([[1.0, 2.0]])
    new = torch.tensor([[0.5, 3.0]])
    got = compute_policy_ratio(old, new, cfg)
    want = torch.exp(
        torch.clamp(old, max=cfg.cfm_clamp) - torch.clamp(new, max=cfg.cfm_clamp)
    )
    assert torch.equal(got, want)


def test_cfm_raw_mode_ignores_the_clamp():
    raw = _cfg(cfm_clamp=0.5, ratio_mode="cfm_raw")
    clamped = dataclasses.replace(raw, ratio_mode="cfm")
    old = torch.full((1, 2), 2.0)
    new = torch.full((1, 2), 3.0)

    # The raw delta is -1. The clamped mode pins both losses at 0.5 and gives 1.
    assert torch.allclose(
        compute_policy_ratio(old, new, raw),
        torch.full((1, 2), torch.exp(torch.tensor(-1.0)).item()),
    )
    assert torch.equal(compute_policy_ratio(old, new, clamped), torch.ones(1, 2))


def test_velocity_modes_need_the_flow_tensors():
    cfg = _cfg(ratio_mode="linear")
    try:
        compute_policy_ratio(torch.ones(1, 1, 1), torch.ones(1, 1, 1), cfg)
    except ValueError as exc:
        assert "velocity_old" in str(exc)
    else:
        raise AssertionError("a velocity mode without velocities must raise")


def test_quadratic_matches_cfm_per_token():
    """``L_old - L_new = -2<r, delta> - |delta|^2`` on the same flow point."""
    cfg = _cfg()
    v_old, v_new, target, t = _flow()
    weight = cfm_loss_weight(t, cfg)
    _, old_token = compute_cfm_loss(v_old, target, cfg, "sample_and_token", weight=weight)
    _, new_token = compute_cfm_loss(v_new, target, cfg, "sample_and_token", weight=weight)

    want = compute_policy_ratio(old_token, new_token, cfg)
    got = compute_policy_ratio(
        old_token,
        new_token,
        dataclasses.replace(cfg, ratio_mode="quadratic"),
        velocity_old=v_old,
        velocity_new=v_new,
        target=target,
        token_weight=weight,
    )
    assert want.shape == got.shape
    assert torch.allclose(got, want, rtol=0.0, atol=1e-12)


def test_quadratic_matches_cfm_after_mc_grouping():
    cfg = _cfg()
    v_old, v_new, target, t = _flow(seed=1)
    weight = cfm_loss_weight(t, cfg)
    old_sample, _ = compute_cfm_loss(v_old, target, cfg, "sample_and_token", weight=weight)
    new_sample, _ = compute_cfm_loss(v_new, target, cfg, "sample_and_token", weight=weight)

    want = compute_policy_ratio(old_sample, new_sample, cfg, group_size=2)
    got = compute_policy_ratio(
        old_sample,
        new_sample,
        dataclasses.replace(cfg, ratio_mode="quadratic"),
        velocity_old=v_old,
        velocity_new=v_new,
        target=target,
        token_weight=weight,
        group_size=2,
    )
    assert got.shape == want.shape == (2, 2)
    assert torch.allclose(got, want, rtol=0.0, atol=1e-12)


def test_linear_mode_is_signed_and_can_exceed_one():
    cfg = _cfg(ratio_mode="linear")
    v_old = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    target = torch.ones_like(v_old)
    # Move halfway toward the target: the first-order log-ratio is positive.
    v_new = v_old - 0.5 * (v_old - target)
    got = compute_policy_ratio(
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        cfg,
        velocity_old=v_old,
        velocity_new=v_new,
        target=target,
    )
    assert float(got) > 1.0


def test_every_mode_returns_one_when_the_velocity_does_not_move():
    for mode in ("cfm", "cfm_raw", "linear", "quadratic"):
        cfg = _cfg(ratio_mode=mode)
        old = torch.full((1, 1, 1), 3.0)
        v = torch.zeros(1, 1, 1, 2)
        got = compute_policy_ratio(
            old, old.clone(), cfg, velocity_old=v, velocity_new=v, target=v
        )
        assert torch.equal(got, torch.ones_like(got)), mode


def test_velocity_modes_mask_padded_tokens():
    cfg = _cfg(ratio_mode="linear")
    v_old = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    target = torch.ones_like(v_old)
    v_new = v_old - 0.5 * (v_old - target)
    pad = torch.tensor([[False, True]])
    got = compute_policy_ratio(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        cfg,
        velocity_old=v_old,
        velocity_new=v_new,
        target=target,
        action_is_pad=pad,
    )
    assert float(got[0, 0, 0]) > 1.0
    assert torch.equal(got[0, 0, 1], torch.tensor(1.0, dtype=got.dtype))


def test_velocity_mode_runs_end_to_end():
    """The stored behaviour velocity reaches the ratio through ``policy_loss``."""
    from fixtures import make_batch, make_policy

    policy = make_policy(ratio_mode="linear")
    batch = make_batch(
        policy,
        obs_dim=5,
        action_dim=2,
        horizon=policy.config.action_horizon,
    )
    loss = policy(batch, check_finite=False)
    assert torch.isfinite(loss).all()
