"""The ratio modes of ``compute_policy_ratio``."""

from __future__ import annotations

import dataclasses

import torch

from myfpo.configuration import MyfpoConfig
from myfpo.fpo_core import compute_policy_ratio


def _cfg(**kw):
    return dataclasses.replace(MyfpoConfig(), cfm_loss_use_huber=False, **kw)


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


def test_velocity_gap_modes_are_gone():
    for mode in ("linear", "quadratic"):
        try:
            MyfpoConfig(ratio_mode=mode)
        except ValueError as exc:
            assert "ratio_mode" in str(exc)
        else:
            raise AssertionError(f"ratio_mode={mode!r} must be rejected")
