"""Weak SIGReg matches covariance to identity. Collapse pays more."""

import dataclasses
import math

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import sigreg_decorr_loss, sigreg_weak_loss
from myfpo.train_sjfpo import build_policy

OBS, ACT, HORIZON, MC = 6, 3, 4, 4


def _policy_cfg(**kw):
    return dataclasses.replace(
        SJFPOConfig(
            action_horizon=HORIZON,
            num_mc_samples=MC,
            hidden_dim=32,
            time_embed_dim=16,
            dit_depth=1,
            dit_heads=2,
            n_obs_steps=2,
            compile=False,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )


def _policy_batch(policy, n=2):
    mc = policy.config.num_mc_samples
    return {
        "observation.state": torch.zeros(n, policy.config.n_obs_steps, OBS),
        "action": torch.zeros(n, HORIZON, ACT),
        "action_is_pad": torch.zeros(n, HORIZON, dtype=torch.bool),
        "normalised_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, mc, HORIZON, 1),
        "mc_eps": torch.randn(n, mc, HORIZON, ACT),
        "behavior_cfm_loss": torch.ones(n, mc),
        "behavior_cfm_loss_token": torch.ones(n, mc, HORIZON),
    }


def _policy(**kw):
    return build_policy(
        OBS, ACT, _policy_cfg(**kw), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def test_gaussian_input_pays_little():
    """Whitened rows already look Gaussian, so both penalties stay small."""
    torch.manual_seed(0)
    z = torch.randn(512, 32)
    assert float(sigreg_weak_loss(z)) < 3.0
    assert float(sigreg_decorr_loss(z)) < 3.0


def test_constant_input_pays_sqrt_dim():
    """Zero covariance leaves ||0 - I||_F, which is exactly sqrt(s)."""
    torch.manual_seed(0)
    z = torch.zeros(512, 32)
    assert float(sigreg_weak_loss(z)) == pytest.approx(math.sqrt(32))


def test_repeated_columns_pay_most():
    """Perfect correlation is the collapse the penalty exists to catch."""
    torch.manual_seed(0)
    z = torch.randn(512, 1).expand(512, 32).clone()
    assert float(sigreg_weak_loss(z)) > 10.0
    assert float(sigreg_decorr_loss(z)) > 10.0


def test_one_row_returns_zero():
    """A covariance from one row is undefined, so the penalty abstains."""
    assert float(sigreg_weak_loss(torch.randn(1, 32))) == 0.0
    assert float(sigreg_decorr_loss(torch.randn(1, 32))) == 0.0


def test_sketching_handles_wide_input():
    """128 columns sketch to 16 without error and stay finite."""
    torch.manual_seed(0)
    z = torch.randn(64, 128)
    for fn in (sigreg_weak_loss, sigreg_decorr_loss):
        assert math.isfinite(float(fn(z, 16)))


def test_zero_coef_adds_nothing():
    """Mode on at coef 0.0 matches mode off exactly, but still reports."""
    policy = _policy(sigreg_mode="weak", sigreg_coef=0.0)
    torch.manual_seed(0)
    batch = _policy_batch(policy)
    loss_on, info_on = policy(batch, compute_info=True)
    assert torch.isfinite(loss_on).all()
    policy.config = dataclasses.replace(policy.config, sigreg_mode="none")
    loss_off, info_off = policy(batch, compute_info=True)
    assert float(loss_on) == float(loss_off)
    assert info_off["sigreg_raw"] == 0.0
    assert math.isfinite(info_on["sigreg_raw"])


def test_positive_coef_runs_and_reports():
    """A live coef keeps the loss finite and the raw term visible."""
    policy = _policy(sigreg_mode="weak", sigreg_coef=1e-3)
    torch.manual_seed(0)
    loss, info = policy(_policy_batch(policy), compute_info=True)
    assert torch.isfinite(loss).all()
    assert math.isfinite(info["sigreg_raw"])


def test_penalty_reaches_the_encoder():
    """The gradient lands in encoder weights, which is the whole point."""
    policy = _policy(sigreg_mode="weak", sigreg_coef=1.0)
    torch.manual_seed(0)
    loss, _ = policy(_policy_batch(policy), compute_info=True)
    loss.backward()
    assert policy.encoder.net.fc1.weight.grad is not None
