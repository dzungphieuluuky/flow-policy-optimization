"""Weak SIGReg matches covariance to identity. Collapse pays more."""

import math

import pytest
import torch

from myfpo.fpo_core import sigreg_decorr_loss, sigreg_weak_loss


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
