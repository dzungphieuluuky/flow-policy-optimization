"""Spectral prior removed."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import sample_noise


def _cfg(**kw):
    return SJFPOConfig(report_to="none", **kw)


def test_normal_mode_is_plain_randn():
    """White == torch.randn, bit for bit, same generator."""
    shape = (4, 8, 3, 3)
    got = sample_noise(shape, _cfg(), torch.device("cpu"), torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    want = torch.randn(*shape, generator=g)
    assert torch.equal(got, want)


def test_uniform_mode_scales_to_unit_variance():
    """Marginal mode is FPO++'s, not spectral."""
    g = torch.Generator().manual_seed(7)
    got = sample_noise((4096,), _cfg(noise_sampling_mode="uniform"), torch.device("cpu"), g)
    assert abs(float(got.std()) - 1.0) < 0.05
    assert float(got.abs().max()) <= 3.0


def test_unknown_marginal_mode_still_raises():
    with pytest.raises(ValueError, match="noise_sampling_mode"):
        sample_noise((2,), _cfg(noise_sampling_mode="laplace"), torch.device("cpu"), None)


def test_generated_noise_is_reproducible():
    """Rollouts draw eps from a seeded generator. Twice must match."""
    shape = (2, 3, 8, 3)
    a = sample_noise(shape, _cfg(), torch.device("cpu"), torch.Generator().manual_seed(3))
    b = sample_noise(shape, _cfg(), torch.device("cpu"), torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))