"""The closed form of ``sample_timesteps``.

The mode name ``fpo_beta`` promises a non-uniform law. This file pins the law
for every mode, and pins the one value where ``fpo_beta`` loses its shape.
"""

import math

import pytest
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import sample_timesteps

# Enough draws that the sampling error sits far below the shape differences
# the tests below assert on.
N = 200_000
# Two standard errors of a uniform draw on [lo, hi] at this N.
TOL = 5e-3


def _cfg(**kwargs) -> SJFPOConfig:
    return SJFPOConfig(**kwargs)


def _draw(**kwargs) -> torch.Tensor:
    cfg = _cfg(**kwargs)
    return sample_timesteps(N, cfg, "cpu", None)


def test_fpo_beta_at_one_equals_uniform():
    """The defect this file exists for: the default beta is the null case."""
    lo, hi = SJFPOConfig().min_timestep, SJFPOConfig().max_timestep
    fpo = _draw(timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=1.0)
    uni = _draw(timestep_sampling_mode="uniform")

    # The closed form, stated directly: (1 - (1 - u) ** (1 / 1.0)) == u.
    assert abs(fpo.mean().item() - (lo + hi) / 2) < TOL
    assert abs(uni.mean().item() - (lo + hi) / 2) < TOL
    assert abs(fpo.mean().item() - uni.mean().item()) < TOL
    assert abs(fpo.std().item() - uni.std().item()) < TOL


def test_fpo_beta_away_from_one_differs_from_uniform():
    beta = 2.0
    fpo = _draw(timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=beta)
    uni = _draw(timestep_sampling_mode="uniform")
    assert abs(fpo.mean().item() - uni.mean().item()) > 10 * TOL


def test_fpo_beta_leans_toward_the_lower_bound():
    """beta > 1 puts more mass near ``min_timestep`` than a uniform draw."""
    lo, hi = SJFPOConfig().min_timestep, SJFPOConfig().max_timestep
    mid = (lo + hi) / 2
    for beta in (0.5, 2.0, 4.0, 8.0):
        draw = _draw(timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=beta)
        below = float((draw < mid).float().mean())
        if beta > 1.0:
            assert below > 0.5, f"beta={beta} gave {below:.4f} of mass below mid"
        else:
            assert below < 0.5, f"beta={beta} gave {below:.4f} of mass below mid"


def test_every_mode_draws_inside_its_range():
    """Every mode draws inside the bounds. The bounds are fp32."""
    cfg = _cfg()
    # The clamp returns the fp32 neighbour of the bound. That neighbour sits
    # just below the Python float when the bound is not representable. So the
    # test compares against the fp32 value, not the Python value.
    lo = torch.tensor(cfg.min_timestep, dtype=torch.float32)
    hi = torch.tensor(cfg.max_timestep, dtype=torch.float32)
    for mode in ("uniform", "normal", "beta", "fpo_beta"):
        draw = _draw(timestep_sampling_mode=mode)
        assert float(draw.min()) >= float(lo), f"{mode} went below lo"
        assert float(draw.max()) <= float(hi), f"{mode} went above hi"


def test_fpo_beta_beta_sweep_is_monotone_in_the_mean():
    """A larger beta moves the mean down. The sweep must have an order."""
    means = [
        float(
            _draw(
                timestep_sampling_mode="fpo_beta",
                timestep_inverse_cdf_beta=beta,
            ).mean()
        )
        for beta in (0.5, 1.0, 2.0, 4.0, 8.0)
    ]
    for earlier, later in zip(means, means[1:]):
        assert earlier > later, f"means did not decrease: {means}"


def test_the_default_config_warns_about_the_null_case():
    """A bare config carries beta=1.0. The warning must name the problem."""
    with pytest.warns(RuntimeWarning, match="uniform"):
        SJFPOConfig()


def test_a_beta_away_from_one_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SJFPOConfig(
            timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=2.0
        )
        SJFPOConfig(timestep_sampling_mode="uniform")
        SJFPOConfig(timestep_sampling_mode="beta")


def test_the_default_is_the_null_case():
    """Record the defect. A later default change must move this assertion."""
    cfg = SJFPOConfig()
    assert cfg.timestep_sampling_mode == "fpo_beta"
    assert cfg.timestep_inverse_cdf_beta == 1.0
    # 1.0 ** (1.0 / 1.0) == 1.0, so the transform is the identity.
    assert math.isclose(1.0 - (1.0 - 0.5) ** (1.0 / 1.0), 0.5)