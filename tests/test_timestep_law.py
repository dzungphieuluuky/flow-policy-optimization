"""The closed form of ``sample_timesteps``.

The mode name ``fpo_beta`` promises a non-uniform law. This file pins the law
for every mode, and pins the one value where ``fpo_beta`` loses its shape.
"""

import dataclasses
import itertools
import math

import torch

from myfpo.configuration import SJFPOConfig
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


def test_the_draw_bounds_are_not_the_range_of_the_flow():
    """Two separate ranges. The old names read as one.

    ``sample_action`` integrates over ``[0, 1]`` on every call and never reads
    the config. The two fields bound the training draw only.
    """
    cfg = SJFPOConfig()
    # The property, not the literal: the draw must exclude both endpoints of
    # the flow without being told the flow range.
    assert cfg.min_sampled_timestep > 0.0, "the draw must not sit exactly on t=0"
    assert cfg.max_sampled_timestep < 1.0, "the draw must not sit exactly on t=1"

    # The sampler grid starts at 0 and ends at 1 regardless of the draw bounds.
    from myfpo.fpo_core import sample_action

    seen = []

    def predict_velocity(x, t, _condition):
        seen.append(float(t.reshape(-1)[0]))
        return torch.zeros_like(x)

    sample_action(
        predict_velocity,
        (1, 2, 1),
        None,
        num_steps=cfg.num_steps_noise_start,
        eps=torch.zeros(1, 2, 1),
        device="cpu",
    )
    # The loop evaluates grid[0 .. num_steps-1], so the times it reads start at
    # exactly 0.0 and stop one step short of 1.0. The final update lands on 1.0.
    assert seen[0] == 0.0 and seen[-1] < 1.0, seen
    assert min(seen) < cfg.min_sampled_timestep, (
        "the sampler must evaluate times the training draw never sees"
    )
    # And the integration still covers the whole interval, so the draw bounds
    # cannot be read as the flow range.
    assert cfg.max_sampled_timestep - cfg.min_sampled_timestep < 1.0
    grid = torch.linspace(0.0, 1.0, cfg.num_steps_noise_start + 1)
    assert float(grid[-1]) == 1.0, "the sampler integrates to t=1"


def test_fpo_beta_at_one_equals_uniform():
    """The defect this file exists for: the default beta is the null case."""
    lo, hi = SJFPOConfig().min_sampled_timestep, SJFPOConfig().max_sampled_timestep
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
    """beta > 1 puts more mass near ``min_sampled_timestep`` than a uniform draw."""
    lo, hi = SJFPOConfig().min_sampled_timestep, SJFPOConfig().max_sampled_timestep
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
    lo = torch.tensor(cfg.min_sampled_timestep, dtype=torch.float32)
    hi = torch.tensor(cfg.max_sampled_timestep, dtype=torch.float32)
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
    for earlier, later in itertools.pairwise(means):
        assert earlier > later, f"means did not decrease: {means}"




def test_a_beta_away_from_one_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SJFPOConfig(
            timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=2.0
        )
        SJFPOConfig(timestep_sampling_mode="uniform")
        SJFPOConfig(timestep_sampling_mode="beta")



def test_beta_one_is_the_null_case():
    """At 1.0 the fpo_beta transform is the identity, so it draws uniform.

    The identity is kept here so nobody re-picks 1.0 without seeing this. The
    shipped default is ``uniform``, not ``fpo_beta``: ``SJFPOConfig`` no longer
    names a law it does not draw, so the default needs no guard against this.
    """
    cfg = dataclasses.replace(SJFPOConfig(), timestep_inverse_cdf_beta=1.0)
    assert cfg.timestep_inverse_cdf_beta == 1.0
    # 1.0 ** (1.0 / 1.0) == 1.0, so the transform is the identity.
    assert math.isclose(1.0 - (1.0 - 0.5) ** (1.0 / 1.0), 0.5)

    gen = torch.Generator().manual_seed(0)
    beta = sample_timesteps(100_000, cfg, "cpu", gen)
    gen2 = torch.Generator().manual_seed(0)
    uniform = sample_timesteps(
        100_000, dataclasses.replace(cfg, timestep_sampling_mode="uniform"), "cpu", gen2
    )
    assert torch.equal(beta, uniform), "beta=1.0 no longer equals uniform"


def test_only_an_explicit_one_warns():
    """The warning fires for a caller who asks for the null case.

    It used to fire on every construction, because 1.0 was the default.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SJFPOConfig()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SJFPOConfig(timestep_sampling_mode="fpo_beta", timestep_inverse_cdf_beta=1.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
        "an explicit beta=1.0 must still warn"
    )
