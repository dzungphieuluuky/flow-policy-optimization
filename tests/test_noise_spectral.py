"""The spectral noise prior, on the action-step grid.

``noise_scaling_mode="freq_weighted_time"`` filters white noise with a 1/f^beta
spectral profile along the token axis. With the DCT off that axis is the chunk's
time steps, so the profile buys temporally correlated noise and the base measure
becomes ``N(0, L L^T)`` instead of ``N(0, I)``.

Two things make this cheap rather than invasive:

* ``sample_noise`` is the only definition of the base sample. ``compute_mc_xt``
  builds ``lerp(noise, action, t)`` and ``compute_target_velocity`` returns
  ``a - eps``, so a coloured ``eps`` makes the interpolant and its velocity
  target consistent with the new base measure with no change at either site.
* The base log-det is constant in theta, so it cancels in ``l_old - l_new`` and
  the FPO ratio is untouched. What the prior *does* change is the ELBO gap the
  ratio is contaminated by, which is the reason to measure it at all.

The sharpest invariant here is :func:`test_realised_spectrum_matches_the_profile`,
which is what settled the DC/Nyquist question: ``compute_freq_weighted_std``
halves bin 0 to orthonormalise the DCT-II basis, which is a basis-scaling
artefact with no counterpart in an rfft spectrum. ``spectral_filter_1d``
therefore passes ``halve_endpoints=False``.
"""

import math

import pytest
import torch

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.fpo_core import (
    compute_freq_weighted_std,
    compute_noise_band_energy,
    sample_noise,
    spectral_filter_1d,
    trajectory_straightness,
)
from spectral_joint_policy.modeling_sjfpo import compute_mc_xt

H, ACT, B, M = 8, 3, 2, 2
SHAPE = (B, M, H, ACT)


def _cfg(**kw):
    base = dict(
        action_horizon=H,
        noise_scaling_mode="freq_weighted_time",
    )
    base.update(kw)
    return SJFPOConfig(**base)


def _white(n=200_000, seed=0):
    # [n, H, 1]: the filter axis is dim=-2, so the time axis has to sit there.
    # A 2-D [n, H] helper would filter across samples instead, and still look
    # like it worked.
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, H, 1, generator=g, dtype=torch.float64)


# --- the sharpest invariant -------------------------------------------------


def test_realised_spectrum_matches_the_profile():
    """Per-bin variance must equal the requested profile squared.

    Dividing by ``horizon`` because torch's rfft is unnormalised; without that
    the assertion is off by a factor of H and would pass on the wrong constant.
    """
    beta = 1.0
    z = _white()
    y = spectral_filter_1d(z, H, beta, torch.device("cpu"))
    emp = compute_noise_band_energy(y)
    want = compute_freq_weighted_std(
        H // 2 + 1, H, torch.device("cpu"), beta, halve_endpoints=False
    ).double() ** 2
    assert emp.tolist() == pytest.approx(want.tolist(), rel=0.05)


def test_the_filter_does_not_attenuate_dc():
    """DC is the chunk's mean action, and it carries data energy.

    The halving in ``compute_freq_weighted_std`` exists to orthonormalise the
    DCT-II basis; reusing it here would silently quarter the mean-action noise.
    """
    y = spectral_filter_1d(_white(), H, 1.0, torch.device("cpu"))
    emp = compute_noise_band_energy(y)
    mag = compute_freq_weighted_std(
        H // 2 + 1, H, torch.device("cpu"), 1.0, halve_endpoints=False
    )
    halved = compute_freq_weighted_std(
        H // 2 + 1, H, torch.device("cpu"), 1.0, halve_endpoints=True
    )
    assert emp[0].item() == pytest.approx(float(mag[0] ** 2), rel=0.05)
    assert float(mag[0] ** 2) == pytest.approx(2.0 * float(halved[0] ** 2))


def test_beta_zero_is_exact_identity():
    """The ablation baseline has to be the current code, not an approximation."""
    z = _white(n=64)
    out = spectral_filter_1d(z, H, 0.0, torch.device("cpu"))
    assert out is z, "beta=0 must return the same tensor, not a copy"


# --- shape, dtype, determinism ---------------------------------------------


def test_shape_and_dtype_are_preserved():
    eps = torch.randn(*SHAPE)
    out = spectral_filter_1d(eps, H, 1.0, torch.device("cpu"))
    assert out.shape == SHAPE
    assert out.dtype == eps.dtype


def test_bf16_survives_the_round_trip():
    """Rollouts store noise in the storage dtype, which is bf16 on GPU."""
    eps = torch.randn(*SHAPE, dtype=torch.bfloat16)
    out = spectral_filter_1d(eps, H, 1.0, torch.device("cpu"))
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_is_deterministic_under_a_fixed_generator():
    cfg = _cfg(noise_scaling_beta=1.0)
    # generator=None draws from the global RNG, which advances between calls,
    # so passing one is what actually pins the draw.
    g = torch.Generator().manual_seed(7)
    a = sample_noise(SHAPE, cfg, torch.device("cpu"), g)
    g = torch.Generator().manual_seed(7)
    b = sample_noise(SHAPE, cfg, torch.device("cpu"), g)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# --- the structure the filter is supposed to create ------------------------


def _analytic_lag1(beta):
    """Lag-1 autocorrelation implied by the profile, via the Fourier sum.

    autocov(1) = (1/H)[ m0^2 + 2 sum_{k<H/2} mk^2 cos(2pi k/H) ] and
    var = (1/H)[ m0^2 + 2 sum_{k<H/2} mk^2 ]. Comparing the estimate against
    this pins the filter exactly, rather than against a guessed threshold.
    """
    m = compute_freq_weighted_std(
        H // 2 + 1, H, torch.device("cpu"), beta, halve_endpoints=False
    ).double() ** 2
    ks = torch.arange(H // 2 + 1, dtype=torch.float64)
    cos = torch.cos(2 * math.pi * ks / H)
    cov = m[0] + 2 * (m[1:-1] * cos[1:-1]).sum() + m[-1] * cos[-1]
    var = m[0] + 2 * m[1:-1].sum() + m[-1]
    return float(cov / var)


def test_temporal_correlation_increases_with_beta():
    """Lag-1 autocorrelation is the observable that separates pink from white.

    A sign or axis error here yields anti-correlated noise, which would pass
    every other test in this file.
    """
    def lag1(beta):
        z = _white()
        y = spectral_filter_1d(z, H, beta, torch.device("cpu"))
        a, b = y[:, :-1, 0], y[:, 1:, 0]
        return float(((a * b).mean()) / (y.var() + 1e-12))

    assert abs(lag1(0.0)) < 0.2, f"white should be ~0, got {lag1(0.0)}"
    assert lag1(1.0) > 0.2, f"pink should be correlated, got {lag1(1.0)}"
    # A 1/f profile over H=8 is a gentle low-pass: the achievable correlation is
    # ~0.26, not ~1. Anything stronger would mean the profile is being misapplied.
    assert lag1(1.0) == pytest.approx(_analytic_lag1(1.0), abs=0.02)
    assert lag1(2.0) > lag1(1.0) > lag1(0.0)


def test_band_energy_decays_with_frequency():
    z = _white()
    emp = compute_noise_band_energy(spectral_filter_1d(z, H, 1.0, torch.device("cpu")))
    assert emp[0] > emp[-1], "1/f profile must fall with frequency"


def test_a_zero_start_stays_zero():
    """``zero_sampling`` starts the flow at eps=0 for eval.

    A linear filter maps 0 -> 0, so the inference start point is unchanged. A
    future non-linear filter would silently alter evaluation, and this is the
    only test that would notice.
    """
    zeros = torch.zeros(*SHAPE)
    out = spectral_filter_1d(zeros, H, 1.0, torch.device("cpu"))
    assert out.abs().max().item() == 0.0


# --- integration with the CFM path -----------------------------------------


def test_beta_zero_sample_noise_matches_white():
    """The baseline must be byte-identical to the pre-existing white path."""
    cfg_white = _cfg(noise_scaling_mode="white")
    cfg_prior = _cfg(noise_scaling_mode="freq_weighted_time", noise_scaling_beta=0.0)
    g = torch.Generator().manual_seed(11)
    a = sample_noise(SHAPE, cfg_white, torch.device("cpu"), g)
    g = torch.Generator().manual_seed(11)
    b = sample_noise(SHAPE, cfg_prior, torch.device("cpu"), g)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_interpolant_still_starts_at_the_coloured_noise():
    """x_tau = tau*a + (1-tau)*L(z): the filtered sample must be the tau=0 end.

    If the filter were applied anywhere but ``sample_noise`` this would not hold,
    because the rest of the CFM path consumes the same tensor.
    """
    g = torch.Generator().manual_seed(3)
    cfg = _cfg(noise_scaling_beta=1.0)
    eps = sample_noise(SHAPE, cfg, torch.device("cpu"), g)
    actions = torch.randn(B, H, ACT)
    tau = torch.zeros(B * M, H, 1)
    flat_eps = eps.reshape(B * M, H * ACT)
    flat_act = actions[:, None].expand(B, M, H, ACT).reshape(B * M, H * ACT)
    x0 = compute_mc_xt(tau, flat_eps, flat_act, (H, ACT))
    torch.testing.assert_close(x0.reshape(B * M, H, ACT), flat_eps.reshape(B * M, H, ACT))


def test_velocity_target_is_action_minus_coloured_noise():
    """v* = a - L(z), the CFM target for base measure N(0, L L^T)."""
    from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy

    eps = torch.randn(*SHAPE)
    actions = torch.randn(B, H, ACT)
    got = SJFPOPolicy.compute_target_velocity(actions, eps)
    torch.testing.assert_close(got, actions.unsqueeze(1) - eps)


# --- straightness diagnostic ------------------------------------------------


def test_straightness_is_one_for_a_constant_field():
    v = [torch.ones(4, 5)] * 5
    assert trajectory_straightness(v) == pytest.approx(1.0)


def test_straightness_is_minus_one_for_a_reversing_field():
    v = [torch.ones(4, 5), -torch.ones(4, 5)]
    assert trajectory_straightness(v) == pytest.approx(-1.0)


def test_straightness_is_nan_for_a_single_step():
    """A 1-step sampler has no consecutive pair, so the metric cannot apply."""
    assert math.isnan(trajectory_straightness([torch.ones(2, 2)]))


def test_straightness_ignores_zero_velocity_pairs():
    v = [torch.zeros(3, 3), torch.ones(3, 3), torch.ones(3, 3)]
    assert trajectory_straightness(v) == pytest.approx(1.0)


# --- config ----------------------------------------------------------------


def test_the_prior_is_reachable_with_the_dct_off():
    cfg = _cfg(noise_scaling_mode="freq_weighted_time")
    cfg.validate()


def test_dct_domain_mode_still_requires_the_dct():
    # __post_init__ validates, so the raise happens at construction.
    with pytest.raises(ValueError, match="no longer exists"):
        _cfg(noise_scaling_mode="freq_weighted")


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="noise_scaling_mode"):
        _cfg(noise_scaling_mode="spectral")


def test_white_remains_the_default():
    """A prior that shipped as the default would invalidate every existing sweep."""
    assert SJFPOConfig().noise_scaling_mode == "white"


# --- metric wiring ---------------------------------------------------------


def _cfm_metrics(**kw):
    from spectral_joint_policy.modeling_sjfpo import compute_cfm_metrics

    n = 4
    return compute_cfm_metrics(
        l_old=torch.zeros(n),
        l_new=torch.ones(n),
        v_star_f=torch.ones(n, H, ACT),
        cfg=_cfg(),
        **kw,
    )


def test_noise_metrics_appear_when_eps_is_supplied():
    eps = sample_noise(SHAPE, _cfg(noise_scaling_beta=1.0), torch.device("cpu"), None)
    out = _cfm_metrics(eps=eps)
    assert {"noise_band_energy_low", "noise_band_energy_high",
            "noise_band_energy_hf_lf_ratio"} <= set(out)


def test_noise_metrics_are_absent_without_eps():
    """Old callers pass no eps; they must keep working unchanged."""
    assert "noise_band_energy_low" not in _cfm_metrics()


def test_every_metric_is_a_scalar():
    """compute_info packs each metric as a 0-d tensor, so a vector breaks it."""
    eps = sample_noise(SHAPE, _cfg(noise_scaling_beta=1.0), torch.device("cpu"), None)
    for key, value in _cfm_metrics(eps=eps).items():
        assert torch.as_tensor(value).reshape(()).numel() == 1, key


def test_the_ratio_reports_a_low_pass_prior_as_less_than_one():
    """Higher bands must carry less energy than lower ones under a 1/f profile."""
    eps = sample_noise(SHAPE, _cfg(noise_scaling_beta=1.0), torch.device("cpu"), None)
    assert _cfm_metrics(eps=eps)["noise_band_energy_hf_lf_ratio"] < 1.0


def test_the_ratio_is_flat_under_white_noise():
    # SHAPE holds only 2*2*3 = 12 realisations, which is not enough to estimate a
    # per-bin variance. White noise has an exactly flat spectrum, so a larger
    # draw should land on 1 to within sampling error.
    big = (64, 8, H, ACT)
    eps = sample_noise(big, _cfg(noise_scaling_mode="white"), torch.device("cpu"), None)
    assert _cfm_metrics(eps=eps)["noise_band_energy_hf_lf_ratio"] == pytest.approx(
        1.0, abs=0.05
    )


def test_the_ratio_separates_white_from_a_low_pass_prior():
    """The metric is only useful if it actually moves when the prior changes."""
    big = (64, 8, H, ACT)
    def ratio(mode):
        eps = sample_noise(big, _cfg(noise_scaling_mode=mode), torch.device("cpu"), None)
        return float(_cfm_metrics(eps=eps)["noise_band_energy_hf_lf_ratio"])
    assert ratio("freq_weighted_time") < ratio("white") - 0.1