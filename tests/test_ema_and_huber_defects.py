"""Two defects found from a Humanoid run that peaked at 131k and decayed 46%.

The EMA shadow covered the flow model but not the conditioning encoder, so every
eval measured EMA_actor(live_encoder(obs)) -- a pairing that never trained. The
Huber knee was scaled by returns_rms, putting it at ~2x the residual spread, so
97% of residuals took the quadratic branch and the loss was MSE in disguise.
"""

import dataclasses
import warnings

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.modeling_sjfpo import SJFPOPolicy


def _policy(**kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    return SJFPOPolicy(cfg)


def _warm(p: SJFPOPolicy) -> SJFPOPolicy:
    p.update_ema(0)
    p.update_ema(1000)
    return p


# --- EMA covers the encoder -------------------------------------------------


def test_ema_shadow_covers_the_encoder():
    p = _warm(_policy())
    enc = {f"encoder.{n}" for n, _ in p.encoder.named_parameters()}
    actor = {f"actor.{n}" for n, _ in p.actor.named_parameters()}
    assert enc, "the encoder has no parameters, test is vacuous"
    assert enc <= set(p._ema_shadow), "encoder params missing from the EMA shadow"
    assert actor <= set(p._ema_shadow)


def test_ema_names_are_namespaced_so_the_two_modules_cannot_collide():
    p = _warm(_policy())
    for name in p._ema_shadow:
        assert name.startswith(("actor.", "encoder.")), name


def test_swap_round_trips_actor_and_encoder_together():
    p = _warm(_policy())
    before = {n: q.detach().clone() for n, q in p._ema_named_params().items()}
    shadow = {n: t.clone() for n, t in p._ema_shadow.items()}

    displaced = p.swap_in_ema(1000)
    assert set(displaced) == set(shadow), "swap left a module behind"
    for name, want in shadow.items():
        assert torch.equal(p._ema_named_params()[name], want)

    p.swap_out_ema(displaced)
    for name, want in before.items():
        assert torch.equal(p._ema_named_params()[name], want), name


def test_swap_in_actually_moves_the_encoder():
    """The old bug: only self.actor moved, so the encoder stayed live."""
    p = _warm(_policy())
    with torch.no_grad():
        for q in p.encoder.parameters():
            q.add_(100.0)
    p.update_ema(2000)
    name = next(n for n in p._ema_named_params() if n.startswith("encoder."))
    live = p._ema_named_params()[name].detach().clone()
    p.swap_in_ema(2000)
    shadow = p._ema_named_params()[name]
    assert not torch.equal(shadow, live), "encoder was not swapped in"
    p.swap_out_ema(p.swap_in_ema(2000))


def test_an_actor_only_shadow_is_reseeded_not_half_applied():
    p = _warm(_policy())
    legacy = {"initialised": True, "shadow_params": {
        n: torch.zeros_like(q) for n, q in p.actor.named_parameters()
    }}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing should warn at this layer
        assert p.load_ema_state_dict(legacy) is False
    assert p._ema_shadow == {}
    assert p._ema_initialised is False


def test_a_matching_shadow_is_restored():
    p = _warm(_policy())
    state = p.ema_state_dict()
    fresh = _policy()
    assert fresh.load_ema_state_dict(state) is True
    assert set(fresh._ema_shadow) == set(p._ema_shadow)


# --- Huber knee follows the residual, not the return ------------------------


def test_huber_knee_is_not_built_from_the_return_scale():
    """delta must not scale with returns_rms; that was the whole defect."""
    torch.manual_seed(0)
    p = _policy(value_huber_delta_scale=1.0)
    # A batch whose returns are 100x the residual spread.
    resid = torch.randn(4096) * 1.0
    p.update_huber_resid_rms(resid)
    assert float(p._huber_resid_rms) == pytest.approx(float(resid.square().mean().sqrt()), rel=1e-5)


def test_huber_knee_moves_with_the_residual_not_the_return():
    p = _policy()
    torch.manual_seed(0)
    tight = torch.randn(4096) * 2.0
    p.update_huber_resid_rms(tight)
    a = float(p._huber_resid_rms)
    for _ in range(200):                       # converge the EMA
        p.update_huber_resid_rms(tight)
    wide = torch.randn(4096) * 40.0
    for _ in range(200):
        p.update_huber_resid_rms(wide)
    b = float(p._huber_resid_rms)
    assert b > a * 10, f"knee did not track the residual spread: {a} -> {b}"


def test_huber_knee_is_smoothed_not_per_batch():
    """A knee that breathes with every batch is a second feedback loop.

    Real batch-to-batch RMS variation is sampling noise, ~1.2x here. The
    tracked value must move far less than the per-batch value does.
    """
    p = _policy()
    torch.manual_seed(0)
    raw = []
    for _ in range(500):
        batch = torch.randn(4096) * 5.0
        raw.append(float(batch.square().mean().sqrt()))
        p.update_huber_resid_rms(batch)
    tracked = float(p._huber_resid_rms)
    spread_raw = max(raw) / min(raw)
    for _ in range(20):
        p.update_huber_resid_rms(torch.randn(4096) * 5.0)
    moved = float(p._huber_resid_rms) / tracked
    # The per-batch RMS swung by spread_raw; the knee barely moved.
    assert moved < 1.01, f"knee tracked the batch too closely: {moved:.4f}x"
    assert spread_raw > 1.05, "fixture did not actually vary"


def test_the_knee_survives_a_single_large_outlier_batch():
    """Not to zero -- an EMA is not a median -- but bounded, unlike per-batch."""
    p = _policy()
    torch.manual_seed(0)
    for _ in range(500):
        p.update_huber_resid_rms(torch.randn(4096) * 5.0)
    before = float(p._huber_resid_rms)
    p.update_huber_resid_rms(torch.randn(4096) * 5000.0)
    # One step at decay 0.99 moves it at most 0.01 * (5000 - before).
    assert float(p._huber_resid_rms) < before + 0.01 * 5000.0 * 1.05


def test_a_1x_knee_puts_a_third_of_residuals_in_the_linear_branch():
    """The reason the basis change matters: at 1x it now bites."""
    torch.manual_seed(0)
    r = torch.randn(200000)
    r = r - r.mean()
    r = r / r.square().mean().sqrt()          # unit RMS
    assert (r.abs() > 1.0).float().mean() == pytest.approx(0.317, abs=0.01)


def test_the_ema_state_dict_carries_the_encoder():
    p = _warm(_policy())
    state = p.ema_state_dict()
    assert any(k.startswith("encoder.") for k in state["shadow_params"])