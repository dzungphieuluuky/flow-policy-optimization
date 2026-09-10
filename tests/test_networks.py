"""Unit tests for src/mfpo/networks.py (plan task 4).

TDD: written BEFORE the implementation exists — the first pytest run must
fail RED on the missing module, then go GREEN once networks.py lands.

Conventions under test (binding, from the plan):
- fp32 everywhere; ELU activations; no BatchNorm / Dropout(p>0) anywhere.
- SpectralTokens uses FROZEN normalization stats (set_frozen_stats /
  capture_stats), never batch statistics (Metis #7).
- sigreg_losses implements Epps-Pulley CF normality test on Z (single scalar).
- Flow heads predict velocity in orthonormal-DCT coefficient space with
  8-dim sinusoidal time embedding cos/sin(t*2^k).
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfpo import config  # noqa: E402

CTX = config.MFPO_CFG.context_len  # 16
ZD = config.MFPO_CFG.Z_dim  # 128
KS = config.MFPO_CFG.K_spectral  # 5
CHUNK = config.MFPO_CFG.chunk_len  # 8

# All network symbols come from mfpo.networks (imported lazily inside a
# helper so that RED phase reports one clean collection error per use).


def _nets():
    import mfpo.networks as nets

    return nets


# ---------------------------------------------------------------------------
# ContextTrunk
# ---------------------------------------------------------------------------


def test_context_trunk_shape_dtype_finite():
    nets = _nets()
    torch.manual_seed(0)
    trunk = nets.ContextTrunk(obs_dim=11)
    assert next(trunk.parameters()).dtype == torch.float32
    x = torch.randn(32, CTX, 11)
    z = trunk(x)
    assert z.shape == (32, CTX, ZD)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_context_trunk_gradient_flows():
    nets = _nets()
    torch.manual_seed(1)
    trunk = nets.ContextTrunk(obs_dim=5)
    z = trunk(torch.randn(4, CTX, 5))
    z.sum().backward()
    grads = [p.grad for p in trunk.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# sigreg_losses — Sketched Isotropic Gaussian Regularizer (SIGReg, Epps-Pulley)
#
# Replicates lucas-maes/le-wm SIGReg.forward(): empirical characteristic
# function of resampled random 1-D projections of Z is penalized toward the
# N(0,1) characteristic function phi(t)=exp(-t^2/2) via trapezoidal quadrature
# on [0,3]. Input is project-layout [N, T, D]; the function permutes to the
# official (T, B, D) then mean-reduces over B and scales by B.
#
# The optional `projection` arg pins the random projection matrix for
# deterministic reference tests and reproducible experiments; when None the
# projections are resampled (unit-norm columns) on every call.
# ---------------------------------------------------------------------------


def _fixed_projection(d, m, seed):
    """Deterministic unit-norm-column projection matrix of shape [d, m]."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, m, generator=g)
    return a / a.norm(dim=0, keepdim=True)


def _epps_pulley_reference(z, projection, knots=17, num_proj=1024, atol=None):
    """Independent in-test replica of the Epps-Pulley SIGReg statistic.

    Mirrors the official formula: permute to (T, B, D), project, trapezoid
    integrate |CF_emp - phi|^2 over t in [0,3], scale by B, mean over (T, M).
    Returns the expected scalar for the given fixed projection.
    """
    zp = z.permute(1, 0, 2)  # (T, B, D)
    t = torch.linspace(0, 3, knots)
    dt = 3.0 / (knots - 1)
    w = torch.full((knots,), 2 * dt)
    w[0] = dt
    w[-1] = dt
    phi = torch.exp(-t.square() / 2.0)
    x_t = (zp @ projection).unsqueeze(-1) * t  # (T, B, M, K)
    err = (x_t.cos().mean(1) - phi).square() + x_t.sin().mean(1).square()
    return ((err @ (w * phi)) * zp.size(-2)).mean()


def test_sigreg_hand_computed_small_fixture():
    """Hand-computed Epps-Pulley case: Z [N=2, T=1, D=2], knots=3, M=1.

    Rows [0.5,-0.5] and [1.0,2.0]; fixed projection [[1],[0]] onto dim 0.
    With a pinned projection the function is deterministic, so the reference
    (independent replica of the official formula) must match to 1e-5.
    """
    nets = _nets()
    z = torch.tensor([[[0.5, -0.5]], [[1.0, 2.0]]], dtype=torch.float32)  # [2,1,2]
    proj = torch.tensor([[1.0], [0.0]])  # [2,1]
    got = nets.sigreg_losses(z, knots=3, num_proj=1, projection=proj)
    assert got.ndim == 0
    expected = _epps_pulley_reference(z, proj, knots=3, num_proj=1)
    assert torch.allclose(got, expected, atol=1e-5)


def test_sigreg_reference_implementation_4x16x128():
    """Reference fixture shape [4,16,128] at full defaults, fixed projection."""
    nets = _nets()
    torch.manual_seed(7)
    z = torch.randn(4, CTX, ZD)
    proj = _fixed_projection(ZD, 1024, seed=42)
    got = nets.sigreg_losses(z, num_proj=1024, projection=proj)
    assert got.ndim == 0
    expected = _epps_pulley_reference(z, proj, knots=17, num_proj=1024)
    assert torch.allclose(got, expected, atol=1e-5)


def test_sigreg_resampled_projections_stay_gaussian_scaled():
    """Without a pinned projection the stochastic estimate is resampled each
    call (unit-norm columns). On a large Gaussian batch it must stay ~1.05
    (well below the degenerate degenerate ~190) and remain a 0-dim scalar."""
    nets = _nets()
    torch.manual_seed(0)
    z_gauss = torch.randn(64, 16, 128)
    vals = [nets.sigreg_losses(z_gauss).item() for _ in range(5)]
    for v in vals:
        assert torch.isfinite(torch.tensor(v))
        assert 0.5 < v < 2.2  # measured gaussian range ~[1.046, 1.056]; 2x margin


def test_sigreg_gradients_flow():
    nets = _nets()
    z = torch.randn(8, CTX, ZD, requires_grad=True)
    loss = nets.sigreg_losses(z, num_proj=32)
    loss.backward()
    assert loss.ndim == 0
    assert z.grad is not None and torch.isfinite(z.grad).all()


# ---------------------------------------------------------------------------
# SpectralTokens — frozen stats, batch invariance, drift guard
# ---------------------------------------------------------------------------


def _make_pinned_spectral(nets):
    """Two fresh constructions under manual_seed(0); bitwise-equal buffers."""
    torch.manual_seed(0)
    st_a = nets.SpectralTokens()
    torch.manual_seed(0)
    st_b = nets.SpectralTokens()
    # determinism precondition: bitwise-equal state between constructions
    for (ka, va), (kb, vb) in zip(st_a.state_dict().items(), st_b.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb)
    mu = torch.zeros(2 * ZD)
    std = torch.ones(2 * ZD)
    st_a.set_frozen_stats(mu, std)
    st_b.set_frozen_stats(mu, std)
    return st_a, st_b


def _pinned_input():
    return torch.arange(CTX * ZD, dtype=torch.float32).reshape(1, CTX, ZD) / 2048.0


def test_spectral_tokens_frozen_stats_batch_invariance():
    nets = _nets()
    st_a, _st_b = _make_pinned_spectral(nets)
    inp = _pinned_input()

    tokens_b1 = st_a(inp)
    assert tokens_b1.shape == (1, KS, 64)  # Dk=64 after projector
    assert tokens_b1.dtype == torch.float32

    # identical input, repeated call -> bitwise identical output
    assert torch.equal(st_a(inp), tokens_b1)

    tokens_b32 = st_a(inp.repeat(32, 1, 1))
    assert tokens_b32.shape == (32, KS, 64)  # Dk=64 after projector
    # plan acceptance criterion (CPU FFT picks kernels by batch size, so
    # cross-batch equality is allclose@1e-6, not bitwise)
    assert torch.allclose(tokens_b1, tokens_b32[0], atol=1e-6) is True

    # huge outlier row prepended: unchanged rows unchanged within 1e-6
    outlier_batch = torch.cat([torch.full((1, CTX, ZD), 1.0e4), inp], dim=0)
    with pytest.warns(UserWarning, match="spectral stats drift"):
        tokens_out = st_a(outlier_batch)
    assert torch.allclose(tokens_out[1], tokens_b1[0], atol=1e-6)


def test_spectral_tokens_deterministic_across_instances():
    nets = _nets()
    st_a, st_b = _make_pinned_spectral(nets)
    inp = _pinned_input()
    assert torch.equal(st_a(inp), st_b(inp))


def test_spectral_tokens_drift_warning_and_clean_pass():
    nets = _nets()
    st, _ = _make_pinned_spectral(nets)
    base = _pinned_input()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st(base)
    drift_msgs = [w for w in caught if "spectral stats drift" in str(w.message)]
    assert drift_msgs == []

    shifted = base + 10.0  # per-dim mean deviates >> 3 frozen stds
    with pytest.warns(UserWarning, match="spectral stats drift"):
        out = st(shifted)
    assert out.shape == (1, KS, 64)  # Dk=64 after projector


def test_spectral_tokens_missing_stats_raises():
    nets = _nets()
    st = nets.SpectralTokens()
    x = torch.randn(1, CTX, ZD) * 0.1
    with pytest.raises(RuntimeError, match="frozen stats not initialized"):
        st(x)


def test_spectral_tokens_capture_stats_matches_set():
    """capture_stats computes frozen stats from FFT-domain features (2*D wide)."""
    from mfpo.dct_utils import dct as _dct  # noqa: F401 — not used, just importing

    nets = _nets()
    torch.manual_seed(3)
    sample = torch.randn(64, CTX, ZD) * 0.1
    st_cap = nets.SpectralTokens()
    st_cap.capture_stats(sample)
    # Stats are captured from rFFT+cat, not raw Z-space
    hann = torch.hann_window(CTX).view(1, -1, 1)
    fft = torch.fft.rfft(sample * hann, dim=1)
    feat = torch.cat([fft.real, fft.imag], dim=-1)  # [64, KS, 2*ZD]
    ref_mu = feat.reshape(-1, 2 * ZD).mean(dim=0)
    ref_std = feat.reshape(-1, 2 * ZD).std(dim=0).clamp_min(1e-4)
    assert torch.allclose(st_cap.frozen_mu, ref_mu, atol=1e-5)
    assert torch.allclose(st_cap.frozen_std, ref_std, atol=1e-5)
    out = st_cap(sample[:4] * 0.1)
    assert out.shape == (4, KS, 64)  # Dk=64 after projector
    assert torch.isfinite(out).all()


def test_spectral_tokens_nan_propagates_detectably():
    """Documented failure behavior: NaN inputs propagate to the output
    (detectable via torch.isnan); no silent swallowing, no crash."""
    nets = _nets()
    st, _ = _make_pinned_spectral(nets)
    x = _pinned_input().clone()
    x[0, 0, 0] = float("nan")
    out = st(x)
    assert torch.isnan(out).any()


# ---------------------------------------------------------------------------
# timestep embedding + flow heads
# ---------------------------------------------------------------------------


def test_timestep_embedding_sin_cos_values():
    nets = _nets()
    emb = nets.timestep_embedding(0.5, dim=8)
    assert emb.shape == (8,)
    expected = [math.sin(0.5 * 2**k) for k in range(4)] + [
        math.cos(0.5 * 2**k) for k in range(4)
    ]
    for got, want in zip(emb.tolist(), expected):
        assert abs(got - want) < 1e-6


def test_policy_flow_head_shape_scalar_t():
    nets = _nets()
    torch.manual_seed(11)
    head = nets.PolicyFlowHead(action_dim=7)
    assert next(head.parameters()).dtype == torch.float32
    x_t = torch.randn(32, CHUNK, 7)  # DCT-coefficient space per Scope
    cond = torch.randn(32, CTX + KS, ZD)  # token seq [Z; fft]
    v = head(x_t, 0.37, cond)
    assert v.shape == (32, CHUNK, 7)
    assert torch.isfinite(v).all()


def test_policy_flow_head_tensor_t_and_grad():
    nets = _nets()
    torch.manual_seed(12)
    head = nets.PolicyFlowHead(action_dim=3)
    x_t = torch.randn(8, CHUNK, 3)
    t = torch.rand(8)
    cond = torch.randn(8, KS, ZD)
    v = head(x_t, t, cond)
    assert v.shape == (8, CHUNK, 3)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_policy_flow_head_cond_from_spectral():
    """Plan: cond from SpectralTokens outputs [B, K, Dk=64]."""
    nets = _nets()
    torch.manual_seed(18)
    Dk = 64
    head = nets.PolicyFlowHead(action_dim=7, cond_dims=(Dk,))
    fft = torch.randn(8, KS, Dk)
    v = head(torch.randn(8, CHUNK, 7), 0.5, fft)
    assert v.shape == (8, CHUNK, 7)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_aux_state_flow_head_shape_and_separate_params():
    nets = _nets()
    torch.manual_seed(13)
    aux = nets.AuxStateFlowHead(state_dim=ZD)
    pol = nets.PolicyFlowHead(action_dim=7)
    x_t = torch.randn(16, CHUNK, ZD)
    cond = torch.randn(16, CTX + KS, ZD)
    v = aux(x_t, 0.5, cond)
    assert v.shape == (16, CHUNK, ZD)
    assert torch.isfinite(v).all()
    # no parameter sharing by identity
    ids_pol = {id(p) for p in pol.parameters()}
    ids_aux = {id(p) for p in aux.parameters()}
    assert ids_pol.isdisjoint(ids_aux)


# ---------------------------------------------------------------------------
# ValueCritic
# ---------------------------------------------------------------------------


def test_value_critic_shape():
    nets = _nets()
    torch.manual_seed(14)
    vc = nets.ValueCritic()
    assert next(vc.parameters()).dtype == torch.float32
    z = torch.randn(32, CTX, ZD)
    v = vc(z)
    assert v.shape == (32,)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


# ---------------------------------------------------------------------------
# EMAProjection
# ---------------------------------------------------------------------------


def test_ema_projection_manual_formula_after_3_steps():
    nets = _nets()
    torch.manual_seed(15)
    ep = nets.EMAProjection(obs_dim=11)
    assert ep.decay == pytest.approx(config.MFPO_CFG.ema_decay)
    assert next(ep.parameters()).dtype == torch.float32

    obs_dim, latent = 11, ZD
    small = 0.1  # keep magnitudes ~0.1 so fp32 rounding stays < 1e-7

    def mk(seed_offset):
        g = torch.Generator().manual_seed(100 + seed_offset)
        w = torch.randn(latent, obs_dim, generator=g) * small
        b = torch.randn(latent, generator=g) * small
        return w, b

    s_w, s_b = mk(0)
    w0, b0 = mk(1)
    w1, b1 = mk(2)
    w2, b2 = mk(3)

    with torch.no_grad():
        ep.online.weight.copy_(s_w)
        ep.online.bias.copy_(s_b)
    ep.copy_from(ep.online)  # hard-set target := online
    assert torch.equal(ep.target.weight, s_w)
    assert torch.equal(ep.target.bias, s_b)

    # three polyak steps with changing online weights:
    # t <- decay*t + (1-decay)*online, closed form computed explicitly
    with torch.no_grad():
        ep.online.weight.copy_(w0)
        ep.online.bias.copy_(b0)
    ep.ema_update(step=1)
    with torch.no_grad():
        ep.online.weight.copy_(w1)
        ep.online.bias.copy_(b1)
    ep.ema_update(step=2)
    with torch.no_grad():
        ep.online.weight.copy_(w2)
        ep.online.bias.copy_(b2)
    ep.ema_update(step=3)

    d = config.MFPO_CFG.ema_decay
    exp_w = (d**3) * s_w.double() + (1 - d) * (
        w2.double() + d * w1.double() + d**2 * w0.double()
    )
    exp_b = (d**3) * s_b.double() + (1 - d) * (
        b2.double() + d * b1.double() + d**2 * b0.double()
    )
    assert torch.allclose(ep.target.weight, exp_w.float(), atol=1e-7)
    assert torch.allclose(ep.target.bias, exp_b.float(), atol=1e-7)


def test_ema_projection_forward_uses_online():
    nets = _nets()
    torch.manual_seed(16)
    ep = nets.EMAProjection(obs_dim=5)
    x = torch.randn(4, 5)
    y = ep(x)
    assert y.shape == (4, ZD)
    expected = torch.nn.functional.linear(x, ep.online.weight, ep.online.bias)
    assert torch.allclose(y, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# global hygiene: fp32-only, no BatchNorm / Dropout
# ---------------------------------------------------------------------------


def test_no_batchnorm_no_dropout_fp32_everywhere():
    nets = _nets()
    torch.manual_seed(17)
    modules = [
        nets.ContextTrunk(obs_dim=9),
        nets.SpectralTokens(),
        nets.PolicyFlowHead(action_dim=7),
        nets.AuxStateFlowHead(state_dim=ZD),
        nets.ValueCritic(),
        nets.EMAProjection(obs_dim=9),
    ]
    for mod in modules:
        for name, sub in mod.named_modules():
            cls = type(sub).__name__.lower()
            assert "batchnorm" not in cls, f"BatchNorm found: {name}"
            if isinstance(sub, torch.nn.Dropout):
                assert sub.p == 0.0, f"Dropout(p>0) found: {name}"
        params = list(mod.parameters())
        if params:
            assert next(iter(params)).dtype == torch.float32
