"""Unit tests for src/sjfpo/networks.py (plan task 4).

TDD: written BEFORE the implementation exists — the first pytest run must
fail RED on the missing module, then go GREEN once networks.py lands.

Conventions under test (binding, from the plan):
- fp32 everywhere; SwiGLU activations; no BatchNorm / Dropout(p>0) anywhere.
- SpectralEncoder needs no calibration: raw DCT low-K bins go straight to
  the projector, never batch statistics (Metis #7).
- sigreg_loss implements Epps-Pulley CF normality test on Z (single scalar).
- Flow heads predict velocity in orthonormal-DCT coefficient space with
  8-dim sinusoidal time embedding cos/sin(t*2^k).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo import config  # noqa: E402

STATE_T = config.SJFPO_CFG.n_obs_steps  # 16
ZD = config.SJFPO_CFG.hidden_size  # 128
KS = max(1, round(config.SJFPO_CFG.top_p_spectral_tokens * STATE_T))  # 5
CHUNK = config.SJFPO_CFG.chunk_size  # 16

# All network symbols come from sjfpo.networks (imported lazily inside a
# helper so that RED phase reports one clean collection error per use).


def make_nets():
    import sjfpo.networks as nets

    return nets

    # ---------------------------------------------------------------------------
    # StateEncoder
    # ---------------------------------------------------------------------------


def test_state_trunk_shape_dtype_finite():
    nets = make_nets()
    torch.manual_seed(0)
    trunk = nets.StateEncoder(obs_dim=11)
    assert next(trunk.parameters()).dtype == torch.float32
    x = torch.randn(32, STATE_T, 11)
    z = trunk(x)
    assert z.shape == (32, STATE_T, ZD)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_state_trunk_gradient_flows():
    nets = make_nets()
    torch.manual_seed(1)
    trunk = nets.StateEncoder(obs_dim=5)
    z = trunk(torch.randn(4, STATE_T, 5))
    z.sum().backward()
    grads = [p.grad for p in trunk.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    # ---------------------------------------------------------------------------
    # sigreg_loss — Sketched Isotropic Gaussian Regularizer (SIGReg, Epps-Pulley)
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


def fixed_projection(d, m, seed):
    """Deterministic unit-norm-column projection matrix of shape [d, m]."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, m, generator=g)
    return a / a.norm(dim=0, keepdim=True)


def epps_pulley_reference(z, projection, knots=17, num_proj=1024, atol=None):
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
    nets = make_nets()
    z = torch.tensor([[[0.5, -0.5]], [[1.0, 2.0]]], dtype=torch.float32)  # [2,1,2]
    proj = torch.tensor([[1.0], [0.0]])  # [2,1]
    got = nets.sigreg_loss(z, knots=3, num_proj=1, projection=proj)
    assert got.ndim == 0
    expected = epps_pulley_reference(z, proj, knots=3, num_proj=1)
    assert torch.allclose(got, expected, atol=1e-5)


def test_sigreg_reference_implementation_4x16x128():
    """Reference fixture shape [4,16,128] at full defaults, fixed projection."""
    nets = make_nets()
    torch.manual_seed(7)
    z = torch.randn(4, STATE_T, ZD)
    proj = fixed_projection(ZD, 1024, seed=42)
    got = nets.sigreg_loss(z, num_proj=1024, projection=proj)
    assert got.ndim == 0
    expected = epps_pulley_reference(z, proj, knots=17, num_proj=1024)
    assert torch.allclose(got, expected, atol=1e-5)


def test_sigreg_resampled_projections_stay_gaussian_scaled():
    """Without a pinned projection the stochastic estimate is resampled each
    call (unit-norm columns). On a large Gaussian batch it must stay ~1.05
    (well below the degenerate degenerate ~190) and remain a 0-dim scalar."""
    nets = make_nets()
    torch.manual_seed(0)
    z_gauss = torch.randn(64, 16, 128)
    vals = [nets.sigreg_loss(z_gauss).item() for _ in range(5)]
    for v in vals:
        assert torch.isfinite(torch.tensor(v))
        assert 0.5 < v < 2.2  # measured gaussian range ~[1.046, 1.056]; 2x margin


def test_sigreg_gradients_flow():
    nets = make_nets()
    z = torch.randn(8, STATE_T, ZD, requires_grad=True)
    loss = nets.sigreg_loss(z, num_proj=32)
    loss.backward()
    assert loss.ndim == 0
    assert z.grad is not None and torch.isfinite(z.grad).all()

    # ---------------------------------------------------------------------------
    # SpectralEncoder — raw DCT low-K bins, batch invariance, determinism
    # ---------------------------------------------------------------------------


def make_pinned_spectral(nets):
    """Two fresh constructions under manual_seed(0); bitwise-equal weights."""
    torch.manual_seed(0)
    st_a = nets.SpectralEncoder()
    torch.manual_seed(0)
    st_b = nets.SpectralEncoder()
    # determinism precondition: bitwise-equal state between constructions
    for (ka, va), (kb, vb) in zip(st_a.state_dict().items(), st_b.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb)
    return st_a, st_b


def pinned_input():
    return (
        torch.arange(STATE_T * ZD, dtype=torch.float32).reshape(1, STATE_T, ZD) / 2048.0
    )


def test_spectral_encoder_batch_invariance():
    nets = make_nets()
    st_a, st_b = make_pinned_spectral(nets)
    inp = pinned_input()

    tokens_b1 = st_a(inp)
    assert tokens_b1.shape == (1, KS, 64)  # spectral_dim=64 after projector
    assert tokens_b1.dtype == torch.float32

    # identical input, repeated call -> bitwise identical output
    assert torch.equal(st_a(inp), tokens_b1)

    tokens_b32 = st_a(inp.repeat(32, 1, 1))
    assert tokens_b32.shape == (32, KS, 64)  # spectral_dim=64 after projector
    # cross-batch equality is allclose@1e-6, not bitwise (matmul kernels
    # may vary by batch size)
    assert torch.allclose(tokens_b1, tokens_b32[0], atol=1e-6) is True

    # huge outlier row prepended: rows are independent (no batch statistics),
    # so unchanged rows are unchanged within 1e-6
    outlier_batch = torch.cat([torch.full((1, STATE_T, ZD), 1.0e4), inp], dim=0)
    tokens_out = st_a(outlier_batch)
    assert torch.allclose(tokens_out[1], tokens_b1[0], atol=1e-6)


def test_spectral_encoder_deterministic_across_instances():
    nets = make_nets()
    st_a, st_b = make_pinned_spectral(nets)
    inp = pinned_input()
    assert torch.equal(st_a(inp), st_b(inp))


def test_spectral_encoder_raw_dct_path():
    """Forward equals manual DCT-II → low-K slice → projector (order lock)."""
    from sjfpo.dct_utils import dct

    nets = make_nets()
    torch.manual_seed(3)
    st = nets.SpectralEncoder()
    sample = torch.randn(4, STATE_T, ZD)
    out = st(sample)
    assert out.shape == (4, KS, 64)  # spectral_dim=64 after projector
    assert torch.isfinite(out).all()
    ref = st.projector(dct(sample, dim=1)[:, :KS, :])
    assert torch.allclose(out, ref, atol=1e-6)


def test_spectral_encoder_nan_propagates_detectably():
    """Documented failure behavior: NaN inputs propagate to the output
    (detectable via torch.isnan); no silent swallowing, no crash."""
    nets = make_nets()
    st, _ = make_pinned_spectral(nets)
    x = pinned_input().clone()
    x[0, 0, 0] = float("nan")
    out = st(x)
    assert torch.isnan(out).any()

    # ---------------------------------------------------------------------------
    # timestep embedding + flow heads
    # ---------------------------------------------------------------------------


def test_get_timestep_embedding_sin_cos_values():
    nets = make_nets()
    emb = nets.get_timestep_embedding(0.5, dim=8)
    assert emb.shape == (8,)
    expected = [math.sin(0.5 * 2**k) for k in range(4)] + [
        math.cos(0.5 * 2**k) for k in range(4)
    ]
    for got, want in zip(emb.tolist(), expected):
        assert abs(got - want) < 1e-6


def test_policy_flow_head_shape_scalar_t():
    nets = make_nets()
    torch.manual_seed(11)
    policy = nets.SJFOPolicyModel(action_dim=7)
    assert next(policy.parameters()).dtype == torch.float32
    x_t = torch.randn(32, CHUNK, 7)  # DCT-coefficient space per Scope
    cond = torch.randn(32, KS, 64)  # live spectral tokens [B, K, spectral_dim]
    v = policy(x_t, 0.37, cond)
    assert v.shape == (32, CHUNK, 7)
    assert torch.isfinite(v).all()


def test_policy_flow_head_tensor_t_and_grad():
    nets = make_nets()
    torch.manual_seed(12)
    policy = nets.SJFOPolicyModel(action_dim=3)
    x_t = torch.randn(8, CHUNK, 3)
    t = torch.rand(8)
    cond = torch.randn(8, KS, 64)  # live spectral tokens [B, K, spectral_dim]
    v = policy(x_t, t, cond)
    assert v.shape == (8, CHUNK, 3)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
    )


def test_policy_flow_head_cond_from_spectral():
    """Plan: cond from SpectralEncoder outputs [B, K, spectral_dim=64]."""
    nets = make_nets()
    torch.manual_seed(18)
    spectral_dim = 64
    policy = nets.SJFOPolicyModel(action_dim=7)
    fft = torch.randn(8, KS, spectral_dim)
    v = policy(torch.randn(8, CHUNK, 7), 0.5, fft)
    assert v.shape == (8, CHUNK, 7)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
    )

    # ---------------------------------------------------------------------------
    # DiT blocks (adaLN-Zero identity init + finite time embeddings)
    # ---------------------------------------------------------------------------


def test_dit_block_identity_at_init():
    nets = make_nets()
    torch.manual_seed(19)
    blk = nets.DiTBlock(128, 4, 4.0)
    x = torch.randn(2, 8, 128)
    c = torch.randn(2, 128)
    assert torch.allclose(blk(x, c), x, atol=1e-5), "adaLN-Zero must start as identity"


def test_dit_timestep_embedding_finite_256():
    nets = make_nets()
    emb = nets.dit_timestep_embedding(torch.rand(16), dim=256)
    assert emb.shape == (16, 256)
    assert torch.isfinite(emb).all(), "256-dim ladder must stay finite in fp32"


def test_policy_dit_zero_velocity_at_init():
    nets = make_nets()
    torch.manual_seed(20)
    policy = nets.SJFOPolicyModel(action_dim=3)
    v = policy(torch.randn(4, CHUNK, 3), 0.5, torch.randn(4, KS, 64))
    assert v.shape == (4, CHUNK, 3)
    assert torch.isfinite(v).all()
    assert float(v.abs().mean()) == pytest.approx(0.0, abs=1e-5)


def test_cross_attention_block_shape():
    nets = make_nets()
    torch.manual_seed(21)
    blk = nets.CrossAttentionBlock(128, 4, 4.0)
    y = blk(torch.randn(3, 8, 128), torch.randn(3, 16, 128))
    assert y.shape == (3, 8, 128)
    assert torch.isfinite(y).all()

    # ---------------------------------------------------------------------------
    # Critic
    # ---------------------------------------------------------------------------


def test_value_critic_shape():
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(act_dim=7)
    assert next(vc.parameters()).dtype == torch.float32
    z = torch.randn(32, STATE_T, ZD)
    a = torch.randn(32, CHUNK, 7)
    v = vc(z, a)
    assert v.shape == (32,)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


def test_value_critic_requires_act_dim():
    nets = make_nets()
    with pytest.raises(ValueError, match="act_dim"):
        nets.Critic(act_dim=0)


def test_value_critic_follows_next_action():
    """V(s,a) must change when the action changes (action conditioning live)."""
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(act_dim=3)
    z = torch.randn(8, STATE_T, ZD)
    v1 = vc(z, torch.zeros(8, CHUNK, 3))
    v2 = vc(z, torch.ones(8, CHUNK, 3))
    assert not torch.equal(v1, v2)

    # ---------------------------------------------------------------------------
    # global hygiene: fp32-only, no BatchNorm / Dropout
    # ---------------------------------------------------------------------------


def test_no_batchnorm_no_dropout_fp32_everywhere():
    nets = make_nets()
    torch.manual_seed(17)
    modules = [
        nets.StateEncoder(obs_dim=9),
        nets.SpectralEncoder(),
        nets.SJFOPolicyModel(action_dim=7),
        nets.Critic(act_dim=7),
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


def test_spectral_encoder_top_p_token_count():
    nets = make_nets()
    assert nets.SpectralEncoder().k == 5  # round(0.3 * 16)
    assert nets.SpectralEncoder(top_p_spectral_tokens=1.0).k == 16
    assert nets.SpectralEncoder(top_p_spectral_tokens=0.01).k == 1
    out = nets.SpectralEncoder(top_p_spectral_tokens=0.5)(torch.randn(2, STATE_T, ZD))
    assert out.shape == (2, 8, 64)  # round(0.5 * 16)
