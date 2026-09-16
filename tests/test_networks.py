"""Unit tests for src/sjfpo/networks.py.

Conventions under test: fp32 params; SwiGLU; no BatchNorm/Dropout;
SpectralEncoder needs no calibration (raw DCT bins, no batch stats);
flow heads predict velocity in DCT space with sinusoidal time embedding.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402

_DEFAULTS = SJFPOConfig()

STATE_T = _DEFAULTS.n_obs_steps  # 16
ZD = _DEFAULTS.hidden_dim  # 128
OBS = 11  # raw obs dim feeding SpectralEncoder (env-like)
KS = max(1, round(_DEFAULTS.top_p_spectral_tokens * STATE_T))  # 5
CHUNK = _DEFAULTS.action_horizon  # 16

# Networks import lazily so a missing module reports one clean collection error.


def _inactive_branch_params(policy):
    """Params of the pooling branch not taken (mutually exclusive modes)."""
    if policy.spectral_condition_mode == "attention":
        return {"cond_proj.weight", "cond_proj.bias"}
    return {
        n for n, _ in policy.named_parameters() if n.startswith("spectral_condition_module.")
    }


def make_nets():
    import sjfpo.networks as nets

    return nets

    # StateEncoder


def test_window_encoder_shape_dtype_finite():
    nets = make_nets()
    torch.manual_seed(0)
    encoder = nets.WindowEncoder(in_dim=11)
    assert next(encoder.parameters()).dtype == torch.float32
    x = torch.randn(32, STATE_T, 11)
    z = encoder(x)
    assert z.shape == (32, STATE_T, ZD)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_window_encoder_gradient_flows():
    nets = make_nets()
    torch.manual_seed(1)
    encoder = nets.WindowEncoder(in_dim=5)
    z = encoder(torch.randn(4, STATE_T, 5))
    z.sum().backward()
    grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

    # RunningMeanStd — running obs stats, checkpointed with the models


def test_running_stats_identity_until_first_update():
    nets = make_nets()
    norm = nets.RunningMeanStd(4)
    x = torch.randn(8, 16, 4) * 100.0
    assert torch.equal(norm(x), x)
    assert int(norm.count) == 0


def test_running_stats_normalize_and_checkpoint():
    nets = make_nets()
    torch.manual_seed(0)
    norm = nets.RunningMeanStd(4)
    batch = torch.randn(64, 16, 4) * 5.0 + 3.0
    norm.update(batch)
    assert int(norm.count) == 64 * 16
    out = norm(batch)
    assert torch.isfinite(out).all()
    assert abs(float(out.mean())) < 0.05
    assert abs(float(out.std()) - 1.0) < 0.05
    assert float(out.abs().max()) <= 10.0
    restored = nets.RunningMeanStd(4)
    restored.load_state_dict(norm.state_dict())
    assert torch.equal(restored(batch), out)


def test_running_stats_rejects_nonpositive_dim():
    nets = make_nets()
    with pytest.raises(ValueError):
        nets.RunningMeanStd(0)

    # SpectralEncoder — raw DCT low-K bins, batch invariance, determinism


def make_pinned_spectral(nets):
    """Two fresh constructions under manual_seed(0); bitwise-equal weights."""
    torch.manual_seed(0)
    st_a = nets.SpectralEncoder(OBS)
    torch.manual_seed(0)
    st_b = nets.SpectralEncoder(OBS)
    # determinism precondition: bitwise-equal state between constructions
    for (ka, va), (kb, vb) in zip(st_a.state_dict().items(), st_b.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb)
    return st_a, st_b


def pinned_input():
    return (
        torch.arange(STATE_T * OBS, dtype=torch.float32).reshape(1, STATE_T, OBS)
        / 2048.0
    )


def test_spectral_encoder_batch_invariance():
    nets = make_nets()
    st_a, st_b = make_pinned_spectral(nets)
    inp = pinned_input()

    tokens_b1 = st_a(inp)
    assert tokens_b1.shape == (1, KS, OBS)  # raw bins, last dim == obs
    assert tokens_b1.dtype == torch.float32

    # identical input, repeated call -> bitwise identical output
    assert torch.equal(st_a(inp), tokens_b1)

    tokens_b32 = st_a(inp.repeat(32, 1, 1))
    assert tokens_b32.shape == (32, KS, OBS)  # raw bins, last dim == obs
    # cross-batch equality is allclose@1e-6, not bitwise (matmul kernels
    # may vary by batch size)
    assert torch.allclose(tokens_b1, tokens_b32[0], atol=1e-6) is True

    # huge outlier row prepended: rows are independent (no batch statistics),
    # so unchanged rows are unchanged within 1e-6
    outlier_batch = torch.cat([torch.full((1, STATE_T, OBS), 1.0e4), inp], dim=0)
    tokens_out = st_a(outlier_batch)
    assert torch.allclose(tokens_out[1], tokens_b1[0], atol=1e-6)


def test_spectral_encoder_deterministic_across_instances():
    nets = make_nets()
    st_a, st_b = make_pinned_spectral(nets)
    inp = pinned_input()
    assert torch.equal(st_a(inp), st_b(inp))


def test_spectral_encoder_raw_dct_path():
    """Forward equals manual DCT-II → low-K slice, no projector (order lock)."""
    from sjfpo.dct_utils import dct

    nets = make_nets()
    torch.manual_seed(3)
    st = nets.SpectralEncoder(OBS)
    assert list(st.parameters()) == []  # parameter-free
    sample = torch.randn(4, STATE_T, OBS)
    out = st(sample)
    assert out.shape == (4, KS, OBS)  # raw bins, last dim == obs
    assert torch.isfinite(out).all()
    ref = dct(sample, dim=1)[:, :KS, :]
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

    # timestep embedding + flow heads


def test_get_timestep_embedding_sin_cos_values():
    nets = make_nets()
    emb = nets.get_timestep_embedding(0.5, dim=8, scale=1.0)
    assert emb.shape == (8,)
    freqs = [math.exp(-math.log(10000.0) * k / 3) for k in range(4)]
    expected = [math.sin(0.5 * f) for f in freqs] + [math.cos(0.5 * f) for f in freqs]
    for got, want in zip(emb.tolist(), expected):
        assert abs(got - want) < 1e-6


def test_policy_flow_head_shape_scalar_t():
    nets = make_nets()
    torch.manual_seed(11)
    policy = nets.SJFOPolicyModel(action_dim=7, obs_dim=OBS)
    assert next(policy.parameters()).dtype == torch.float32
    x_t = torch.randn(32, CHUNK, 7)  # DCT-coefficient space per Scope
    condition = torch.randn(32, KS, OBS)  # raw spectral tokens [B, K, obs]
    v = policy(x_t, 0.37, condition, torch.randn(32, STATE_T, OBS))
    assert v.shape == (32, CHUNK, 7)
    assert torch.isfinite(v).all()


def test_policy_flow_head_tensor_t_and_grad():
    nets = make_nets()
    torch.manual_seed(12)
    policy = nets.SJFOPolicyModel(action_dim=3, obs_dim=OBS)
    x_t = torch.randn(8, CHUNK, 3)
    t = torch.rand(8)
    condition = torch.randn(8, KS, OBS)  # raw spectral tokens [B, K, obs]
    v = policy(x_t, t, condition, torch.randn(8, STATE_T, OBS))
    assert v.shape == (8, CHUNK, 3)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in policy.named_parameters()
        if n not in _inactive_branch_params(policy)
    )


def test_policy_flow_head_cond_from_spectral():
    """Plan: condition from SpectralEncoder outputs [B, K, obs] raw bins."""
    nets = make_nets()
    torch.manual_seed(18)
    policy = nets.SJFOPolicyModel(action_dim=7, obs_dim=OBS)
    fft = torch.randn(8, KS, OBS)
    v = policy(torch.randn(8, CHUNK, 7), 0.5, fft, torch.randn(8, STATE_T, OBS))
    assert v.shape == (8, CHUNK, 7)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in policy.named_parameters()
        if n not in _inactive_branch_params(policy)
    )

    # DiT blocks (adaLN-Zero identity init + finite time embeddings)


def test_dit_block_identity_at_init():
    nets = make_nets()
    torch.manual_seed(19)
    block = nets.DiTBlock(128, 4, 4.0)
    x = torch.randn(2, 8, 128)
    c = torch.randn(2, 128)
    assert torch.allclose(block(x, c), x, atol=1e-5), (
        "adaLN-Zero must start as identity"
    )


def test_timestep_scale_resolves_flow_times():
    nets = make_nets()
    a = nets.get_timestep_embedding(0.20, dim=32)
    b = nets.get_timestep_embedding(0.25, dim=32)
    assert float((a - b).abs().max()) > 0.5, (
        "default scale must separate nearby flow times (unscaled diff < 0.05)"
    )


def test_get_timestep_embedding_finite_256():
    nets = make_nets()
    emb = nets.get_timestep_embedding(torch.rand(16), dim=256)
    assert emb.shape == (16, 256)
    assert torch.isfinite(emb).all(), "256-dim ladder must stay finite in fp32"


def test_policy_dit_zero_velocity_at_init():
    nets = make_nets()
    torch.manual_seed(20)
    policy = nets.SJFOPolicyModel(action_dim=3, obs_dim=OBS)
    v = policy(
        torch.randn(4, CHUNK, 3),
        0.5,
        torch.randn(4, KS, OBS),
        torch.randn(4, STATE_T, OBS),
    )
    assert v.shape == (4, CHUNK, 3)
    assert torch.isfinite(v).all()
    assert float(v.abs().mean()) == pytest.approx(0.0, abs=1e-5)


def test_cross_attention_block_shape():
    nets = make_nets()
    torch.manual_seed(21)
    block = nets.CrossAttentionBlock(128, 4, 4.0)
    y = block(torch.randn(3, 8, 128), torch.randn(3, 16, 128))
    assert y.shape == (3, 8, 128)
    assert torch.isfinite(y).all()

    # Critic


def test_value_critic_shape():
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(OBS, 7)
    assert next(vc.parameters()).dtype == torch.float32
    s = torch.randn(32, STATE_T, OBS)
    a = torch.randn(32, CHUNK, 7)
    v = vc(s, a)
    assert v.shape == (32,)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


def test_value_critic_requires_action_dim():
    nets = make_nets()
    with pytest.raises(ValueError, match="action_dim"):
        nets.Critic(OBS, 0)


def test_value_critic_requires_obs_dim():
    nets = make_nets()
    with pytest.raises(ValueError, match="obs_dim"):
        nets.Critic(0, 7)


def test_value_critic_follows_next_action():
    """V(s,a) must change when the action changes (action conditioning live)."""
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(OBS, 3)
    s = torch.randn(8, STATE_T, OBS)
    v1 = vc(s, torch.zeros(8, CHUNK, 3))
    v2 = vc(s, torch.ones(8, CHUNK, 3))
    assert not torch.equal(v1, v2)


def test_value_critic_action_order_sensitive():
    """Flipped action order changes V (action PE lock: mean would hide it)."""
    nets = make_nets()
    torch.manual_seed(15)
    vc = nets.Critic(OBS, 3)
    s = torch.randn(4, STATE_T, OBS)
    a = torch.randn(4, CHUNK, 3)
    v1 = vc(s, a)
    v2 = vc(s, a.flip(dims=(1,)))
    assert not torch.allclose(v1, v2, atol=1e-6)

    # global hygiene: fp32-only, no BatchNorm / Dropout


def test_no_batchnorm_no_dropout_fp32_everywhere():
    nets = make_nets()
    torch.manual_seed(17)
    modules = [
        nets.WindowEncoder(in_dim=9),
        nets.SpectralEncoder(obs_dim=9),
        nets.SJFOPolicyModel(action_dim=7, obs_dim=9),
        nets.Critic(9, 7),
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
    assert nets.SpectralEncoder(OBS).k == 5  # round(0.3 * 16)
    assert nets.SpectralEncoder(OBS, top_p_spectral_tokens=1.0).k == 16
    assert nets.SpectralEncoder(OBS, top_p_spectral_tokens=0.01).k == 1
    out = nets.SpectralEncoder(OBS, top_p_spectral_tokens=0.5)(
        torch.randn(2, STATE_T, OBS)
    )
    assert out.shape == (2, 8, OBS)  # round(0.5 * 16)


def test_window_encoder_position_sensitive():
    """Identical frames still map to distinct latents per position (PE lock)."""
    nets = make_nets()
    torch.manual_seed(2)
    encoder = nets.WindowEncoder(in_dim=6)
    x = torch.ones(2, STATE_T, 6)  # no order info in the input itself
    z = encoder(x)
    assert not torch.allclose(z[:, 0, :], z[:, 1, :], atol=1e-6)


def test_window_encoder_mask_hides_tails():
    """Garbage tails cannot move real rows: masked as attention keys."""
    nets = make_nets()
    torch.manual_seed(20)
    enc = nets.WindowEncoder(in_dim=6)
    x = torch.randn(2, STATE_T, 6)
    garb = x.clone()
    garb[:, 8:, :] = 100.0
    z1 = enc(x, [8, 8])
    z2 = enc(garb, [8, 8])
    assert torch.allclose(z1[:, :8, :], z2[:, :8, :], atol=1e-6)


def test_critic_value_ignores_padded_tails():
    """Garbage tails must not move V: key-masked in the encoder, dropped in pooling."""
    nets = make_nets()
    torch.manual_seed(21)
    critic = nets.Critic(OBS, 3)
    s = torch.randn(2, STATE_T, OBS)
    a = torch.randn(2, CHUNK, 3)
    garb = a.clone()
    garb[:, 8:, :] = 100.0
    v1 = critic(s, a, [8, 8])
    v2 = critic(s, garb, [8, 8])
    assert torch.allclose(v1, v2, atol=1e-6)


def test_critic_rejects_zero_executed_steps():
    nets = make_nets()
    critic = nets.Critic(OBS, 3)
    s = torch.randn(2, STATE_T, OBS)
    a = torch.randn(2, CHUNK, 3)
    with pytest.raises(ValueError, match=">= 1 executed step"):
        critic(s, a, [0, 0])


def test_positional_encoding_shape():
    nets = make_nets()
    out = nets.PositionalEncoding()(torch.zeros(2, STATE_T, ZD))
    assert out.shape == (2, STATE_T, ZD)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])  # positions differ


def test_policy_bin_order_sensitive():
    """Identical bin values still map to distinct rows (frequency-slot lock).

    Without bin codes, symmetric inputs give symmetric outputs and attention
    can only mix bins by content magnitude; the encoding breaks the symmetry
    so inter-bin relations are frequency-addressable."""
    nets = make_nets()
    torch.manual_seed(32)
    policy = nets.SJFOPolicyModel(action_dim=3, obs_dim=OBS)
    # Wake the zero-init (adaLN gates + final out) so inputs matter.
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    ka = max(1, round(0.3 * CHUNK))  # 5
    x_t = torch.ones(2, ka, 3)  # no freq info in the input itself
    v = policy(
        x_t,
        0.5,
        torch.randn(2, ka, OBS),
        torch.randn(2, STATE_T, OBS),
    )
    assert v.shape == (2, ka, 3)
    assert not torch.allclose(v[:, 0, :], v[:, 1, :], atol=1e-6)


def test_policy_reads_raw_state_prefix():
    """DiT output changes with the raw state window (prefix-token lock)."""
    nets = make_nets()
    torch.manual_seed(30)
    policy = nets.SJFOPolicyModel(action_dim=3, obs_dim=OBS)
    # Wake the zero-init (adaLN gates + final out) so inputs matter.
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    x_t = torch.randn(4, CHUNK, 3)
    condition = torch.randn(4, KS, OBS)
    s1 = torch.randn(4, STATE_T, OBS)
    v1 = policy(x_t, 0.5, condition, s1)
    assert v1.shape == (4, CHUNK, 3)  # only action positions read out
    v2 = policy(x_t, 0.5, condition, torch.randn(4, STATE_T, OBS))
    assert not torch.allclose(v1, v2, atol=1e-6)


def test_policy_forward_truncated_coeff_space():
    """DiT denoises Ka=top-p rows; zero-pad restores full H (shape lock)."""
    nets = make_nets()
    torch.manual_seed(31)
    policy = nets.SJFOPolicyModel(action_dim=3, obs_dim=OBS)
    ka = max(1, round(0.3 * CHUNK))  # 5
    v = policy(
        torch.randn(4, ka, 3),
        0.5,
        torch.randn(4, ka, OBS),
        torch.randn(4, STATE_T, OBS),
    )
    assert v.shape == (4, ka, 3)
    padded = torch.nn.functional.pad(v, (0, 0, 0, CHUNK - ka))
    assert padded.shape == (4, CHUNK, 3)
    assert (padded[:, ka:, :] == 0).all()  # missing bins are exactly zero
