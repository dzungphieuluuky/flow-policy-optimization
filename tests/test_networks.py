"""Unit tests for src/sjfpo/networks.py.

Conventions under test: fp32 params; SwiGLU; no BatchNorm/Dropout;
state spectral path needs no calibration (full DCT bins, no batch stats);
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
OBS = 11  # raw obs dim feeding the state spectral path (env-like)
CHUNK = _DEFAULTS.action_horizon  # 16

# Networks import lazily so a missing module reports one clean collection error.


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

    # Spectral conditioning — state keeps full DCT bins; the K-lowest
    # filter applies to the action path only (utils.filter_low_freq).


def pinned_input():
    return (
        torch.arange(STATE_T * OBS, dtype=torch.float32).reshape(1, STATE_T, OBS)
        / 2048.0
    )


def test_state_spectral_full_bins_no_truncation():
    """State path keeps all T bins: encode_dct == manual DCT-II, no slicing."""
    from sjfpo.dct_utils import dct
    from sjfpo.utils import encode_dct

    torch.manual_seed(3)
    sample = torch.randn(4, STATE_T, OBS)
    out = encode_dct(sample)
    assert out.shape == (4, STATE_T, OBS)  # full bins, last dim == obs
    assert torch.isfinite(out).all()
    assert torch.allclose(out, dct(sample, dim=1), atol=1e-6)


def test_state_spectral_batch_invariance():
    from sjfpo.utils import encode_dct

    inp = pinned_input()

    tokens_b1 = encode_dct(inp)
    assert tokens_b1.shape == (1, STATE_T, OBS)  # full bins, last dim == obs

    # identical input, repeated call -> bitwise identical output
    assert torch.equal(encode_dct(inp), tokens_b1)

    tokens_b32 = encode_dct(inp.repeat(32, 1, 1))
    assert tokens_b32.shape == (32, STATE_T, OBS)
    # cross-batch equality is allclose@1e-6, not bitwise (matmul kernels
    # may vary by batch size)
    assert torch.allclose(tokens_b1, tokens_b32[0], atol=1e-6) is True

    # huge outlier row prepended: rows are independent (no batch statistics),
    # so unchanged rows are unchanged within 1e-6
    outlier_batch = torch.cat([torch.full((1, STATE_T, OBS), 1.0e4), inp], dim=0)
    tokens_out = encode_dct(outlier_batch)
    assert torch.allclose(tokens_out[1], tokens_b1[0], atol=1e-6)


def test_state_spectral_nan_propagates_detectably():
    """Documented failure behavior: NaN inputs propagate to the output
    (detectable via torch.isnan); no silent swallowing, no crash."""
    from sjfpo.utils import encode_dct

    x = pinned_input().clone()
    x[0, 0, 0] = float("nan")
    out = encode_dct(x)
    assert torch.isnan(out).any()


def test_action_low_freq_filter_keeps_ka_bins():
    """K-lowest filter is action-only: [B, H, Ad] -> [B, Ka, Ad]."""
    from sjfpo.dct_utils import dct
    from sjfpo.utils import filter_low_freq

    ka = max(1, round(_DEFAULTS.top_p_spectral_tokens * CHUNK))  # 5
    assert ka == 5
    a = torch.randn(2, CHUNK, 3)
    out = filter_low_freq(dct(a, dim=1), ka, dim=1)
    assert out.shape == (2, ka, 3)
    assert torch.allclose(out, dct(a, dim=1)[:, :ka, :], atol=1e-6)

    # timestep embedding + flow heads


def test_timestep_embedder_sin_cos_values():
    nets = make_nets()
    emb = nets.TimestepEmbedder(time_dim=8, scale=1.0)(0.5)
    assert emb.shape == (8,)
    freqs = [math.exp(-math.log(10000.0) * k / 3) for k in range(4)]
    expected = [math.sin(0.5 * f) for f in freqs] + [math.cos(0.5 * f) for f in freqs]
    for got, want in zip(emb.tolist(), expected):
        assert abs(got - want) < 1e-6


def test_policy_flow_head_shape_scalar_t():
    nets = make_nets()
    torch.manual_seed(11)
    policy = nets.SJFPOFlowModel(action_dim=7, obs_dim=OBS)
    assert next(policy.parameters()).dtype == torch.float32
    x_t = torch.randn(32, CHUNK, 7)  # DCT-coefficient space per Scope
    condition = torch.randn(32, STATE_T, OBS)  # full state bins [B, T, obs]
    v = policy(x_t, 0.37, condition, torch.randn(32, STATE_T, OBS))
    assert v.shape == (32, CHUNK, 7)
    assert torch.isfinite(v).all()


def test_policy_flow_head_tensor_t_and_grad():
    nets = make_nets()
    torch.manual_seed(12)
    policy = nets.SJFPOFlowModel(action_dim=3, obs_dim=OBS)
    x_t = torch.randn(8, CHUNK, 3)
    t = torch.rand(8)
    condition = torch.randn(8, STATE_T, OBS)  # full state bins [B, T, obs]
    v = policy(x_t, t, condition, torch.randn(8, STATE_T, OBS))
    assert v.shape == (8, CHUNK, 3)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in policy.parameters()
    )


def test_policy_flow_head_cond_from_spectral():
    """Plan: condition carries full state DCT bins [B, T, obs]."""
    nets = make_nets()
    torch.manual_seed(18)
    policy = nets.SJFPOFlowModel(action_dim=7, obs_dim=OBS)
    fft = torch.randn(8, STATE_T, OBS)
    v = policy(torch.randn(8, CHUNK, 7), 0.5, fft, torch.randn(8, STATE_T, OBS))
    assert v.shape == (8, CHUNK, 7)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in policy.parameters()
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
    a = nets.TimestepEmbedder(time_dim=32)(0.20)
    b = nets.TimestepEmbedder(time_dim=32)(0.25)
    assert float((a - b).abs().max()) > 0.5, (
        "default scale must separate nearby flow times (unscaled diff < 0.05)"
    )


def test_timestep_embedder_finite_256():
    nets = make_nets()
    emb = nets.TimestepEmbedder(time_dim=256)(torch.rand(16))
    assert emb.shape == (16, 256)
    assert torch.isfinite(emb).all(), "256-dim ladder must stay finite in fp32"


def test_policy_dit_zero_velocity_at_init():
    nets = make_nets()
    torch.manual_seed(20)
    policy = nets.SJFPOFlowModel(action_dim=3, obs_dim=OBS)
    v = policy(
        torch.randn(4, CHUNK, 3),
        0.5,
        torch.randn(4, STATE_T, OBS),
        torch.randn(4, STATE_T, OBS),
    )
    assert v.shape == (4, CHUNK, 3)
    assert torch.isfinite(v).all()
    assert float(v.abs().mean()) == pytest.approx(0.0, abs=1e-5)


    # Critic (PPO state-value baseline V(s))


def test_value_critic_shape():
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(OBS)
    assert next(vc.parameters()).dtype == torch.float32
    v = vc(torch.randn(32, STATE_T, OBS))
    assert v.shape == (32,)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


def test_value_critic_requires_obs_dim():
    nets = make_nets()
    with pytest.raises(ValueError, match="obs_dim"):
        nets.Critic(0)


def test_value_critic_follows_state():
    """V(s) must change when the state changes (state conditioning live)."""
    nets = make_nets()
    torch.manual_seed(14)
    vc = nets.Critic(OBS)
    v1 = vc(torch.zeros(8, STATE_T, OBS))
    v2 = vc(torch.ones(8, STATE_T, OBS))
    assert not torch.equal(v1, v2)


def test_value_critic_gradient_flows():
    nets = make_nets()
    torch.manual_seed(15)
    vc = nets.Critic(OBS)
    vc(torch.randn(4, STATE_T, OBS)).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in vc.parameters()
    )

    # global hygiene: fp32-only, no BatchNorm / Dropout


def test_no_batchnorm_no_dropout_fp32_everywhere():
    nets = make_nets()
    torch.manual_seed(17)
    modules = [
        nets.WindowEncoder(in_dim=9),
        nets.SJFPOFlowModel(action_dim=7, obs_dim=9),
        nets.Critic(9),
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


def test_action_top_p_token_count():
    """top_p maps onto the action horizon only: Ka = round(p * H)."""
    from sjfpo.dct_utils import dct
    from sjfpo.utils import filter_low_freq

    assert max(1, round(0.3 * CHUNK)) == 5
    assert max(1, round(1.0 * CHUNK)) == CHUNK
    assert max(1, round(0.01 * CHUNK)) == 1
    a = torch.randn(2, CHUNK, OBS)
    out = filter_low_freq(dct(a, dim=1), max(1, round(0.5 * CHUNK)), dim=1)
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


def test_critic_state_order_sensitive():
    """Flipped state order changes V(s) (state PE lock: mean would hide it)."""
    nets = make_nets()
    torch.manual_seed(21)
    critic = nets.Critic(OBS)
    s = torch.randn(2, STATE_T, OBS)
    v1 = critic(s)
    v2 = critic(s.flip(dims=(1,)))
    assert not torch.allclose(v1, v2, atol=1e-6)


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
    policy = nets.SJFPOFlowModel(action_dim=3, obs_dim=OBS)
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
    policy = nets.SJFPOFlowModel(action_dim=3, obs_dim=OBS)
    # Wake the zero-init (adaLN gates + final out) so inputs matter.
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    x_t = torch.randn(4, CHUNK, 3)
    condition = torch.randn(4, STATE_T, OBS)  # full state bins [B, T, obs]
    s1 = torch.randn(4, STATE_T, OBS)
    v1 = policy(x_t, 0.5, condition, s1)
    assert v1.shape == (4, CHUNK, 3)  # only action positions read out
    v2 = policy(x_t, 0.5, condition, torch.randn(4, STATE_T, OBS))
    assert not torch.allclose(v1, v2, atol=1e-6)


def test_policy_forward_truncated_coeff_space():
    """DiT denoises Ka=top-p rows; zero-pad restores full H (shape lock)."""
    nets = make_nets()
    torch.manual_seed(31)
    policy = nets.SJFPOFlowModel(action_dim=3, obs_dim=OBS)
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
