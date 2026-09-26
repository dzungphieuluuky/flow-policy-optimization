"""Unit tests for src/sjfpo/networks.py.

Conventions under test: fp32 params; SwiGLU; no BatchNorm/Dropout;
StateConditioner fuses raw states + DCT bins with no calibration;
flow heads take a pre-fused [B, D] vector with sinusoidal time embedding.
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
OBS = 11  # raw obs dim feeding the conditioner (env-like)
CHUNK = _DEFAULTS.action_horizon  # 16

# Networks import lazily so a missing module reports one clean collection error.


def make_nets():
    import sjfpo.networks as nets

    return nets


def make_flow(nets, action_dim):
    return nets.SJFPOFlowModel(
        action_dim,
        action_horizon=_DEFAULTS.action_horizon,
        hidden_dim=_DEFAULTS.hidden_dim,
        dit_depth=_DEFAULTS.dit_depth,
        dit_heads=_DEFAULTS.dit_heads,
        dit_mlp_ratio=_DEFAULTS.dit_mlp_ratio,
        time_embed_dim=_DEFAULTS.time_embed_dim,
        timestep_scale=_DEFAULTS.timestep_scale,
    )


def make_window(nets, in_dim):
    return nets.WindowEncoder(
        in_dim,
        hidden_dim=_DEFAULTS.hidden_dim,
        n_heads=_DEFAULTS.dit_heads,
        mlp_ratio=_DEFAULTS.dit_mlp_ratio,
    )


def make_conditioner(nets, obs_dim):
    return nets.StateConditioner(obs_dim, hidden_dim=_DEFAULTS.hidden_dim)


def make_critic(nets, obs_dim):
    return nets.Critic(obs_dim, hidden_dim=_DEFAULTS.hidden_dim)

    # StateEncoder


def test_self_attention_matches_math_reference():
    """SDPA module equals softmax(QK'/sqrt(d))V through its own projections."""
    nets = make_nets()
    torch.manual_seed(40)
    attn = nets.SelfAttention(32, n_heads=4)
    x = torch.randn(3, 9, 32)
    with torch.no_grad():
        q, k, v = attn.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(3, 9, 4, 8).transpose(1, 2)

        ref = torch.softmax(
            heads(q) @ heads(k).transpose(-1, -2) / math.sqrt(8), dim=-1
        )
        ref = attn.out((ref @ heads(v)).transpose(1, 2).reshape(3, 9, 32))
    got = attn(x)
    assert torch.allclose(got, ref, atol=1e-5)


def test_self_attention_mask_hides_keys():
    """attention_mask=True rows are valid; False rows are ignored."""
    nets = make_nets()
    torch.manual_seed(41)
    attn = nets.SelfAttention(32, n_heads=4)
    x = torch.randn(2, 9, 32)
    garb = x.clone()
    garb[:, 5:, :] = 100.0
    mask = torch.arange(9).unsqueeze(0) < torch.tensor([5, 5]).unsqueeze(1)
    z1 = attn(x, mask)
    z2 = attn(garb, mask)
    assert torch.allclose(z1[:, :5, :], z2[:, :5, :], atol=1e-4)


def test_self_attention_all_padding_is_finite_and_shape_is_enforced():
    nets = make_nets()
    attn = nets.SelfAttention(32, n_heads=4)
    x = torch.randn(2, 9, 32)
    output = attn(x, torch.zeros(2, 9, dtype=torch.bool))
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError, match=r"attention_mask must have shape"):
        attn(x, torch.ones(2, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"attention_mask must have shape"):
        attn(x, torch.ones(2, 9, 9, dtype=torch.bool))


def test_self_attention_rejects_invalid_head_count():
    nets = make_nets()
    with pytest.raises(ValueError, match="n_heads must be positive"):
        nets.SelfAttention(32, n_heads=0)


def test_window_encoder_shape_dtype_finite():
    nets = make_nets()
    torch.manual_seed(0)
    encoder = make_window(nets, 11)
    assert next(encoder.parameters()).dtype == torch.float32
    x = torch.randn(32, STATE_T, 11)
    z = encoder(x)
    assert z.shape == (32, STATE_T, ZD)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_window_encoder_gradient_flows():
    nets = make_nets()
    torch.manual_seed(1)
    encoder = make_window(nets, 5)
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

    # Spectral utils — K-lowest filter is action-only (utils.filter_low_freq).


def test_action_low_freq_filter_keeps_ka_bins():
    """K-lowest filter is action-only: [B, H, Ad] -> [B, Ka, Ad]."""
    from sjfpo.dct_utils import extract_dct_coeffs
    from sjfpo.utils import filter_low_freq

    ka = max(1, round(_DEFAULTS.top_p_spectral_tokens * CHUNK))
    assert ka == CHUNK
    a = torch.randn(2, CHUNK, 3)
    # M = CHUNK-1 keeps every bin, so this is the full orthonormal transform.
    coeffs = extract_dct_coeffs(a, M=CHUNK - 1)
    out = filter_low_freq(coeffs, ka, dim=1)
    assert out.shape == (2, ka, 3)
    assert torch.allclose(out, coeffs[:, :ka, :], atol=1e-6)

    # timestep embedding + flow heads


def test_timestep_embedder_sin_cos_values():
    nets = make_nets()
    emb = nets.SinusoidalEncoding(8, scale=1.0).encode_times(0.5)
    assert emb.shape == (8,)
    freqs = [math.exp(-math.log(10000.0) * k / 3) for k in range(4)]
    expected = [math.sin(0.5 * f) for f in freqs] + [math.cos(0.5 * f) for f in freqs]
    for got, want in zip(emb.tolist(), expected):
        assert abs(got - want) < 1e-6


def test_policy_flow_head_shape_scalar_t():
    nets = make_nets()
    torch.manual_seed(11)
    policy = make_flow(nets, 7)
    assert next(policy.parameters()).dtype == torch.float32
    x_t = torch.randn(32, CHUNK, 7)  # DCT-coefficient space per Scope
    condition = torch.randn(32, ZD)  # pre-fused state vector [B, D]
    v = policy(x_t, 0.37, condition)
    assert v.shape == (32, CHUNK, 7)
    assert torch.isfinite(v).all()


def test_policy_flow_head_tensor_t_and_grad():
    nets = make_nets()
    torch.manual_seed(12)
    policy = make_flow(nets, 3)
    x_t = torch.randn(8, CHUNK, 3)
    t = torch.rand(8)
    condition = torch.randn(8, ZD)  # pre-fused state vector [B, D]
    v = policy(x_t, t, condition)
    assert v.shape == (8, CHUNK, 3)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
    )


def test_policy_flow_head_per_token_timesteps():
    nets = make_nets()
    torch.manual_seed(13)
    policy = make_flow(nets, 3)
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.adaLN_out.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    x_t = torch.randn(2, CHUNK, 3)
    condition = torch.randn(2, ZD)
    t_a = torch.rand(2, CHUNK, 1)
    t_b = t_a.clone()
    t_b[:, 1, 0] += 0.25
    v_a = policy(x_t, t_a, condition)
    v_b = policy(x_t, t_b, condition)
    assert v_a.shape == (2, CHUNK, 3)
    assert v_b.shape == (2, CHUNK, 3)
    assert not torch.allclose(v_a[:, 1], v_b[:, 1])
    v_a.sum().backward()
    assert policy.time_mlp.fc1.weight.grad is not None


def test_policy_flow_head_cond_from_conditioner():
    """Conditioner output plugs straight into the flow head [B, D]."""
    nets = make_nets()
    torch.manual_seed(18)
    policy = make_flow(nets, 7)
    conditioner = make_conditioner(nets, OBS)
    fused = conditioner(torch.randn(8, STATE_T, OBS))
    assert fused.shape == (8, ZD)
    v = policy(torch.randn(8, CHUNK, 7), 0.5, fused)
    assert v.shape == (8, CHUNK, 7)
    v.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()
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
    a = nets.SinusoidalEncoding(32, scale=_DEFAULTS.timestep_scale).encode_times(0.20)
    b = nets.SinusoidalEncoding(32, scale=_DEFAULTS.timestep_scale).encode_times(0.25)
    assert float((a - b).abs().max()) > 0.5, (
        "default scale must separate nearby flow times (unscaled diff < 0.05)"
    )


def test_timestep_embedder_finite_256():
    nets = make_nets()
    emb = nets.SinusoidalEncoding(256).encode_times(torch.rand(16))
    assert emb.shape == (16, 256)
    assert torch.isfinite(emb).all(), "256-dim ladder must stay finite in fp32"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rms_norm_preserves_input_dtype(dtype):
    nets = make_nets()
    norm = nets._rms_norm(8).to(dtype=dtype)
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    output = norm(x)
    assert output.dtype == dtype
    output.square().mean().backward()
    assert x.grad is not None and x.grad.dtype == dtype
    assert torch.isfinite(x.grad).all()


def test_sinusoidal_encoding_rejects_dim_one_and_clears_cache():
    nets = make_nets()
    with pytest.raises(ValueError, match="dim must be >= 2"):
        nets.SinusoidalEncoding(1)
    encoding = nets.SinusoidalEncoding(8)
    encoding.encode_times(torch.zeros(2))
    assert len(encoding._freq_cache) == 1
    encoding.clear_cache()
    assert not encoding._freq_cache


def test_policy_dit_zero_velocity_at_init():
    nets = make_nets()
    torch.manual_seed(20)
    policy = make_flow(nets, 3)
    v = policy(
        torch.randn(4, CHUNK, 3),
        0.5,
        torch.randn(4, ZD),
    )
    assert v.shape == (4, CHUNK, 3)
    assert torch.isfinite(v).all()
    assert float(v.abs().mean()) == pytest.approx(0.0, abs=1e-5)


def test_actor_ema_warmup_and_update():
    from sjfpo.configuration_sjfpo import SJFPOConfig
    from sjfpo.modeling_sjfpo import SJFPOPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        ema_decay=0.5,
        ema_warmup_steps=2,
    )
    policy = SJFPOPolicy(cfg)
    name, param = next(iter(policy.actor.named_parameters()))
    initial = param.detach().clone()
    policy.update_ema(1)
    shadow = policy.ema_state_dict()["shadow_params"][name]
    assert torch.equal(shadow, initial)
    with torch.no_grad():
        param.add_(1.0)
    policy.update_ema(2)
    shadow = policy.ema_state_dict()["shadow_params"][name]
    assert torch.equal(shadow, param.detach())
    with torch.no_grad():
        param.add_(1.0)
    policy.update_ema(3)
    shadow = policy.ema_state_dict()["shadow_params"][name]
    assert torch.allclose(shadow, param.detach() - 0.5)


def test_policy_first_backward_stages_zero_output_initialization():
    nets = make_nets()
    torch.manual_seed(21)
    policy = make_flow(nets, 3)
    policy(torch.randn(2, CHUNK, 3), 0.5, torch.randn(2, ZD)).sum().backward()
    assert policy.out.weight.grad is not None
    assert torch.count_nonzero(policy.out.weight.grad) > 0
    assert policy.adaLN_out.weight.grad is None or torch.count_nonzero(
        policy.adaLN_out.weight.grad
    ) == 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in policy.blocks.parameters()
    )

    # Critic (PPO state-value baseline V(s))


def test_value_critic_shape():
    nets = make_nets()
    torch.manual_seed(14)
    vc = make_critic(nets, OBS)
    assert next(vc.parameters()).dtype == torch.float32
    v = vc(torch.randn(32, STATE_T, OBS))
    assert v.shape == (32,)
    assert v.dtype == torch.float32
    assert torch.isfinite(v).all()


def test_value_critic_requires_obs_dim():
    nets = make_nets()
    with pytest.raises(ValueError, match="obs_dim"):
        nets.Critic(0, hidden_dim=ZD)


def test_value_critic_follows_state():
    """V(s) must change when the state changes (state conditioning live)."""
    nets = make_nets()
    torch.manual_seed(14)
    vc = make_critic(nets, OBS)
    v1 = vc(torch.zeros(8, STATE_T, OBS))
    v2 = vc(torch.ones(8, STATE_T, OBS))
    assert not torch.equal(v1, v2)


def test_value_critic_gradient_flows():
    nets = make_nets()
    torch.manual_seed(15)
    vc = make_critic(nets, OBS)
    vc(torch.randn(4, STATE_T, OBS)).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in vc.parameters()
    )

    # global hygiene: fp32-only, no BatchNorm / Dropout


def test_no_batchnorm_no_dropout_fp32_everywhere():
    nets = make_nets()
    torch.manual_seed(17)
    modules = [
        make_window(nets, 9),
        make_conditioner(nets, 9),
        make_flow(nets, 7),
            make_critic(nets, 9),

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
    from sjfpo.dct_utils import extract_dct_coeffs
    from sjfpo.utils import filter_low_freq

    assert max(1, round(0.3 * CHUNK)) == 5
    assert max(1, round(1.0 * CHUNK)) == CHUNK
    assert max(1, round(0.01 * CHUNK)) == 1
    a = torch.randn(2, CHUNK, OBS)
    coeffs = extract_dct_coeffs(a, M=CHUNK - 1)
    out = filter_low_freq(coeffs, max(1, round(0.5 * CHUNK)), dim=1)
    assert out.shape == (2, 8, OBS)  # round(0.5 * 16)


def test_window_encoder_position_sensitive():
    """Identical frames still map to distinct latents per position (PE lock)."""
    nets = make_nets()
    torch.manual_seed(2)
    encoder = make_window(nets, 6)
    x = torch.ones(2, STATE_T, 6)  # no order info in the input itself
    z = encoder(x)
    assert not torch.allclose(z[:, 0, :], z[:, 1, :], atol=1e-6)


def test_window_encoder_mask_hides_tails():
    """Garbage tails cannot move real rows: masked as attention keys."""
    nets = make_nets()
    torch.manual_seed(20)
    enc = make_window(nets, 6)
    x = torch.randn(2, STATE_T, 6)
    garb = x.clone()
    garb[:, 8:, :] = 100.0
    attention_mask = torch.arange(STATE_T).unsqueeze(0) < torch.tensor([8, 8]).unsqueeze(1)
    z1 = enc(x, attention_mask)
    z2 = enc(garb, attention_mask)
    assert torch.allclose(z1[:, :8, :], z2[:, :8, :], atol=1e-6)


def test_critic_state_order_sensitive():
    """Flipped state order changes V(s) (state PE lock: mean would hide it)."""
    nets = make_nets()
    torch.manual_seed(21)
    critic = make_critic(nets, OBS)
    s = torch.randn(2, STATE_T, OBS)
    v1 = critic(s)
    v2 = critic(s.flip(dims=(1,)))
    assert not torch.allclose(v1, v2, atol=1e-6)


def test_positional_encoding_shape():
    nets = make_nets()
    out = nets.SinusoidalEncoding(ZD)(torch.zeros(2, STATE_T, ZD))
    assert out.shape == (2, STATE_T, ZD)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])  # positions differ


def test_policy_bin_order_sensitive():
    """Identical action rows still map to distinct outputs (slot-code lock).

    Without position codes, symmetric inputs give symmetric outputs; the
    encoding breaks the symmetry so inter-row relations are addressable."""
    nets = make_nets()
    torch.manual_seed(32)
    policy = make_flow(nets, 3)
    # Wake the zero-init (adaLN gates + final out) so inputs matter.
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.adaLN_out.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    ka = max(1, round(0.3 * CHUNK))  # 5
    x_t = torch.ones(2, ka, 3)  # no order info in the input itself
    v = policy(x_t, 0.5, torch.randn(2, ZD))
    assert v.shape == (2, ka, 3)
    assert not torch.allclose(v[:, 0, :], v[:, 1, :], atol=1e-6)


def test_policy_rejects_more_tokens_than_action_horizon():
    nets = make_nets()
    policy = make_flow(nets, 3)
    with pytest.raises(ValueError, match="action token count"):
        policy(
            torch.randn(1, _DEFAULTS.action_horizon + 1, 3),
            0.5,
            torch.randn(1, ZD),
        )


def test_policy_reads_fused_state():
    """DiT output changes with the fused state vector (conditioning live)."""
    nets = make_nets()
    torch.manual_seed(30)
    policy = make_flow(nets, 3)
    conditioner = make_conditioner(nets, OBS)
    # Wake the zero-init (adaLN gates + final out) so inputs matter.
    with torch.no_grad():
        for block in policy.blocks:
            block.adaLN.fc2.weight.normal_(std=0.1)
        policy.out.weight.normal_(std=0.1)
    x_t = torch.randn(4, CHUNK, 3)
    c1 = conditioner(torch.randn(4, STATE_T, OBS))
    v1 = policy(x_t, 0.5, c1)
    assert v1.shape == (4, CHUNK, 3)  # only action positions read out
    v2 = policy(x_t, 0.5, conditioner(torch.randn(4, STATE_T, OBS)))
    assert not torch.allclose(v1, v2, atol=1e-6)


def test_state_conditioner_shape_and_grad():
    """[B, T, obs] -> [B, D], finite, grads reach every parameter."""
    nets = make_nets()
    torch.manual_seed(33)
    conditioner = make_conditioner(nets, OBS)
    assert next(conditioner.parameters()).dtype == torch.float32
    out = conditioner(torch.randn(4, STATE_T, OBS))
    assert out.shape == (4, ZD)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in conditioner.parameters()
    )


def test_state_conditioner_rejects_invalid_state_shapes():
    nets = make_nets()
    conditioner = make_conditioner(nets, OBS)
    with pytest.raises(ValueError, match="states must have shape"):
        conditioner(torch.randn(4, OBS))
    with pytest.raises(ValueError, match="positive batch and time"):
        conditioner(torch.randn(4, 0, OBS))
    with pytest.raises(ValueError, match="feature dimension"):
        conditioner(torch.randn(4, STATE_T, OBS + 1))


def test_state_conditioner_sees_bin_order():
    """Permuting state rows changes the fused vector (order-sensitive)."""
    nets = make_nets()
    torch.manual_seed(34)
    conditioner = make_conditioner(nets, OBS)
    s = torch.randn(2, STATE_T, OBS)
    v1 = conditioner(s)
    v2 = conditioner(s.flip(dims=(1,)))
    assert not torch.allclose(v1, v2, atol=1e-6)


def test_policy_validates_condition_and_timestep_shapes():
    nets = make_nets()
    policy = make_flow(nets, 3)
    x_t = torch.randn(4, CHUNK, 3)
    condition = torch.randn(4, ZD)
    with pytest.raises(ValueError, match="condition must have shape"):
        policy(x_t, 0.5, torch.randn(3, ZD))
    with pytest.raises(ValueError, match="t must contain"):
        policy(x_t, torch.rand(3), condition)
    with pytest.raises(ValueError, match="t must be scalar"):
        policy(x_t, torch.rand(4, 1), condition)
    with pytest.raises(ValueError, match="x_t must have shape"):
        policy(torch.randn(4, CHUNK), 0.5, condition)


def test_policy_forward_truncated_coeff_space():
    """DiT denoises Ka=top-p rows; zero-pad restores full H (shape lock)."""
    nets = make_nets()
    torch.manual_seed(31)
    policy = make_flow(nets, 3)
    ka = max(1, round(0.3 * CHUNK))  # 5
    v = policy(
        torch.randn(4, ka, 3),
        0.5,
        torch.randn(4, ZD),
    )
    assert v.shape == (4, ka, 3)
    padded = torch.nn.functional.pad(v, (0, 0, 0, CHUNK - ka))
    assert padded.shape == (4, CHUNK, 3)
    assert (padded[:, ka:, :] == 0).all()  # missing bins are exactly zero
