"""Critic trunk: LayerNorm + Mish, and a two-member ensemble by default.

Follows TD-MPC2 (arXiv:2310.16828): every hidden layer is Linear -> LayerNorm
-> Mish. The old trunk was bare Linear -> ReLU with no normalisation at all.
TD-MPC2 ships 5 ensemble members; this repo runs 2, and
``--myfpo.num_critics 5`` restores the paper's width.

The action and state inputs are the two places the trunk departs from the paper.
The action is a learned affine, not a LayerNorm, because a LayerNorm over the
chunk erases the chunk's offset. The state norm divides by the RMS and keeps
the mean, for the same reason. See ``myfpo.critic.ActionAffine`` and
``StateAffine``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType
from torch import nn

from myfpo.configuration import MyfpoConfig
from myfpo.networks import Critic, CriticMember, ActionAffine, StateAffine
from myfpo.fpo_core import compute_value_metrics
from myfpo.modeling import MyfpoPolicy

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _cfg(**kw) -> MyfpoConfig:
    return dataclasses.replace(
        MyfpoConfig(
            action_horizon=H,
            n_action_steps=H,
            hidden_dim=HIDDEN,
            n_obs_steps=STEPS,
            dit_depth=1,
            dit_heads=2,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (OBS,))
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )


def _member(num_layers: int = 2, dropout: float = 0.0) -> CriticMember:
    torch.manual_seed(0)
    return CriticMember(
        OBS, ACT, HIDDEN, STEPS, H, num_layers=num_layers, dropout=dropout
    )


@pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
def test_every_hidden_layer_is_linear_layernorm_mish(num_layers):
    mods = list(_member(num_layers).mlp)
    hidden = mods[:-1]
    assert len(hidden) == 3 * num_layers
    for i in range(num_layers):
        assert isinstance(hidden[3 * i], nn.Linear)
        assert isinstance(hidden[3 * i + 1], nn.LayerNorm)
        assert isinstance(hidden[3 * i + 2], nn.Mish)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_the_output_layer_is_a_bare_linear(num_layers):
    assert isinstance(_member(num_layers).mlp[-1], nn.Linear)
    assert _member(num_layers).mlp[-1].out_features == 1


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_there_is_no_relu_left(num_layers):
    assert not any(isinstance(m, nn.ReLU) for m in _member(num_layers).mlp)


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_the_layernorms_are_on_the_hidden_width(num_layers):
    for m in _member(num_layers).mlp:
        if isinstance(m, nn.LayerNorm):
            assert m.normalized_shape == (HIDDEN,)


def test_the_first_linear_takes_the_pool_and_not_the_chunk():
    """The pool is the last state token and the mean action token."""
    first = _member(2).mlp[0]
    assert first.in_features == 2 * HIDDEN + 1


def test_the_default_ensemble_is_two():
    """The repo default is 2, not TD-MPC2's 5. The paper's width is a flag."""
    assert MyfpoConfig().num_critics == 2
    assert Critic(OBS, HIDDEN, ACT, STEPS, H, 2).num_critics == 2


def test_the_members_stay_independent():
    torch.manual_seed(0)
    critic = Critic(OBS, HIDDEN, ACT, STEPS, H, num_critics=5)
    assert len(critic.networks) == 5
    w0 = critic.networks[0].mlp[0].weight
    for other in critic.networks[1:]:
        assert not torch.equal(w0, other.mlp[0].weight)


def test_a_saved_config_round_trips_with_the_new_ensemble():
    cfg = dataclasses.replace(MyfpoConfig(), num_critics=5)
    assert cfg.num_critics == 5
    cfg.validate()


def test_the_dropout_sits_directly_after_the_first_linear():
    mods = list(_member(2, dropout=0.1).mlp)
    kinds = [type(m).__name__ for m in mods]
    assert kinds == [
        "Linear", "Dropout", "LayerNorm", "Mish",
        "Linear", "LayerNorm", "Mish",
        "Linear",
    ]


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_there_is_exactly_one_dropout(num_layers):
    assert sum(isinstance(m, nn.Dropout) for m in _member(num_layers, 0.1).mlp) == 1


def test_zero_dropout_leaves_no_dropout_module():
    assert not any(isinstance(m, nn.Dropout) for m in _member(3).mlp)


def test_the_constructors_default_to_off_so_a_bare_critic_is_deterministic():
    """The config default is on; the bare constructor turns it off."""
    assert Critic(OBS, HIDDEN, ACT, STEPS, H, 2).dropout == 0.0
    assert MyfpoPolicy(_cfg()).critic.dropout == 0.01


def test_dropout_is_active_in_train_mode_and_off_in_eval():
    net = _member(2, dropout=0.5)
    x = torch.randn(8, net.input_dim)
    net.train()
    a, b = net.mlp(x), net.mlp(x)
    net.eval()
    c, d = net.mlp(x), net.mlp(x)
    assert not torch.equal(a, b)
    assert torch.equal(c, d)


def test_the_members_get_independent_masks_in_one_pass():
    torch.manual_seed(0)
    critic = Critic(OBS, HIDDEN, ACT, STEPS, H, num_critics=5, dropout=0.5).train()
    out = critic(
        torch.randn(4, STEPS, OBS),
        torch.randn(4, H, ACT),
        torch.full((4,), 0.5),
    )
    assert out.shape == (5, 4)
    assert not torch.equal(out[0], out[1])


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_the_config_rejects_a_dropout_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match=r"critic_dropout=.* must be in \[0, 1\)"):
        dataclasses.replace(MyfpoConfig(), critic_dropout=bad)


def test_dropout_adds_no_parameters():
    a = sum(p.numel() for p in Critic(OBS, HIDDEN, ACT, STEPS, H, 5).parameters())
    a = sum(p.numel() for p in Critic(OBS, HIDDEN, ACT, STEPS, H, 5).parameters())
    b = sum(
        p.numel() for p in Critic(OBS, HIDDEN, ACT, STEPS, H, 5, dropout=0.5).parameters()
    )
    assert a == b


# --- per-part input norms ------------------------------------------------------


def test_the_state_and_the_action_are_treated_separately():
    """The state gets an RMS affine. The action gets a learned affine.

    A LayerNorm over the chunk would subtract the chunk's mean, so a chunk of
    all +5 and a chunk of all 0 would reach the critic as the same input.
    """
    m = _member()
    assert isinstance(m.norm_state, StateAffine)
    assert isinstance(m.norm_action, ActionAffine)
    assert m.norm_state.weight.shape == (STEPS * OBS,)
    assert m.norm_state.bias.shape == (STEPS * OBS,)
    assert m.norm_action.weight.shape == (H * ACT,)
    assert m.norm_action.bias.shape == (H * ACT,)


def test_the_output_is_invariant_to_a_global_scale_on_the_observation():
    """The property the state norm buys: the state's scale does not move Q.

    The offset does, and that is the point of ``StateAffine``: a LayerNorm
    subtracted it, so a raised state and the original state were one input.
    """
    torch.manual_seed(1)
    m = _member().eval()
    obs = torch.randn(4, STEPS, OBS)
    act = torch.randn(4, H, ACT)
    frac = torch.full((4,), 0.5)
    base = m(obs, act, frac)
    assert torch.allclose(m(2.0 * obs, act, frac), base, atol=1e-5)
    assert not torch.allclose(m(obs + 3.0, act, frac), base, atol=1e-5)


def test_the_output_is_not_invariant_to_a_global_affine_on_the_action():
    """The property the action affine buys.

    A LayerNorm over the chunk subtracted the chunk's mean, so the offset of
    the chunk never reached the critic: a sustained push and no push were the
    same input.
    """
    torch.manual_seed(1)
    m = _member().eval()
    obs = torch.randn(4, STEPS, OBS)
    act = torch.randn(4, H, ACT)
    frac = torch.full((4,), 0.5)
    base = m(obs, act, frac)
    assert not torch.allclose(m(obs, act + 3.0, frac), base, atol=1e-5)
    assert not torch.allclose(m(obs, 0.5 * act, frac), base, atol=1e-5)


def test_a_constant_chunk_keeps_its_offset():
    """The sustained-push case.

    A LayerNorm mapped every constant chunk to the same input, so the critic
    could not tell a push from no push.
    """
    torch.manual_seed(1)
    m = _member().eval()
    obs = torch.randn(4, STEPS, OBS)
    frac = torch.full((4,), 0.5)
    push = torch.full((4, H, ACT), 5.0)
    idle = torch.zeros(4, H, ACT)
    assert not torch.allclose(m(obs, push, frac), m(obs, idle, frac), atol=1e-5)


def test_the_action_scale_loads_a_layernorm_state_dict():
    """A checkpoint written before this change must still load."""
    m = _member()
    m.norm_action.load_state_dict(nn.LayerNorm(H * ACT).state_dict())


def test_the_gradient_reaches_the_action_scale():
    """Guards a detached or accidentally frozen scale."""
    m = _member()
    m(torch.randn(4, STEPS, OBS), torch.randn(4, H, ACT), torch.full((4,), 0.5)).sum().backward()
    assert m.norm_action.weight.grad is not None
    assert m.norm_action.weight.grad.abs().sum() > 0
    assert m.norm_action.bias.grad.abs().sum() > 0


def test_the_action_affine_runs_once_per_forward():
    """One affine pass over the chunk.

    A second pass squares the scale: two passes compute ``w * (w * x + b) + b``,
    so the chunk reaches the critic with the weight applied twice. The value
    changes and the gradient on the weight doubles. The count is the check.
    """
    m = _member()
    calls: list[tuple] = []
    original = m.norm_action.forward
    m.norm_action.forward = lambda x: (calls.append(tuple(x.shape)), original(x))[1]
    m(torch.randn(3, STEPS, OBS), torch.randn(3, H, ACT), torch.full((3,), 0.5))
    assert len(calls) == 1


def test_the_chunk_reaches_the_action_tokens_through_one_affine():
    """The affine is applied once, so the chunk keeps its own scale."""
    torch.manual_seed(3)
    m = _member().eval()
    with torch.no_grad():
        m.norm_action.weight.copy_(
            torch.linspace(0.5, 2.0, H * ACT).reshape(H * ACT)
        )
        m.norm_action.bias.copy_(torch.linspace(-0.3, 0.4, H * ACT))
    seen: dict[str, torch.Tensor] = {}
    m.act_in.register_forward_pre_hook(
        lambda _module, args: seen.__setitem__("x", args[0].detach().clone())
    )
    obs = torch.randn(2, STEPS, OBS)
    act = torch.randn(2, H, ACT)
    m(obs, act, torch.full((2,), 0.5))
    expected = (
        act.reshape(2, H * ACT) * m.norm_action.weight + m.norm_action.bias
    ).view(2, H, ACT)
    assert torch.equal(seen["x"], expected)


def test_the_input_parts_add_exactly_their_own_widths():
    """Two affines, a chunk attention block, two cross blocks, and the token
    projection, beyond the MLP."""
    m = _member()
    with_parts = sum(p.numel() for p in m.parameters())
    mlp = sum(p.numel() for p in m.mlp.parameters())
    encoder = sum(p.numel() for p in m.state_encoder.parameters())
    blocks = (
        sum(p.numel() for p in m.obs_to_act.parameters())
        + sum(p.numel() for p in m.act_to_obs.parameters())
        + sum(p.numel() for p in m.cross_norm_obs.parameters())
        + sum(p.numel() for p in m.cross_norm_act.parameters())
        + sum(p.numel() for p in m.act_attn.parameters())
        + sum(p.numel() for p in m.act_attn_norm.parameters())
        + sum(p.numel() for p in m.act_in.parameters())
    )
    assert with_parts == mlp + encoder + blocks + 2 * (STEPS * OBS) + 2 * (H * ACT)


def test_the_shared_encoder_branch_skips_the_state_trunk():
    torch.manual_seed(0)
    m = CriticMember(OBS, ACT, HIDDEN, 1, H, condition_dim=HIDDEN)
    assert m.norm_state is None
    assert m.state_encoder is None
    assert isinstance(m.norm_action, ActionAffine)
    assert m.input_dim == 2 * HIDDEN + 1


def test_the_state_norm_receives_the_flattened_window_not_the_chunk():
    m = _member()
    seen = {}
    original = m.norm_state.forward
    m.norm_state.forward = lambda x: (seen.setdefault("shape", tuple(x.shape)), original(x))[1]
    m(torch.randn(3, STEPS, OBS), torch.randn(3, H, ACT), torch.full((3,), 0.5))
    assert seen["shape"] == (3, STEPS * OBS)


def test_the_member_gap_p95_flags_a_collapsed_ensemble():
    """The gap is the only signal that the ensemble has not collapsed onto one function."""
    n = 64
    collapsed = torch.zeros(5, n)
    healthy = torch.arange(5, dtype=torch.float32).unsqueeze(1).expand(5, n).clone()
    assert _gap_p95(collapsed) == pytest.approx(0.0)
    assert _gap_p95(healthy) == pytest.approx(4.0)
    assert _gap_p95(healthy) > _gap_p95(collapsed)


def _gap_p95(pred: torch.Tensor) -> float:
    out = compute_value_metrics(
        raw_value_pred=pred,
        raw_returns_flat=torch.zeros(pred.shape[1]),
        policy_per_chunk=torch.zeros(pred.shape[1]),
        value_loss_per_chunk=torch.zeros(pred.shape[1]),
    )
    return float(out["value_pred_member_gap_p95"])


def test_the_member_correlation_flags_a_collapsed_ensemble():
    """A correlation near one means the members agree on every state."""
    n = 64
    base = torch.randn(n)
    collapsed = base.unsqueeze(0).expand(5, n).clone()
    independent = torch.randn(5, n)
    assert _member_corr(collapsed) == pytest.approx(1.0)
    assert _member_corr(independent) < 0.5
    assert _member_corr(collapsed) > _member_corr(independent)


def test_the_correlation_and_the_gap_read_a_collapse_together():
    """The correlation is invariant to a shift and a scale. The gap is not.

    So a member that is the same function with an offset reads correlation one
    and a nonzero gap. A collapse is correlation one and a gap of zero.
    """
    n = 64
    base = torch.randn(n)
    offset = torch.stack([base, base + 0.5])
    identical = base.unsqueeze(0).expand(2, n).clone()
    assert _member_corr(offset) == pytest.approx(1.0)
    assert _member_gap(offset) > 0.0
    assert _member_corr(identical) == pytest.approx(1.0)
    assert _member_gap(identical) == pytest.approx(0.0)


def test_one_member_has_no_pair_so_the_correlation_is_nan():
    """NaN is how the repo says "unmeasured"."""
    out = compute_value_metrics(
        raw_value_pred=torch.zeros(1, 8),
        raw_returns_flat=torch.zeros(8),
        policy_per_chunk=torch.zeros(8),
        value_loss_per_chunk=torch.zeros(8),
    )
    assert math.isnan(float(out["value_pred_member_corr"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device mixing bites on cuda")
def test_the_nan_branch_sits_on_the_input_device():
    """The nan branch must not build its tensor on the default device.

    One cpu tensor among cuda ones makes the stack in compute_policy_info
    raise "Expected all tensors to be on the same device". This is a regression
    test for that abort, so it only runs where the failure can.
    """
    values = torch.zeros(1, 8, device="cuda")
    out = compute_value_metrics(
        raw_value_pred=values,
        raw_returns_flat=torch.zeros(8, device="cuda"),
        policy_per_chunk=torch.zeros(8, device="cuda"),
        value_loss_per_chunk=torch.zeros(8, device="cuda"),
    )
    assert out["value_pred_member_corr"].is_cuda


def test_a_constant_member_has_no_correlation_to_measure():
    """A member that gives the same value on every state has no spread."""
    pred = torch.randn(2, 8)
    pred[1] = 3.0
    out = compute_value_metrics(
        raw_value_pred=pred,
        raw_returns_flat=torch.zeros(8),
        policy_per_chunk=torch.zeros(8),
        value_loss_per_chunk=torch.zeros(8),
    )
    assert math.isnan(float(out["value_pred_member_corr"]))


def _member_corr(pred: torch.Tensor) -> float:
    out = compute_value_metrics(
        raw_value_pred=pred,
        raw_returns_flat=torch.zeros(pred.shape[1]),
        policy_per_chunk=torch.zeros(pred.shape[1]),
        value_loss_per_chunk=torch.zeros(pred.shape[1]),
    )
    return float(out["value_pred_member_corr"])


def _member_gap(pred: torch.Tensor) -> float:
    out = compute_value_metrics(
        raw_value_pred=pred,
        raw_returns_flat=torch.zeros(pred.shape[1]),
        policy_per_chunk=torch.zeros(pred.shape[1]),
        value_loss_per_chunk=torch.zeros(pred.shape[1]),
    )
    return float(out["value_pred_member_gap"])