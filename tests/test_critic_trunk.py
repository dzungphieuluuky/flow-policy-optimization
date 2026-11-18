"""Critic trunk: LayerNorm + Mish, single head.

Every hidden layer is Linear -> LayerNorm -> Mish. The action and state inputs
are the two places the trunk departs from a vanilla MLP: the action is a
learned affine, not a LayerNorm, because a LayerNorm over the chunk erases the
chunk's offset; the state norm divides by the RMS and keeps the mean, for the
same reason. See ``ActionAffine`` and ``StateAffine``.
"""

from __future__ import annotations


import pytest
import torch
from torch import nn

from flow_policy_optimization.flow_policy_optimization import Critic, ActionAffine, StateAffine
from conftest import unzero_critic_head

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _member(num_layers: int = 2) -> Critic:
    torch.manual_seed(0)
    member = Critic(OBS, HIDDEN, ACT, STEPS, H, num_layers=num_layers, state_hidden=HIDDEN)
    unzero_critic_head(member)
    return member


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
    assert with_parts == mlp + encoder + blocks + 2 * (STEPS * OBS) + 2 * (H * ACT) + (HIDDEN + 1)


def test_the_state_norm_receives_the_flattened_window_not_the_chunk():
    m = _member()
    seen = {}
    original = m.norm_state.forward
    m.norm_state.forward = lambda x: (seen.setdefault("shape", tuple(x.shape)), original(x))[1]
    m(torch.randn(3, STEPS, OBS), torch.randn(3, H, ACT), torch.full((3,), 0.5))
    assert seen["shape"] == (3, STEPS * OBS)
