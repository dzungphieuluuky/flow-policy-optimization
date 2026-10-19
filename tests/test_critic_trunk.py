"""Critic trunk: LayerNorm + Mish, and a five-member ensemble by default.

Follows TD-MPC2 (arXiv:2310.16828): every hidden layer is Linear -> LayerNorm
-> Mish, and the ensemble is 5 wide. The old trunk was bare Linear -> ReLU with
no normalisation at all.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.critic import Critic, CriticMember

OBS, HIDDEN, STEPS, ACT, H = 11, 32, 8, 3, 6


def _member(num_layers: int = 2) -> CriticMember:
    torch.manual_seed(0)
    return CriticMember(OBS, ACT, HIDDEN, STEPS, H, num_layers=num_layers)


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


def test_the_first_linear_takes_the_concatenated_input():
    first = _member(2).mlp[0]
    assert first.in_features == STEPS * OBS + H * ACT + 1


def test_the_default_ensemble_is_five():
    assert SJFPOConfig().num_critics == 5
    assert Critic(OBS, HIDDEN, ACT, STEPS, H).num_critics == 5


def test_the_members_stay_independent():
    torch.manual_seed(0)
    critic = Critic(OBS, HIDDEN, ACT, STEPS, H, num_critics=5)
    assert len(critic.networks) == 5
    w0 = critic.networks[0].mlp[0].weight
    for other in critic.networks[1:]:
        assert not torch.equal(w0, other.mlp[0].weight)


def test_a_saved_config_round_trips_with_the_new_ensemble():
    cfg = dataclasses.replace(SJFPOConfig(), num_critics=5)
    assert cfg.num_critics == 5
    cfg.validate()


def test_num_critics_must_be_positive():
    with pytest.raises(ValueError, match="num_critics must be >= 1"):
        Critic(OBS, HIDDEN, ACT, STEPS, H, num_critics=0)