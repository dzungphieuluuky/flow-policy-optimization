"""``actor_arch: mlp`` must be FPO++'s actor, not a smaller DiT."""

import dataclasses

import pytest
import torch

from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.flow import SJFPOFlowMLP, SJFPOFlowModel
from myfpo.train_sjfpo import build_policy

OBS, ACT, HORIZON, HIDDEN, TIME_EMBED = 6, 3, 8, 32, 16


def _cfg(**kw):
    return dataclasses.replace(
        SJFPOConfig(
            action_horizon=HORIZON,
            hidden_dim=HIDDEN,
            time_embed_dim=TIME_EMBED,
            dit_depth=1,
            dit_heads=2,
            n_obs_steps=2,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )


def _mlp(hidden_dims=(512, 512, 512)):
    return SJFPOFlowMLP(ACT, HORIZON, HIDDEN, TIME_EMBED, hidden_dims)


def test_the_first_layer_is_the_reference_concatenation():
    """``K*D`` chunk + the time embedding + the conditioning, nothing else."""
    mlp = _mlp()
    first = mlp.mlp[0]
    assert first.in_features == HORIZON * ACT + HORIZON * TIME_EMBED + HIDDEN
    assert isinstance(mlp.mlp[1], torch.nn.Mish)


def test_the_last_layer_maps_back_to_the_chunk():
    last = _mlp().mlp[-1]
    assert last.out_features == HORIZON * ACT


def test_shapes_match_the_dit_contract():
    """Same signature as SJFPOFlowModel, so ``actor_arch`` swaps nothing else."""
    mlp, dit = _mlp(), SJFPOFlowModel(
        ACT, HORIZON, HIDDEN, 1, 2, 2.0, TIME_EMBED
    )
    x = torch.randn(3, HORIZON, ACT)
    t = torch.rand(3, HORIZON, 1)
    cond = torch.randn(3, HIDDEN)
    for net in (mlp, dit):
        assert net(x, t, cond).shape == x.shape


def test_untrained_predicts_zero_velocity_like_the_dit():
    """Both zero their output layer, so a fresh policy is the identity flow."""
    mlp = _mlp()
    out = mlp(torch.randn(3, HORIZON, ACT), torch.rand(3, HORIZON, 1),
              torch.randn(3, HIDDEN))
    assert torch.equal(out, torch.zeros_like(out))


def test_a_scalar_time_is_broadcast_over_the_chunk():
    mlp = _mlp()
    x, cond = torch.randn(3, HORIZON, ACT), torch.randn(3, HIDDEN)
    assert mlp(x, torch.tensor(0.4), cond).shape == x.shape
    assert mlp(x, 0.4, cond).shape == x.shape


def test_a_per_chunk_time_is_broadcast_over_the_tokens():
    mlp = _mlp()
    x, cond = torch.randn(3, HORIZON, ACT), torch.randn(3, HIDDEN)
    assert mlp(x, torch.rand(3), cond).shape == x.shape


def test_every_token_reaches_the_output():
    """The chunk is flattened, so a late token cannot be ignored."""
    torch.manual_seed(0)
    mlp = _mlp(hidden_dims=(64,))
    torch.nn.init.normal_(mlp.mlp[-1].weight, std=0.1)
    x = torch.zeros(2, HORIZON, ACT)
    cond = torch.randn(2, HIDDEN)

    base = mlp(x, torch.rand(2, HORIZON, 1), cond)
    bumped = x.clone()
    bumped[0, HORIZON - 1, 0] = 1.0
    moved = mlp(bumped, torch.rand(2, HORIZON, 1) * 0, cond)
    assert not torch.allclose(moved[0], base[0], atol=1e-6)


def test_the_policy_builds_either_actor():
    for arch, cls in (("dit", SJFPOFlowModel), ("mlp", SJFPOFlowMLP)):
        # compile=False: torch.compile wraps the module, so the class check would.
        policy = build_policy(
            OBS, ACT, _cfg(actor_arch=arch, compile=False), device="cpu",
            action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
        )
        assert isinstance(policy.actor, cls), arch


def test_an_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="actor_arch"):
        _cfg(actor_arch="unet").validate()