"""Actor and critic gradients clip independently.

One shared clip lets whichever side has the larger gradient shrink the other.
The regression these guard is a large actor gradient silently attenuating a
small critic gradient, which the joint clip does at the cost of both.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import SJFPOConfig
from myfpo.modeling import SJFPOPolicy
from myfpo.training import split_grad_params

OBS, ACT, HORIZON, STEPS = 6, 3, 8, 4


def _policy(**kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(
            action_horizon=HORIZON,
            hidden_dim=32,
            n_obs_steps=STEPS,
            dit_depth=1,
            dit_heads=2,
            num_critics=5,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )
    torch.manual_seed(0)
    return SJFPOPolicy(cfg)


def _set_grad(params, norm: float) -> None:
    """Give every element of every tensor the value that makes the group norm exact."""
    elements = sum(p.numel() for p in params)
    for p in params:
        p.grad = torch.full_like(p, norm / elements**0.5)


def _grad_norm(params) -> float:
    grads = [p.grad.reshape(-1) for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads)))


def test_the_split_partitions_every_trainable_parameter():
    policy = _policy()
    groups = split_grad_params(policy)
    assert set(groups) == {"actor", "critic"}
    flat = [id(p) for params in groups.values() for p in params]
    assert len(flat) == len(set(flat)), "a parameter is in both groups"
    trainable = {id(p) for p in policy.parameters() if p.requires_grad}
    assert set(flat) == trainable


def test_the_ensemble_width_does_not_change_the_split():
    one = split_grad_params(_policy(num_critics=1))
    five = split_grad_params(_policy(num_critics=5))
    assert len(one["critic"]) != len(five["critic"])
    assert len(one["actor"]) == len(five["actor"])


def test_a_large_actor_gradient_leaves_a_small_critic_gradient_alone():
    groups = split_grad_params(_policy())
    _set_grad(groups["actor"], 10.0)
    _set_grad(groups["critic"], 1e-3)
    for params in groups.values():
        torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert _grad_norm(groups["actor"]) == pytest.approx(1.0, rel=1e-3)
    assert _grad_norm(groups["critic"]) == pytest.approx(1e-3, rel=1e-3)


def test_the_joint_clip_would_have_attenuated_both():
    """The old behaviour, kept as the contrast that makes the test above mean something."""
    groups = split_grad_params(_policy())
    _set_grad(groups["actor"], 10.0)
    _set_grad(groups["critic"], 1e-3)
    flat = [p for params in groups.values() for p in params]
    torch.nn.utils.clip_grad_norm_(flat, 1.0)
    assert _grad_norm(groups["critic"]) < 1e-3


@pytest.mark.parametrize("scale", [1.0, 37.0, 1e-4])
def test_the_logged_total_equals_the_norms_over_the_whole_parameter_set(scale):
    groups = split_grad_params(_policy())
    _set_grad(groups["actor"], scale)
    _set_grad(groups["critic"], scale * 3)
    flat = [p for params in groups.values() for p in params]
    joint = _grad_norm(flat)
    logged = sum(_grad_norm(params) ** 2 for params in groups.values()) ** 0.5
    # rel=1e-3, not tighter: the two routes sum ~1.5M fp32 elements in a
    # different order. A wrong grouping would disagree by O(1), not 1e-3.
    assert logged == pytest.approx(joint, rel=1e-3)