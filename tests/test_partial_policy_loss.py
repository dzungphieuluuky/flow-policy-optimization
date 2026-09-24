"""Regression tests for action-chunk validity in receding-horizon training."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402


OBS_DIM = 3
ACT_DIM = 2
HORIZON = 4


def make_policy(use_dct: bool, vf_coef: float = 1.0) -> SJFPOPolicy:
    config = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS_DIM,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
        n_obs_steps=4,
        action_horizon=HORIZON,
        n_action_steps=HORIZON,
        num_mc_samples=2,
        hidden_dim=8,
        dit_depth=1,
        dit_heads=2,
        dit_mlp_ratio=2.0,
        dit_time_dim=8,
        num_euler_steps=2,
        num_inference_steps=2,
        use_dct_transform=use_dct,
        vf_coef=vf_coef,
    )
    return SJFPOPolicy(config)


def make_batch(all_partial: bool) -> dict[str, torch.Tensor]:
    is_pad = torch.zeros(2, HORIZON, dtype=torch.bool)
    if all_partial:
        is_pad[:] = True
    else:
        is_pad[1, -2:] = True
    return {
        "observation.state": torch.randn(2, 4, OBS_DIM),
        "action": torch.randn(2, HORIZON, ACT_DIM).clamp(-1.0, 1.0),
        "action_is_pad": is_pad,
        "advantages": torch.tensor([0.4, 0.4]),
        "returns": torch.tensor([1.0, 1.0]),
    }


@pytest.mark.parametrize("use_dct", [False, True])
def test_empty_action_chunks_do_not_receive_policy_gradient(use_dct: bool):
    policy = make_policy(use_dct, vf_coef=1.0)
    loss, _ = policy(make_batch(all_partial=True), compute_info=False)
    loss.backward()

    actor_grads = [p.grad for p in policy.actor.parameters()]
    assert actor_grads
    assert all(grad is None or float(grad.abs().sum()) == 0.0 for grad in actor_grads)
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0
        for p in policy.critic.parameters()
    )


def test_receding_horizon_chunks_receive_policy_gradient():
    policy = make_policy(use_dct=True, vf_coef=0.0)
    loss_vector, _ = policy(
        make_batch(all_partial=False), reduction="none", compute_info=False
    )
    assert loss_vector[0].abs() > 0.0
    assert loss_vector[1].abs() > 0.0
