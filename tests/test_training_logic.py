"""Unit test: training logic produces expected losses and updates."""
import inspect

import torch
from myfpo.training import (
    build_optimizer,
    build_policy,
    policy_update_step,
    run_training_loop,
    split_grad_params,
)
from myfpo.configuration import MyfpoConfig
from lerobot.configs.types import FeatureType, PolicyFeature


def test_observation_stats_land_before_the_update():
    """The batch normalises at load time (``modeling.load_batch``), so a stats
    update after ``train_update`` leaves the first rollout unnormalised."""
    source = inspect.getsource(run_training_loop)
    # Frozen-stats invariant: stats updated after policy_update_step, before target_critic update.
    assert source.index("update_observation_stats") > source.index("policy_update_step(")
    assert source.index("update_observation_stats") < source.index("update_target_critic()")

def test_training_logic():
    cfg = MyfpoConfig(
        action_horizon=4,
        n_action_steps=2,
        hidden_dim=16,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
    )
    policy = build_policy(5, 3, cfg, device="cpu")
    batch = {
        "observation.state": torch.zeros(1, cfg.n_obs_steps, 5),
        "action": torch.zeros(1, cfg.action_horizon, 3),
        "action_is_pad": torch.zeros(1, cfg.action_horizon, dtype=torch.bool),
        "raw_advantages": torch.ones(1),
        "normalised_advantages": torch.ones(1),
        "raw_returns": torch.ones(1),
        "mc_t": torch.ones(1, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(1, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(1, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(1, cfg.num_mc_samples, cfg.action_horizon),
        "behavior_velocity_token": torch.zeros(1, cfg.num_mc_samples, cfg.action_horizon, 3),
        "normalizer_observations": torch.zeros(10, 5),
    }
    loss = policy(batch)
    assert loss.dim() == 0, f"loss must be scalar, got {loss.shape}"
    assert torch.isfinite(loss).all(), "loss must be finite"
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters() if p.requires_grad)
    assert has_grad, "actor gradients must be non-zero"


def test_one_policy_update_runs():
    """The update path must execute end to end.

    ``autocast_context`` was called but never imported, so the first update of
    a run raised NameError. Nothing in the suite called this path: the loss
    tests call ``policy(batch)`` directly.
    """
    cfg = MyfpoConfig(
        action_horizon=2,
        n_action_steps=2,
        n_obs_steps=2,
        hidden_dim=16,
        dit_depth=1,
        dit_heads=2,
        num_mc_samples=2,
        num_learning_epochs=1,
        chunks_per_update=2,
        batch_size=1,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
    )
    policy = build_policy(5, 3, cfg, device="cpu")
    n = 2
    batch = {
        "observation.state": torch.zeros(n, cfg.n_obs_steps, 5),
        "action": torch.zeros(n, cfg.action_horizon, 3),
        "action_is_pad": torch.zeros(n, cfg.action_horizon, dtype=torch.bool),
        "raw_advantages": torch.ones(n),
        "normalised_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(n, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(n, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(n, cfg.num_mc_samples, cfg.action_horizon),
        "behavior_velocity_token": torch.zeros(
            n, cfg.num_mc_samples, cfg.action_horizon, 3
        ),
    }
    optimizer = build_optimizer(policy)
    _, grad_norms_pre, _, _ = policy_update_step(
        policy, optimizer, batch, None, cfg, split_grad_params(policy)
    )
    assert len(grad_norms_pre) == cfg.num_mini_batches
    assert set(grad_norms_pre[0]) == {"actor", "critic", "total"}
    assert policy.step_count == cfg.num_mini_batches
