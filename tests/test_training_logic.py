"""Unit test: training logic produces expected losses and updates."""
import torch
from myfpo.training import build_policy
from myfpo.configuration import MyfpoConfig
from lerobot.configs.types import FeatureType, PolicyFeature

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
        "normalizer_observations": torch.zeros(10, 5),
    }
    loss, _ = policy(batch, compute_info=True)
    assert loss.dim() == 0, f"loss must be scalar, got {loss.shape}"
    assert torch.isfinite(loss).all(), "loss must be finite"
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters() if p.requires_grad)
    assert has_grad, "actor gradients must be non-zero"
