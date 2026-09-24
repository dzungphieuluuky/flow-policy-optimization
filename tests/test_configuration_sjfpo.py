"""Tests for src/sjfpo/configuration_sjfpo.py (LeRobot-native SJFPO config)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402


def test_registered_as_sjfpo_choice():
    assert SJFPOConfig().type == "sjfpo"


def test_canonical_defaults():
    cfg = SJFPOConfig()
    assert (cfg.n_obs_steps, cfg.action_horizon, cfg.hidden_dim) == (16, 16, 128)
    assert cfg.n_action_steps == cfg.action_horizon
    assert (cfg.policy_lr, cfg.optimizer_weight_decay) == (3e-4, 1e-2)
    assert cfg.grad_clip_norm == 0.5
    assert cfg.num_epochs == 4
    assert cfg.chunks_per_update == 256


def test_post_init_rejects_bad_geometry():
    with pytest.raises(ValueError, match="n_action_steps"):
        SJFPOConfig(n_action_steps=17)
    with pytest.raises(ValueError, match="n_action_steps"):
        SJFPOConfig(n_action_steps=0)
    with pytest.raises(ValueError, match="top_p_spectral_tokens"):
        SJFPOConfig(top_p_spectral_tokens=1.5)
    with pytest.raises(ValueError, match="top_p_spectral_tokens"):
        SJFPOConfig(top_p_spectral_tokens=0.0)


def test_delta_indices():
    cfg = SJFPOConfig()
    assert cfg.observation_delta_indices == list(range(-15, 1))
    assert cfg.action_delta_indices == list(range(16))
    assert cfg.reward_delta_indices is None


def test_optimizer_preset_and_no_scheduler():
    cfg = SJFPOConfig()
    opt = cfg.get_optimizer_preset()
    assert (opt.lr, opt.weight_decay, opt.grad_clip_norm) == (
        cfg.policy_lr,
        cfg.optimizer_weight_decay,
        cfg.grad_clip_norm,
    )
    assert cfg.get_scheduler_preset() is None


def test_validate_features_needs_state_and_action():
    with pytest.raises(ValueError, match="observation.state"):
        SJFPOConfig().validate_features()
    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (17,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
    )
    cfg.validate_features()


def test_post_init_rejects_bad_optimizer():
    with pytest.raises(ValueError, match="policy_lr"):
        SJFPOConfig(policy_lr=0.0)


def test_post_init_rejects_bad_timestep_range():
    with pytest.raises(ValueError, match="min_timestep"):
        SJFPOConfig(min_timestep=0.5, max_timestep=0.1)
    with pytest.raises(ValueError, match="min_timestep"):
        SJFPOConfig(min_timestep=1.0, max_timestep=1.0)
    with pytest.raises(ValueError, match="epsilon_clip"):
        SJFPOConfig(epsilon_clip=0.0)
    with pytest.raises(ValueError, match="cfm_clamp"):
        SJFPOConfig(cfm_clamp=0.0)
    # OFF endpoints are valid.
    assert SJFPOConfig(min_timestep=0.005, max_timestep=0.995).min_timestep == 0.005


def test_per_bin_clip_defaults_off_and_validated():
    cfg = SJFPOConfig()
    assert cfg.use_per_bin_clip is False
    assert (cfg.epsilon_clip_lowfreq, cfg.epsilon_clip_highfreq) == (0.02, 0.10)
    with pytest.raises(ValueError, match="epsilon_clip_lowfreq"):
        SJFPOConfig(
            use_dct_transform=True, use_per_bin_clip=True, epsilon_clip_lowfreq=0.0
        )
    with pytest.raises(ValueError, match="epsilon_clip_lowfreq"):
        SJFPOConfig(
            use_dct_transform=True,
            use_per_bin_clip=True,
            epsilon_clip_lowfreq=0.2,
            epsilon_clip_highfreq=0.1,
        )
    with pytest.raises(ValueError, match="epsilon_clip_highfreq"):
        SJFPOConfig(
            use_dct_transform=True, use_per_bin_clip=True, epsilon_clip_highfreq=1.0
        )
    assert SJFPOConfig(
        use_dct_transform=True,
        use_per_bin_clip=True,
        epsilon_clip_lowfreq=0.05,
        epsilon_clip_highfreq=0.05,
    )
