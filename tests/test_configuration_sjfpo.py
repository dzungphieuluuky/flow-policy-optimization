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
    assert (cfg.n_obs_steps, cfg.horizon, cfg.hidden_dim) == (16, 16, 128)
    assert cfg.n_action_steps == round(cfg.executed_vs_predicted_ratio * cfg.horizon)
    assert (cfg.optimizer_lr, cfg.optimizer_weight_decay) == (1e-4, 1e-4)
    assert cfg.grad_clip_norm == 1.0
    assert cfg.num_epochs == 10
    assert cfg.chunks_per_update == 128


def test_post_init_rejects_bad_geometry():
    with pytest.raises(ValueError, match="n_action_steps"):
        SJFPOConfig(n_action_steps=17)
    with pytest.raises(ValueError, match="n_action_steps"):
        SJFPOConfig(n_action_steps=0)
    with pytest.raises(ValueError, match="executed_vs_predicted_ratio"):
        SJFPOConfig(executed_vs_predicted_ratio=1.5)
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
        cfg.optimizer_lr,
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
