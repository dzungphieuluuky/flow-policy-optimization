"""Tests for skrl PPO/SAC baseline drivers (gymnasium FakeEnv so wrap_env works)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tests.fixtures.fake_env import FakeGymEnv

from sjfpo.config import PPOConfig, SACConfig

# Config tests


class TestPPOConfig:
    """Verify PPOConfig fields match skrl 2.1.0 PPO_CFG dataclass defaults."""

    def test_ppo_config_field_values(self):
        from skrl.agents.torch.ppo import PPO_CFG
        import dataclasses

        cfg = PPOConfig()
        skrl_fields = {f.name: f.default for f in dataclasses.fields(PPO_CFG)}

        # Our config maps to skrl keys with known provenance
        assert cfg.rollouts == 2048  # plan §Must-have
        assert cfg.learning_epochs == 10  # plan §Must-have
        assert cfg.mini_batches == 8  # plan §Must-have
        assert cfg.lr == 3e-4  # paper lr
        assert cfg.kl_threshold == 0.01  # KLAdaptiveLR default
        assert cfg.grad_norm_clip == 1.0
        assert cfg.discount == 0.99  # matches skrl default
        assert cfg.gae_lambda == 0.95  # matches skrl default
        assert cfg.ratio_clip == 0.2  # PPO clip default
        assert cfg.hidden == (256, 256, 256)

    def test_skrl_ppo_cfg_has_expected_keys(self):
        """Ensure the skrl PPO_CFG dataclass has the keys we need."""
        from skrl.agents.torch.ppo import PPO_CFG
        import dataclasses

        fields = {f.name for f in dataclasses.fields(PPO_CFG)}
        for key in [
            "rollouts",
            "learning_epochs",
            "mini_batches",
            "learning_rate",
            "kl_threshold",
            "grad_norm_clip",
            "discount_factor",
            "gae_lambda",
            "ratio_clip",
            "value_loss_scale",
            "entropy_loss_scale",
        ]:
            assert key in fields, f"PPO_CFG missing key: {key}"


class TestSACConfig:
    """Verify SACConfig fields match skrl 2.1.0 SAC_CFG dataclass defaults."""

    def test_sac_config_field_values(self):
        cfg = SACConfig()
        assert cfg.lr == 3e-4
        assert cfg.batch_size == 256
        assert cfg.memory_size == 300_000
        assert cfg.gradient_steps == 1
        assert cfg.polyak == 0.005
        assert cfg.discount == 0.99
        assert cfg.initial_random_steps == 5000
        assert cfg.hidden == (256, 256, 256)

    def test_skrl_sac_cfg_has_expected_keys(self):
        from skrl.agents.torch.sac import SAC_CFG
        import dataclasses

        fields = {f.name for f in dataclasses.fields(SAC_CFG)}
        for key in [
            "gradient_steps",
            "batch_size",
            "discount_factor",
            "polyak",
            "learning_rate",
            "random_timesteps",
            "learning_starts",
            "grad_norm_clip",
        ]:
            assert key in fields, f"SAC_CFG missing key: {key}"


class TestProvenanceTable:
    """Verify the provenance table renders with required (method,key,value,source) quadruples."""

    def test_provenance_table_contains_required_quadruples(self):
        from sjfpo.baselines import PROVENANCE_TABLE

        # PPO quadruples: (method, key, value, source)
        for method, key, value, source in [
            ("PPO", "rollouts", "2048", "default"),
            ("PPO", "learning_epochs", "10", "default"),
            ("PPO", "mini_batches", "8", "default"),
            ("PPO", "learning_rate", "3e-4", "paper"),
            ("PPO", "kl_threshold", "0.01", "default"),
            ("PPO", "grad_norm_clip", "1.0", "default"),
            ("PPO", "discount_factor", "0.99", "adaptation"),
            ("PPO", "gae_lambda", "0.95", "adaptation"),
            ("PPO", "ratio_clip", "0.2", "default"),
        ]:
            assert (method, key, value, source) in PROVENANCE_TABLE, (
                f"Missing provenance: ({method}, {key}, {value}, {source})"
            )

        # SAC quadruples
        for method, key, value, source in [
            ("SAC", "learning_rate", "3e-4", "default"),
            ("SAC", "batch_size", "256", "default"),
            ("SAC", "memory_size", "300000", "adaptation"),
            ("SAC", "gradient_steps", "1", "default"),
            ("SAC", "polyak", "0.005", "default"),
            ("SAC", "initial_random_steps", "5000", "adaptation"),
        ]:
            assert (method, key, value, source) in PROVENANCE_TABLE, (
                f"Missing provenance: ({method}, {key}, {value}, {source})"
            )


# Tiny smoke tests (require skrl + gymnasium but NOT mujoco)


class TestPPOTinySmoke:
    """PPO on FakeGymEnv should complete and produce result dict."""

    def test_ppo_train_run_completes(self, tmp_path, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from sjfpo.baselines import ppo_train_run

        env = FakeGymEnv(obs_dim=4, act_dim=2, max_steps=50)
        cfg = PPOConfig(
            rollouts=32,
            learning_epochs=2,
            mini_batches=2,
            hidden=(64, 64),
        )

        result = ppo_train_run(
            env=env,
            out_dir=tmp_path / "ppo_run",
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=200,
            ckpt_interval=200,
        )

        assert result["method"] == "ppo"
        assert result["seed"] == 0
        assert isinstance(result["reward_mean_last10"], float)
        assert result["fps"] > 0
        assert isinstance(result["artifacts"], list)

        # Checkpoint should exist
        for artifact in result["artifacts"]:
            assert Path(artifact).exists(), f"artifact missing: {artifact}"


class TestSACTinySmoke:
    """SAC on FakeGymEnv should complete and produce result dict."""

    def test_sac_train_run_completes(self, tmp_path, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from sjfpo.baselines import sac_train_run

        env = FakeGymEnv(obs_dim=4, act_dim=2, max_steps=50)
        cfg = SACConfig(
            batch_size=32,
            memory_size=500,
            hidden=(64, 64),
            initial_random_steps=0,
        )

        result = sac_train_run(
            env=env,
            out_dir=tmp_path / "sac_run",
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=200,
            ckpt_interval=200,
        )

        assert result["method"] == "sac"
        assert result["seed"] == 0
        assert isinstance(result["reward_mean_last10"], float)
        assert result["fps"] > 0
        assert isinstance(result["artifacts"], list)

        for artifact in result["artifacts"]:
            assert Path(artifact).exists(), f"artifact missing: {artifact}"
