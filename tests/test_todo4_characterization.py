"""Characterization tests for fn-consolidation Todo 4 (written BEFORE splits).

Pins current observable behavior of the baselines loop/train fns, trainer
assemble_update/compute_log_likelihood, and math helpers (gae, inference,
checkpointing) so behavior-identical pure-extraction splits
cannot change results. Fast unit-level only; skrl training paths are proven
by the 500-step PPO/SAC smoke runs in task-4 evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestChunkGaeGolden:
    def test_single_chunk_hand_computed(self):
        from sjfpo.gae import chunk_gae

        adv, ret = chunk_gae(
            [1.0, 1.0],
            [0.5],
            [0.0],
            [2],
            [False],
            gamma=0.99,
            lam=0.95,
            center=False,
        )
        assert adv == pytest.approx([1.49], abs=1e-9)
        assert ret == pytest.approx([1.99], abs=1e-9)

    def test_two_chunk_recursion_hand_computed(self):
        from sjfpo.gae import chunk_gae

        adv, ret = chunk_gae(
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1, 1],
            [False, False],
            gamma=0.99,
            lam=0.95,
            center=False,
        )
        assert adv == pytest.approx([2.881, 2.0], abs=1e-9)
        assert ret == pytest.approx([2.881, 2.0], abs=1e-9)

    def test_terminated_cuts_bootstrap(self):
        from sjfpo.gae import chunk_gae

        adv, _ = chunk_gae(
            [1.0],
            [0.5],
            [9.0],
            [1],
            [True],
            gamma=0.99,
            lam=0.95,
            center=False,
        )
        assert adv == pytest.approx([0.5], abs=1e-9)

    def test_center_zero_mean(self):
        from sjfpo.gae import chunk_gae

        adv, _ = chunk_gae(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1, 1, 1],
            [False, False, False],
            gamma=0.99,
            lam=0.95,
            center=True,
        )
        assert float(np.mean(adv)) == pytest.approx(0.0, abs=1e-9)


class TestCheckpointGolden:
    def test_roundtrip_and_manifest(self, tmp_path):
        import torch

        from sjfpo.checkpointing import load_checkpoint, save_checkpoint

        state = {"w": torch.tensor([1.0, 2.0]), "step": 7}
        save_checkpoint(tmp_path, state)
        payload = load_checkpoint(tmp_path, ["w", "step"])
        assert torch.equal(payload["w"], state["w"])
        assert payload["step"] == 7

    def test_unexpected_keys_raise(self, tmp_path):
        import torch

        from sjfpo.checkpointing import ResumeError, load_checkpoint, save_checkpoint

        save_checkpoint(tmp_path, {"w": torch.tensor([1.0])})
        with pytest.raises(ResumeError):
            load_checkpoint(tmp_path, ["w", "nope"])


class TestTrainerGolden:
    def trainer(self):
        from sjfpo.config import SJFPOConfig
        from sjfpo.networks import (
            StateEncoder,
            SJFOPolicyModel,
            SpectralEncoder,
            Critic,
        )
        from sjfpo.trainer import SJFPOTrainer

        obs_dim, act_dim = 17, 4
        cfg = SJFPOConfig(num_mc_samples=2)
        policy_state_encoder = StateEncoder(
            obs_dim, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
        )
        policy_head = SJFOPolicyModel(act_dim, hidden_size=cfg.hidden_size)
        critic = Critic(hidden_size=cfg.hidden_size, act_dim=act_dim)
        spectral = SpectralEncoder(
            hidden_size=cfg.hidden_size, n_obs_steps=cfg.n_obs_steps
        )
        spectral.set_frozen_stats(
            torch.zeros(cfg.hidden_size), torch.ones(cfg.hidden_size)
        )
        critic_state_encoder = StateEncoder(
            obs_dim, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
        )
        return (
            SJFPOTrainer(
                policy_state_encoder=policy_state_encoder,
                policy_head=policy_head,
                critic=critic,
                spectral=spectral,
                critic_state_encoder=critic_state_encoder,
                cfg=cfg,
            ),
            cfg,
            obs_dim,
            act_dim,
        )

    def chunks(self, cfg, obs_dim, act_dim, n=4):
        torch.manual_seed(0)
        chunks = []
        for _ in range(n):
            chunks.append(
                {
                    "observation.state": torch.randn(cfg.n_obs_steps, obs_dim),
                    "action": torch.randn(cfg.chunk_size, act_dim) * 0.5,
                    "next.action": torch.randn(cfg.chunk_size, act_dim) * 0.5,
                    "next.observation.state": torch.randn(cfg.n_obs_steps, obs_dim),
                    "next.reward": torch.rand(cfg.n_action_steps),
                    "next.done": torch.tensor(False),
                    "next.truncated": torch.tensor(False),
                    "n_action_steps": torch.tensor(cfg.n_action_steps),
                    "clip_count": torch.tensor(0),
                }
            )
        meta = {
            "advantages": torch.randn(n) * 0.1,
        }
        return chunks, meta

    def test_assemble_update_keys_finite(self):
        trainer, cfg, obs_dim, act_dim = self.trainer()
        chunks, meta = self.chunks(cfg, obs_dim, act_dim)
        diag = trainer.assemble_update(chunks, meta)
        assert set(diag) == {
            "policy_loss",
            "value_loss",
            "sigreg_loss",
            "clip_fraction",
            "mean_log_ratio",
            "clamp_engagement",
        }
        for v in diag.values():
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_compute_log_likelihood_shape(self):
        trainer, cfg, obs_dim, act_dim = self.trainer()
        dev = trainer.device
        torch.manual_seed(1)
        actions = (torch.randn(4, cfg.chunk_size, act_dim) * 0.5).to(dev)
        states = torch.randn(4, cfg.n_obs_steps, obs_dim).to(dev)
        z = trainer.policy_state_encoder(states)
        cond = trainer.spectral(z)
        ll = trainer.compute_log_likelihood(
            actions, z, cond, act_dim=act_dim, chunk_size=cfg.chunk_size
        )
        assert ll.shape == (4, cfg.num_mc_samples)
        assert torch.isfinite(ll).all()


class TestBaselinesHelpersGolden:
    def test_frame_stack_shape(self):
        import gymnasium as gym

        from sjfpo.baselines import FrameStackWrapper
        from tests.fixtures.fake_env import FakeGymEnv

        env = FrameStackWrapper(FakeGymEnv(obs_dim=4, act_dim=2), n_obs_steps=16)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4 * 16,)
        assert obs.dtype == np.float32

    def test_import_skrl_idempotent(self):
        from sjfpo import baselines

        baselines.import_skrl_dependencies()
        first = baselines.PPOPolicy
        baselines.import_skrl_dependencies()
        assert baselines.PPOPolicy is first
        assert baselines.PPOPolicy is not None
        assert baselines.SACCritic is not None

    def test_seed_and_wrap_dims(self):
        from sjfpo import baselines

        baselines.import_skrl_dependencies()
        from tests.fixtures.fake_env import FakeGymEnv

        wrapped, skrl_env, obs_dim, act_dim = baselines.seed_and_wrap_env(
            FakeGymEnv(obs_dim=4, act_dim=2), seed=0, n_obs_steps=16
        )
        assert obs_dim == 4 * 16
        assert act_dim == 2
        assert wrapped.observation_space.shape[0] == obs_dim
        wrapped.close()
