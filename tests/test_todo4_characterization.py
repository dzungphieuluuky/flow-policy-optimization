"""Characterization tests for fn-consolidation Todo 4 (written BEFORE splits).

Pins current observable behavior of the baselines loop/train fns, trainer
assemble_update/compute_log_likelihood, and math helpers (gae, inference,
calibration, checkpointing) so behavior-identical pure-extraction splits
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
        from mfpo.gae import chunk_gae

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
        from mfpo.gae import chunk_gae

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
        from mfpo.gae import chunk_gae

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
        from mfpo.gae import chunk_gae

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


class TestEulerGolden:
    def test_constant_velocity_exact(self):
        from mfpo.inference import euler_integrate_joint

        z0 = torch.zeros(1, 2, 3)
        Z = torch.randn(1, 4, 8)
        out = euler_integrate_joint(lambda z, t, Z: torch.ones_like(z), z0, Z, steps=4)
        assert torch.allclose(out, torch.ones(1, 2, 3))

    def test_steps_zero_raises(self):
        from mfpo.inference import euler_integrate_joint

        with pytest.raises(ValueError):
            euler_integrate_joint(
                lambda z, t, Z: z, torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), steps=0
            )

    def test_decode_joint_gripper_discrete(self):
        from mfpo.inference import decode_joint

        torch.manual_seed(4)
        X_hat = torch.randn(2, 8, 7 + 128)
        A_hat, S_hat = decode_joint(X_hat)
        assert A_hat.shape == (2, 8, 7)
        assert S_hat.shape == (2, 8, 128)
        uniq = set(torch.unique(A_hat[..., -2:]).tolist())
        assert uniq.issubset({0.0, 1.0})


class TestCalibrateGolden:
    def test_gate_pass_high_fps(self):
        from mfpo import config
        from mfpo.calibration import calibrate_and_gate
        from mfpo.config import build_legacy_run_matrix

        report = calibrate_and_gate(
            build_legacy_run_matrix(), {"mfpo": 50000.0, "ppo": 80000.0, "sac": 60000.0}
        )
        assert report.gate_pass is True
        assert report.fallback_ladder == []
        assert report.total_hours < config.BUDGET_TOTAL_HOURS

    def test_gate_fail_low_fps(self):
        from mfpo import config
        from mfpo.calibration import calibrate_and_gate
        from mfpo.config import build_legacy_run_matrix

        report = calibrate_and_gate(
            build_legacy_run_matrix(), {"mfpo": 80.0, "ppo": 100.0, "sac": 90.0}
        )
        assert report.gate_pass is False
        assert report.total_hours > config.BUDGET_TOTAL_HOURS
        assert "USER SIGN-OFF REQUIRED" in report.markdown


class TestCheckpointGolden:
    def test_roundtrip_and_manifest(self, tmp_path):
        import torch

        from mfpo.checkpointing import load_checkpoint, save_checkpoint

        state = {"w": torch.tensor([1.0, 2.0]), "step": 7}
        save_checkpoint(tmp_path, state)
        payload = load_checkpoint(tmp_path, ["w", "step"])
        assert torch.equal(payload["w"], state["w"])
        assert payload["step"] == 7

    def test_unexpected_keys_raise(self, tmp_path):
        import torch

        from mfpo.checkpointing import ResumeError, load_checkpoint, save_checkpoint

        save_checkpoint(tmp_path, {"w": torch.tensor([1.0])})
        with pytest.raises(ResumeError):
            load_checkpoint(tmp_path, ["w", "nope"])


class TestTrainerGolden:
    def _trainer(self):
        from mfpo.collector import ChunkRecord  # noqa: F401 (import seam pin)
        from mfpo.config import MFPOConfig
        from mfpo.networks import (
            AuxStateFlowHead,
            ContextTrunk,
            EMAProjection,
            PolicyFlowHead,
            SpectralTokens,
            ValueCritic,
        )
        from mfpo.trainer import MFPOTrainer

        obs_dim, act_dim = 17, 4
        cfg = MFPOConfig(N_mc=2)
        trunk = ContextTrunk(obs_dim, context_len=cfg.context_len, z_dim=cfg.Z_dim)
        policy_head = PolicyFlowHead(act_dim, z_dim=cfg.Z_dim, cond_dims=(64,))
        aux_head = AuxStateFlowHead(
            state_dim=cfg.Z_dim, z_dim=cfg.Z_dim, cond_dims=(64,)
        )
        critic = ValueCritic(z_dim=cfg.Z_dim)
        ema_proj = EMAProjection(obs_dim, latent=cfg.Z_dim)
        spectral = SpectralTokens(z_dim=cfg.Z_dim, context_len=cfg.context_len)
        spectral.set_frozen_stats(
            torch.zeros(2 * cfg.Z_dim), torch.ones(2 * cfg.Z_dim)
        )
        return (
            MFPOTrainer(
                trunk=trunk,
                policy_head=policy_head,
                aux_head=aux_head,
                critic=critic,
                ema_proj=ema_proj,
                spectral=spectral,
                cfg=cfg,
            ),
            cfg,
            obs_dim,
            act_dim,
        )

    def _chunks(self, cfg, obs_dim, act_dim, n=4):
        from mfpo.collector import ChunkRecord

        torch.manual_seed(0)
        chunks = []
        for _ in range(n):
            chunks.append(
                ChunkRecord(
                    context=torch.randn(cfg.context_len, obs_dim),
                    action_dct=torch.randn(cfg.chunk_len, act_dim) * 0.5,
                    rewards=torch.rand(8).tolist(),
                    done="continue",
                    bootstrap=False,
                    clip_count=0,
                    exec_len=cfg.chunk_len,
                )
            )
        meta = {
            "returns": torch.randn(n) * 0.1,
            "advantages": torch.randn(n) * 0.1,
            "values": torch.randn(n) * 0.1,
        }
        return chunks, meta

    def test_assemble_update_keys_finite(self):
        trainer, cfg, obs_dim, act_dim = self._trainer()
        chunks, meta = self._chunks(cfg, obs_dim, act_dim)
        diag = trainer.assemble_update(chunks, meta)
        assert set(diag) == {
            "policy_loss",
            "value_loss",
            "aux_loss",
            "sigreg_loss",
            "clip_fraction",
            "mean_log_ratio",
            "clamp_engagement",
        }
        for v in diag.values():
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_compute_log_likelihood_shape(self):
        trainer, cfg, obs_dim, act_dim = self._trainer()
        torch.manual_seed(1)
        action_dcts = torch.randn(4, cfg.chunk_len, act_dim) * 0.5
        contexts = torch.randn(4, cfg.context_len, obs_dim)
        z = trainer.trunk(contexts)
        cond = trainer.spectral(z)
        ll = trainer.compute_log_likelihood(
            action_dcts, z, cond, act_dim=act_dim, chunk_len=cfg.chunk_len
        )
        assert ll.shape == (4, cfg.N_mc)
        assert torch.isfinite(ll).all()


class TestBaselinesHelpersGolden:
    def test_frame_stack_shape(self):
        import gymnasium as gym

        from mfpo.baselines import FrameStackWrapper
        from tests.fixtures.fake_env import FakeGymEnv

        env = FrameStackWrapper(FakeGymEnv(obs_dim=4, act_dim=2), context_len=16)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4 * 16,)
        assert obs.dtype == np.float32

    def test_import_skrl_idempotent(self):
        from mfpo import baselines

        baselines.import_skrl_dependencies()
        first = baselines.PPOPolicy
        baselines.import_skrl_dependencies()
        assert baselines.PPOPolicy is first
        assert baselines.PPOPolicy is not None
        assert baselines.SACCritic is not None

    def test_seed_and_wrap_dims(self):
        from mfpo import baselines

        baselines.import_skrl_dependencies()
        from tests.fixtures.fake_env import FakeGymEnv

        wrapped, skrl_env, obs_dim, act_dim = baselines.seed_and_wrap_env(
            FakeGymEnv(obs_dim=4, act_dim=2), seed=0, context_len=16
        )
        assert obs_dim == 4 * 16
        assert act_dim == 2
        assert wrapped.observation_space.shape[0] == obs_dim
        wrapped.close()
