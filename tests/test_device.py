"""Device handling: CUDA default with CPU fallback (SJFPO_DEVICE override)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.utils import resolve_device  # noqa: E402


class TestResolveDevice:
    def test_returns_torch_device(self):
        assert isinstance(resolve_device(), torch.device)

    def test_explicit_cpu_wins(self, monkeypatch):
        monkeypatch.setenv("SJFPO_DEVICE", "cpu")
        assert resolve_device() == torch.device("cpu")

    def test_unknown_value_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setenv("SJFPO_DEVICE", "tpu")
        assert isinstance(resolve_device(), torch.device)

    def test_cuda_when_available(self, monkeypatch):
        monkeypatch.setenv("SJFPO_DEVICE", "auto")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device() == torch.device("cuda")

    def test_cpu_fallback_when_no_cuda(self, monkeypatch):
        monkeypatch.delenv("SJFPO_DEVICE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device() == torch.device("cpu")


class TestTrainerDevice:
    def trainer(self, **kw):
        from sjfpo.config import SJFPOConfig
        from sjfpo.networks import (
            SpectralEncoder,
            SJFOPolicyModel,
            StateEncoder,
            Critic,
        )
        from sjfpo.trainer import SJFPOTrainer

        cfg = SJFPOConfig(num_mc_samples=2)
        policy = SJFOPolicyModel(2, 4, hidden_size=cfg.hidden_size)
        critic = Critic(hidden_size=cfg.hidden_size, act_dim=2)
        spectral = SpectralEncoder(4, n_obs_steps=16)
        critic_state_encoder = StateEncoder(
            4, n_obs_steps=16, hidden_size=cfg.hidden_size
        )
        return SJFPOTrainer(
            policy,
            critic,
            spectral,
            critic_state_encoder,
            cfg=cfg,
            **kw,
        )

    def test_modules_live_on_trainer_device(self):
        t = self.trainer(device="cpu")
        assert t.device == torch.device("cpu")
        for m in (
            t.policy_head,
            t.critic,
            t.critic_state_encoder,
        ):
            assert next(m.parameters()).device == torch.device("cpu")
        assert list(t.spectral.parameters()) == []  # parameter-free DCT

    def test_update_on_default_device(self):
        t = self.trainer()
        chunks = [
            {
                "observation.state": torch.randn(16, 4),
                "action": torch.randn(16, 2) * 0.5,
                "next.action": torch.randn(16, 2) * 0.5,
                "next.observation.state": torch.randn(16, 4),
                "next.reward": torch.rand(8),
                "next.done": torch.tensor(False),
                "next.truncated": torch.tensor(False),
                "n_action_steps": torch.tensor(8),
                "clip_count": torch.tensor(0),
            }
            for _ in range(4)
        ]
        diag = t.assemble_update(chunks, {"advantages": torch.randn(4) * 0.1})
        import math

        for k in ("policy_loss", "value_loss", "sigreg_loss"):
            assert math.isfinite(diag[k]), f"{k} not finite on {t.device}"

    def test_infer_returns_cpu_chunk(self):
        from sjfpo.config import SJFPOConfig
        from sjfpo.runner import infer_chunk_actions

        t = self.trainer()
        cfg = SJFPOConfig()
        out = infer_chunk_actions(
            torch.randn(16, 4), t.spectral, t.policy_head, cfg
        )
        assert out.device == torch.device("cpu")
        assert out.shape == (cfg.chunk_size, 2)
