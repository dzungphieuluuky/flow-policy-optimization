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


def test_build_policy_cpu_fp32_inference(monkeypatch):
    from sjfpo.configuration_sjfpo import SJFPOConfig
    from sjfpo.networks import SJFPOFlowModel
    from sjfpo.train_sjfpo import build_policy

    monkeypatch.setenv("SJFPO_COMPILE", "0")
    policy = build_policy(4, 2, SJFPOConfig(), device="cpu")
    assert isinstance(policy.actor, SJFPOFlowModel)
    assert all(p.dtype == torch.float32 for p in policy.parameters())
    assert policy.obs_normalizer.mean.dtype == torch.float32
    actions = policy.predict_action_chunk(
        {"observation.state": torch.randn(1, 16, 4)}
    )
    assert actions.shape == (1, 16, 2)
    assert torch.isfinite(actions).all()


class TestPolicyDevice:
    def policy(self, **kw):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from sjfpo.configuration_sjfpo import SJFPOConfig
        from sjfpo.modeling_sjfpo import SJFPOPolicy

        device = kw.pop("device", None)
        cfg = SJFPOConfig(
            num_mc_samples=2,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (4,))
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        )
        policy = SJFPOPolicy(cfg)
        if device is not None:
            policy = policy.to(device)
        return policy

    def test_modules_live_on_policy_device(self):
        t = self.policy(device="cpu")
        for m in (
            t.actor,
            t.critic,
        ):
            assert next(m.parameters()).device == torch.device("cpu")
        assert next(t.critic.state_encoder.parameters()).device == torch.device("cpu")
        assert list(t.spectral.parameters()) == []  # parameter-free DCT

    def test_update_on_default_device(self):
        t = self.policy()
        batch = {
            "observation.state": torch.randn(4, 16, 4),
            "action": torch.randn(4, 16, 2) * 0.5,
            "action_is_pad": torch.zeros(4, 16, dtype=torch.bool),
            "advantages": torch.randn(4) * 0.1,
            "returns": torch.randn(4) * 10.0,
        }
        loss, info = t(batch)
        import math

        for k in ("policy_loss", "value_loss"):
            assert math.isfinite(info[k]), f"{k} not finite"

    def test_infer_returns_cpu_chunk(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from sjfpo.configuration_sjfpo import SJFPOConfig
        from sjfpo.modeling_sjfpo import SJFPOPolicy

        policy = SJFPOPolicy(
            SJFPOConfig(
                input_features={
                    "observation.state": PolicyFeature(FeatureType.STATE, (4,))
                },
                output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
            )
        )
        out = policy.predict_action_chunk({"observation.state": torch.randn(1, 16, 4)})
        assert out.device == torch.device("cpu")
        assert out.shape == (1, 16, 2)
