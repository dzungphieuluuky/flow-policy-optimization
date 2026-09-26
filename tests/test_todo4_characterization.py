"""Golden behavior pins: chunk GAE numerics and trainer assembly.

Hand-computed GAE values plus trainer assembly outputs, so later refactors
cannot silently change math. Fast unit-level only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestTrainerGolden:
    def policy(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from sjfpo.configuration_sjfpo import SJFPOConfig
        from sjfpo.modeling_sjfpo import SJFPOPolicy

        obs_dim, action_dim = 17, 4
        cfg = SJFPOConfig(
            num_mc_samples=2,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))
            },
            output_features={
                "action": PolicyFeature(FeatureType.ACTION, (action_dim,))
            },
        )
        return SJFPOPolicy(cfg), cfg, obs_dim, action_dim

    def chunks(self, cfg, obs_dim, action_dim, n=4):
        torch.manual_seed(0)
        chunks = []
        for _ in range(n):
            chunks.append(
                {
                    "observation.state": torch.randn(cfg.n_obs_steps, obs_dim),
                    "action": torch.randn(cfg.action_horizon, action_dim) * 0.5,
                    "next.observation.state": torch.randn(cfg.n_obs_steps, obs_dim),
                    "next.reward": torch.rand(8),
                    "next.done": torch.tensor(False),
                    "next.truncated": torch.tensor(False),
                    "executed_n_action_steps": torch.tensor(8),
                    "clip_count": torch.tensor(0),
                }
            )
        meta = {
            "advantages": torch.randn(n) * 0.1,
        }
        return chunks, meta

    def batch(self, chunks, meta, n):
        lengths = [int(c["executed_n_action_steps"]) for c in chunks]
        h = chunks[0]["action"].shape[0]
        return {
            "observation.state": torch.stack([c["observation.state"] for c in chunks]),
            "action": torch.stack([c["action"] for c in chunks]),
            "action_is_pad": torch.arange(h).unsqueeze(0)
            >= torch.tensor(lengths).unsqueeze(1),
            "advantages": meta["advantages"],
            "returns": torch.randn(n) * 10.0,
        }

    def test_assemble_update_keys_finite(self):
        policy, cfg, obs_dim, action_dim = self.policy()
        chunks, meta = self.chunks(cfg, obs_dim, action_dim)
        loss, info = policy(self.batch(chunks, meta, 4))
        assert set(info) == {
            "policy_loss",
            "value_loss",
            "explained_var",
            "mean_log_ratio",
            "flow_rho_mean",
            "flow_rho_std",
            "flow_rho_bin_range",
            "flow_rho_bin_lowfreq",
            "flow_rho_bin_highfreq",
            "flow_rho_bin_outside_frac",
            "flow_rho_bin_low_outside_frac",
            "flow_rho_bin_high_outside_frac",
            "flow_v_star_rms",
            "cond_temb",
            "cond_fused",
            "cond_c",
            "cond_gate_msa",
            "cond_gate_mlp",
            "cond_scale_msa",
            "cond_scale_mlp",
        }
        for v in info.values():
            assert isinstance(v, float)
            assert np.isfinite(v)
        assert torch.isfinite(loss)

    def test_compute_log_likelihood_shape(self):
        policy, cfg, obs_dim, action_dim = self.policy()
        device = next(policy.parameters()).device
        torch.manual_seed(1)
        top_k_spectral_action_tokens = max(
            1, round(cfg.top_p_spectral_tokens * cfg.action_horizon)
        )
        x_t = torch.randn(
            4 * cfg.num_mc_samples, top_k_spectral_action_tokens, action_dim
        ).to(device)
        tau = torch.rand(4 * cfg.num_mc_samples, device=device)
        states = torch.randn(4 * cfg.num_mc_samples, cfg.n_obs_steps, obs_dim).to(
            device
        )
        v = policy.compute_predicted_velocity(x_t, tau, states)
        assert v.shape == (
            4 * cfg.num_mc_samples,
            top_k_spectral_action_tokens,
            action_dim,
        )
        assert torch.isfinite(v).all()
