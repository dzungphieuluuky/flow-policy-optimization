"""Golden behavior pins: chunk GAE numerics, trainer assembly, baselines helpers.

Hand-computed GAE values plus trainer/baselines assembly outputs, so later
refactors cannot silently change math. Fast unit-level only.
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
        from sjfpo.gae import compute_advantages

        advantage, ret = compute_advantages(
            [1.0, 1.0],
            [0.5],
            [0.0],
            [2],
            [False],
            gamma=0.99,
            lam=0.95,
            normalize=False,
        )
        assert advantage == pytest.approx([1.49], abs=1e-9)
        assert ret == pytest.approx([1.99], abs=1e-9)

    def test_two_chunk_recursion_hand_computed(self):
        from sjfpo.gae import compute_advantages

        advantage, ret = compute_advantages(
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1, 1],
            [False, False],
            gamma=0.99,
            lam=0.95,
            normalize=False,
        )
        assert advantage == pytest.approx([2.881, 2.0], abs=1e-9)
        assert ret == pytest.approx([2.881, 2.0], abs=1e-9)

    def test_terminated_cuts_bootstrap(self):
        from sjfpo.gae import compute_advantages

        advantage, _ = compute_advantages(
            [1.0],
            [0.5],
            [9.0],
            [1],
            [True],
            gamma=0.99,
            lam=0.95,
            normalize=False,
        )
        assert advantage == pytest.approx([0.5], abs=1e-9)

    def test_normalize_zero_mean(self):
        from sjfpo.gae import compute_advantages

        advantage, _ = compute_advantages(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1, 1, 1],
            [False, False, False],
            gamma=0.99,
            lam=0.95,
            normalize=True,
        )
        assert float(np.mean(advantage)) == pytest.approx(0.0, abs=1e-9)


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
            output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
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
                    "next.action": torch.randn(cfg.action_horizon, action_dim) * 0.5,
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
            "mean_log_ratio",
            "flow_rho_mean",
            "flow_rho_std",
            "flow_rho_bin_range",
            "flow_rho_bin_lowfreq",
            "flow_rho_bin_highfreq",
            "flow_rho_bin_outside_frac",
            "flow_v_star_rms",
            "cond_temb",
            "cond_pooled",
            "cond_z_state",
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
        top_k_spectral_action_tokens = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
        x_t = torch.randn(4 * cfg.num_mc_samples, top_k_spectral_action_tokens, action_dim).to(device)
        tau = torch.rand(4 * cfg.num_mc_samples, device=device)
        condition = torch.randn(4 * cfg.num_mc_samples, 5, obs_dim).to(device)
        states = torch.randn(4 * cfg.num_mc_samples, cfg.n_obs_steps, obs_dim).to(
            device
        )
        v = policy.compute_predicted_velocity(x_t, tau, condition, states)
        assert v.shape == (4 * cfg.num_mc_samples, top_k_spectral_action_tokens, action_dim)
        assert torch.isfinite(v).all()


class TestBaselinesHelpersGolden:
    def test_frame_stack_shape(self):
        import gymnasium as gym

        from sjfpo.baselines import FrameStackWrapper
        from tests.fixtures.fake_env import FakeGymEnv

        env = FrameStackWrapper(FakeGymEnv(obs_dim=4, action_dim=2), n_obs_steps=16)
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

        wrapped, skrl_env, obs_dim, action_dim = baselines.seed_and_wrap_env(
            FakeGymEnv(obs_dim=4, action_dim=2), seed=0, n_obs_steps=16
        )
        assert obs_dim == 4 * 16
        assert action_dim == 2
        assert wrapped.observation_space.shape[0] == obs_dim
        wrapped.close()
