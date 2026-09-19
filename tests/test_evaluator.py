"""Tests for shared evaluator.

Covers:
  (a) drives fake policies of all three kinds (callable, zero-noise SJFPO,
      mean-action PPO, deterministic SAC) producing CSV rows
  (b) determinism: same weights+eval_seed twice → identical episode returns
  (c) acting batch B=1 works out of the box (no calibration)
  (d) 12-episode default + max-steps cap respected
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.networks import SJFPOFlowModel, StateConditioner
from sjfpo.fpo_core import sample_action
from sjfpo.evaluator import Evaluator

# Fake environment


class FakeEnv:
    """Minimal fake MuJoCo-like env for evaluator tests."""

    def __init__(self, obs_dim: int = 17, action_dim: int = 4, max_steps: int = 50):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.rng = np.random.RandomState(0)
        self.observation_space = type("Space", (), {"shape": (obs_dim,)})()
        self.action_space = type("Space", (), {"shape": (action_dim,)})()

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        obs = self.rng.randn(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        obs = self.rng.randn(self.obs_dim).astype(np.float32)
        reward = float(self.rng.randn() * 0.1)
        terminated = self.step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

        # Policy adapters


def callable_policy(obs: np.ndarray) -> np.ndarray:
    """Simple callable returning random actions."""
    return np.random.randn(4).astype(np.float32)


class SJFPOPolicyAdapter:
    """SJFPO-style adapter with zero-noise eval and internal state window."""

    def __init__(self, policy_head, conditioner, cfg):
        from collections import deque

        self.policy_head = policy_head
        self.conditioner = conditioner
        self.cfg = cfg
        self.window = deque(maxlen=cfg.n_obs_steps)
        self.first_obs = None

    def reset(self):
        self.window.clear()
        self.first_obs = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if self.first_obs is None:
            self.first_obs = obs_t
            for _ in range(self.cfg.n_obs_steps):
                self.window.append(self.first_obs.clone())
        self.window.append(obs_t)
        state = torch.stack(list(self.window), dim=0).unsqueeze(0)  # [1, T, obs]
        with torch.no_grad():
            condition = self.conditioner(state)  # [1, D] fused outside the head
            # Policy model (zero noise via eps=zeros for deterministic eval)
            top_k_spectral_action_tokens = max(1, round(self.cfg.top_p_spectral_tokens * self.cfg.action_horizon))
            chunk = sample_action(
                lambda x, t, c: self.policy_head(x, t, c),
                (1, top_k_spectral_action_tokens, self.action_dim),
                condition,
                num_inference_steps=self.cfg.num_inference_steps,
                eps=torch.zeros(1, top_k_spectral_action_tokens, self.action_dim),
            )  # [1, Ka, action_dim]
            chunk = torch.nn.functional.pad(
                chunk, (0, 0, 0, self.cfg.action_horizon - top_k_spectral_action_tokens)
            )
        return chunk.squeeze(0).cpu().numpy()

    def set_action_dim(self, dim: int):
        self.action_dim = dim


class PPOPolicyAdapter:
    """PPO-style adapter: mean action (no sampling noise)."""

    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class SACPolicyAdapter:
    """SAC-style adapter: deterministic action."""

    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.ones(self.action_dim, dtype=np.float32) * 0.5

        # (a) drives fake policies of all three kinds


class TestPolicyKinds:
    def test_callable_policy(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=3)
        results = evaluator.evaluate(callable_policy)
        assert len(results) == 3
        for r in results:
            assert "return" in r
            assert "length" in r
            assert isinstance(r["return"], float)
            assert isinstance(r["length"], int)

    def test_ppo_mean_action(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=3)
        adapter = PPOPolicyAdapter(action_dim=4)
        results = evaluator.evaluate(adapter)
        assert len(results) == 3

    def test_sac_deterministic(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=3)
        adapter = SACPolicyAdapter(action_dim=4)
        results = evaluator.evaluate(adapter)
        assert len(results) == 3

    def test_sjfpo_zero_noise(self):
        cfg = SJFPOConfig()
        policy_head = SJFPOFlowModel(4, hidden_dim=cfg.hidden_dim)
        conditioner = StateConditioner(17, hidden_dim=cfg.hidden_dim)

        env = FakeEnv(obs_dim=17, action_dim=4)
        adapter = SJFPOPolicyAdapter(policy_head, conditioner, cfg)
        adapter.action_dim = 4

        evaluator = Evaluator(env, eval_seed=42, max_episodes=2)
        results = evaluator.evaluate(adapter)
        assert len(results) == 2

    def test_csv_schema(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator.evaluate(callable_policy, out_dir=tmpdir)
            csv_path = Path(tmpdir) / "eval_history.csv"
            assert csv_path.exists()
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            for row in rows:
                assert "return" in row
                assert "length" in row

                # (b) determinism: same weights+eval_seed twice → identical episode returns


class TestDeterminism:
    def test_same_seed_same_returns(self):
        env1 = FakeEnv()
        env2 = FakeEnv()
        ev1 = Evaluator(env1, eval_seed=42, max_episodes=5)
        ev2 = Evaluator(env2, eval_seed=42, max_episodes=5)

        r1 = ev1.evaluate(callable_policy)
        r2 = ev2.evaluate(callable_policy)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["return"] == pytest.approx(b["return"]), (
                "returns differ with same seed"
            )
            assert a["length"] == b["length"]

    def test_different_seeds_differ(self):
        env1 = FakeEnv()
        env2 = FakeEnv()
        ev1 = Evaluator(env1, eval_seed=42, max_episodes=5)
        ev2 = Evaluator(env2, eval_seed=99, max_episodes=5)

        r1 = ev1.evaluate(callable_policy)
        r2 = ev2.evaluate(callable_policy)

        # With different seeds, at least one episode return should differ
        returns1 = [r["return"] for r in r1]
        returns2 = [r["return"] for r in r2]
        assert returns1 != returns2, "different seeds produced identical returns"

        # (c) acting batch B=1 works out of the box (no calibration)


class TestBatchSizeOne:
    def test_b1_no_calibration_needed(self):
        cfg = SJFPOConfig()
        policy_head = SJFPOFlowModel(4, hidden_dim=cfg.hidden_dim)
        conditioner = StateConditioner(17, hidden_dim=cfg.hidden_dim)

        env = FakeEnv(obs_dim=17, action_dim=4)
        adapter = SJFPOPolicyAdapter(policy_head, conditioner, cfg)
        adapter.action_dim = 4

        evaluator = Evaluator(env, eval_seed=0, max_episodes=2)
        results = evaluator.evaluate(adapter)
        assert len(results) == 2
        for r in results:
            assert math.isfinite(r["return"])

            # (d) 12-episode default + max-steps cap respected


class TestEpisodeAndStepLimits:
    def test_max_episodes_respected(self):
        env = FakeEnv(max_steps=100)
        evaluator = Evaluator(env, eval_seed=42, max_episodes=12)
        results = evaluator.evaluate(callable_policy)
        assert len(results) == 12

    def test_max_steps_cap(self):
        env = FakeEnv(max_steps=100)  # long episodes
        evaluator = Evaluator(env, eval_seed=42, max_episodes=100, max_steps=50)
        results = evaluator.evaluate(callable_policy)
        total_steps = sum(r["length"] for r in results)
        assert total_steps <= 50, f"total steps {total_steps} > max_steps 50"


class TestSmoothnessMetrics:
    def test_action_jerk_zero_for_constant_and_linear(self):
        from sjfpo.evaluator import action_jerk

        assert action_jerk(np.zeros((10, 4))) == 0.0
        ramp = np.arange(10, dtype=float).reshape(-1, 1) * np.ones((1, 3))
        assert action_jerk(ramp) == 0.0  # constant velocity, no jerk
        assert action_jerk(np.zeros((2, 4))) is None  # < 3 steps undefined

    def test_action_jerk_positive_for_alternation(self):
        from sjfpo.evaluator import action_jerk

        bang = np.array([[1.0], [-1.0]] * 5)  # bang-bang switching
        assert action_jerk(bang) == pytest.approx(16.0)

    def test_extract_success_absent_and_present(self):
        from sjfpo.evaluator import extract_success

        assert extract_success({}) is None
        assert extract_success({"success": True}) == 1.0
        assert extract_success({"success": 0}) == 0.0

    def test_mean_chunk_jerk_masks_tails(self):
        from sjfpo.evaluator import action_jerk, mean_chunk_jerk

        actions = np.zeros((2, 8, 2))
        actions[0, 4] = 1.0  # spike inside the executed prefix
        is_pad = np.zeros((2, 8), dtype=bool)
        is_pad[0, 5:] = True  # executed prefix of 5
        is_pad[1, :] = True  # fully padded chunk contributes nothing
        assert mean_chunk_jerk(actions, is_pad) == pytest.approx(
            action_jerk(actions[0, :5])
        )
        assert mean_chunk_jerk(actions, np.ones((2, 8), dtype=bool)) is None

    def test_rollout_reports_success_and_jerk(self):
        from sjfpo.evaluator import Evaluator

        class SuccessEnv(FakeEnv):
            def step(self, action):
                obs, reward, terminated, truncated, _ = super().step(action)
                info = {"success": True} if terminated else {}
                return obs, reward, terminated, truncated, info

        ev = Evaluator(SuccessEnv(max_steps=5), eval_seed=0, max_episodes=2)
        results = ev.evaluate(lambda obs: np.zeros(4, dtype=np.float32))
        assert [r["success"] for r in results] == [1.0, 1.0]
        assert all(r["jerk"] == 0.0 for r in results)

        # Need this import for TestBatchSizeOne


import math
