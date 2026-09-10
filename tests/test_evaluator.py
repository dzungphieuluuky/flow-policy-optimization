"""Tests for shared evaluator (plan task 10 — RED phase).

Covers:
  (a) drives fake policies of all three kinds (callable, zero-noise MFPO,
      mean-action PPO, deterministic SAC) producing CSV rows
  (b) determinism: same weights+eval_seed twice → identical episode returns
  (c) acting batch B=1 works WITH frozen stats
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

from mfpo.config import MFPOConfig
from mfpo.networks import ContextTrunk, EMAProjection, SpectralTokens
from mfpo.networks import PolicyFlowHead
from mfpo.fpo_core import euler_sample
from mfpo.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Fake environment
# ---------------------------------------------------------------------------


class FakeEnv:
    """Minimal fake MuJoCo-like env for evaluator tests."""

    def __init__(self, obs_dim: int = 17, action_dim: int = 4, max_steps: int = 50):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.RandomState(0)
        self.observation_space = type("Space", (), {"shape": (obs_dim,)})()
        self.action_space = type("Space", (), {"shape": (action_dim,)})()

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        obs = self._rng.randn(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.randn(self.obs_dim).astype(np.float32)
        reward = float(self._rng.randn() * 0.1)
        terminated = self._step_count >= self._max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# Policy adapters
# ---------------------------------------------------------------------------


def _callable_policy(obs: np.ndarray) -> np.ndarray:
    """Simple callable returning random actions."""
    return np.random.randn(4).astype(np.float32)


class MFPOPolicyAdapter:
    """MFPO-style adapter with zero-noise eval and internal context window."""

    def __init__(self, trunk, policy_head, spectral, ema_proj, cfg):
        from collections import deque

        self.trunk = trunk
        self.policy_head = policy_head
        self.spectral = spectral
        self.ema_proj = ema_proj
        self.cfg = cfg
        self._window = deque(maxlen=cfg.context_len)
        self._first_obs = None

    def reset(self):
        self._window.clear()
        self._first_obs = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if self._first_obs is None:
            self._first_obs = obs_t
            for _ in range(self.cfg.context_len):
                self._window.append(self._first_obs.clone())
        self._window.append(obs_t)
        context = torch.stack(list(self._window), dim=0).unsqueeze(0)  # [1, T, obs]
        with torch.no_grad():
            z = self.trunk(context)
            cond = self.spectral(z)
            # EMA projection online side
            latent = self.ema_proj(obs_t.unsqueeze(0))
            # Policy head (zero noise via eps=zeros for deterministic eval)
            chunk = euler_sample(
                self.policy_head,
                (1, self.cfg.chunk_len, self.action_dim),
                cond,
                steps=self.cfg.euler_eval,
                eps=torch.zeros(1, self.cfg.chunk_len, self.action_dim),
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


# ---------------------------------------------------------------------------
# (a) drives fake policies of all three kinds
# ---------------------------------------------------------------------------


class TestPolicyKinds:
    def test_callable_policy(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=3)
        results = evaluator.evaluate(_callable_policy)
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

    def test_mfpo_zero_noise(self):
        cfg = MFPOConfig()
        trunk = ContextTrunk(17, context_len=cfg.context_len, z_dim=cfg.Z_dim)
        policy_head = PolicyFlowHead(4, z_dim=cfg.Z_dim, cond_dims=(64,))
        spectral = SpectralTokens(z_dim=cfg.Z_dim, context_len=cfg.context_len)
        spectral.set_frozen_stats(torch.zeros(2 * cfg.Z_dim), torch.ones(2 * cfg.Z_dim))
        ema_proj = EMAProjection(17, latent=cfg.Z_dim)

        env = FakeEnv(obs_dim=17, action_dim=4)
        adapter = MFPOPolicyAdapter(trunk, policy_head, spectral, ema_proj, cfg)
        adapter.action_dim = 4

        evaluator = Evaluator(env, eval_seed=42, max_episodes=2)
        results = evaluator.evaluate(adapter)
        assert len(results) == 2

    def test_csv_schema(self):
        env = FakeEnv()
        evaluator = Evaluator(env, eval_seed=42, max_episodes=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator.evaluate(_callable_policy, out_dir=tmpdir)
            csv_path = Path(tmpdir) / "eval_history.csv"
            assert csv_path.exists()
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            for row in rows:
                assert "return" in row
                assert "length" in row


# ---------------------------------------------------------------------------
# (b) determinism: same weights+eval_seed twice → identical episode returns
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_returns(self):
        env1 = FakeEnv()
        env2 = FakeEnv()
        ev1 = Evaluator(env1, eval_seed=42, max_episodes=5)
        ev2 = Evaluator(env2, eval_seed=42, max_episodes=5)

        r1 = ev1.evaluate(_callable_policy)
        r2 = ev2.evaluate(_callable_policy)

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

        r1 = ev1.evaluate(_callable_policy)
        r2 = ev2.evaluate(_callable_policy)

        # With different seeds, at least one episode return should differ
        returns1 = [r["return"] for r in r1]
        returns2 = [r["return"] for r in r2]
        assert returns1 != returns2, "different seeds produced identical returns"


# ---------------------------------------------------------------------------
# (c) acting batch B=1 works WITH frozen stats
# ---------------------------------------------------------------------------


class TestBatchSizeOne:
    def test_b1_with_frozen_stats(self):
        cfg = MFPOConfig()
        trunk = ContextTrunk(17, context_len=cfg.context_len, z_dim=cfg.Z_dim)
        policy_head = PolicyFlowHead(4, z_dim=cfg.Z_dim, cond_dims=(64,))
        spectral = SpectralTokens(z_dim=cfg.Z_dim, context_len=cfg.context_len)
        spectral.set_frozen_stats(torch.zeros(2 * cfg.Z_dim), torch.ones(2 * cfg.Z_dim))
        ema_proj = EMAProjection(17, latent=cfg.Z_dim)

        env = FakeEnv(obs_dim=17, action_dim=4)
        adapter = MFPOPolicyAdapter(trunk, policy_head, spectral, ema_proj, cfg)
        adapter.action_dim = 4

        evaluator = Evaluator(env, eval_seed=0, max_episodes=2)
        results = evaluator.evaluate(adapter)
        assert len(results) == 2
        for r in results:
            assert math.isfinite(r["return"])

    def test_unset_frozen_stats_raises(self):
        cfg = MFPOConfig()
        trunk = ContextTrunk(17, context_len=cfg.context_len, z_dim=cfg.Z_dim)
        policy_head = PolicyFlowHead(4, z_dim=cfg.Z_dim, cond_dims=(64,))
        spectral = SpectralTokens(z_dim=cfg.Z_dim, context_len=cfg.context_len)
        # Do NOT set frozen stats
        ema_proj = EMAProjection(17, latent=cfg.Z_dim)

        env = FakeEnv(obs_dim=17, action_dim=4)
        adapter = MFPOPolicyAdapter(trunk, policy_head, spectral, ema_proj, cfg)
        adapter.action_dim = 4

        evaluator = Evaluator(env, eval_seed=0, max_episodes=1)
        with pytest.raises(RuntimeError, match="frozen stats"):
            evaluator.evaluate(adapter)


# ---------------------------------------------------------------------------
# (d) 12-episode default + max-steps cap respected
# ---------------------------------------------------------------------------


class TestEpisodeAndStepLimits:
    def test_max_episodes_respected(self):
        env = FakeEnv(max_steps=100)
        evaluator = Evaluator(env, eval_seed=42, max_episodes=12)
        results = evaluator.evaluate(_callable_policy)
        assert len(results) == 12

    def test_max_steps_cap(self):
        env = FakeEnv(max_steps=100)  # long episodes
        evaluator = Evaluator(env, eval_seed=42, max_episodes=100, max_steps=50)
        results = evaluator.evaluate(_callable_policy)
        total_steps = sum(r["length"] for r in results)
        assert total_steps <= 50, f"total steps {total_steps} > max_steps 50"


# Need this import for TestBatchSizeOne
import math
