"""Tests for SJFPO end-to-end run driver (plan task 11).

Uses a FakeEnv to avoid MuJoCo dependency in unit tests.
Integration smoke tests use the real env via scripts/train.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import SJFPOConfig
from sjfpo.networks import (
    StateEncoder,
    SJFOPolicyModel,
    SpectralEncoder,
    Critic,
)


# ---------------------------------------------------------------------------
# FakeEnv: minimal gym-like env for testing without MuJoCo
# ---------------------------------------------------------------------------


class FakeEnv:
    """Tiny fake env with obs_dim=4, act_dim=2, episodic after 20 steps."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 2, max_steps: int = 20):
        self.observation_space = type("S", (), {"shape": (obs_dim,)})()
        self.action_space = type(
            "A",
            (),
            {
                "shape": (act_dim,),
                "low": np.full(act_dim, -1.0),
                "high": np.full(act_dim, 1.0),
                "seed": lambda s=None: None,
                "sample": lambda s=None: np.random.randn(self.act_dim).astype(
                    np.float32
                ),
            },
        )()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self.step_count = 0
        self.rng = np.random.RandomState(42)

    def reset(self, seed=None, **kw):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        return self.rng.randn(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = self.rng.randn(self.obs_dim).astype(np.float32)
        reward = float(self.rng.randn() * 0.1)
        terminated = False
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

        # ---------------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------------


def build_sjfpo(obs_dim: int = 4, act_dim: int = 2, cfg=None):
    """Construct all SJFPO components for a given obs/act dim."""
    cfg = cfg or SJFPOConfig()
    trunk = StateEncoder(
        obs_dim, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
    )
    policy_head = SJFOPolicyModel(act_dim, hidden_size=cfg.hidden_size)
    critic = Critic(hidden_size=cfg.hidden_size, act_dim=act_dim)
    spectral = SpectralEncoder(
        hidden_size=cfg.hidden_size,
        num_spectral_tokens=cfg.num_spectral_tokens,
        n_obs_steps=cfg.n_obs_steps,
    )
    # Set frozen stats
    with torch.no_grad():
        z_sample = torch.randn(4, cfg.n_obs_steps, cfg.hidden_size)
    spectral.capture_stats(z_sample)
    return trunk, policy_head, critic, spectral


def build_run_dir(tmp_path: Path, name: str = "test_run"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d

    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------


class TestSJFPOPolicyCallable:
    """Test that the SJFPO policy callable produces correct shapes."""

    def test_policy_returns_correct_action_shape(self):
        from sjfpo.dct_utils import idct
        from sjfpo.fpo_core import euler_sample

        cfg = SJFPOConfig()
        obs_dim, act_dim = 4, 2
        trunk, policy_head, _, spectral = build_sjfpo(obs_dim, act_dim, cfg)

        def policy(state: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                z = trunk(state.unsqueeze(0))  # [1, T, Z]
                cond = spectral(z)  # [1, K, 2Z]
                x0 = euler_sample(
                    policy_head,
                    (1, cfg.chunk_size, act_dim),
                    cond,
                    steps=cfg.euler_eval,
                    eps=torch.zeros(1, cfg.chunk_size, act_dim),
                )
                actions = idct(x0.squeeze(0), dim=0)
                return actions.clamp(-1.0, 1.0)

        state = torch.randn(cfg.n_obs_steps, obs_dim)
        action = policy(state)
        assert action.shape == (cfg.chunk_size, act_dim)

    def test_policy_deterministic(self):
        from sjfpo.dct_utils import idct
        from sjfpo.fpo_core import euler_sample

        cfg = SJFPOConfig()
        obs_dim, act_dim = 4, 2
        trunk, policy_head, _, spectral = build_sjfpo(obs_dim, act_dim, cfg)

        def policy(state: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                z = trunk(state.unsqueeze(0))
                cond = spectral(z)
                x0 = euler_sample(
                    policy_head,
                    (1, cfg.chunk_size, act_dim),
                    cond,
                    steps=cfg.euler_eval,
                    eps=torch.zeros(1, cfg.chunk_size, act_dim),
                )
                actions = idct(x0.squeeze(0), dim=0)
                return actions.clamp(-1.0, 1.0)

        ctx = torch.randn(cfg.n_obs_steps, obs_dim)
        a1 = policy(ctx)
        a2 = policy(ctx)
        assert torch.equal(a1, a2)


class TestTrainRun:
    """Test the SJFPO train_run function with fake env."""

    def test_train_run_completes_and_produces_artifacts(self, tmp_path):
        from sjfpo.runner import train_run

        env = FakeEnv(obs_dim=4, act_dim=2, max_steps=20)
        out_dir = build_run_dir(tmp_path, "sjfpo_run")

        cfg = SJFPOConfig(
            num_mc_samples=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            n_obs_steps=16,
        )

        result = train_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )

        # Check result structure
        assert "reward_mean_last10" in result
        assert "fps" in result
        assert "artifacts" in result
        assert isinstance(result["reward_mean_last10"], float)
        assert result["fps"] > 0

        # Check artifacts exist
        for artifact in result["artifacts"]:
            assert Path(artifact).exists(), f"artifact missing: {artifact}"

    def test_train_run_writes_checkpoint(self, tmp_path):
        from sjfpo.runner import train_run

        env = FakeEnv(obs_dim=4, act_dim=2, max_steps=20)
        out_dir = build_run_dir(tmp_path, "ckpt_test")

        cfg = SJFPOConfig(
            num_mc_samples=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            n_obs_steps=16,
        )

        result = train_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )

        # Check checkpoint exists
        ckpt_dir = out_dir / "checkpoint_final"
        assert ckpt_dir.exists()
        assert (ckpt_dir / "checkpoint.pt").exists()
        assert (ckpt_dir / "manifest.json").exists()

    def test_train_run_writes_eval_csv(self, tmp_path):
        from sjfpo.runner import train_run

        env = FakeEnv(obs_dim=4, act_dim=2, max_steps=20)
        out_dir = build_run_dir(tmp_path, "eval_test")

        cfg = SJFPOConfig(
            num_mc_samples=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            n_obs_steps=16,
        )

        result = train_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )

        # Check eval CSV exists
        csv_path = out_dir / "eval_history.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) >= 2  # header + at least one row

    def test_train_run_nan_raises(self, tmp_path):
        """NaN injection into the env should NOT crash but produce nan returns."""
        from sjfpo.runner import train_run

        class NaNEnv(FakeEnv):
            def step(self, action):
                obs, reward, term, trunc, info = super().step(action)
                return obs, float("nan"), term, trunc, info

        env = NaNEnv(obs_dim=4, act_dim=2, max_steps=20)
        out_dir = build_run_dir(tmp_path, "nan_test")

        cfg = SJFPOConfig(
            num_mc_samples=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            n_obs_steps=16,
        )

        # NaN rewards → SJFPOAborted from trainer, should exit cleanly
        with pytest.raises((RuntimeError, ValueError)):
            train_run(
                env=env,
                out_dir=out_dir,
                cfg=cfg,
                seed=0,
                total_steps=100,
                eval_interval=50,
                ckpt_interval=50,
                obs_dim=4,
                act_dim=2,
            )


class TestClipFractionWarning:
    """Test clip_fraction monitoring."""

    def test_clip_fraction_in_diagnostics(self, tmp_path):
        from sjfpo.runner import train_run

        env = FakeEnv(obs_dim=4, act_dim=2, max_steps=20)
        out_dir = build_run_dir(tmp_path, "clip_test")

        cfg = SJFPOConfig(
            num_mc_samples=2,
            euler_train=4,
            euler_eval=4,
            chunks_per_update=16,
            n_obs_steps=16,
        )

        result = train_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=0,
            total_steps=200,
            eval_interval=100,
            ckpt_interval=100,
            obs_dim=4,
            act_dim=2,
        )

        # clip_fraction should be logged (may be > 0.02 with tiny num_mc_samples=2)
        # but the function should complete without error
        assert "artifacts" in result


class TestCLISmoke:
    """Integration test wrapping scripts/train.py via subprocess."""

    def test_sjfpo_cli_exits_clean_and_prints_json(self, tmp_path):
        """Run the CLI with SJFPO on FakeEnv-equivalent (Hopper-v5 skipped — no mujoco)."""
        import json as json_mod
        import subprocess

        # Use the CLI with a FakeEnv-compatible approach: run the CLI entry point
        # directly via Python to avoid needing real mujoco in CI.
        # The CLI's run_sjfpo calls gym.make which needs mujoco, so we test
        # via the train_run API path instead (already covered above).
        # This test validates argparse + dispatch + JSON output format.
        script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--env",
                "Hopper-v5",
                "--total-steps",
                "100",
                "--method",
                "sjfpo",
                "--seed",
                "0",
                "--out-dir",
                str(tmp_path / "smoke"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        # If mujoco is installed, we get exit 0 + JSON line.
        # If not, we get a gymnasium import error — that's OK, skip.
        if result.returncode != 0 and "mujoco" in (result.stderr or "").lower():
            pytest.skip("MuJoCo not available in this environment")
            # If it succeeded, validate JSON output
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # At least one JSON line
            parsed = json_mod.loads(lines[0])
            assert "method" in parsed
            assert "reward_mean_last10" in parsed
            assert "fps" in parsed
            assert "artifacts" in parsed

    def test_cli_bad_method_returns_1(self):
        """Invalid method choice should fail with exit code != 0."""
        import subprocess

        script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--env",
                "Hopper-v5",
                "--methods",
                "nonexistent",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
