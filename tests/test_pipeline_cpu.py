"""CPU pipeline invariants — T4 of mujoco-training-pipeline-rocm7.14.md

Locks the CPU-first contract so regressions are caught:
- baselines and SJFPO write five-column evaluation CSV files
"""

from __future__ import annotations

import os
import subprocess
import sys
import csv
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from sjfpo.config import PPOConfig, SACConfig
from sjfpo.configuration_sjfpo import SJFPOConfig
from tests.fixtures.fake_env import FakeGymEnv


def test_baselines_write_5col_csv(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from sjfpo.baselines import ppo_train_run

    env = FakeGymEnv()
    cfg = PPOConfig(rollouts=32, learning_epochs=2, mini_batches=2, hidden=(64, 64))
    result = ppo_train_run(
        env=env,
        out_dir=tmp_path / "ppo_3col",
        cfg=cfg,
        seed=0,
        total_steps=200,
        eval_steps=200,
        save_steps=200,
    )
    # result dict keys per T4
    assert result["method"] == "ppo"
    assert "reward_mean_last10" in result
    assert "artifacts" in result
    assert "diverged" in result
    # find csv artifact
    csv_artifacts = [p for p in result["artifacts"] if p.endswith("eval_history.csv")]
    assert csv_artifacts, "no csv artifact"
    with open(csv_artifacts[0], newline="") as f:
        header = next(csv.reader(f))
        assert header == ["episode", "return", "length", "success", "jerk"]


def test_sjfpo_train_run_writes_5col_csv(tmp_path):
    from sjfpo.train_sjfpo import train_sjfpo_run

    env = FakeGymEnv()
    out = tmp_path / "sjfpo_3col"
    cfg = SJFPOConfig(
        num_mc_samples=2,
        num_euler_steps=4,
        num_inference_steps=4,
        chunks_per_update=16,
        n_obs_steps=16,
        dit_depth=2,
    )
    result = train_sjfpo_run(
        env=env,
        out_dir=out,
        cfg=cfg,
        seed=0,
        total_steps=200,
        eval_steps=100,
        save_steps=100,
        obs_dim=4,
        action_dim=2,
    )
    assert result["method"] == "sjfpo"
    assert any(str(a).endswith("model.safetensors") for a in result["artifacts"])
    csv_path = out / "eval_history.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["episode", "return", "length", "success", "jerk"], (
            f"header mismatch: {header}"
        )
        rows = list(reader)
        assert len(rows) >= 1
        for r in rows:
            assert len(r) == 5


def test_wandb_soft_import_no_crash():
    """Subprocess guard: fresh Python, deleted-gate contract → import succeeds, wl.wandb mockable."""
    code = (
        "import os, sys; "
        "os.environ.pop('SJFPO_DISABLE_WANDB', None); "
        "sys.path.insert(0, 'src'); "
        "import sjfpo.wandb_logging as wl; "
        "assert wl.wandb_disabled() is False; "
        "from unittest.mock import MagicMock; wl.wandb = MagicMock(); "
        "wl.log_metrics({'x': 1}, step=0); "
        "wl.wandb.log.assert_called_once(); "
        "print('wandb default-on guard ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0, (
        f"default-on guard failed: {proc.stderr}\n{proc.stdout}"
    )
    assert "wandb default-on guard ok" in proc.stdout
