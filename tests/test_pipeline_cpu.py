"""CPU pipeline invariants — T4 of mujoco-training-pipeline-rocm7.14.md

Locks the CPU-first contract so regressions are caught:
- train_run and baselines write 3-col CSV [episode,return,length]
- RUN_MATRIX cardinality 45 (and TINY 3)
"""

from __future__ import annotations

import os
import subprocess
import sys
import csv
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from sjfpo.config import build_run_matrix, SJFPOConfig, PPOConfig, SACConfig
from tests.fixtures.fake_env import FakeGymEnv


def test_train_run_writes_3col_csv(tmp_path):
    from sjfpo.runner import train_run

    env = FakeGymEnv()
    out = tmp_path / "sjfpo_3col"
    cfg = SJFPOConfig(
        num_mc_samples=2,
        euler_train=4,
        euler_eval=4,
        chunks_per_update=16,
        n_obs_steps=16,
    )
    train_run(
        env=env,
        out_dir=out,
        cfg=cfg,
        seed=0,
        total_steps=200,
        eval_interval=100,
        ckpt_interval=100,
        obs_dim=4,
        act_dim=2,
    )
    csv_path = out / "eval_history.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["episode", "return", "length"], f"header mismatch: {header}"
        rows = list(reader)
        assert len(rows) >= 1
        for r in rows:
            assert len(r) == 3


def test_baselines_write_3col_csv(tmp_path, monkeypatch):
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
        eval_interval=200,
        ckpt_interval=200,
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
        assert header == ["episode", "return", "length"]


def test_run_matrix_cardinality_is_45():
    from sjfpo.config import RUN_MATRIX, TINY_MATRIX

    assert len(build_run_matrix()) == 45
    assert len(RUN_MATRIX) == 45
    assert TINY_MATRIX is False
    # tiny env
    os.environ["SJFPO_TINY_MATRIX"] = "1"
    try:
        import importlib
        import sjfpo.config as cfg_mod

        importlib.reload(cfg_mod)
        assert len(cfg_mod.RUN_MATRIX) == 3
        for s in cfg_mod.RUN_MATRIX:
            assert s.env == "Hopper-v5"
        importlib.reload(
            cfg_mod
        )  # reload back will be handled by fixture in other tests, but restore here
        # force back to default
        if "SJFPO_TINY_MATRIX" in os.environ:
            del os.environ["SJFPO_TINY_MATRIX"]
        importlib.reload(cfg_mod)
    finally:
        if "SJFPO_TINY_MATRIX" in os.environ:
            del os.environ["SJFPO_TINY_MATRIX"]
        import importlib
        import sjfpo.config as cfg_mod

        importlib.reload(cfg_mod)


# ---------------------------------------------------------------------------
# T7 — wandb integration invariants (mujoco-wandb-logging plan)
# Recovered from pre-existing uncommitted working-tree additions (see tmp/t8-noisy).
# ---------------------------------------------------------------------------


def test_wandb_logging_does_not_affect_csv(tmp_path, monkeypatch):
    """SJFPO_DISABLE_WANDB=1 → train_run still writes 3-col CSV [episode,return,length]."""
    monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
    from sjfpo.runner import train_run

    env = FakeGymEnv()
    out = tmp_path / "wandb_disabled_csv"
    cfg = SJFPOConfig(
        num_mc_samples=2,
        euler_train=4,
        euler_eval=4,
        chunks_per_update=16,
        n_obs_steps=16,
    )
    train_run(
        env=env,
        out_dir=out,
        cfg=cfg,
        seed=0,
        total_steps=200,
        eval_interval=100,
        ckpt_interval=100,
        obs_dim=4,
        act_dim=2,
    )
    csv_path = out / "eval_history.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["episode", "return", "length"], f"header mismatch: {header}"
        rows = list(reader)
        assert len(rows) >= 1
        for r in rows:
            assert len(r) == 3


def test_wandb_logging_does_not_change_run_matrix():
    """RUN_MATRIX cardinality remains 45 regardless of wandb integration."""
    assert len(build_run_matrix()) == 45


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


def test_wandb_disabled_in_train_run(tmp_path, monkeypatch):
    """SJFPO_DISABLE_WANDB=1 → train_run return dict has expected keys, no wandb artifacts."""
    monkeypatch.setenv("SJFPO_DISABLE_WANDB", "1")
    from sjfpo.runner import train_run

    env = FakeGymEnv()
    out = tmp_path / "wandb_unset_return"
    cfg = SJFPOConfig(
        num_mc_samples=2,
        euler_train=4,
        euler_eval=4,
        chunks_per_update=16,
        n_obs_steps=16,
    )
    result = train_run(
        env=env,
        out_dir=out,
        cfg=cfg,
        seed=0,
        total_steps=200,
        eval_interval=100,
        ckpt_interval=100,
        obs_dim=4,
        act_dim=2,
    )
    expected_keys = {
        "method",
        "env_name",
        "seed",
        "reward_mean_last10",
        "fps",
        "artifacts",
        "diverged",
        "global_step",
    }
    assert expected_keys == set(result.keys()), (
        f"missing keys: {expected_keys - set(result.keys())}"
    )
    assert result["method"] == "sjfpo"
    assert isinstance(result["artifacts"], list)
    # no wandb-specific artifacts (wandb/, *.wandb files) when disabled
    wandb_artifacts = [
        a
        for a in result["artifacts"]
        if "/wandb/" in str(a) or str(a).endswith(".wandb")
    ]
    assert not wandb_artifacts, f"unexpected wandb artifacts: {wandb_artifacts}"
