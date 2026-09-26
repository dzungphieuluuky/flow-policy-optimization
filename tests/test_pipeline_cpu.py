"""CPU pipeline invariants — T4 of mujoco-training-pipeline-rocm7.14.md

Locks the CPU-first contract so regressions are caught:
- SJFPO writes five-column evaluation CSV files
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

from sjfpo.configuration_sjfpo import SJFPOConfig
from tests.fixtures.fake_env import FakeGymEnv


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


def test_sjfpo_aborts_when_cfm_loss_saturates_cfm_clamp(tmp_path):
    """The training loop must not sit in the clamp dead zone silently.

    With cfm_clamp below the achievable CFM loss, clamp() zeroes the ratio
    gradient and the run can only plateau, so the loop aborts instead.
    """
    from sjfpo.modeling_sjfpo import SJFPOAborted
    from sjfpo.train_sjfpo import train_sjfpo_run

    cfg = SJFPOConfig(
        num_mc_samples=2,
        num_euler_steps=4,
        num_inference_steps=4,
        chunks_per_update=8,
        n_obs_steps=16,
        dit_depth=1,
        cfm_clamp=0.01,
    )
    with pytest.raises(SJFPOAborted, match="cfm_clamp"):
        train_sjfpo_run(
            env=FakeGymEnv(obs_dim=4, action_dim=2, max_steps=50),
            out_dir=tmp_path / "dead",
            cfg=cfg,
            seed=0,
            total_steps=4000,
            eval_steps=10000,
            save_steps=10000,
            obs_dim=4,
            action_dim=2,
            wandb_enabled=False,
        )


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
