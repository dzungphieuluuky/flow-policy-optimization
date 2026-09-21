"""MJX GPU rollout env: numpy semantics + opt-in gating (CPU-safe).

The jitted physics itself needs a CUDA box (Blackwell); these tests pin the
parts that run anywhere: observation/reward math, gating flags, and loud
failure when MJX is requested but unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.mjx_env import (  # noqa: E402
    halfcheetah_obs,
    halfcheetah_reward,
    mjx_available,
    mjx_collect_chunks,
    resolve_env_name,
    want_mjx,
)


def test_obs_drops_rootx():
    qpos = np.arange(9, dtype=np.float64)
    qvel = np.arange(9, dtype=np.float64) + 100
    obs = halfcheetah_obs(qpos, qvel)
    assert obs.shape == (17,)
    assert obs[0] == 1.0  # rootz, not rootx
    assert obs[8] == 100.0  # rootx velocity kept


def test_reward_forward_minus_ctrl():
    # dx=0.05 over dt=0.05 -> v=1.0; ctrl=0.1*||a||^2 over 6 ones = 0.6
    r = halfcheetah_reward(0.0, 0.05, 0.05, np.ones(6))
    assert r == pytest.approx(1.0 - 0.6)


def test_want_mjx_gating(monkeypatch):
    monkeypatch.delenv("SJFPO_MJX", raising=False)
    assert want_mjx() is False
    monkeypatch.setenv("SJFPO_MJX", "1")
    assert want_mjx() is True


def test_available_never_raises():
    assert isinstance(mjx_available(), bool)


def test_resolve_env_name_single_and_vec():
    """Regression: default num_envs=2 makes a SyncVectorEnv (spec=None)."""
    import gymnasium as gym

    env = gym.make("HalfCheetah-v5")
    vec = gym.vector.SyncVectorEnv(
        [lambda: gym.make("HalfCheetah-v5") for _ in range(2)]
    )
    try:
        assert resolve_env_name(env) == "HalfCheetah-v5"
        assert resolve_env_name(vec) == "HalfCheetah-v5"
        assert vec.spec is None  # the trap that caused 'unknown'
    finally:
        vec.close()
        env.close()


def test_unsupported_env_rejected_without_gpu():
    with pytest.raises(ValueError, match="supports"):
        mjx_collect_chunks(
            object(), "Hopper-v5", 16, -1.0, 1.0,
            n_obs_steps=16, n_action_steps=16, num_envs=1, seed=0,
        )


def test_explicit_opt_in_fails_loud_on_cpu_box():
    if mjx_available():
        pytest.skip("GPU MJX box: fallback path not exercised")
    with pytest.raises(RuntimeError, match="SJFPO_MJX=1 needs"):
        mjx_collect_chunks(
            object(), "HalfCheetah-v5", 16, -1.0, 1.0,
            n_obs_steps=16, n_action_steps=16, num_envs=1, seed=0,
        )


def test_module_imports_without_jax_or_mjx():
    """Heavy deps are lazy: importing mjx_env must never require them."""
    import importlib

    import sjfpo.mjx_env as m

    importlib.reload(m)
    assert isinstance(m.STORAGE_DTYPE, torch.dtype)  # resolves on any box
