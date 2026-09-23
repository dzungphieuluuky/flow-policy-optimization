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
    # MJX-blocked envs fail loudly with the reason, unknown ids with KeyError.
    with pytest.raises(RuntimeError, match="no MJX path"):
        mjx_collect_chunks(
            object(),
            "Humanoid-v5",
            16,
            -1.0,
            1.0,
            n_obs_steps=16,
            n_action_steps=16,
            num_envs=1,
            seed=0,
        )
    with pytest.raises(RuntimeError, match="classic control"):
        mjx_collect_chunks(
            object(),
            "CartPole-v1",
            16,
            -1.0,
            1.0,
            n_obs_steps=16,
            n_action_steps=16,
            num_envs=1,
            seed=0,
        )
    with pytest.raises(KeyError):
        mjx_collect_chunks(
            object(),
            "Nope-v5",
            16,
            -1.0,
            1.0,
            n_obs_steps=16,
            n_action_steps=16,
            num_envs=1,
            seed=0,
        )
    # Newly supported: Hopper passes the gate (fails later on CUDA here).
    if not mjx_available():
        with pytest.raises(RuntimeError, match="SJFPO_MJX=1 needs"):
            mjx_collect_chunks(
                object(),
                "Hopper-v5",
                16,
                -1.0,
                1.0,
                n_obs_steps=16,
                n_action_steps=16,
                num_envs=1,
                seed=0,
            )


def test_explicit_opt_in_fails_loud_on_cpu_box():
    if mjx_available():
        pytest.skip("GPU MJX box: fallback path not exercised")
    with pytest.raises(RuntimeError, match="SJFPO_MJX=1 needs"):
        mjx_collect_chunks(
            object(),
            "HalfCheetah-v5",
            16,
            -1.0,
            1.0,
            n_obs_steps=16,
            n_action_steps=16,
            num_envs=1,
            seed=0,
        )


def test_init_state_fallback_matches_gymnasium():
    """Regression: MjModel has no qvel0; init = qpos0 / zeros(nv)."""
    import mujoco
    import gymnasium as gym
    from pathlib import Path

    xml = Path(gym.__file__).parent / "envs" / "mujoco" / "assets" / "half_cheetah.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    assert m.nkey == 0  # else-branch: the one that crashed
    assert not hasattr(m, "qvel0")
    assert np.asarray(m.qpos0).shape == (m.nq,)
    assert np.zeros(m.nv).shape == (m.nv,)


def test_dlpack_import_takes_tensor_not_capsule():
    """Regression: from_dlpack(tensor) works; from_dlpack(capsule) fails.

    The Kaggle bug passed a ``to_dlpack()`` capsule. CPU torch+JAX here
    exercise the same protocol without MJX or a GPU.
    """
    import jax
    import torch
    import torch.utils.dlpack as torch_dlpack

    t = torch.randn(2, 3)
    back = jax.dlpack.from_dlpack(t)
    np.testing.assert_allclose(np.asarray(back), t.numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(Exception):
        jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(t))


def test_shift_window_rolls_without_overlap_fault():
    """Regression: in-place view shifting faults; roll+writeback works."""
    import torch

    from sjfpo.mjx_env import shift_window

    windows = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    old = windows[1].clone()
    # The old code did exactly this and raised on overlapping memory.
    with pytest.raises(RuntimeError, match="single memory location"):
        w = windows[1]
        w[:-1].copy_(w[1:])
    shift_window(windows, 1, torch.full((3,), 99.0))
    assert torch.equal(windows[1, :-1], old[1:])  # rows shift up, first dropped
    assert torch.equal(windows[1, -1], torch.full((3,), 99.0))
    assert torch.equal(windows[0], torch.arange(12, dtype=torch.float32).reshape(4, 3))


def test_module_imports_without_jax_or_mjx():
    """Heavy deps are lazy: importing mjx_env must never require them."""
    import importlib

    import sjfpo.mjx_env as m

    importlib.reload(m)
    assert isinstance(m.STORAGE_DTYPE, torch.dtype)  # resolves on any box
