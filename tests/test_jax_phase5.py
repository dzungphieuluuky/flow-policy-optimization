"""Phase 5: train_jax_run end-to-end on CPU (stub vec + gym eval)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax.training import spec_dims, train_jax_run  # noqa: E402


class StubVec17:
    """Deterministic JAX vec, HalfCheetah obs dim, term at ep>=5."""

    obs_dim = 17

    def __init__(self, env_name, num_envs, seed):
        self.num_envs = num_envs
        self.ep = jnp.zeros((num_envs,), dtype=jnp.int32)

    def reset(self):
        self.ep = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        return jnp.zeros((self.num_envs, self.obs_dim))

    def step(self, ctrl):
        self.ep = self.ep + 1
        term = self.ep >= 5
        trunc = self.ep >= 1000
        done = term | trunc
        obs = jnp.ones((self.num_envs, self.obs_dim)) * self.ep[:, None].astype(jnp.float32)
        if bool(jnp.any(done)):
            self.ep = jnp.where(done, 0, self.ep)
            obs = jnp.zeros((self.num_envs, self.obs_dim))
        return obs, jnp.ones((self.num_envs,)), term, trunc


def _tiny_run(out_dir, **overrides):
    import os

    os.environ["SJFPO_DISABLE_WANDB"] = "1"
    eval_env = gym.make("HalfCheetah-v5")
    kwargs = dict(
        env_name="HalfCheetah-v5", eval_env=eval_env,
        out_dir=out_dir, seed=0, total_steps=16,
        eval_steps=16, save_steps=16, num_envs=1, n_obs_steps=4,
        action_horizon=8, hidden_dim=8, dit_depth=1, dit_heads=2,
        dit_mlp_ratio=2.0, dit_time_dim=8, num_epochs=1, num_minibatches=1,
        chunks_per_update=2, num_mc_samples=2, num_euler_steps=2,
        num_inference_steps=2, eval_episodes=1, vec_factory=StubVec17,
    )
    kwargs.update(overrides)
    try:
        return train_jax_run(**kwargs)
    finally:
        eval_env.close()


def test_profiler_trace_gated_by_env(tmp_path, monkeypatch):
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("SJFPO_PROFILE_DIR", str(trace_dir))
    result = _tiny_run(tmp_path / "run")
    assert result["profile_dir"] == str(trace_dir)
    assert any(trace_dir.iterdir()), "trace files expected"
    monkeypatch.delenv("SJFPO_PROFILE_DIR")
    result = _tiny_run(tmp_path / "run2")
    assert "profile_dir" not in result


def test_spec_dims_match_registry():
    assert spec_dims("HalfCheetah-v5") == (17, 6)
    assert spec_dims("Hopper-v5") == (11, 3)


def test_train_jax_run_smoke(tmp_path):
    import os

    os.environ["SJFPO_DISABLE_WANDB"] = "1"
    eval_env = gym.make("HalfCheetah-v5")
    try:
        result = train_jax_run(
            env_name="HalfCheetah-v5", eval_env=eval_env,
            out_dir=tmp_path / "run", seed=0, total_steps=64,
            eval_steps=64, save_steps=64, num_envs=1, n_obs_steps=4,
            action_horizon=8, hidden_dim=8, dit_depth=1, dit_heads=2,
            dit_mlp_ratio=2.0, dit_time_dim=8, num_epochs=1, num_minibatches=2,
            chunks_per_update=4, num_mc_samples=2, num_euler_steps=2,
            num_inference_steps=2, eval_episodes=1, vec_factory=StubVec17,
        )
    finally:
        eval_env.close()
    assert result["method"] == "sjfpo_jax"
    assert result["global_step"] == 64
    assert result["diverged"] is False
    for artifact in result["artifacts"]:
        assert Path(artifact).exists(), artifact
