"""Phase 3b: env-semantics parity vs live gymnasium + rollout loop on a stub."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax import policy as jpol  # noqa: E402
from sjfpo_jax import rollout as jroll  # noqa: E402
from sjfpo_jax.envs import (  # noqa: E402
    get_spec,
    is_healthy,
    obs,
    reset_state,
    reward,
    terminated,
)


def test_unknown_env_rejected():
    with pytest.raises(KeyError):
        get_spec("Nope-v5")


def test_semantics_match_live_gym():
    import gymnasium as gym

    for env_name in ("HalfCheetah-v5", "Hopper-v5"):
        spec = get_spec(env_name)
        env = gym.make(env_name)
        try:
            obs0, _ = env.reset(seed=0)
            d = env.unwrapped.data
            np.testing.assert_allclose(
                obs(spec, d.qpos, d.qvel), np.asarray(obs0, dtype=np.float64),
                rtol=1e-5, atol=1e-5,
            )
            rng = np.random.default_rng(1)
            action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float64)
            x_before = float(d.qpos[0])
            dt = float(env.unwrapped.dt)
            o, r, term, trunc, info = env.step(action)
            d2 = env.unwrapped.data
            healthy = is_healthy(spec, d2.qpos, d2.qvel)
            assert term == terminated(spec, healthy)
            assert r == pytest.approx(
                reward(spec, x_before, float(d2.qpos[0]), dt, action, healthy),
                abs=1e-5,
            )
            np.testing.assert_allclose(
                obs(spec, d2.qpos, d2.qvel), np.asarray(o, dtype=np.float64),
                rtol=1e-5, atol=1e-5,
            )
            # Unhealthy state terminates hopper, never the cheetah.
            bad = np.asarray(d2.qpos); bad[1] = 0.1
            h = is_healthy(spec, bad, np.asarray(d2.qvel))
            assert terminated(spec, h) == (env_name == "Hopper-v5")
            # Reset sampler shapes/ranges.
            qpos, qvel = reset_state(
                spec, np.random.default_rng(0),
                np.zeros(env.unwrapped.model.nq), np.zeros(env.unwrapped.model.nv),
            )
            assert qpos.shape == (env.unwrapped.model.nq,)
            assert qvel.shape == (env.unwrapped.model.nv,)
            assert bool(np.all(np.abs(qpos) <= spec["reset_noise_scale"] + 1e-12))
        finally:
            env.close()


class StubVec:
    """Deterministic JAX vec (no MJX): constant obs, term at ep>=5."""

    obs_dim = 6

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


def test_generate_batch_on_stub():
    pj, _ = jpol.build_policy(6, 2, seed=0, hidden_dim=8, dit_depth=1, dit_heads=2)
    batch = jroll.generate_batch(
        pj, "HalfCheetah-v5", 40, jnp.full((2,), -1.0), jnp.full((2,), 1.0),
        jax.random.PRNGKey(0), n_obs_steps=4, action_horizon=8, num_envs=2,
        num_mc_samples=2, top_p_spectral_tokens=1.0, num_euler_steps=2,
        vec_factory=StubVec,
    )
    n = batch["advantages"].shape[0]
    assert n > 0
    assert batch["observation.state"].shape == (n, 4, 6)
    assert batch["action"].shape == (n, 8, 2)
    assert batch["eps"].shape == (n, 2, 8, 2)
    assert batch["tau"].shape == (n, 2)
    # Executed steps match the stub's 5-step episodes + budget tail.
    assert int((~batch["action_is_pad"]).sum()) == 40
    assert bool(np.all(np.isfinite(np.asarray(batch["advantages"]))))
    # Termination path exercised (stub ends episodes at 5 steps).
    assert batch["returns"].shape == (n,)
