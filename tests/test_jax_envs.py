"""Env semantics parity vs live gymnasium + rollout loop on a stub (all MJX envs)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax import policy as jpol  # noqa: E402
from sjfpo_jax import rollout as jroll  # noqa: E402
from sjfpo_jax.envs import (  # noqa: E402
    ENV_SPECS,
    MJX_UNSUPPORTED,
    get_spec,
    is_healthy,
    obs,
    reset_state,
    reward,
    terminated,
)

MJX_ENVS = sorted(ENV_SPECS)


def _forward_before(env, env_name):
    d = env.unwrapped.data
    if env_name == "Ant-v5":
        return float(d.body("torso").xpos[0])
    if env_name in ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
                    "Swimmer-v5", "InvertedDoublePendulum-v5"):
        return float(d.qpos[0])
    return 0.0  # unused by this env's reward (reacher/pusher/pendulum)


def _forward_after(env, env_name):
    return _forward_before(env, env_name)


def _extras(env, spec):
    """Sensor extras from live mujoco data (mirror of the MJX rollout dict)."""
    d = env.unwrapped.data
    extra = {}
    if spec["obs_kind"] == "ant" or spec["reward_kind"] == "locomotion" and spec.get("contact_weight"):
        extra["cfrc_ext"] = np.asarray(d.cfrc_ext)
    if spec["obs_kind"] == "double_pendulum":
        extra["qfrc_constraint"] = np.asarray(d.qfrc_constraint)
        extra["site_xpos0"] = np.asarray(d.site_xpos)[0]
        extra["qvel"] = np.asarray(d.qvel)
    if spec["obs_kind"] == "reacher" or spec["reward_kind"] == "reacher":
        extra["fingertip"] = np.asarray(d.body("fingertip").xpos)
        extra["target"] = np.asarray(d.body("target").xpos)
    if spec["obs_kind"] == "pusher" or spec["reward_kind"] == "pusher":
        extra["tips_arm"] = np.asarray(d.body("tips_arm").xpos)
        extra["object"] = np.asarray(d.body("object").xpos)
        extra["goal"] = np.asarray(d.body("goal").xpos)
    return extra


def _health_kwargs(env, spec, gym_obs):
    if spec["done_kind"] == "pole":
        return {"obs": np.asarray(gym_obs, dtype=np.float64)}
    if spec["done_kind"] == "site_y":
        d = env.unwrapped.data
        return {"site_y": float(np.asarray(d.site_xpos)[0][2])}
    return {}


@pytest.mark.parametrize("env_name", MJX_ENVS)
def test_semantics_match_live_gym(env_name):
    spec = get_spec(env_name)
    env = gym.make(env_name)
    try:
        obs0, _ = env.reset(seed=0)
        d = env.unwrapped.data
        np.testing.assert_allclose(
            obs(spec, d.qpos, d.qvel, _extras(env, spec)),
            np.asarray(obs0, dtype=np.float64), rtol=1e-5, atol=1e-5,
        )
        rng = np.random.default_rng(1)
        action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float64)
        before = _forward_before(env, env_name)
        dt = float(env.unwrapped.dt)
        o, r, term, trunc, _ = env.step(action)
        d2 = env.unwrapped.data
        extra2 = _extras(env, spec)
        kw = _health_kwargs(env, spec, o)
        healthy = is_healthy(spec, d2.qpos, d2.qvel, **kw)
        assert term == terminated(spec, healthy, **kw)
        after = _forward_after(env, env_name)
        assert r == pytest.approx(
            reward(spec, before, after, dt, action, healthy, extra2), abs=1e-5
        )
        np.testing.assert_allclose(
            obs(spec, d2.qpos, d2.qvel, extra2),
            np.asarray(o, dtype=np.float64), rtol=1e-5, atol=1e-5,
        )
        # Registry dims match the mirrored obs (configs/envs.yaml gate).
        with open(Path(__file__).resolve().parents[1] / "configs" / "envs.yaml") as fh:
            registry = yaml.safe_load(fh)
        entry = registry[env_name]
        assert len(obs(spec, d2.qpos, d2.qvel, extra2)) == entry["obs_dim"]
        assert env.action_space.shape[0] == entry["action_dim"]
    finally:
        env.close()


def test_forced_unhealthy_terminates():
    def _break(q, v, idx, val):
        q = np.asarray(q, dtype=np.float64).copy()
        q[idx] = val
        return q, v

    cases = {
        "Hopper-v5": lambda q, v: _break(q, v, 1, 0.1),
        "Walker2d-v5": lambda q, v: _break(q, v, 1, 0.1),
        "Ant-v5": lambda q, v: _break(q, v, 2, 5.0),  # high: no contact pop-back
        "InvertedPendulum-v5": lambda q, v: _break(q, v, 1, 0.5),
    }
    for env_name, break_fn in cases.items():
        spec = get_spec(env_name)
        env = gym.make(env_name)
        try:
            env.reset(seed=0)
            d = env.unwrapped.data
            qpos, qvel = break_fn(np.asarray(d.qpos), np.asarray(d.qvel))
            env.unwrapped.set_state(qpos, qvel)
            o, _, term, _, _ = env.step(np.zeros(env.action_space.shape))
            kw = _health_kwargs(env, spec, o)
            d2 = env.unwrapped.data
            assert term is True
            assert terminated(spec, is_healthy(spec, d2.qpos, d2.qvel, **kw), **kw) is True
        finally:
            env.close()


def test_double_pendulum_site_rule_unit():
    spec = get_spec("InvertedDoublePendulum-v5")
    q = np.zeros(3)
    v = np.zeros(3)
    assert is_healthy(spec, q, v, site_y=1.5) is True
    assert is_healthy(spec, q, v, site_y=0.5) is False
    assert terminated(spec, False, site_y=0.5) is True


def test_reset_sampler_properties():
    rng = np.random.default_rng(0)
    r = reset_state(get_spec("Reacher-v5"), rng, np.zeros(4), np.zeros(4))
    assert np.linalg.norm(r[0][-2:]) < 0.2 and bool((r[1][-2:] == 0).all())
    p = reset_state(get_spec("Pusher-v5"), rng, np.zeros(11), np.zeros(11))
    assert np.linalg.norm(p[0][-4:-2] - p[0][-2:]) > 0.17
    assert bool((p[1][-4:] == 0).all())
    h = reset_state(get_spec("Hopper-v5"), rng, np.zeros(6), np.zeros(6))
    assert bool(np.all(np.abs(h[0]) <= 5e-3 + 1e-12))


def test_mjx_unsupported_raises_loudly():
    assert set(MJX_UNSUPPORTED) >= {
        "CartPole-v1", "Pendulum-v1", "MountainCar-v0", "Humanoid-v5",
        "AdroitHandDoor-v1", "AdroitHandPen-v1", "AdroitHandHammer-v1",
    }
    for env_name in MJX_UNSUPPORTED:
        with pytest.raises(RuntimeError, match="no MJX path"):
            get_spec(env_name)
    with pytest.raises(KeyError):
        get_spec("Nope-v5")


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
