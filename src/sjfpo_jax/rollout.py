"""JAX-native MJX chunk rollout (no DLPack; physics and policy both JAX).

``MjxVec`` runs ``jit(vmap)`` physics for one env type (see ``envs.py`` for
the replicated gymnasium semantics); ``generate_batch`` executes the
open-loop chunk loop on-device and returns a ``compute_loss``-ready batch.
All heavy imports are lazy so this module imports anywhere; without MJX or
a GPU backend, construction raises with install hints.
"""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from sjfpo_jax.envs import get_spec
from sjfpo_jax.gae import compute_advantages, validate_stream

__all__ = ["rollout_available", "allow_cpu_rollout", "MjxVec", "generate_batch"]


def allow_cpu_rollout() -> bool:
    """Explicit CPU prototyping opt-in: ``SJFPO_JAX_CPU=1`` (slow)."""
    return os.environ.get("SJFPO_JAX_CPU") == "1"


def rollout_available() -> bool:
    """True when MJX imports and the backend is usable (lazy, never raises).

    GPU always; CPU only with ``SJFPO_JAX_CPU=1`` (prototyping, slow).
    """
    try:
        from mujoco import mjx  # noqa: F401
    except Exception:
        return False
    try:
        backend = jax.default_backend()
    except Exception:
        return False
    return backend == "gpu" or (backend == "cpu" and allow_cpu_rollout())


def _require_rollout():
    if rollout_available():
        if jax.default_backend() == "cpu":
            warnings.warn("SJFPO_JAX_CPU=1: MJX on CPU is for prototyping only")
        return
    if allow_cpu_rollout():
        raise RuntimeError(
            "SJFPO_JAX_CPU=1 needs mujoco.mjx installed (pip install -U mujoco)"
        )
    raise RuntimeError(
        "JAX MJX rollout needs a CUDA box with mujoco.mjx + JAX-GPU "
        '(pip install -U "mujoco>=3.3" "jax[cuda12]")'
    )


class MjxVec:
    """Batched MJX env with autoreset; jnp in/out, all on device."""

    def __init__(self, env_name: str, num_envs: int, seed: int):
        _require_rollout()
        import mujoco
        from mujoco import mjx

        import gymnasium as gym
        from pathlib import Path

        if num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be >= 1")
        from sjfpo_jax.envs import obs_dim as _obs_dim_fn

        self.spec = get_spec(env_name)
        xml = (
            Path(gym.__file__).resolve().parent
            / "envs"
            / "mujoco"
            / "assets"
            / self.spec["xml"]
        )
        mj_model = mujoco.MjModel.from_xml_path(str(xml))
        self.nq, self.nv, self.nu = mj_model.nq, mj_model.nv, mj_model.nu
        self.obs_dim = _obs_dim_fn(self.spec, mj_model.nq, mj_model.nv, mj_model.nbody)
        self.dt = float(mj_model.opt.timestep * self.spec["frame_skip"])
        # Named bodies for com/forward sensors (ant torso, reacher/pusher frames).
        want_bodies = list(self.spec.get("com_bodies", []))
        if self.spec.get("forward_source") == "body_x":
            want_bodies.append(self.spec["main_body"])
        self._body_ids = {
            name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in want_bodies
        }
        if mj_model.nkey > 0:
            self._init_qpos = jnp.asarray(mj_model.key_qpos[0], dtype=jnp.float32)
            self._init_qvel = jnp.asarray(mj_model.key_qvel[0], dtype=jnp.float32)
        else:
            # MjModel has qpos0 but no qvel0 (mirrors gymnasium data-at-load).
            self._init_qpos = jnp.asarray(mj_model.qpos0, dtype=jnp.float32)
            self._init_qvel = jnp.zeros((mj_model.nv,), dtype=jnp.float32)

        self.mx = mjx.put_model(mj_model)
        base = mjx.put_data(mj_model, mujoco.MjData(mj_model))

        def _tile(x):
            if not hasattr(x, "ndim"):
                return x
            if x.ndim == 0:
                return jnp.broadcast_to(x, (num_envs,))
            return jnp.tile(x, (num_envs,) + (1,) * x.ndim)

        self.num_envs = num_envs
        self._template = jax.tree.map(_tile, base)
        fs = self.spec["frame_skip"]

        def _physics(mx, dx, ctrl):
            dx = dx.replace(ctrl=ctrl)
            return jax.lax.fori_loop(0, fs, lambda _, d: mjx.step(mx, d), dx)

        self._step_fn = jax.jit(jax.vmap(_physics, in_axes=(None, 0, 0)))
        self._fwd_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self._rng = jax.random.PRNGKey(int(seed))
        self._ep_len = jnp.zeros((num_envs,), dtype=jnp.int32)
        self.reset()

    def _sample_state(self, rng, n: int):
        import jax

        kind = self.spec["reset_kind"]
        s = self.spec.get("reset_noise_scale", 0.0)
        if kind in ("uniform_normal", "uniform_uniform"):
            rng, k1, k2 = jax.random.split(rng, 3)
            qpos = self._init_qpos + jax.random.uniform(
                k1, (n, self.nq), minval=-s, maxval=s
            )
            if self.spec.get("qvel_dist", "uniform") == "normal":
                qvel = self._init_qvel + s * jax.random.normal(k2, (n, self.nv))
            else:
                qvel = self._init_qvel + jax.random.uniform(
                    k2, (n, self.nv), minval=-s, maxval=s
                )
        elif kind == "reacher":
            # Goal resampling is host-side (reset is eager, never jitted).
            qpos = np.zeros((n, self.nq))
            qvel = np.zeros((n, self.nv))
            for i in range(n):
                rng, k1, k2, k3 = jax.random.split(rng, 4)
                qpos[i] = np.asarray(self._init_qpos) + np.asarray(
                    jax.random.uniform(k1, (self.nq,), minval=-0.1, maxval=0.1)
                )
                while True:
                    rng, kg = jax.random.split(rng)
                    goal = np.asarray(jax.random.uniform(kg, (2,), minval=-0.2, maxval=0.2))
                    if np.linalg.norm(goal) < 0.2:
                        break
                qpos[i, -2:] = goal
                qvel[i] = np.asarray(self._init_qvel) + np.asarray(
                    jax.random.uniform(k2, (self.nv,), minval=-0.005, maxval=0.005)
                )
                qvel[i, -2:] = 0
            qpos = jnp.asarray(qpos, dtype=jnp.float32)
            qvel = jnp.asarray(qvel, dtype=jnp.float32)
        elif kind == "pusher":
            qpos = np.zeros((n, self.nq))
            qvel = np.zeros((n, self.nv))
            for i in range(n):
                rng, k1, k2 = jax.random.split(rng, 3)
                qpos[i] = np.asarray(self._init_qpos)
                goal = np.zeros(2)
                while True:
                    rng, kg1, kg2 = jax.random.split(rng, 3)
                    cyl = np.concatenate([
                        np.asarray(jax.random.uniform(kg1, (1,), minval=-0.3, maxval=0)),
                        np.asarray(jax.random.uniform(kg2, (1,), minval=-0.2, maxval=0.2)),
                    ])
                    if np.linalg.norm(cyl - goal) > 0.17:
                        break
                qpos[i, -4:-2] = cyl
                qpos[i, -2:] = goal
                qvel[i] = np.asarray(self._init_qvel) + np.asarray(
                    jax.random.uniform(k2, (self.nv,), minval=-0.005, maxval=0.005)
                )
                qvel[i, -4:] = 0
            qpos = jnp.asarray(qpos, dtype=jnp.float32)
            qvel = jnp.asarray(qvel, dtype=jnp.float32)
        else:
            raise ValueError(f"reset_kind={kind!r} unknown")
        return qpos.astype(jnp.float32), qvel.astype(jnp.float32)

    def _extras_jax(self, dx) -> dict:
        extra = {}
        if self.spec["obs_kind"] == "ant" or self.spec.get("contact_weight"):
            lo, hi = self.spec["contact_force_range"]
            extra["cfrc_ext"] = jnp.clip(dx.cfrc_ext, lo, hi)
        if self.spec["obs_kind"] == "double_pendulum":
            extra["qfrc_constraint"] = dx.qfrc_constraint
        for name in self.spec.get("com_bodies", []):
            extra[name] = dx.xpos[:, self._body_ids[name]]
        return extra

    def _obs_jax(self, dx):
        kind = self.spec["obs_kind"]
        if kind == "qpos_qvel":
            qpos = dx.qpos[:, self.spec["obs_drop"]:]
            qvel = dx.qvel
            if self.spec["qvel_clip"] is not None:
                qvel = jnp.clip(qvel, -self.spec["qvel_clip"], self.spec["qvel_clip"])
            return jnp.concatenate([qpos, qvel], axis=1)
        if kind == "full":
            return jnp.concatenate([dx.qpos, dx.qvel], axis=1)
        if kind == "ant":
            qpos = dx.qpos[:, self.spec["obs_drop"]:]
            lo, hi = self.spec["contact_force_range"]
            cfrc = jnp.clip(dx.cfrc_ext, lo, hi)[:, 1:].reshape(dx.qpos.shape[0], -1)
            return jnp.concatenate([qpos, dx.qvel, cfrc], axis=1)
        if kind == "reacher":
            theta = dx.qpos[:, :2]
            extra = self._extras_jax(dx)
            return jnp.concatenate([
                jnp.cos(theta), jnp.sin(theta), dx.qpos[:, 2:], dx.qvel[:, :2],
                (extra["fingertip"] - extra["target"])[:, :2],
            ], axis=1)
        if kind == "double_pendulum":
            return jnp.concatenate([
                dx.qpos[:, :1], jnp.sin(dx.qpos[:, 1:]), jnp.cos(dx.qpos[:, 1:]),
                jnp.clip(dx.qvel, -10, 10),
                jnp.clip(dx.qfrc_constraint, -10, 10)[:, :1],
            ], axis=1)
        if kind == "pusher":
            extra = self._extras_jax(dx)
            return jnp.concatenate([
                dx.qpos[:, :7], dx.qvel[:, :7],
                extra["tips_arm"], extra["object"], extra["goal"],
            ], axis=1)
        raise ValueError(f"obs_kind={kind!r} unknown")

    def _healthy_jax(self, dx, obs):
        kind = self.spec["done_kind"]
        if kind == "never":
            return jnp.ones((dx.qpos.shape[0],), dtype=bool)
        if kind == "healthy":
            z, angle = dx.qpos[:, 1], dx.qpos[:, 2]
            ok = (self.spec["healthy_z_range"][0] < z) & (z < self.spec["healthy_z_range"][1])
            ok = ok & (self.spec["healthy_angle_range"][0] < angle) & (angle < self.spec["healthy_angle_range"][1])
            if self.spec.get("healthy_state_range") is not None:
                s_lo, s_hi = self.spec["healthy_state_range"]
                state = jnp.concatenate([dx.qpos, dx.qvel], axis=1)[:, 2:]
                ok = ok & jnp.all((s_lo < state) & (state < s_hi), axis=1)
            return ok
        if kind == "ant":
            state = jnp.concatenate([dx.qpos, dx.qvel], axis=1)
            z_lo, z_hi = self.spec["healthy_z_range"]
            return jnp.isfinite(state).all(axis=1) & (z_lo <= state[:, 2]) & (state[:, 2] <= z_hi)
        if kind == "pole":
            return jnp.isfinite(obs).all(axis=1) & (jnp.abs(obs[:, 1]) <= 0.2)
        if kind == "site_y":
            return dx.site_xpos[:, self.spec.get("site_index", 0), 2] > 1
        raise ValueError(f"done_kind={kind!r} unknown")

    def _forward_before(self, dx):
        src = self.spec.get("forward_source", "none")
        if src == "qpos_x":
            return dx.qpos[:, 0]
        if src == "body_x":
            return dx.xpos[:, self._body_ids[self.spec["main_body"]], 0]
        return jnp.zeros((dx.qpos.shape[0],))

    def reset(self) -> jnp.ndarray:
        import jax

        self._rng, sub = jax.random.split(self._rng)
        qpos, qvel = self._sample_state(sub, self.num_envs)
        dx = self._template.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((self.num_envs, self.nu), dtype=jnp.float32),
        )
        self._dx = self._fwd_fn(self.mx, dx)
        self._ep_len = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        return self._obs_jax(self._dx)

    def step(
        self, ctrl: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Step all envs; returns (obs, rew, terminated, truncated), autoreset."""
        import jax

        ctrl = ctrl.astype(jnp.float32)
        x_before = self._forward_before(self._dx)
        self._dx = self._step_fn(self.mx, self._dx, ctrl)
        obs = self._obs_jax(self._dx)
        healthy = self._healthy_jax(self._dx, obs)
        rk = self.spec["reward_kind"]
        if rk == "locomotion":
            x_after = self._forward_before(self._dx)
            fwd = self.spec["forward_weight"] * (x_after - x_before) / self.dt
            survive = jnp.where(healthy, self.spec["healthy_reward"], 0.0)
            ctrl_cost = self.spec["ctrl_weight"] * jnp.sum(ctrl * ctrl, axis=1)
            contact = jnp.zeros_like(fwd)
            if self.spec["contact_weight"]:
                lo, hi = self.spec["contact_force_range"]
                cfrc = jnp.clip(self._dx.cfrc_ext, lo, hi)
                contact = self.spec["contact_weight"] * jnp.sum(cfrc * cfrc, axis=(1, 2))
            rew = (fwd + survive - ctrl_cost - contact).astype(jnp.float32)
        elif rk == "survive":
            rew = jnp.where(healthy, 1.0, 0.0).astype(jnp.float32)
        elif rk == "double_pendulum":
            v1, v2 = self._dx.qvel[:, 1], self._dx.qvel[:, 2]
            site = self._dx.site_xpos[:, self.spec.get("site_index", 0)]
            dist = 0.01 * site[:, 0] ** 2 + (site[:, 2] - 2) ** 2
            vel = 1e-3 * v1**2 + 5e-3 * v2**2
            rew = (self.spec["healthy_reward"] * healthy.astype(jnp.float32) - dist - vel).astype(jnp.float32)
        elif rk == "reacher":
            extra = self._extras_jax(self._dx)
            vec = extra["fingertip"] - extra["target"]
            rew = (-jnp.linalg.norm(vec, axis=1) * self.spec["reward_dist_weight"]
                   - jnp.sum(ctrl * ctrl, axis=1) * self.spec["reward_control_weight"]).astype(jnp.float32)
        elif rk == "pusher":
            extra = self._extras_jax(self._dx)
            near = -jnp.linalg.norm(extra["object"] - extra["tips_arm"], axis=1) * self.spec["reward_near_weight"]
            dist = -jnp.linalg.norm(extra["object"] - extra["goal"], axis=1) * self.spec["reward_dist_weight"]
            ctrl_cost = jnp.sum(ctrl * ctrl, axis=1) * self.spec["reward_control_weight"]
            rew = (dist - ctrl_cost + near).astype(jnp.float32)
        else:
            raise ValueError(f"reward_kind={rk!r} unknown")
        terminated = (~healthy) & self.spec.get("terminate_when_unhealthy", False)
        terminated = jnp.asarray(terminated)
        self._ep_len = self._ep_len + 1
        truncated = self._ep_len >= self.spec["max_episode_steps"]
        done = terminated | truncated
        if bool(jnp.any(done)):
            self._rng, sub = jax.random.split(self._rng)
            qpos, qvel = self._sample_state(sub, self.num_envs)
            fresh = self._fwd_fn(self.mx, self._template.replace(qpos=qpos, qvel=qvel))
            take = done[:, None] if fresh.qpos.ndim == 2 else done
            self._dx = jax.tree.map(
                lambda new, old: jnp.where(take, new, old)
                if getattr(new, "ndim", 0) == 2
                else old,
                fresh,
                self._dx,
            )
            self._ep_len = jnp.where(done, 0, self._ep_len)
            obs = self._obs_jax(self._dx)
        return obs, rew, terminated, truncated


def generate_batch(
    policy,
    env_name: str,
    total_steps: int,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    key: jax.Array,
    *,
    n_obs_steps: int = 16,
    action_horizon: int = 16,
    num_envs: int = 2,
    seed: int = 0,
    num_mc_samples: int = 16,
    top_p_spectral_tokens: float = 1.0,
    num_euler_steps: int = 16,
    gamma: float = 0.999,
    gae_lambda: float = 0.995,
    min_timestep: float = 0.005,
    max_timestep: float = 0.995,
    vec_factory=None,
) -> dict[str, jnp.ndarray]:
    """On-device chunk rollout + critic scoring + GAE -> loss-ready batch.

    ``vec_factory`` (default :class:`MjxVec`) builds ``(env_name, num_envs,
    seed)``; tests inject a stub. Raises without MJX/GPU when defaulted.
    """
    from sjfpo_jax import policy as jpol

    get_spec(env_name)  # KeyError on unsupported ids (works without MJX).
    vec = (vec_factory or MjxVec)(env_name, num_envs, seed)
    device_low = jnp.asarray(action_low, dtype=jnp.float32).reshape(-1)
    device_high = jnp.asarray(action_high, dtype=jnp.float32).reshape(-1)
    act_dim = int(device_low.shape[0])
    target = -(-total_steps // num_envs)
    first = vec.reset().astype(jnp.float32)
    windows = jnp.broadcast_to(first[:, None, :], (num_envs, n_obs_steps, first.shape[-1])).copy()
    executed_total = [0] * num_envs
    chunk_states, chunk_actions, chunk_next, chunk_rews, chunk_lens = [], [], [], [], []
    chunk_done, chunk_trunc = [], []
    last_idx = [-1] * num_envs
    last_live = [False] * num_envs

    while any(e < target for e in executed_total):
        need = [i for i in range(num_envs) if executed_total[i] < target]
        states = windows[jnp.asarray(need)].copy()
        key, sub = jax.random.split(key)
        proposed = jpol.predict_action_chunk(
            policy, states, sub, action_horizon=action_horizon,
            top_p_spectral_tokens=top_p_spectral_tokens,
            num_euler_steps=num_euler_steps, noise=True,
        )
        executed = jnp.clip(proposed, device_low, device_high)
        scaled = 2.0 * (executed - device_low) / (device_high - device_low) - 1.0
        pos = [0] * len(need)
        rews: list[list] = [[] for _ in need]
        live = [True] * len(need)
        end_term = [False] * len(need)
        end_trunc = [False] * len(need)
        for _ in range(action_horizon):
            active = [
                k for k in range(len(need))
                if live[k] and executed_total[need[k]] + pos[k] < target
            ]
            if not active:
                break
            acts = jnp.zeros((num_envs, act_dim), dtype=jnp.float32)
            for k in active:
                acts = acts.at[need[k]].set(executed[k, pos[k]])
            obs, rew, term, trunc = vec.step(acts)
            for k in active:
                rews[k].append(rew[k])
                pos[k] += 1
                if bool(term[k]):
                    end_term[k] = True
                    live[k] = False
                elif bool(trunc[k]):
                    end_trunc[k] = True
                    live[k] = False
            for k in range(len(need)):
                if not live[k] and (end_term[k] or end_trunc[k]):
                    # Autoreset obs returned: rebuild warmup (CPU parity).
                    windows = windows.at[need[k]].set(
                        jnp.broadcast_to(obs[need[k]][None, :], (n_obs_steps, obs.shape[-1]))
                    )
                elif live[k] or k in active:
                    w = windows[need[k]]
                    w = jnp.roll(w, shift=-1, axis=0)
                    w = w.at[-1].set(obs[need[k]])
                    windows = windows.at[need[k]].set(w)
            if not any(live[k] and executed_total[need[k]] + pos[k] < target for k in range(len(need))):
                break
        for k, i in enumerate(need):
            n_exec = pos[k]
            if n_exec == 0:
                continue
            # Terminated wins; truncation closes the chunk too (CPU parity).
            term_k, trunc_k = end_term[k], end_trunc[k] and not end_term[k]
            chunk_states.append(states[k])
            chunk_actions.append(scaled[k])
            chunk_next.append(windows[i].copy())
            chunk_rews.append(jnp.stack(rews[k][:n_exec]))
            chunk_lens.append(n_exec)
            chunk_done.append(term_k)
            chunk_trunc.append(trunc_k)
            last_idx[i] = len(chunk_states) - 1
            last_live[i] = not (term_k or trunc_k)
            executed_total[i] += n_exec
    # Budget exhausted mid-episode: bootstrap the cut (CPU-vec parity).
    for i in range(num_envs):
        if last_live[i] and last_idx[i] >= 0:
            chunk_trunc[last_idx[i]] = True
    n_chunks = len(chunk_states)
    states_b = jnp.stack(chunk_states)
    next_b = jnp.stack(chunk_next)
    lengths = jnp.asarray(chunk_lens, dtype=jnp.int32)
    terminated_b = jnp.asarray(chunk_done)
    truncated_b = jnp.asarray(chunk_trunc)
    all_rew = jnp.concatenate(chunk_rews)
    validate_stream(all_rew, jnp.zeros((n_chunks,)), lengths)
    states_n = policy.normalizer(states_b)
    next_n = policy.normalizer(next_b)
    values = policy.critic(states_n)
    v_next_all = policy.critic(next_n)
    next_values = jnp.where(terminated_b, 0.0, v_next_all)
    advantages, returns = compute_advantages(
        all_rew, values, next_values, lengths, terminated_b,
        gamma=gamma, lam=gae_lambda, normalize=True, truncated=truncated_b,
    )
    key, k1, k2 = jax.random.split(key, 3)
    tau = jax.random.uniform(k1, (num_mc_samples,)) * (max_timestep - min_timestep) + min_timestep
    token_dim = max(1, round(top_p_spectral_tokens * action_horizon))
    eps = jax.random.normal(k2, (n_chunks, num_mc_samples, token_dim, act_dim))
    is_pad = jnp.arange(action_horizon)[None, :] >= lengths[:, None]
    return {
        "observation.state": states_b.astype(jnp.float32),
        "action": jnp.stack(chunk_actions).astype(jnp.float32),
        "action_is_pad": is_pad,
        "advantages": advantages.astype(jnp.float32),
        "returns": returns.astype(jnp.float32),
        "values": values.astype(jnp.float32),
        "tau": jnp.broadcast_to(tau[None, :], (n_chunks, num_mc_samples)),
        "eps": eps.astype(jnp.float32),
    }
