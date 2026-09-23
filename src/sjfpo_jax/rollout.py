"""JAX-native MJX chunk rollout (no DLPack; physics and policy both JAX).

``MjxVec`` runs ``jit(vmap)`` physics for one env type (see ``envs.py`` for
the replicated gymnasium semantics); ``generate_batch`` executes the
open-loop chunk loop on-device and returns a ``compute_loss``-ready batch.
All heavy imports are lazy so this module imports anywhere; without MJX or
a GPU backend, construction raises with install hints.

Performance notes
-----------------
* ``MjxVec.step`` fuses obs/healthy/reward/termination into a single jitted
  ``_post_step`` call — one dispatch per step instead of ~5.
* Autoreset for uniform-kind envs goes through ``jax.lax.cond``, eliminating
  the per-step device→host sync of ``bool(jnp.any(done))``. Host-side
  reset kinds (reacher, pusher) keep the eager path.
* ``generate_batch`` reads ``terminated``/``truncated``/``rew`` as a single
  batched ``np.asarray`` per step (one sync) instead of per-env ``bool(...)``.
* Action scatter, window roll-update, normalizer calls, and critic calls are
  vectorized across the wavefront of active envs.
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
    """True when MJX imports and the backend is usable (lazy, never raises)."""
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


# ---------------------------------------------------------------------------
# MjxVec
# ---------------------------------------------------------------------------


class MjxVec:
    """Batched MJX env with autoreset; jnp in/out, all on device."""

    _JIT_RESET_KINDS = frozenset({"uniform_normal", "uniform_uniform"})

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
        # Fused post-step: obs + healthy + reward + termination in one dispatch.
        self._post_step_fn = jax.jit(self._post_step_impl)

        self._reset_jit_able = self.spec["reset_kind"] in self._JIT_RESET_KINDS
        self._rng = jax.random.PRNGKey(int(seed))
        self._ep_len = jnp.zeros((num_envs,), dtype=jnp.int32)
        self._max_ep_steps = self.spec["max_episode_steps"]
        self._terminate_when_unhealthy = bool(
            self.spec.get("terminate_when_unhealthy", False)
        )
        self.reset()

    # ------------------------------------------------------------------
    # Reset sampling
    # ------------------------------------------------------------------

    def _sample_state_jit(self, rng, n: int):
        """Jit-able reset for ``uniform_*`` kinds; raises for others."""
        kind = self.spec["reset_kind"]
        if kind not in self._JIT_RESET_KINDS:
            raise ValueError(f"reset_kind={kind!r} is not jit-able")
        s = self.spec.get("reset_noise_scale", 0.0)
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
        return qpos.astype(jnp.float32), qvel.astype(jnp.float32)

    def _sample_state(self, rng, n: int):
        """Eager reset for all kinds (used at construction and for host kinds)."""
        kind = self.spec["reset_kind"]
        s = self.spec.get("reset_noise_scale", 0.0)
        if kind in self._JIT_RESET_KINDS:
            return self._sample_state_jit(rng, n)
        if kind == "reacher":
            qpos = np.zeros((n, self.nq))
            qvel = np.zeros((n, self.nv))
            for i in range(n):
                rng, k1, k2, k3 = jax.random.split(rng, 4)
                qpos[i] = np.asarray(self._init_qpos) + np.asarray(
                    jax.random.uniform(k1, (self.nq,), minval=-0.1, maxval=0.1)
                )
                while True:
                    rng, kg = jax.random.split(rng)
                    goal = np.asarray(
                        jax.random.uniform(kg, (2,), minval=-0.2, maxval=0.2)
                    )
                    if np.linalg.norm(goal) < 0.2:
                        break
                qpos[i, -2:] = goal
                qvel[i] = np.asarray(self._init_qvel) + np.asarray(
                    jax.random.uniform(k2, (self.nv,), minval=-0.005, maxval=0.005)
                )
                qvel[i, -2:] = 0
            return jnp.asarray(qpos, dtype=jnp.float32), jnp.asarray(
                qvel, dtype=jnp.float32
            )
        if kind == "pusher":
            qpos = np.zeros((n, self.nq))
            qvel = np.zeros((n, self.nv))
            for i in range(n):
                rng, k1, k2 = jax.random.split(rng, 3)
                qpos[i] = np.asarray(self._init_qpos)
                goal = np.zeros(2)
                while True:
                    rng, kg1, kg2 = jax.random.split(rng, 3)
                    cyl = np.concatenate(
                        [
                            np.asarray(
                                jax.random.uniform(kg1, (1,), minval=-0.3, maxval=0)
                            ),
                            np.asarray(
                                jax.random.uniform(kg2, (1,), minval=-0.2, maxval=0.2)
                            ),
                        ]
                    )
                    if np.linalg.norm(cyl - goal) > 0.17:
                        break
                qpos[i, -4:-2] = cyl
                qpos[i, -2:] = goal
                qvel[i] = np.asarray(self._init_qvel) + np.asarray(
                    jax.random.uniform(k2, (self.nv,), minval=-0.005, maxval=0.005)
                )
                qvel[i, -4:] = 0
            return jnp.asarray(qpos, dtype=jnp.float32), jnp.asarray(
                qvel, dtype=jnp.float32
            )
        raise ValueError(f"reset_kind={kind!r} unknown")

    # ------------------------------------------------------------------
    # Observation / extras / health / reward (all pure functions of dx)
    # ------------------------------------------------------------------

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
            qpos = dx.qpos[:, self.spec["obs_drop"] :]
            qvel = dx.qvel
            if self.spec["qvel_clip"] is not None:
                qvel = jnp.clip(qvel, -self.spec["qvel_clip"], self.spec["qvel_clip"])
            return jnp.concatenate([qpos, qvel], axis=1)
        if kind == "full":
            return jnp.concatenate([dx.qpos, dx.qvel], axis=1)
        if kind == "ant":
            qpos = dx.qpos[:, self.spec["obs_drop"] :]
            lo, hi = self.spec["contact_force_range"]
            cfrc = jnp.clip(dx.cfrc_ext, lo, hi)[:, 1:].reshape(dx.qpos.shape[0], -1)
            return jnp.concatenate([qpos, dx.qvel, cfrc], axis=1)
        if kind == "reacher":
            theta = dx.qpos[:, :2]
            extra = self._extras_jax(dx)
            return jnp.concatenate(
                [
                    jnp.cos(theta),
                    jnp.sin(theta),
                    dx.qpos[:, 2:],
                    dx.qvel[:, :2],
                    (extra["fingertip"] - extra["target"])[:, :2],
                ],
                axis=1,
            )
        if kind == "double_pendulum":
            return jnp.concatenate(
                [
                    dx.qpos[:, :1],
                    jnp.sin(dx.qpos[:, 1:]),
                    jnp.cos(dx.qpos[:, 1:]),
                    jnp.clip(dx.qvel, -10, 10),
                    jnp.clip(dx.qfrc_constraint, -10, 10)[:, :1],
                ],
                axis=1,
            )
        if kind == "pusher":
            extra = self._extras_jax(dx)
            return jnp.concatenate(
                [
                    dx.qpos[:, :7],
                    dx.qvel[:, :7],
                    extra["tips_arm"],
                    extra["object"],
                    extra["goal"],
                ],
                axis=1,
            )
        raise ValueError(f"obs_kind={kind!r} unknown")

    def _healthy_jax(self, dx, obs):
        kind = self.spec["done_kind"]
        if kind == "never":
            return jnp.ones((dx.qpos.shape[0],), dtype=bool)
        if kind == "healthy":
            z, angle = dx.qpos[:, 1], dx.qpos[:, 2]
            ok = (self.spec["healthy_z_range"][0] < z) & (
                z < self.spec["healthy_z_range"][1]
            )
            ok = (
                ok
                & (self.spec["healthy_angle_range"][0] < angle)
                & (angle < self.spec["healthy_angle_range"][1])
            )
            if self.spec.get("healthy_state_range") is not None:
                s_lo, s_hi = self.spec["healthy_state_range"]
                state = jnp.concatenate([dx.qpos, dx.qvel], axis=1)[:, 2:]
                ok = ok & jnp.all((s_lo < state) & (state < s_hi), axis=1)
            return ok
        if kind == "ant":
            state = jnp.concatenate([dx.qpos, dx.qvel], axis=1)
            z_lo, z_hi = self.spec["healthy_z_range"]
            return (
                jnp.isfinite(state).all(axis=1)
                & (z_lo <= state[:, 2])
                & (state[:, 2] <= z_hi)
            )
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

    def _reward_jax(self, dx, ctrl, x_before, healthy):
        rk = self.spec["reward_kind"]
        if rk == "locomotion":
            x_after = self._forward_before(dx)
            fwd = self.spec["forward_weight"] * (x_after - x_before) / self.dt
            survive = jnp.where(healthy, self.spec["healthy_reward"], 0.0)
            ctrl_cost = self.spec["ctrl_weight"] * jnp.sum(jnp.square(ctrl), axis=1)
            if self.spec["contact_weight"]:
                lo, hi = self.spec["contact_force_range"]
                cfrc = jnp.clip(dx.cfrc_ext, lo, hi)
                contact = self.spec["contact_weight"] * jnp.sum(
                    jnp.square(cfrc), axis=(1, 2)
                )
            else:
                contact = jnp.zeros_like(fwd)
            return (fwd + survive - ctrl_cost - contact).astype(jnp.float32)
        if rk == "survive":
            return jnp.where(healthy, 1.0, 0.0).astype(jnp.float32)
        if rk == "double_pendulum":
            v1, v2 = dx.qvel[:, 1], dx.qvel[:, 2]
            site = dx.site_xpos[:, self.spec.get("site_index", 0)]
            dist = 0.01 * jnp.square(site[:, 0]) + jnp.square(site[:, 2] - 2)
            vel = 1e-3 * jnp.square(v1) + 5e-3 * jnp.square(v2)
            return (
                self.spec["healthy_reward"] * healthy.astype(jnp.float32) - dist - vel
            ).astype(jnp.float32)
        if rk == "reacher":
            extra = self._extras_jax(dx)
            vec = extra["fingertip"] - extra["target"]
            return (
                -jnp.linalg.norm(vec, axis=1) * self.spec["reward_dist_weight"]
                - jnp.sum(jnp.square(ctrl), axis=1) * self.spec["reward_control_weight"]
            ).astype(jnp.float32)
        if rk == "pusher":
            extra = self._extras_jax(dx)
            near = (
                -jnp.linalg.norm(extra["object"] - extra["tips_arm"], axis=1)
                * self.spec["reward_near_weight"]
            )
            dist = (
                -jnp.linalg.norm(extra["object"] - extra["goal"], axis=1)
                * self.spec["reward_dist_weight"]
            )
            ctrl_cost = (
                jnp.sum(jnp.square(ctrl), axis=1) * self.spec["reward_control_weight"]
            )
            return (dist - ctrl_cost + near).astype(jnp.float32)
        raise ValueError(f"reward_kind={rk!r} unknown")

    def _post_step_impl(self, dx, ctrl, x_before, ep_len):
        """Fused obs + healthy + reward + termination (one jit dispatch)."""
        obs = self._obs_jax(dx)
        healthy = self._healthy_jax(dx, obs)
        rew = self._reward_jax(dx, ctrl, x_before, healthy)
        terminated = jnp.asarray(~healthy & self._terminate_when_unhealthy)
        truncated = ep_len >= self._max_ep_steps
        return obs, rew, terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> jnp.ndarray:
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

    def _make_reset_branch(self, sub, done):
        """Closure-captured jit-able reset branch for ``lax.cond``."""

        def do_reset(state):
            dx, ep_len = state
            qpos, qvel = self._sample_state_jit(sub, self.num_envs)
            fresh = self._fwd_fn(self.mx, self._template.replace(qpos=qpos, qvel=qvel))
            take = done[:, None] if fresh.qpos.ndim == 2 else done
            new_dx = jax.tree.map(
                lambda new, old: (
                    jnp.where(take, new, old) if getattr(new, "ndim", 0) == 2 else old
                ),
                fresh,
                dx,
            )
            return new_dx, jnp.where(done, 0, ep_len)

        return do_reset

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, ctrl: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Step all envs; returns (obs, rew, terminated, truncated), autoreset."""
        ctrl = ctrl.astype(jnp.float32)
        x_before = self._forward_before(self._dx)
        self._dx = self._step_fn(self.mx, self._dx, ctrl)

        ep_next = self._ep_len + 1
        obs, rew, terminated, truncated = self._post_step_fn(
            self._dx, ctrl, x_before, ep_next
        )
        done = terminated | truncated

        if self._reset_jit_able:
            # Single jitted branch: no device→host sync.
            self._rng, sub = jax.random.split(self._rng)
            do_reset = self._make_reset_branch(sub, done)
            self._dx, self._ep_len = jax.lax.cond(
                jnp.any(done),
                do_reset,
                lambda s: s,
                (self._dx, ep_next),
            )
            obs = self._obs_jax(self._dx)
        else:
            # Host-side reset (reacher/pusher): eager fallback.
            self._ep_len = ep_next
            if bool(jnp.any(done)):
                self._rng, sub = jax.random.split(self._rng)
                qpos, qvel = self._sample_state(sub, self.num_envs)
                fresh = self._fwd_fn(
                    self.mx, self._template.replace(qpos=qpos, qvel=qvel)
                )
                take = done[:, None] if fresh.qpos.ndim == 2 else done
                self._dx = jax.tree.map(
                    lambda new, old: (
                        jnp.where(take, new, old)
                        if getattr(new, "ndim", 0) == 2
                        else old
                    ),
                    fresh,
                    self._dx,
                )
                self._ep_len = jnp.where(done, 0, self._ep_len)
                obs = self._obs_jax(self._dx)

        return obs, rew, terminated, truncated


# ---------------------------------------------------------------------------
# Rollout -> loss-ready batch
# ---------------------------------------------------------------------------


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
    """On-device chunk rollout + critic scoring + GAE -> loss-ready batch."""
    from sjfpo_jax import policy as jpol

    get_spec(env_name)  # KeyError on unsupported ids (works without MJX).
    vec = (vec_factory or MjxVec)(env_name, num_envs, seed)
    device_low = jnp.asarray(action_low, dtype=jnp.float32).reshape(-1)
    device_high = jnp.asarray(action_high, dtype=jnp.float32).reshape(-1)
    act_dim = int(device_low.shape[0])
    scale = device_high - device_low
    target = -(-total_steps // num_envs)

    first = vec.reset().astype(jnp.float32)
    windows = jnp.broadcast_to(
        first[:, None, :], (num_envs, n_obs_steps, first.shape[-1])
    )
    windows = jnp.array(windows)  # materialize once; .at[]-updates keep it dense

    executed_total = [0] * num_envs
    chunk_states: list = []
    chunk_actions: list = []
    chunk_next: list = []
    chunk_rews: list = []
    chunk_lens: list = []
    chunk_done: list = []
    chunk_trunc: list = []
    last_idx = [-1] * num_envs
    last_live = [False] * num_envs

    while any(e < target for e in executed_total):
        need = [i for i in range(num_envs) if executed_total[i] < target]
        n_need = len(need)
        need_arr = jnp.asarray(need, dtype=jnp.int32)
        states = windows[need_arr]  # advanced indexing already copies

        key, sub = jax.random.split(key)
        proposed = jpol.predict_action_chunk(
            policy,
            states,
            sub,
            action_horizon=action_horizon,
            top_p_spectral_tokens=top_p_spectral_tokens,
            num_euler_steps=num_euler_steps,
            noise=True,
        )
        executed = jnp.clip(proposed, device_low, device_high)
        scaled = 2.0 * (executed - device_low) / scale - 1.0  # [n_need, H, act]

        pos = [0] * n_need
        live = [True] * n_need
        end_term = [False] * n_need
        end_trunc = [False] * n_need
        rews_k: list[list] = [[] for _ in range(n_need)]  # device scalars

        for _ in range(action_horizon):
            active_k = [
                k
                for k in range(n_need)
                if live[k] and executed_total[need[k]] + pos[k] < target
            ]
            if not active_k:
                break

            active_arr = jnp.asarray(active_k, dtype=jnp.int32)
            active_envs = need_arr[active_arr]  # [n_active]
            active_pos = jnp.asarray([pos[k] for k in active_k], dtype=jnp.int32)

            # Single scatter for all active envs.
            acts = jnp.zeros((num_envs, act_dim), dtype=jnp.float32)
            acts = acts.at[active_envs].set(executed[active_arr, active_pos])

            obs, rew, term, trunc = vec.step(acts)

            # One device→host sync per step (batched).
            term_np = np.asarray(term)
            trunc_np = np.asarray(trunc)

            # Python-side bookkeeping; rews stay on device.
            for k in active_k:
                env_i = need[k]
                rews_k[k].append(rew[env_i])
                pos[k] += 1
                if term_np[env_i]:
                    end_term[k] = True
                    live[k] = False
                elif trunc_np[env_i]:
                    end_trunc[k] = True
                    live[k] = False

            # Vectorized window update for all active envs.
            aw = windows[active_envs]  # [n_act, T, D]
            rolled = jnp.concatenate([aw[:, 1:], obs[active_envs][:, None, :]], axis=1)
            fresh = jnp.broadcast_to(
                obs[active_envs][:, None, :],
                (len(active_k), n_obs_steps, obs.shape[-1]),
            )
            done_list = [bool(term_np[need[k]] or trunc_np[need[k]]) for k in active_k]
            done_mask = jnp.asarray(done_list)[:, None, None]
            windows = windows.at[active_envs].set(jnp.where(done_mask, fresh, rolled))

        # Close out this wavefront's chunks.
        for k, i in enumerate(need):
            n_exec = pos[k]
            if n_exec == 0:
                continue
            term_k = end_term[k]
            trunc_k = end_trunc[k] and not end_term[k]
            chunk_states.append(states[k])
            chunk_actions.append(scaled[k])
            chunk_next.append(windows[i])
            chunk_rews.append(
                rews_k[k][0][None] if n_exec == 1 else jnp.stack(rews_k[k])
            )
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

    # One normalizer pass for states + next_states (concatenate along batch).
    combined = jnp.concatenate([states_b, next_b], axis=0)
    combined_n = policy.normalizer(combined)
    states_n = combined_n[:n_chunks]
    next_n = combined_n[n_chunks:]

    # One critic pass for both.
    combined_v = policy.critic(combined_n)
    values = combined_v[:n_chunks]
    v_next_all = combined_v[n_chunks:]
    next_values = jnp.where(terminated_b, 0.0, v_next_all)

    advantages, returns = compute_advantages(
        all_rew,
        values,
        next_values,
        lengths,
        terminated_b,
        gamma=gamma,
        lam=gae_lambda,
        normalize=True,
        truncated=truncated_b,
    )

    key, k1, k2 = jax.random.split(key, 3)
    tau = (
        jax.random.uniform(k1, (num_mc_samples,)) * (max_timestep - min_timestep)
        + min_timestep
    )
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
