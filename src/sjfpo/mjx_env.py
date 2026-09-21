"""GPU-batched MJX rollout env (Blackwell training box only).

Wraps the ENTIRE chunk rollout on GPU: ``jit(vmap(mjx.step))`` physics (the
canonical pattern from google-deepmind/mujoco + vmoens/mujoco-torch), the
torch DiT policy on the same CUDA device, zero-copy exchange via DLPack.
Fills the preallocated :class:`sjfpo.rollout_storage.RolloutStorage` on
device, then converts to legacy ``Transition`` dicts so ``generate_rollout``
downstream is unchanged.

HalfCheetah-v5 semantics replicated from gymnasium (see installed
``half_cheetah_v5.py``): obs ``qpos[1:]+qvel`` (17,), ``reward =
forward - ctrl`` with weights (1.0, 0.1), ``frame_skip=5``, reset noise
``U(+-0.1)``/``0.1*N(0,1)``, never terminates, truncate at 1000
(TimeLimit equivalent).

Opt-in only: ``SJFPO_MJX=1`` + CUDA + ``mujoco.mjx``/JAX-GPU. Import-time
safe on CPU-only machines (all heavy imports are lazy). ThinkPad setup:
``pip install -U "mujoco>=3.3" "jax[cuda12]"``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from sjfpo.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()  # cached once

__all__ = [
    "mjx_available",
    "want_mjx",
    "resolve_env_name",
    "halfcheetah_obs",
    "halfcheetah_reward",
    "shift_window",
    "mjx_collect_chunks",
]

# HalfCheetah-v5 defaults for the numpy mirrors below (physics now reads
# per-env specs; see sjfpo_jax.envs).
_FORWARD_WEIGHT = 1.0
_CTRL_WEIGHT = 0.1


def mjx_available() -> bool:
    """True when MJX + JAX-GPU are importable (lazy, never raises)."""
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except Exception:
        return False
    try:
        import jax

        return jax.default_backend() == "gpu"
    except Exception:
        return False


def want_mjx() -> bool:
    """Explicit opt-in: ``SJFPO_MJX=1`` (default off; CPU path unchanged)."""
    return os.environ.get("SJFPO_MJX") == "1"


def resolve_env_name(env) -> str:
    """Env id from single or vector envs (``SyncVectorEnv.spec`` is None).

    Vector envs expose per-sub-env specs via ``get_attr('spec')``; single
    envs via ``unwrapped.spec``. Returns ``'unknown'`` when unresolvable.
    """
    get_attr = getattr(env, "get_attr", None)
    if callable(get_attr):
        try:
            specs = get_attr("spec")
            if specs and getattr(specs[0], "id", None):
                return specs[0].id
        except Exception:
            pass
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    env_id = getattr(spec, "id", None) if spec is not None else None
    return env_id or "unknown"


def halfcheetah_obs(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """``qpos[1:]+qvel`` observation (numpy mirror, CPU-testable)."""
    return np.concatenate([np.asarray(qpos).ravel()[1:], np.asarray(qvel).ravel()])


def halfcheetah_reward(
    x_before: float, x_after: float, dt: float, action: np.ndarray
) -> float:
    """``forward - ctrl`` reward (numpy mirror, CPU-testable)."""
    forward = _FORWARD_WEIGHT * (x_after - x_before) / dt
    ctrl = _CTRL_WEIGHT * float(np.sum(np.square(action)))
    return float(forward - ctrl)


def shift_window(windows: torch.Tensor, idx: int, new_obs: torch.Tensor) -> None:
    """Roll one env's ``[T, obs]`` window forward, appending ``new_obs``.

    ``torch.roll`` returns a fresh tensor: shifting the ``windows[idx]``
    view in place faults on overlapping memory ("some elements ... refer
    to a single memory location").
    """
    w = torch.roll(windows[idx], shifts=-1, dims=0)
    w[-1].copy_(new_obs.to(dtype=w.dtype))
    windows[idx] = w


class MjxVec:
    """Batched MJX env on GPU; torch-CUDA in/out via DLPack (zero-copy).

    Spec-driven over every ``sjfpo_jax.envs`` entry (same formulas as the
    JAX ``MjxVec``; only the I/O is torch). ``reset() -> [M, obs]``;
    ``step(ctrl) -> (obs, rew, terminated, truncated)`` with autoreset.
    """

    def __init__(self, env_name: str, num_envs: int, seed: int, device: torch.device):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        import gymnasium as gym

        from sjfpo_jax.envs import get_spec
        from sjfpo_jax.envs import obs_dim as _obs_dim_fn

        if num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be >= 1")
        self.spec = get_spec(env_name)  # KeyError/RuntimeError on unsupported
        xml = (
            __import__("pathlib").Path(gym.__file__).resolve().parent
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
        # Mirror gymnasium MujocoEnv: init from data state at load
        # (= qpos0 / zeros); MjModel has no qvel0 attribute.
        if mj_model.nkey > 0:
            init_qpos = np.asarray(mj_model.key_qpos[0])
            init_qvel = np.asarray(mj_model.key_qvel[0])
        else:
            init_qpos = np.asarray(mj_model.qpos0)
            init_qvel = np.zeros(mj_model.nv)
        self._init_qpos = jnp.asarray(init_qpos, dtype=jnp.float32)
        self._init_qvel = jnp.asarray(init_qvel, dtype=jnp.float32)

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

        frame_skip = self.spec["frame_skip"]

        def _physics(mx, dx, ctrl):
            dx = dx.replace(ctrl=ctrl)
            return jax.lax.fori_loop(0, frame_skip, lambda _, d: mjx.step(mx, d), dx)

        self._step_fn = jax.jit(jax.vmap(_physics, in_axes=(None, 0, 0)))
        self._fwd_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self._rng = jax.random.PRNGKey(int(seed))
        self._ep_len = jnp.zeros((num_envs,), dtype=jnp.int32)
        self._device = torch.device(device)
        self.reset()

    def _torch(self, x):
        """JAX-GPU -> torch-CUDA, zero-copy via DLPack.

        Per the JAX library code, ``jax.dlpack`` exposes only ``from_dlpack``
        (import); export uses the ``__dlpack__`` protocol on ``jax.Array``,
        which torch consumes directly.
        """
        import jax
        import torch.utils.dlpack as dlpack

        jax.block_until_ready(x)
        return dlpack.from_dlpack(x)

    def _jax(self, t: torch.Tensor):
        """torch-CUDA -> JAX-GPU, zero-copy via DLPack.

        ``jax.dlpack.from_dlpack`` takes the tensor itself (it exposes
        ``__dlpack__``); passing a ``to_dlpack()`` capsule fails. Mirror
        image of the earlier ``to_dlpack`` export bug.
        """
        import jax

        t = t.to(device="cuda", dtype=torch.float32).contiguous()
        return jax.dlpack.from_dlpack(t)

    def _sample_state(self, rng):
        """Initial (qpos, qvel) [M, ...] with per-env noise/goal sampling."""
        import jax
        import jax.numpy as jnp

        n = self.num_envs
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
        elif kind in ("reacher", "pusher"):
            # Goal resampling is host-side (reset is eager, never jitted).
            rng_np = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31 - 1)))
            from sjfpo_jax.envs import reset_state as _reset_state

            qpos = np.zeros((n, self.nq))
            qvel = np.zeros((n, self.nv))
            for i in range(n):
                qp, qv = _reset_state(
                    self.spec, rng_np, np.asarray(self._init_qpos), np.asarray(self._init_qvel)
                )
                qpos[i], qvel[i] = qp, qv
            rng, _ = jax.random.split(rng)
            qpos = jnp.asarray(qpos, dtype=jnp.float32)
            qvel = jnp.asarray(qvel, dtype=jnp.float32)
        else:
            raise ValueError(f"reset_kind={kind!r} unknown")
        return qpos.astype(jnp.float32), qvel.astype(jnp.float32)

    def _extras_jax(self, dx) -> dict:
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

        src = self.spec.get("forward_source", "none")
        if src == "qpos_x":
            return dx.qpos[:, 0]
        if src == "body_x":
            return dx.xpos[:, self._body_ids[self.spec["main_body"]], 0]
        return jnp.zeros((dx.qpos.shape[0],))

    def reset(self) -> torch.Tensor:
        import jax

        self._rng, sub = jax.random.split(self._rng)
        qpos, qvel = self._sample_state(sub)
        dx = self._template.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jax.numpy.zeros((self.num_envs, self.nu), dtype=jax.numpy.float32),
        )
        self._dx = self._fwd_fn(self.mx, dx)
        self._ep_len = jax.numpy.zeros((self.num_envs,), dtype=jax.numpy.int32)
        return self._torch(self._obs_jax(self._dx)).to(dtype=torch.float32)

    def step(
        self, ctrl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all envs; returns (obs, rew, terminated, truncated), autoreset."""
        import jax
        import jax.numpy as jnp

        ctrl_j = self._jax(ctrl)
        x_before = self._forward_before(self._dx)
        self._dx = self._step_fn(self.mx, self._dx, ctrl_j)
        obs = self._obs_jax(self._dx)
        healthy = self._healthy_jax(self._dx, obs)
        rk = self.spec["reward_kind"]
        if rk == "locomotion":
            x_after = self._forward_before(self._dx)
            fwd = self.spec["forward_weight"] * (x_after - x_before) / self.dt
            survive = jnp.where(healthy, self.spec["healthy_reward"], 0.0)
            ctrl_cost = self.spec["ctrl_weight"] * jnp.sum(ctrl_j * ctrl_j, axis=1)
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
                   - jnp.sum(ctrl_j * ctrl_j, axis=1) * self.spec["reward_control_weight"]).astype(jnp.float32)
        elif rk == "pusher":
            extra = self._extras_jax(self._dx)
            near = -jnp.linalg.norm(extra["object"] - extra["tips_arm"], axis=1) * self.spec["reward_near_weight"]
            dist = -jnp.linalg.norm(extra["object"] - extra["goal"], axis=1) * self.spec["reward_dist_weight"]
            ctrl_cost = jnp.sum(ctrl_j * ctrl_j, axis=1) * self.spec["reward_control_weight"]
            rew = (dist - ctrl_cost + near).astype(jnp.float32)
        else:
            raise ValueError(f"reward_kind={rk!r} unknown")
        terminated = (~healthy) & self.spec.get("terminate_when_unhealthy", False)
        terminated = jnp.asarray(terminated)
        self._ep_len = self._ep_len + 1
        truncated = self._ep_len >= self.spec["max_episode_steps"]
        done = terminated | truncated
        # Autoreset done slots (fresh obs returned, flags kept).
        if bool(jnp.any(done)):
            self._rng, sub = jax.random.split(self._rng)
            qpos, qvel = self._sample_state(sub)
            fresh = self._fwd_fn(
                self.mx, self._template.replace(qpos=qpos, qvel=qvel)
            )
            take = done[:, None]
            self._dx = jax.tree.map(
                lambda new, old: jnp.where(take, new, old)
                if new.ndim == 2
                else old,
                fresh,
                self._dx,
            )
            self._ep_len = jnp.where(done, 0, self._ep_len)
            obs = self._obs_jax(self._dx)
        return (
            self._torch(obs).to(dtype=torch.float32),
            self._torch(rew).to(dtype=torch.float32),
            self._torch(terminated),
            self._torch(truncated),
        )


def mjx_collect_chunks(
    policy,
    env_name: str,
    total_steps: int,
    action_low,
    action_high,
    *,
    n_obs_steps: int,
    n_action_steps: int,
    num_envs: int,
    seed: int,
) -> list[dict]:
    """Entire chunk rollout on GPU; returns legacy ``Transition`` dicts.

    One batched policy call per chunk + ``H`` jitted physics steps, all on
    CUDA; rewards/windows/flags never touch the host until the final
    ``to_transition_list`` copy (~100KB). Raises loudly when MJX/CUDA is
    missing (explicit opt-in must never silently fall back).
    """
    import torch

    from sjfpo.rollout_storage import RolloutStorage

    from sjfpo_jax.envs import get_spec as _get_spec

    _get_spec(env_name)  # KeyError/RuntimeError on unsupported (no silent CPU).
    if not mjx_available():
        raise RuntimeError(
            "SJFPO_MJX=1 needs a CUDA box with mujoco.mjx + JAX-GPU "
            '(ThinkPad: pip install -U "mujoco>=3.3" "jax[cuda12]")'
        )
    device = torch.device("cuda")
    low = torch.as_tensor(action_low, device=device, dtype=torch.float32).reshape(-1)
    high = torch.as_tensor(action_high, device=device, dtype=torch.float32).reshape(-1)
    if bool(((high - low) <= 0).any()):
        raise ValueError("action bounds must satisfy high > low in every dim")
    act_dim = int(low.numel())
    vec = MjxVec(env_name, num_envs, seed, device)
    obs_dim = vec.obs_dim
    target = -(-total_steps // num_envs)  # ceil per-env budget
    max_chunks = target // min(n_action_steps, 1) + 2
    storage = RolloutStorage(
        num_envs, max_chunks, n_obs_steps, n_action_steps, obs_dim, act_dim,
        device=device, dtype=STORAGE_DTYPE,
    )
    from lerobot.utils.constants import OBS_STATE

    first = vec.reset().to(dtype=STORAGE_DTYPE)  # [M, obs]
    windows = first.unsqueeze(1).expand(num_envs, n_obs_steps, -1).clone()
    executed_total = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_idx = [-1] * num_envs  # last chunk slot per env (budget-tail fixup)
    prev_live = [False] * num_envs
    with torch.inference_mode():
        while bool((executed_total < target).any()):
            need = [i for i in range(num_envs) if executed_total[i] < target]
            states = windows[need].clone()
            proposed = policy.predict_action_chunk({OBS_STATE: states}, noise=True)
            executed = torch.clamp(proposed, min=low, max=high)
            scaled = 2.0 * (executed - low) / (high - low) - 1.0
            clip = ((proposed < low) | (proposed > high)).sum(dim=(1, 2))
            # Open-loop execution, all on GPU (CPU-vec parity: every env
            # steps each round; live envs record, truncated rebuild warmup).
            step_idx = torch.zeros(len(need), dtype=torch.long, device=device)
            rews: list = [[] for _ in need]
            dones = [False] * len(need)
            end_term = [False] * len(need)
            end_trunc = [False] * len(need)
            for _ in range(n_action_steps):
                live = [
                    k for k in range(len(need))
                    if not dones[k]
                    and executed_total[need[k]] + int(step_idx[k]) < target
                ]
                if not live:
                    break
                acts = torch.zeros(num_envs, act_dim, device=device)
                for k in live:
                    acts[need[k]] = executed[k, int(step_idx[k])]
                obs, rew, term, trunc = vec.step(acts)
                for k in live:
                    rews[k].append(rew[k])
                    step_idx[k] += 1
                    if bool(term[k]):
                        end_term[k] = True
                        dones[k] = True
                    elif bool(trunc[k]):
                        end_trunc[k] = True
                        dones[k] = True
                    if dones[k]:
                        # Autoreset obs returned: rebuild warmup (CPU parity).
                        windows[need[k]] = obs[need[k]].unsqueeze(0).expand(
                            n_obs_steps, -1
                        ).clone().to(dtype=STORAGE_DTYPE)
                    else:
                        shift_window(windows, need[k], obs[need[k]])
            for k, i in enumerate(need):
                n_exec = int(step_idx[k])
                if n_exec == 0:
                    continue
                r = torch.stack(rews[k][:n_exec])
                term_k, trunc_k = end_term[k], end_trunc[k] and not end_term[k]
                storage.add_chunk(
                    i, states[k].to(dtype=STORAGE_DTYPE),
                    scaled[k].to(dtype=STORAGE_DTYPE),
                    windows[i].clone().to(dtype=STORAGE_DTYPE),
                    r, term_k, trunc_k,
                )
                # Stash GPU-computed clip counts (storage writes 0; patched below).
                if not hasattr(storage, "_clip"):
                    storage._clip = {}  # type: ignore[attr-defined]
                storage._clip[(int(storage.cursor[i].item()) - 1, i)] = int(clip[k])  # type: ignore[attr-defined]
                prev_idx[i] = int(storage.cursor[i].item()) - 1
                prev_live[i] = not bool(dones[k])
                executed_total[i] += n_exec
        # Budget exhausted mid-episode: bootstrap the cut (CPU-vec parity).
        for i in range(num_envs):
            if prev_live[i] and prev_idx[i] >= 0:
                storage.truncated[prev_idx[i], i] = True
    chunks = storage.to_transition_list()
    # Patch real clip counts (storage writes 0; computed on GPU above).
    # to_transition_list emits [env, chunk] order; keys match that layout.
    clips = getattr(storage, "_clip", {})
    if clips:
        from lerobot.processor.converters import to_tensor

        idx = 0
        for e in range(num_envs):
            for c in range(int(storage.cursor[e].item())):
                if (c, e) in clips:
                    chunks[idx]["clip_count"] = to_tensor(clips[(c, e)], dtype=torch.int64)
                idx += 1
    return chunks
