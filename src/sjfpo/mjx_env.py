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
    "mjx_collect_chunks",
]

# HalfCheetah-v5 defaults (mirrors gymnasium; assumption, documented).
_FRAME_SKIP = 5
_FORWARD_WEIGHT = 1.0
_CTRL_WEIGHT = 0.1
_RESET_NOISE_SCALE = 0.1
_MAX_EPISODE_STEPS = 1000
_SUPPORTED_ENVS = ("HalfCheetah-v5",)


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


class MjxHalfCheetahVec:
    """Batched HalfCheetah on GPU; torch-CUDA in/out via DLPack (zero-copy).

    ``reset() -> [M, obs]``; ``step(ctrl[M, act]) -> (obs, rew, trunc)``
    with in-place autoreset on truncation (fresh reset obs returned).
    """

    def __init__(self, num_envs: int, seed: int, device: torch.device):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        import gymnasium as gym

        if num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be >= 1")
        xml = (
            __import__("pathlib").Path(gym.__file__).resolve().parent
            / "envs"
            / "mujoco"
            / "assets"
            / "half_cheetah.xml"
        )
        mj_model = mujoco.MjModel.from_xml_path(str(xml))
        self.nq, self.nv, self.nu = mj_model.nq, mj_model.nv, mj_model.nu
        self.obs_dim = self.nq - 1 + self.nv
        self.dt = float(mj_model.opt.timestep * _FRAME_SKIP)
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

        def _physics(mx, dx, ctrl):
            dx = dx.replace(ctrl=ctrl)
            return jax.lax.fori_loop(0, _FRAME_SKIP, lambda _, d: mjx.step(mx, d), dx)

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

    def _sample_state(self, rng, n: int):
        import jax
        import jax.numpy as jnp

        k1, k2 = jax.random.split(rng)
        qpos = self._init_qpos + jax.random.uniform(
            k1, (n, self.nq), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        qvel = self._init_qvel + _RESET_NOISE_SCALE * jax.random.normal(k2, (n, self.nv))
        return qpos.astype(jnp.float32), qvel.astype(jnp.float32)

    def _obs_jax(self, dx):
        import jax.numpy as jnp

        return jnp.concatenate([dx.qpos[:, 1:], dx.qvel], axis=1)

    def reset(self) -> torch.Tensor:
        import jax

        self._rng, sub = jax.random.split(self._rng)
        qpos, qvel = self._sample_state(sub, self.num_envs)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import jax
        import jax.numpy as jnp

        ctrl_j = self._jax(ctrl)
        x_before = self._dx.qpos[:, 0]
        self._dx = self._step_fn(self.mx, self._dx, ctrl_j)
        x_after = self._dx.qpos[:, 0]
        rew = (
            _FORWARD_WEIGHT * (x_after - x_before) / self.dt
            - _CTRL_WEIGHT * jnp.sum(ctrl_j * ctrl_j, axis=1)
        ).astype(jnp.float32)
        self._ep_len = self._ep_len + 1
        trunc = self._ep_len >= _MAX_EPISODE_STEPS
        obs = self._obs_jax(self._dx)
        # Autoreset truncated slots (fresh obs returned, flag kept).
        if bool(jnp.any(trunc)):
            self._rng, sub = jax.random.split(self._rng)
            qpos, qvel = self._sample_state(sub, self.num_envs)
            fresh = self._fwd_fn(
                self.mx, self._template.replace(qpos=qpos, qvel=qvel)
            )
            take = trunc[:, None]
            self._dx = jax.tree.map(
                lambda new, old: jnp.where(take, new, old)
                if new.ndim == 2
                else old,
                fresh,
                self._dx,
            )
            self._ep_len = jnp.where(trunc, 0, self._ep_len)
            obs = self._obs_jax(self._dx)
        return (
            self._torch(obs).to(dtype=torch.float32),
            self._torch(rew).to(dtype=torch.float32),
            self._torch(trunc),
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

    if env_name not in _SUPPORTED_ENVS:
        raise ValueError(f"SJFPO_MJX supports {list(_SUPPORTED_ENVS)}, got {env_name!r}")
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
    vec = MjxHalfCheetahVec(num_envs, seed, device)
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
                obs, rew, trunc = vec.step(acts)
                for k in live:
                    rews[k].append(rew[k])
                    step_idx[k] += 1
                    if bool(trunc[k]):
                        # Autoreset obs returned: rebuild warmup (CPU parity).
                        windows[need[k]] = obs[need[k]].unsqueeze(0).expand(
                            n_obs_steps, -1
                        ).clone().to(dtype=STORAGE_DTYPE)
                        dones[k] = True
                    else:
                        w = windows[need[k]]
                        w[:-1].copy_(w[1:])
                        w[-1].copy_(obs[need[k]].to(dtype=STORAGE_DTYPE))
            for k, i in enumerate(need):
                n_exec = int(step_idx[k])
                if n_exec == 0:
                    continue
                r = torch.stack(rews[k][:n_exec])
                storage.add_chunk(
                    i, states[k].to(dtype=STORAGE_DTYPE),
                    scaled[k].to(dtype=STORAGE_DTYPE),
                    windows[i].clone().to(dtype=STORAGE_DTYPE),
                    r, False, bool(dones[k]),
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
