"""MJX non-visual adapter — draft stub gated behind MFPO_USE_MJX.

Default (MFPO_USE_MJX != "1"): pure-numpy stub with is_stub=True, no
jax/mujoco.mjx dependency. Gated path (MFPO_USE_MJX=="1"): function-local
imports of mujoco.mjx/jax, MjxModelWrapper with put_model + jax.jit step,
exact-dim asserts.

is_stub = True signals draft not wired in runner/baselines.
"""

from __future__ import annotations

import os

import numpy as np
from gymnasium import spaces

# Gated flag — default OFF, stub by default.
_USE_MJX = os.environ.get("MFPO_USE_MJX") == "1"

is_stub = True


def require_mjx_gate() -> None:
    """Guard for real MJX path — raise if gate on but deps/arch unverified."""
    if not _USE_MJX:
        raise RuntimeError("MJX gated off: set MFPO_USE_MJX=1 to enable real MJX path")
    # Arch verification guard: require a verifiable GPU backend when gate is on.
    # On APU gfx1103 the stub repo only publishes CPU/host fallback; attempting
    # MJX on unverified arch risks silent miscompilation (DCT/CFM).
    # This check is function-local so stub import never touches jax.
    try:
        import jax  # noqa: WPS433 - gated local import
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MJX gate on but jax unavailable: {exc}") from exc
    try:
        devs = jax.devices()
        if not devs or devs[0].platform not in ("gpu", "cuda"):
            raise RuntimeError(f"MJX gate on but no GPU jax device: {devs!r} — verify PYTORCH_ROCM_ARCH=gfx1103")
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MJX arch verification failed: {exc}") from exc


class MjxObsAdapter:
    """Non-visual MuJoCo MJX observation adapter (draft).

    Flattens MuJoCo MJX state (qpos + qvel) to a single float32 vector.
    For dict obs with 'qpos'/'qvel' keys, concatenates them; for flat
    arrays, returns as float32. Stub path allows pad/truncate for smoke
    stability; gated real path asserts exact dims.

    Attributes:
        is_stub: Always True for this draft (no real MJX env wired).
        nq: Position dimension (from mjx_model.nq).
        nv: Velocity dimension (from mjx_model.nv).
        total_dim: nq + nv.
        observation_space: Box(total_dim, fp32).
    """

    is_stub = True

    def __init__(self, mjx_model=None, nq: int | None = None, nv: int | None = None):
        if mjx_model is not None:
            # If gate on, require exact dims via mjx_model; no fallback.
            if _USE_MJX:
                require_mjx_gate()
                nq_raw = getattr(mjx_model, "nq", None)
                nv_raw = getattr(mjx_model, "nv", None)
                assert nq_raw is not None and nv_raw is not None, "mjx_model must expose nq/nv in gated mode"
                nq = int(nq_raw)
                nv = int(nv_raw)
            else:
                nq = int(getattr(mjx_model, "nq", nq or 0))
                nv = int(getattr(mjx_model, "nv", nv or 0))
        # fallback to common MuJoCo env dims when not gated
        nq = int(nq if nq is not None else 9)
        nv = int(nv if nv is not None else 8)
        self.nq = nq
        self.nv = nv
        self.total_dim = nq + nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_dim,), dtype=np.float32
        )

    def __call__(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            if "qpos" in obs and "qvel" in obs:
                qpos = np.asarray(obs["qpos"], dtype=np.float32).flatten()
                qvel = np.asarray(obs["qvel"], dtype=np.float32).flatten()
                if _USE_MJX:
                    assert qpos.size == self.nq and qvel.size == self.nv, (
                        f"exact-dim assert gated MJX: qpos {qpos.size} != nq {self.nq} or qvel {qvel.size} != nv {self.nv}"
                    )
                return np.concatenate([qpos, qvel]).astype(np.float32)
            parts = [np.asarray(v, dtype=np.float32).flatten() for v in obs.values()]
            out = np.concatenate(parts).astype(np.float32) if parts else np.zeros((self.total_dim,), dtype=np.float32)
            if _USE_MJX:
                assert out.size == self.total_dim, f"gated MJX exact-dim assert: {out.size} != {self.total_dim}"
            return out
        arr = np.asarray(obs, dtype=np.float32).flatten()
        if arr.size == self.total_dim:
            return arr
        if _USE_MJX:
            raise AssertionError(f"gated MJX exact-dim assert: {arr.size} != {self.total_dim} — no pad/truncate in gated mode")
        # stub path: pad/truncate for smoke stability
        out = np.zeros((self.total_dim,), dtype=np.float32)
        n = min(arr.size, self.total_dim)
        out[:n] = arr[:n]
        return out


class MjxWrapper:
    """Wraps a flat obs env to expose MjxObsAdapter observation_space."""

    is_stub = True

    def __init__(self, env, adapter: MjxObsAdapter):
        self.env = env
        self.adapter = adapter
        self.action_space = env.action_space
        self.observation_space = adapter.observation_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.adapter(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self.adapter(obs), rew, term, trunc, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class MjxModelWrapper:
    """Gated real MJX model wrapper (only when MFPO_USE_MJX==1).

    Lazy imports, put_model + jax.jit step. Not used by runner
    (stub-by-default contract). Instantiation requires MFPO_USE_MJX=1
    and raises RuntimeError otherwise.
    """

    is_stub = True  # still stub from wiring perspective; real step() is gated

    def __init__(self, mjcf_xml: str | None = None, mjx_model=None):
        require_mjx_gate()
        # Function-local imports — never at top-level.
        import jax  # noqa: WPS433
        import mujoco  # noqa: WPS433
        import mujoco.mjx  # noqa: WPS433

        if mjx_model is not None:
            self.mjx_model = mjx_model
        elif mjcf_xml is not None:
            mj_model = mujoco.MjModel.from_xml_string(mjcf_xml)  # type: ignore[attr-defined]
            self.mjx_model = mujoco.mjx.put_model(mj_model)
        else:
            raise ValueError("MjxModelWrapper requires mjcf_xml or mjx_model")

        # JIT-compiled step for batched rollouts
        self._jit_step = jax.jit(mujoco.mjx.step)
        self._jax = jax
        self._mjx = mujoco.mjx

    def step(self, data, action):
        """Single MJX step — exact-dim gated path."""
        require_mjx_gate()
        return self._jit_step(self.mjx_model, data, action)
