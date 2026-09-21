"""MuJoCo env semantics for the JAX rollout path (backend-agnostic numpy).

Each entry mirrors its gymnasium ``*_v5.py`` source exactly (defaults only):
observation construction, reward terms, health checks, reset noise. The JAX
MJX vec (``rollout.py``) implements these same formulas in ``jnp``; these
numpy mirrors exist so CPU tests can assert parity against LIVE gymnasium
envs without needing MJX or a GPU.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ENV_SPECS", "get_spec", "obs", "is_healthy", "reward", "terminated", "reset_state"]


ENV_SPECS: dict[str, dict] = {
    "HalfCheetah-v5": {
        "xml": "half_cheetah.xml",
        "frame_skip": 5,
        "forward_weight": 1.0,
        "ctrl_weight": 0.1,
        "healthy_reward": 0.0,
        "terminate_when_unhealthy": False,
        "reset_noise_scale": 0.1,
        "qvel_dist": "normal",  # qvel = init + scale * N(0, 1)
        "obs_drop_first": True,
        "qvel_clip": None,
        "max_episode_steps": 1000,
    },
    "Hopper-v5": {
        "xml": "hopper.xml",
        "frame_skip": 4,
        "forward_weight": 1.0,
        "ctrl_weight": 1e-3,
        "healthy_reward": 1.0,
        "terminate_when_unhealthy": True,
        "reset_noise_scale": 5e-3,
        "qvel_dist": "uniform",  # qvel = init + U(-scale, scale)
        "obs_drop_first": True,
        "qvel_clip": 10.0,
        "healthy_z_range": (0.7, float("inf")),
        "healthy_angle_range": (-0.2, 0.2),
        "healthy_state_range": (-100.0, 100.0),
        "max_episode_steps": 1000,
    },
}


def get_spec(env_name: str) -> dict:
    """Spec dict; KeyError on unknown ids (add the env's formulas first)."""
    try:
        return ENV_SPECS[env_name]
    except KeyError:
        raise KeyError(f"JAX rollout supports {sorted(ENV_SPECS)}, got {env_name!r}")


def obs(spec: dict, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """``qpos[1:] + clipped qvel`` (both envs drop rootx by default)."""
    qpos = np.asarray(qpos, dtype=np.float32).ravel()
    qvel = np.asarray(qvel, dtype=np.float32).ravel()
    if spec["obs_drop_first"]:
        qpos = qpos[1:]
    if spec["qvel_clip"] is not None:
        qvel = np.clip(qvel, -spec["qvel_clip"], spec["qvel_clip"])
    return np.concatenate([qpos, qvel])


def is_healthy(spec: dict, qpos: np.ndarray, qvel: np.ndarray) -> bool:
    """Post-step health (HalfCheetah: always True)."""
    if not spec["terminate_when_unhealthy"] and spec["healthy_reward"] == 0.0:
        return True
    qpos = np.asarray(qpos, dtype=np.float32).ravel()
    qvel = np.asarray(qvel, dtype=np.float32).ravel()
    z, angle = qpos[1], qpos[2]
    state = np.concatenate([qpos, qvel])[2:]
    s_lo, s_hi = spec["healthy_state_range"]
    z_lo, z_hi = spec["healthy_z_range"]
    a_lo, a_hi = spec["healthy_angle_range"]
    return bool(
        np.all((s_lo < state) & (state < s_hi))
        and (z_lo < z < z_hi)
        and (a_lo < angle < a_hi)
    )


def reward(
    spec: dict,
    x_before: float,
    x_after: float,
    dt: float,
    action: np.ndarray,
    healthy: bool,
) -> float:
    """``forward + healthy - ctrl`` (healthy from POST-step state)."""
    forward = spec["forward_weight"] * (x_after - x_before) / dt
    survive = spec["healthy_reward"] if healthy else 0.0
    ctrl = spec["ctrl_weight"] * float(np.sum(np.square(action)))
    return float(forward + survive - ctrl)


def terminated(spec: dict, healthy: bool) -> bool:
    """Episode end (truncation is a step-count concern, handled by rollout)."""
    return (not healthy) and spec["terminate_when_unhealthy"]


def reset_state(
    spec: dict,
    rng: np.random.Generator,
    init_qpos: np.ndarray,
    init_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial (qpos, qvel) with per-env noise (mirrors reset_model)."""
    s = spec["reset_noise_scale"]
    qpos = np.asarray(init_qpos, dtype=np.float32) + rng.uniform(-s, s, len(init_qpos))
    if spec["qvel_dist"] == "normal":
        qvel = np.asarray(init_qvel, dtype=np.float32) + s * rng.standard_normal(len(init_qvel))
    elif spec["qvel_dist"] == "uniform":
        qvel = np.asarray(init_qvel, dtype=np.float32) + rng.uniform(-s, s, len(init_qvel))
    else:
        raise ValueError(f"qvel_dist={spec['qvel_dist']!r} unknown")
    return qpos, qvel
