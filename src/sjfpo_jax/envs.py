"""MuJoCo env semantics for the MJX rollout paths (backend-agnostic numpy).

Each entry mirrors its gymnasium ``*_v5.py`` source exactly (defaults only).
``obs``/``reward`` take small ``extra`` dicts for sensor-heavy envs
(contact forces, site positions, body frames); the MJX vecs build the same
dicts from ``mjx.Data``. CPU tests assert parity against LIVE gymnasium envs.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ENV_SPECS", "MJX_UNSUPPORTED", "get_spec", "zero_extras", "obs_dim",
    "obs", "is_healthy", "reward", "terminated", "reset_state",
]


# Envs with no MJX path + the reason (raised loudly, never silently wrong).
MJX_UNSUPPORTED = {
    # Classic control: not MuJoCo at all (no MJX model exists).
    "CartPole-v1": "classic control, not a MuJoCo env",
    "Pendulum-v1": "classic control, not a MuJoCo env",
    "MountainCar-v0": "classic control, not a MuJoCo env",
    # MJX limitations (verified against the installed mujoco.mjx).
    "Humanoid-v5": "model uses the PGS solver, which MJX rejects",
    "AdroitHandDoor-v1": "cylinder/box collisions unimplemented in MJX",
    "AdroitHandPen-v1": "cylinder/box collisions unimplemented in MJX",
    "AdroitHandHammer-v1": "cylinder/box collisions unimplemented in MJX",
}


ENV_SPECS: dict[str, dict] = {
    "HalfCheetah-v5": {
        "xml": "half_cheetah.xml", "frame_skip": 5,
        "obs_kind": "qpos_qvel", "obs_drop": 1, "qvel_clip": None,
        "reward_kind": "locomotion", "forward_weight": 1.0, "ctrl_weight": 0.1,
        "healthy_reward": 0.0, "contact_weight": 0.0,
        "done_kind": "never", "terminate_when_unhealthy": False,
        "reset_kind": "uniform_normal", "reset_noise_scale": 0.1,
        "forward_source": "qpos_x",
        "max_episode_steps": 1000,
    },
    "Hopper-v5": {
        "xml": "hopper.xml", "frame_skip": 4,
        "obs_kind": "qpos_qvel", "obs_drop": 1, "qvel_clip": 10.0,
        "reward_kind": "locomotion", "forward_weight": 1.0, "ctrl_weight": 1e-3,
        "healthy_reward": 1.0, "contact_weight": 0.0,
        "done_kind": "healthy", "terminate_when_unhealthy": True,
        "healthy_z_range": (0.7, float("inf")),
        "healthy_angle_range": (-0.2, 0.2),
        "healthy_state_range": (-100.0, 100.0),
        "reset_kind": "uniform_uniform", "reset_noise_scale": 5e-3,
        "forward_source": "qpos_x",
        "max_episode_steps": 1000,
    },
    "Walker2d-v5": {
        "xml": "walker2d_v5.xml", "frame_skip": 4,
        "obs_kind": "qpos_qvel", "obs_drop": 1, "qvel_clip": 10.0,
        "reward_kind": "locomotion", "forward_weight": 1.0, "ctrl_weight": 1e-3,
        "healthy_reward": 1.0, "contact_weight": 0.0,
        "done_kind": "healthy", "terminate_when_unhealthy": True,
        "healthy_z_range": (0.8, 2.0),
        "healthy_angle_range": (-1.0, 1.0),
        "healthy_state_range": None,  # walker checks z/angle only
        "reset_kind": "uniform_uniform", "reset_noise_scale": 5e-3,
        "forward_source": "qpos_x",
        "max_episode_steps": 1000,
    },
    "Swimmer-v5": {
        "xml": "swimmer.xml", "frame_skip": 4,
        "obs_kind": "qpos_qvel", "obs_drop": 2, "qvel_clip": None,
        "reward_kind": "locomotion", "forward_weight": 1.0, "ctrl_weight": 1e-4,
        "healthy_reward": 0.0, "contact_weight": 0.0,
        "done_kind": "never", "terminate_when_unhealthy": False,
        "reset_kind": "uniform_uniform", "reset_noise_scale": 0.1,
        "forward_source": "qpos_x",
        "max_episode_steps": 1000,
    },
    "Ant-v5": {
        "xml": "ant.xml", "frame_skip": 5,
        "obs_kind": "ant", "obs_drop": 2, "qvel_clip": None,
        "contact_force_range": (-1.0, 1.0),
        "reward_kind": "locomotion", "forward_weight": 1.0, "ctrl_weight": 0.5,
        "healthy_reward": 1.0, "contact_weight": 5e-4,
        "done_kind": "ant", "terminate_when_unhealthy": True,
        "healthy_z_range": (0.2, 1.0),
        "reset_kind": "uniform_normal", "reset_noise_scale": 0.1,
        "main_body": "torso",
        "forward_source": "body_x",
        "max_episode_steps": 1000,
    },
    "InvertedPendulum-v5": {
        "xml": "inverted_pendulum.xml", "frame_skip": 2,
        "obs_kind": "full",
        "reward_kind": "survive",
        "done_kind": "pole",
        "reset_kind": "uniform_uniform", "reset_noise_scale": 0.01,
        "forward_source": "none",
        "max_episode_steps": 1000,
    },
    "InvertedDoublePendulum-v5": {
        "xml": "inverted_double_pendulum.xml", "frame_skip": 5,
        "obs_kind": "double_pendulum",
        "reward_kind": "double_pendulum", "healthy_reward": 10.0,
        "done_kind": "site_y",
        "reset_kind": "uniform_normal", "reset_noise_scale": 0.1,
        "forward_source": "none", "site_index": 0,
        "max_episode_steps": 1000,
    },
    "Reacher-v5": {
        "xml": "reacher.xml", "frame_skip": 2,
        "obs_kind": "reacher",
        "reward_kind": "reacher", "reward_dist_weight": 1.0,
        "reward_control_weight": 1.0,
        "done_kind": "never",
        "reset_kind": "reacher",
        "forward_source": "none",
        "com_bodies": ["fingertip", "target"],
        "max_episode_steps": 1000,
    },
    "Pusher-v5": {
        "xml": "pusher.xml", "frame_skip": 5,
        "obs_kind": "pusher",
        "reward_kind": "pusher", "reward_near_weight": 0.5,
        "reward_dist_weight": 1.0, "reward_control_weight": 0.1,
        "done_kind": "never",
        "reset_kind": "pusher",
        "forward_source": "none",
        "com_bodies": ["tips_arm", "object", "goal"],
        "max_episode_steps": 1000,
    },
}


def zero_extras(spec: dict, nbody: int, nv: int) -> dict:
    """Zero sensor dict with the shapes ``obs`` needs (dim measurement)."""
    extra: dict = {}
    if spec["obs_kind"] == "ant":
        extra["cfrc_ext"] = np.zeros((nbody, 6))
    if spec["obs_kind"] == "double_pendulum":
        extra["qfrc_constraint"] = np.zeros(nv)
    for name in spec.get("com_bodies", []):
        extra[name] = np.zeros(3)
    return extra


def obs_dim(spec: dict, nq: int, nv: int, nbody: int) -> int:
    """Observation dim without running physics (registry gate)."""
    return len(obs(spec, np.zeros(nq), np.zeros(nv), zero_extras(spec, nbody, nv)))


def get_spec(env_name: str) -> dict:
    """Spec dict; KeyError on unknown ids, RuntimeError when MJX-blocked."""
    try:
        return ENV_SPECS[env_name]
    except KeyError:
        pass
    if env_name in MJX_UNSUPPORTED:
        raise RuntimeError(f"no MJX path for {env_name}: {MJX_UNSUPPORTED[env_name]}")
    raise KeyError(f"MJX rollout supports {sorted(ENV_SPECS)}, got {env_name!r}")


def obs(spec: dict, qpos: np.ndarray, qvel: np.ndarray,
        extra: dict | None = None) -> np.ndarray:
    """Observation; ``extra`` carries cfrc_ext/site/body frames for exotics."""
    extra = extra or {}
    kind = spec["obs_kind"]
    qpos = np.asarray(qpos, dtype=np.float64).ravel()
    qvel = np.asarray(qvel, dtype=np.float64).ravel()
    if kind == "qpos_qvel":
        position = qpos[spec["obs_drop"]:]
        if spec["qvel_clip"] is not None:
            qvel = np.clip(qvel, -spec["qvel_clip"], spec["qvel_clip"])
        return np.concatenate([position, qvel])
    if kind == "full":
        return np.concatenate([qpos, qvel])
    if kind == "ant":
        position = qpos[spec["obs_drop"]:]
        lo, hi = spec["contact_force_range"]
        cfrc = np.clip(np.asarray(extra["cfrc_ext"], dtype=np.float64), lo, hi)
        return np.concatenate([position, qvel, cfrc[1:].flatten()])
    if kind == "reacher":
        theta = qpos[:2]
        return np.concatenate([
            np.cos(theta), np.sin(theta), qpos[2:], qvel[:2],
            (np.asarray(extra["fingertip"]) - np.asarray(extra["target"]))[:2],
        ])
    if kind == "double_pendulum":
        return np.concatenate([
            qpos[:1], np.sin(qpos[1:]), np.cos(qpos[1:]),
            np.clip(qvel, -10, 10),
            np.clip(np.asarray(extra["qfrc_constraint"]), -10, 10)[:1],
        ]).ravel()
    if kind == "pusher":
        return np.concatenate([
            qpos[:7], qvel[:7],
            np.asarray(extra["tips_arm"]), np.asarray(extra["object"]),
            np.asarray(extra["goal"]),
        ])
    raise ValueError(f"obs_kind={kind!r} unknown")


def is_healthy(spec: dict, qpos: np.ndarray, qvel: np.ndarray,
               obs: np.ndarray | None = None, site_y: float | None = None) -> bool:
    """Post-step health ( locomotion z/angle/state, ant z, pole, site_y)."""
    kind = spec["done_kind"]
    if kind == "never":
        return True
    qpos = np.asarray(qpos, dtype=np.float64).ravel()
    qvel = np.asarray(qvel, dtype=np.float64).ravel()
    if kind == "healthy":
        z, angle = qpos[1], qpos[2]
        z_lo, z_hi = spec["healthy_z_range"]
        a_lo, a_hi = spec["healthy_angle_range"]
        if not (z_lo < z < z_hi and a_lo < angle < a_hi):
            return False
        if spec.get("healthy_state_range") is not None:
            s_lo, s_hi = spec["healthy_state_range"]
            state = np.concatenate([qpos, qvel])[2:]
            if not bool(np.all((s_lo < state) & (state < s_hi))):
                return False
        return True
    if kind == "ant":
        state = np.concatenate([qpos, qvel])
        z_lo, z_hi = spec["healthy_z_range"]
        return bool(np.isfinite(state).all() and z_lo <= state[2] <= z_hi)
    if kind == "pole":
        if obs is None:
            raise ValueError("pole done_kind needs the post-step obs")
        return bool(np.isfinite(np.asarray(obs)).all() and abs(float(obs[1])) <= 0.2)
    if kind == "site_y":
        if site_y is None:
            raise ValueError("site_y done_kind needs the tip height")
        return bool(float(site_y) > 1)
    raise ValueError(f"done_kind={kind!r} unknown")


def reward(spec: dict, x_before: float, x_after: float, dt: float,
           action: np.ndarray, healthy: bool, extra: dict | None = None) -> float:
    """Step reward (healthy/dist terms read POST-step state, like gymnasium)."""
    extra = extra or {}
    kind = spec["reward_kind"]
    action = np.asarray(action, dtype=np.float64)
    if kind == "locomotion":
        forward = spec["forward_weight"] * (x_after - x_before) / dt
        survive = spec["healthy_reward"] if healthy else 0.0
        ctrl = spec["ctrl_weight"] * float(np.sum(np.square(action)))
        contact = 0.0
        if spec["contact_weight"]:
            lo, hi = spec["contact_force_range"]
            cfrc = np.clip(np.asarray(extra["cfrc_ext"], dtype=np.float64), lo, hi)
            contact = spec["contact_weight"] * float(np.sum(np.square(cfrc)))
        return float(forward + survive - ctrl - contact)
    if kind == "survive":
        return 1.0 if healthy else 0.0
    if kind == "double_pendulum":
        v1, v2 = np.asarray(extra["qvel"])[1:3]
        site_x, site_y = float(extra["site_xpos0"][0]), float(extra["site_xpos0"][2])
        dist = 0.01 * site_x**2 + (site_y - 2) ** 2
        vel = 1e-3 * v1**2 + 5e-3 * v2**2
        return float(spec["healthy_reward"] * int(healthy) - dist - vel)
    if kind == "reacher":
        vec = np.asarray(extra["fingertip"]) - np.asarray(extra["target"])
        dist = -float(np.linalg.norm(vec)) * spec["reward_dist_weight"]
        ctrl = -float(np.square(action).sum()) * spec["reward_control_weight"]
        return float(dist + ctrl)
    if kind == "pusher":
        tips, obj, goal = (np.asarray(extra[k]) for k in ("tips_arm", "object", "goal"))
        near = -float(np.linalg.norm(obj - tips)) * spec["reward_near_weight"]
        dist = -float(np.linalg.norm(obj - goal)) * spec["reward_dist_weight"]
        ctrl = -float(np.square(action).sum()) * spec["reward_control_weight"]
        return float(dist + ctrl + near)
    raise ValueError(f"reward_kind={kind!r} unknown")


def terminated(spec: dict, healthy: bool, obs: np.ndarray | None = None,
               site_y: float | None = None) -> bool:
    """Episode end (truncation is a step-count concern, handled by rollout)."""
    # Every terminating family defines healthy == True  <=>  not done.
    if spec["done_kind"] == "never":
        return False
    return not healthy


def reset_state(spec: dict, rng: np.random.Generator,
                init_qpos: np.ndarray, init_qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initial (qpos, qvel) with per-env noise and goal sampling."""
    kind = spec["reset_kind"]
    s = spec.get("reset_noise_scale", 0.0)
    nq, nv = len(init_qpos), len(init_qvel)
    if kind == "uniform_normal":
        qpos = np.asarray(init_qpos, dtype=np.float64) + rng.uniform(-s, s, nq)
        qvel = np.asarray(init_qvel, dtype=np.float64) + s * rng.standard_normal(nv)
    elif kind == "uniform_uniform":
        qpos = np.asarray(init_qpos, dtype=np.float64) + rng.uniform(-s, s, nq)
        qvel = np.asarray(init_qvel, dtype=np.float64) + rng.uniform(-s, s, nv)
    elif kind == "reacher":
        qpos = np.asarray(init_qpos, dtype=np.float64) + rng.uniform(-0.1, 0.1, nq)
        while True:
            goal = rng.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(goal) < 0.2:
                break
        qpos[-2:] = goal
        qvel = np.asarray(init_qvel, dtype=np.float64) + rng.uniform(-0.005, 0.005, nv)
        qvel[-2:] = 0
    elif kind == "pusher":
        qpos = np.asarray(init_qpos, dtype=np.float64)
        goal = np.zeros(2)
        while True:
            cyl = np.concatenate([
                rng.uniform(low=-0.3, high=0, size=1),
                rng.uniform(low=-0.2, high=0.2, size=1),
            ])
            if np.linalg.norm(cyl - goal) > 0.17:
                break
        qpos[-4:-2] = cyl
        qpos[-2:] = goal
        qvel = np.asarray(init_qvel, dtype=np.float64) + rng.uniform(-0.005, 0.005, nv)
        qvel[-4:] = 0
    else:
        raise ValueError(f"reset_kind={kind!r} unknown")
    return qpos, qvel
