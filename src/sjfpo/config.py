"""Benchmark config registry: PPO/SAC hyperparameters + run matrix.

Dataclasses are frozen with tuple defaults; RUN_MATRIX is a tuple of frozen
RunSpec built over methods × envs × seeds. Mode switches read env vars once
at import. MuJoCo-only (flat Box).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Mode switches (plain module-level constants; read env at import time ONLY)

IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
TINY_MATRIX = os.environ.get("SJFPO_TINY_MATRIX") == "1"

# Run-matrix axes

METHODS = ("sjfpo", "ppo", "sac")
ENVS = ("HalfCheetah-v5", "Walker2d-v5", "Hopper-v5")
PROTOTYPE_ENVS = (
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Swimmer-v5",
    "Pusher-v5",
    "Ant-v5",
    "Humanoid-v5",
    "AdroitHandDoor-v1",
    "AdroitHandPen-v1",
    "AdroitHandHammer-v1",
)
KNOWN_ENVS = ENVS + PROTOTYPE_ENVS  # --env choices; matrix still uses ENVS only
ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"
SEEDS = (0, 1, 2, 3, 4)

TOTAL_STEPS = 1_000_000
EVAL_STEPS = 50_000
CKPT_INTERVAL = 25_000


# Method hyperparameters (Scope §Must-have, verbatim values)


@dataclass(frozen=True)
class PPOConfig:
    """skrl 2.1.0 PPO baseline keys."""

    rollouts: int = 2048
    learning_epochs: int = 10
    mini_batches: int = 8
    lr: float = 3e-4
    kl_threshold: float = 0.01  # KLAdaptiveLR
    grad_norm_clip: float = 1.0
    discount: float = 0.99
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_loss_scale: float = 1.0
    hidden: tuple[int, ...] = (256, 256, 256)


@dataclass(frozen=True)
class SACConfig:
    """skrl 2.1.0 SAC baseline keys."""

    lr: float = 3e-4
    batch_size: int = 256
    memory_size: int = 300_000
    gradient_steps: int = 1
    polyak: float = 0.005
    discount: float = 0.99
    initial_random_steps: int = 5000
    hidden: tuple[int, ...] = (256, 256, 256)


PPO_CFG = PPOConfig()
SAC_CFG = SACConfig()


# Run matrix


@dataclass(frozen=True)
class RunSpec:
    """Single run: method × env × seed + budgets."""

    method: str
    env: str
    seed: int
    total_steps: int
    eval_steps: int
    ckpt_interval: int


def build_run_matrix() -> tuple[RunSpec, ...]:
    """Legacy MuJoCo matrix (45 runs: 3 methods × 3 envs × 5 seeds)."""
    specs: list[RunSpec] = []
    for method in METHODS:
        for env in ENVS:
            for seed in SEEDS:
                specs.append(
                    RunSpec(
                        method=method,
                        env=env,
                        seed=seed,
                        total_steps=TOTAL_STEPS,
                        eval_steps=EVAL_STEPS,
                        ckpt_interval=CKPT_INTERVAL,
                    )
                )
    return tuple(specs)


RUN_MATRIX = build_run_matrix()  # 45 runs


def build_filtered_matrix(
    methods: Iterable[str],
    envs: Iterable[str],
    seeds: Iterable[int],
    total_steps: int,
    eval_steps: int,
    ckpt_interval: int,
) -> tuple[RunSpec, ...]:
    """Cartesian matrix in RUN_MATRIX order (method x env x seed) for CLI filters."""
    return tuple(
        RunSpec(
            method=m,
            env=e,
            seed=s,
            total_steps=total_steps,
            eval_steps=eval_steps,
            ckpt_interval=ckpt_interval,
        )
        for m in [x for x in METHODS if x in methods]
        for e in [x for x in KNOWN_ENVS if x in envs]
        for s in seeds
    )


if TINY_MATRIX:
    tiny: list[RunSpec] = []
    for method in METHODS:
        tiny.append(
            RunSpec(
                method=method,
                env="Hopper-v5",
                seed=0,
                total_steps=20_000,
                eval_steps=10_000,
                ckpt_interval=10_000,
            )
        )
    RUN_MATRIX = tuple(tiny)  # 3 runs


def validate_run_matrix(specs: Iterable[RunSpec]) -> None:
    """Raise AssertionError on any duplicate (method, env, seed)."""
    seen: set[tuple[str, str, int]] = set()
    for spec in specs:
        key = (spec.method, spec.env, spec.seed)
        assert key not in seen, f"duplicate (method, env, seed) in run matrix: {key}"
        seen.add(key)


def parse_version_tuple(version: str) -> tuple[int, ...]:
    """Version string -> (major, minor, patch, ...) for comparison."""
    parts: list[int] = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def assert_runtime(allow_fp16: bool = False) -> None:
    """Fail fast on dependency pins and unsupported precision flags.

    bf16 *autocast* (fp32 storage, bf16 matmuls) is allowed and needs no
    flag; full half/bfloat STORAGE raises.
    """
    import importlib.metadata

    gym_version = importlib.metadata.version("gymnasium")
    if parse_version_tuple(gym_version)[:2] < (1, 0):
        raise RuntimeError(
            f"gymnasium {gym_version} has major.minor < 1.0; "
            "this project requires gymnasium[mujoco]>=1.0,<1.3"
        )

    skrl_version = importlib.metadata.version("skrl")
    if skrl_version != "2.1.0":
        raise RuntimeError(f"skrl {skrl_version} != 2.1.0; exact pin required")

    if allow_fp16:
        raise RuntimeError(
            "fp16/bf16 storage requested but unsupported: flow-matching "
            "Euler rollouts and CFM losses require fp32 storage; use "
            "SJFPO_DTYPE=bf16 for bf16 autocast (fp32 params, bf16 matmuls)"
        )


validate_run_matrix(RUN_MATRIX)


def lookup_env_dims(env_id: str) -> tuple[int, int]:
    """(obs_dim, action_dim) from configs/envs.yaml; KeyError on unknown ids."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml required for --env lookup (pip install pyyaml)")
    with open(ENV_REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    try:
        entry = registry[env_id]
    except KeyError:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])
