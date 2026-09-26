"""Benchmark config registry: run matrix + runtime dependency guards."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
TINY_MATRIX = os.environ.get("SJFPO_TINY_MATRIX") == "1"

METHODS = ("sjfpo",)
ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"

try:
    import yaml

    with open(ENV_REGISTRY_PATH) as f:
        _ENV_REGISTRY = yaml.safe_load(f) or {}
except Exception:
    _ENV_REGISTRY = {}

_REGISTRY_KEYS = list(_ENV_REGISTRY.keys()) if isinstance(_ENV_REGISTRY, dict) else []
ENVS = tuple(k for k in _REGISTRY_KEYS if _ENV_REGISTRY[k].get("tier") == "core")
PROTOTYPE_ENVS = tuple(
    k for k in _REGISTRY_KEYS if _ENV_REGISTRY[k].get("tier") != "core"
)
KNOWN_ENVS = ENVS + PROTOTYPE_ENVS
SEEDS = (0, 1, 2, 3, 4)

TOTAL_STEPS = 1_000_000
EVAL_STEPS = 50_000
SAVE_STEPS = 25_000


@dataclass(frozen=True)
class RunSpec:
    method: str
    env: str
    seed: int
    total_steps: int
    eval_steps: int
    save_steps: int


def build_run_matrix() -> tuple[RunSpec, ...]:
    """Legacy MuJoCo matrix (20 runs: 1 method × 4 envs × 5 seeds)."""
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
                        save_steps=SAVE_STEPS,
                    )
                )
    return tuple(specs)


RUN_MATRIX = build_run_matrix()  # 20 runs


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
                save_steps=10_000,
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
    """Fail fast on dependency pins and unsupported precision flags.      bf16 *autocast* (fp32 storage, bf16 matmuls) is allow"""
    import importlib.metadata

    gym_version = importlib.metadata.version("gymnasium")
    if parse_version_tuple(gym_version)[:2] < (1, 0):
        raise RuntimeError(
            f"gymnasium {gym_version} has major.minor < 1.0; "
            "this project requires gymnasium[mujoco]>=1.0,<1.3"
        )

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
