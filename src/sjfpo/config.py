"""Environment registry and run-matrix helpers for the SJFPO benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

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


@dataclass(frozen=True)
class RunSpec:
    method: str
    env: str
    seed: int
    total_steps: int
    eval_steps: int
    save_steps: int


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
