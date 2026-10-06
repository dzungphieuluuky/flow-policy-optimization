"""Environment registry and run-matrix helpers for the SJFPO benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml

ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"

with open(ENV_REGISTRY_PATH) as f:
    _ENV_REGISTRY = yaml.safe_load(f) or {}

_REGISTRY_KEYS = list(_ENV_REGISTRY.keys())
KNOWN_ENVS = tuple(_REGISTRY_KEYS)


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id`` from ``configs/envs.yaml``.

    ``RuntimeError`` if PyYAML is missing. ``KeyError`` if ``env_id`` is not
    in the registry.
    """
    with open(ENV_REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)
    if not registry:
        raise KeyError(f"registry empty at {ENV_REGISTRY_PATH}")
    entry = registry.get(env_id)
    if entry is None:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])
