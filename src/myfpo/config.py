"""Environment registry and run-matrix helpers for the SJFPO benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml

ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"

with open(ENV_REGISTRY_PATH) as f:
    _ENV_REGISTRY = yaml.safe_load(f) or {}

KNOWN_ENVS = tuple(_ENV_REGISTRY.keys())


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id`` from ``configs/envs.yaml``.

    Reads the registry parsed once at import, so a call cannot disagree with
    ``KNOWN_ENVS`` about what exists.

    ``KeyError`` if the registry is empty or ``env_id`` is not in it.
    """
    if not _ENV_REGISTRY:
        raise KeyError(f"registry empty at {ENV_REGISTRY_PATH}")
    entry = _ENV_REGISTRY.get(env_id)
    if entry is None:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])
