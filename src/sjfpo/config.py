"""Environment registry and run-matrix helpers for the SJFPO benchmark."""

from __future__ import annotations

from pathlib import Path


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


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id`` from ``configs/envs.yaml``.

    ``RuntimeError`` if PyYAML is missing. ``KeyError`` if ``env_id`` is not
    in the registry.
    """
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
