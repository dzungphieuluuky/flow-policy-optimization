"""YAML<->dataclass parity: every method YAML key must be a real config field.

Locks the deslop Wave 3 single-truth gate: silently-dropped keys (e.g. the
removed ``n_obs_steps`` in ppo/sac.yaml) fail here instead of drifting.
"""

import dataclasses
from pathlib import Path

import yaml

from sjfpo.config import PPOConfig, SACConfig
from sjfpo.configuration_sjfpo import SJFPOConfig

CASES = [
    ("configs/sjfpo.yaml", SJFPOConfig),
    ("configs/ppo.yaml", PPOConfig),
    ("configs/sac.yaml", SACConfig),
]


def load(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / name
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def test_yaml_keys_are_dataclass_fields():
    for name, cls in CASES:
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = [k for k in load(name) if k not in fields]
        assert not unknown, f"{name} has keys no dataclass accepts: {unknown}"
