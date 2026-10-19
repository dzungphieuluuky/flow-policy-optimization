"""``configs/myfpo.yaml`` must equal the dataclass defaults."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from myfpo.configuration import MyfpoConfig

ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = ROOT / "configs" / "myfpo.yaml"


@pytest.fixture(scope="module")
def loaded() -> dict:
    if not YAML_PATH.is_file():
        pytest.skip(f"{YAML_PATH} not present")
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}


def test_every_yaml_key_is_a_real_config_field(loaded: dict):
    """A typo'd key used to be dropped with a warning and the run looked configured."""
    known = {f.name for f in dataclasses.fields(MyfpoConfig)}
    assert not set(loaded) - known, f"unknown keys: {sorted(set(loaded) - known)}"


def test_yaml_matches_the_dataclass_defaults(loaded: dict):
    fresh = dataclasses.replace(MyfpoConfig())
    drift = {}
    for key, raw_value in loaded.items():
        default = getattr(fresh, key, None)
        value = raw_value
        # YAML has no tuple type. Adam_betas arrives as a list.
        if isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        if default != value:
            drift[key] = (value, default)
    assert not drift, (
        "configs/myfpo.yaml has drifted from the dataclass. Regenerate with "
        f".venv/bin/python scripts/gen_config.py\ndrifted: {drift}"
    )


def test_generator_is_deterministic(loaded: dict):
    """Running the generator must be a no-op on a committed tree."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_config", ROOT / "scripts" / "gen_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regenerated = yaml.safe_load(module.render())
    assert regenerated == loaded, (
        "regenerating the file changed it; the committed file is stale"
    )