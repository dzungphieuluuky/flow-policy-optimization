"""The JAX config stays a faithful mirror of the torch one.

Guards two things that drift silently: the field set, and the properties the
learner reads. A field added to torch and not here is a silent drop; a default
that differs is a different experiment.
"""
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sjfpo_jax.configuration_sjfpo import (
    FEATURE_PROPERTIES,
    LEROBOT_ONLY_FIELDS,
    SJFPOConfig,
)
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig as TorchConfig


def field_names(cls):
    return {f.name for f in dataclasses.fields(cls)}


def test_field_set_is_torch_minus_lerobot_plus_the_two_dims():
    torch_fields = field_names(TorchConfig)
    jax_fields = field_names(SJFPOConfig)
    assert jax_fields == (torch_fields - LEROBOT_ONLY_FIELDS) | {"obs_dim", "action_dim"}


def test_defaults_are_identical_where_the_fields_overlap():
    """Same experiment, different framework. A differing default is a bug."""
    torch_cfg, jax_cfg = TorchConfig(), SJFPOConfig(obs_dim=11, action_dim=2)
    for name in field_names(TorchConfig) - LEROBOT_ONLY_FIELDS:
        torch_value = getattr(torch_cfg, name)
        jax_value = getattr(jax_cfg, name)
        assert jax_value == torch_value or (
            isinstance(torch_value, float) and abs(jax_value - torch_value) < 1e-12
        ), f"{name}: torch {torch_value!r} vs jax {jax_value!r}"


def test_no_feature_derived_properties_survive():
    """These read input_features, which does not exist here."""
    for name in FEATURE_PROPERTIES:
        assert not hasattr(SJFPOConfig, name), name


def test_the_properties_the_learner_reads_agree():
    torch_cfg, jax_cfg = TorchConfig(), SJFPOConfig(obs_dim=11, action_dim=2)
    for name in ("num_mini_batches", "num_flow_tokens", "observation_delta_indices",
                 "action_delta_indices", "reward_delta_indices"):
        assert getattr(jax_cfg, name) == getattr(torch_cfg, name), name
    assert jax_cfg.num_mini_batches == 8
    assert jax_cfg.num_flow_tokens == 4


def test_num_flow_tokens_ignores_num_kept_tokens_without_the_dct():
    """Without the DCT the token axis is time, so there is nothing to keep."""
    torch_cfg, jax_cfg = TorchConfig(), SJFPOConfig(obs_dim=11, action_dim=2)
    assert jax_cfg.num_flow_tokens == 4
    for cfg in (torch_cfg, jax_cfg):
        cfg.use_dct_transform = False
        cfg.num_kept_tokens = 8
    assert jax_cfg.num_flow_tokens == jax_cfg.action_horizon == torch_cfg.num_flow_tokens


def test_n_action_steps_falls_back_to_the_horizon():
    cfg = SJFPOConfig(obs_dim=11, action_dim=2, n_action_steps=None)
    assert cfg.n_action_steps == cfg.action_horizon


def test_the_load_bearing_invariants_raise():
    """These four compute something else rather than erroring, so they are checked."""
    base = {"obs_dim": 11, "action_dim": 2}
    with pytest.raises(ValueError, match="num_kept_tokens"):
        SJFPOConfig(**base, action_horizon=8, num_kept_tokens=16)
    with pytest.raises(ValueError, match="batch_size"):
        SJFPOConfig(**base, chunks_per_update=64, batch_size=128)
    with pytest.raises(ValueError, match="divisible"):
        SJFPOConfig(**base, hidden_dim=130, dit_heads=4)
    with pytest.raises(ValueError, match="gamma"):
        SJFPOConfig(**base, gamma=0.0)
    with pytest.raises(ValueError, match="min_timestep"):
        SJFPOConfig(**base, min_timestep=0.9, max_timestep=0.5)
    with pytest.raises(ValueError, match="obs_dim"):
        SJFPOConfig(action_dim=2)


def test_it_stays_a_plain_dataclass():
    """draccus parses it and `build_tracker` calls `dataclasses.asdict`."""
    assert dataclasses.is_dataclass(SJFPOConfig)
    as_dict = dataclasses.asdict(SJFPOConfig(obs_dim=11, action_dim=2))
    assert isinstance(as_dict, dict) and as_dict["policy_lr"] == 1e-5


def test_it_does_not_import_torch():
    """The whole point: the JAX process must not drag torch in."""
    import sys as _sys

    for module in ("torch", "lerobot"):
        assert module not in _sys.modules or _sys.modules[module] is not None
    source = (ROOT / "src" / "sjfpo_jax" / "configuration_sjfpo.py").read_text()
    assert "import torch" not in source
    assert "lerobot" not in source.split('"""', 2)[2]
