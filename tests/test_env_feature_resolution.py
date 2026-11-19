"""Passing only ``--env`` must be enough to size and feature the policy."""

from pathlib import Path

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig

ROOT = Path(__file__).resolve().parents[1]


def test_a_bare_config_constructs_without_dims():
    """Dims are not config fields; the policy takes them as constructor arguments."""
    cfg = FlowPolicyOptimizationConfig()
    assert not hasattr(cfg, "obs_dim")
    assert not hasattr(cfg, "action_dim")


def test_the_env_name_is_not_a_config_field():
    """Auto-detection keyed on this never had a value to key on."""
    assert not hasattr(FlowPolicyOptimizationConfig(), "env_name")


def test_train_py_documents_underscore_flag_names():
    """draccus rejects --obs-dim. Dimensions derive from the live env spaces."""
    train_py = (ROOT / "scripts" / "train.py").read_text()
    assert "--obs-dim" not in train_py
    assert "--act-dim" not in train_py
