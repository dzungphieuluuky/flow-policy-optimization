"""Unit tests for src/sjfpo/config.py.

Mode switches are read once at import; env-var semantics are tested via
importlib.reload under monkeypatch, with an autouse fixture restoring
clean-env state after every test.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo import config  # noqa: E402
from sjfpo.configuration_sjfpo import SJFPOConfig

ENV_VARS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "SJFPO_TINY_MATRIX",
)


@pytest.fixture(autouse=True)
def clean_module_state():
    """Every test starts from (and leaves behind) a clean-env reloaded module."""
    for var in ENV_VARS:
        os.environ.pop(var, None)
    importlib.reload(config)
    yield
    for var in ENV_VARS:
        os.environ.pop(var, None)
    importlib.reload(config)


def reload_with_env(monkeypatch: pytest.MonkeyPatch, **env: str):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    return importlib.reload(config)


def fake_version(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, str]) -> None:
    real = importlib.metadata.version

    def fake(dist: str, *args, **kwargs):
        if dist in overrides:
            return overrides[dist]
        return real(dist, *args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "version", fake)

    # 1. Run matrix shape / uniqueness / seeds


def test_run_matrix_45_unique_tuples_and_seeds():
    assert len(config.RUN_MATRIX) == 45
    keys = [(s.method, s.env, s.seed) for s in config.RUN_MATRIX]
    assert len(keys) == len(set(keys)), "collision in (method, env, seed)"
    assert {s.seed for s in config.RUN_MATRIX} == {0, 1, 2, 3, 4}
    assert {s.method for s in config.RUN_MATRIX} == {"sjfpo", "ppo", "sac"}
    assert {s.env for s in config.RUN_MATRIX} == {
        "HalfCheetah-v5",
        "Walker2d-v5",
        "Hopper-v5",
    }

    # 2. Env registry: measured dims + prototype envs outside the matrix


def test_lookup_env_dims_measured():
    assert config.lookup_env_dims("Hopper-v5") == (11, 3)
    assert config.lookup_env_dims("Ant-v5") == (105, 8)
    assert config.lookup_env_dims("Humanoid-v5") == (348, 17)
    assert config.lookup_env_dims("AdroitHandDoor-v1") == (39, 28)


def test_lookup_env_dims_unknown_raises():
    with pytest.raises(KeyError):
        config.lookup_env_dims("NotAnEnv-v0")


def test_known_envs_extends_matrix_envs():
    assert set(config.ENVS) <= set(config.KNOWN_ENVS)
    assert len(config.RUN_MATRIX) == 45  # prototypes stay out of the default matrix


def test_every_config_field_non_null():
    for instance in (SJFPOConfig(), config.PPO_CFG, config.SAC_CFG):
        for field in dataclasses.fields(instance):
            if field.default is None:
                continue  # Hub-inherited optional (repo_id, ...) unset pre-push
            value = getattr(instance, field.name)
            assert value is not None, (
                f"{type(instance).__name__}.{field.name} must not be None"
            )


def test_sjfpo_config_rejects_bad_top_p():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_p_spectral_tokens"):
            SJFPOConfig(top_p_spectral_tokens=bad)


def test_sjfpo_config_documented_ranges():
    c = SJFPOConfig()
    assert c.num_mc_samples >= 2
    assert c.num_euler_steps >= 1
    assert c.timestep_scale > 0
    assert c.num_inference_steps >= 1
    assert c.chunks_per_update >= 1
    assert c.num_epochs >= 1
    assert c.optimizer_lr > 0
    assert c.optimizer_weight_decay >= 0
    assert 0 < c.gamma < 1
    assert 0 < c.gae_lambda < 1
    assert 0 < c.epsilon_clip < 1
    lo, hi = c.min_timestep, c.max_timestep
    assert 0 <= lo < hi <= 1
    assert c.vf_coef >= 0
    assert c.dit_depth >= 1
    assert c.dit_heads >= 1
    assert c.dit_mlp_ratio > 0
    assert c.dit_time_dim >= 8
    assert c.grad_clip_norm > 0
    assert c.n_obs_steps == 16
    assert c.action_horizon == 16
    assert c.executed_vs_predicted_ratio == 0.5
    assert c.top_p_spectral_tokens == 0.3
    assert c.hidden_dim > 0


def test_ppo_config_documented_ranges():
    c = config.PPO_CFG
    assert c.rollouts > 0
    assert c.learning_epochs >= 1
    assert c.mini_batches >= 1
    assert c.lr > 0
    assert c.kl_threshold > 0
    assert c.grad_norm_clip > 0
    assert 0 < c.discount < 1
    assert 0 < c.gae_lambda < 1
    assert 0 < c.ratio_clip < 1
    assert c.value_loss_scale > 0
    assert len(c.hidden) > 0 and all(h > 0 for h in c.hidden)


def test_sac_config_documented_ranges():
    c = config.SAC_CFG
    assert c.lr > 0
    assert c.batch_size >= 1
    assert c.memory_size > 0
    assert c.gradient_steps >= 1
    assert 0 < c.polyak < 1
    assert 0 < c.discount < 1
    assert c.initial_random_steps >= 0
    assert len(c.hidden) > 0 and all(h > 0 for h in c.hidden)

    # 3. assert_runtime(): dependency pins + precision flag


def test_assert_runtime_passes_on_pinned_env():
    config.assert_runtime()
    config.assert_runtime(allow_fp16=False)


def test_assert_runtime_gymnasium_below_1_0(monkeypatch):
    fake_version(monkeypatch, {"gymnasium": "0.29.1"})
    with pytest.raises(RuntimeError, match="gymnasium"):
        config.assert_runtime()


def test_assert_runtime_skrl_not_2_1_0(monkeypatch):
    fake_version(monkeypatch, {"skrl": "2.0.9"})
    with pytest.raises(RuntimeError, match="skrl"):
        config.assert_runtime()


def test_assert_runtime_rejects_fp16_request():
    with pytest.raises(RuntimeError, match="fp16|bf16"):
        config.assert_runtime(allow_fp16=True)

        # 4. Env-var semantics via importlib.reload under monkeypatch


def test_tiny_matrix_env_var_swaps_run_matrix(monkeypatch):
    m = reload_with_env(monkeypatch, SJFPO_TINY_MATRIX="1")
    assert m.TINY_MATRIX is True
    assert len(m.RUN_MATRIX) == 3
    for spec in m.RUN_MATRIX:
        assert spec.env == "Hopper-v5"
        assert spec.seed == 0
        assert spec.total_steps == 20_000
        assert spec.eval_steps == 10_000
        assert spec.save_steps == 10_000
    m_default = reload_with_env(monkeypatch)
    assert m_default.TINY_MATRIX is False
    assert len(m_default.RUN_MATRIX) == 45


def test_is_kaggle_true_for_any_value(monkeypatch):
    for value in ("notebook", "0"):
        m = reload_with_env(monkeypatch, KAGGLE_KERNEL_RUN_TYPE=value)
        assert m.IS_KAGGLE is True
    m = reload_with_env(monkeypatch)
    assert m.IS_KAGGLE is False

    # 5. validate_run_matrix(): duplicate-seed injection raises AssertionError


def test_validate_run_matrix_accepts_current_matrix():
    config.validate_run_matrix(config.RUN_MATRIX)


def test_duplicate_seed_injection_raises_assertion_error():
    duplicated = list(config.RUN_MATRIX) + [dataclasses.replace(config.RUN_MATRIX[0])]
    keys = [(s.method, s.env, s.seed) for s in duplicated]
    assert len(keys) != len(set(keys)), "fixture itself must contain a duplicate"
    with pytest.raises(AssertionError):
        config.validate_run_matrix(duplicated)
