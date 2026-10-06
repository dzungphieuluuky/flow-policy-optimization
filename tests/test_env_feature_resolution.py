"""Passing only ``--env`` must be enough to size and feature the policy.

The feature lookup lived in ``SJFPOConfig.__post_init__`` and read an
``env_name`` field the dataclass never had, so it never fired and its bare
``except Exception: pass`` hid that. Meanwhile ``validate()`` demanded
``robot_state_feature`` on a bare config, so ``SJFPOConfig()`` could not be
constructed at all and draccus reported the failure as "couldn't instantiate
TrainConfig" -- naming a missing feature rather than the real cause.

Dimensions now come from ``configs/envs.yaml`` via ``train.py:run_single``,
which also cross-checks them against the live gym spaces. These tests pin that
contract, plus the two failure modes worth a loud error: a drifted registry and
an explicitly mismatched override.
"""

import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_joint_policy.config import (  # noqa: E402
    ENV_REGISTRY_PATH,
    KNOWN_ENVS,
    get_observation_action_dim,
)
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig  # noqa: E402

# Tier "smoke": pure gymnasium, no mujoco, so these run anywhere. All have
# continuous (Box) action spaces -- see the note in configs/envs.yaml on why
# the Discrete-action toy envs were removed.
SMOKE_ENVS = ("Pendulum-v1", "Reacher-v5")


def _gym_env(env_id):
    gym = pytest.importorskip("gymnasium")
    return gym.make(env_id)


def test_a_bare_config_constructs_without_features():
    """The regression: this used to raise, breaking every CLI parse."""
    cfg = SJFPOConfig()
    assert cfg.input_features == {}
    assert cfg.output_features == {}
    # Feature requirement is still enforced, just later -- see the two tests below.
    with pytest.raises(ValueError, match="observation.state"):
        cfg.validate_features()


def test_the_env_name_is_not_a_config_field():
    """Auto-detection keyed on this never had a value to key on."""
    assert not hasattr(SJFPOConfig(), "env_name")


@pytest.mark.parametrize("env_id", SMOKE_ENVS)
def test_env_name_alone_resolves_registry_dims(env_id):
    obs_dim, action_dim = get_observation_action_dim(env_id)
    assert obs_dim > 0 and action_dim > 0


@pytest.mark.parametrize("env_id", SMOKE_ENVS)
def test_registry_dims_match_the_live_env_space(env_id):
    """This is the check train.py runs at startup; keep it true."""
    env = _gym_env(env_id)
    try:
        live = (
            int(math.prod(env.observation_space.shape)),
            int(math.prod(env.action_space.shape)),
        )
    finally:
        env.close()
    assert get_observation_action_dim(env_id) == live, (
        f"{env_id} registry dims disagree with the live gym space; "
        f"update {ENV_REGISTRY_PATH.name}"
    )


@pytest.mark.parametrize("env_id", SMOKE_ENVS)
def test_build_policy_attaches_features_from_dims_alone(env_id):
    """The runner's only inputs: an env name and its resolved dims."""
    from spectral_joint_policy.train_sjfpo import build_policy

    obs_dim, action_dim = get_observation_action_dim(env_id)
    cfg = SJFPOConfig()
    policy = build_policy(obs_dim=obs_dim, action_dim=action_dim, cfg=cfg, device="cpu")
    assert tuple(policy.config.robot_state_feature.shape) == (obs_dim,)
    assert tuple(policy.config.action_feature.shape) == (action_dim,)
    # dataclasses.replace inside build_policy: the caller's config is untouched.
    assert cfg.input_features == {}


def test_no_discrete_action_envs_are_registered():
    """Guards the bug that made CartPole/MountainCar unrunnable.

    train.py measures live action dims with math.prod(action_space.shape),
    which is 1 for a Discrete space (its shape is ()), not Discrete.n. An
    entry recording the true n therefore disagreed with the live space and
    tripped the startup check. SJFPO emits continuous actions regardless, so
    only Box-action envs belong in the registry at all.
    """
    gym = pytest.importorskip("gymnasium")
    for env_id in KNOWN_ENVS:
        entry = yaml.safe_load(ENV_REGISTRY_PATH.read_text())[env_id]
        if entry.get("requires"):
            continue  # optional extra, not installed in every environment
        try:
            env = gym.make(env_id)
        except Exception as exc:  # mujoco / robotics wheels may be absent
            pytest.skip(f"{env_id} unavailable: {exc}")
        try:
            space = env.action_space
        finally:
            env.close()
        assert not isinstance(
            space, gym.spaces.Discrete
        ), f"{env_id} has a Discrete action space; SJFPO needs continuous actions"
        assert int(math.prod(space.shape)) == entry["action_dim"], (
            f"{env_id} action_dim={entry['action_dim']} != live "
            f"{int(math.prod(space.shape))}"
        )


def test_every_registry_entry_declares_both_dims():
    """Keys are consumed positionally; a missing dim must not read as None."""
    registry = yaml.safe_load(ENV_REGISTRY_PATH.read_text())
    assert registry, f"{ENV_REGISTRY_PATH} is empty"
    for env_id, entry in registry.items():
        assert isinstance(entry.get("obs_dim"), int), f"{env_id}: obs_dim"
        assert isinstance(entry.get("action_dim"), int), f"{env_id}: action_dim"
    assert set(registry) == set(KNOWN_ENVS)


def test_the_removed_discrete_toy_envs_are_gone():
    """Keeps the removal honest against a stale clone or reverted edit."""
    registry = yaml.safe_load(ENV_REGISTRY_PATH.read_text())
    for env_id in ("CartPole-v1", "MountainCar-v0"):
        assert env_id not in registry
        assert env_id not in KNOWN_ENVS
        # The rationale must survive the deletion.
        assert env_id in ENV_REGISTRY_PATH.read_text()


def test_train_py_documents_underscore_flag_names():
    """draccus rejects --obs-dim; the old envs.yaml header advertised it."""
    train_py = (ROOT / "scripts" / "train.py").read_text()
    assert "--obs-dim" not in train_py
    assert "--act-dim" not in train_py
    # The registry remains the single source of dims; the CLI only overrides.
    assert "get_observation_action_dim(env_id)" in train_py