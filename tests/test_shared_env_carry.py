"""The rollout's carried windows must not survive an eval on the same env.

A single-env run evaluates on the env the collector drives. The evaluator resets
it and runs whole episodes, so the carried windows describe a state that no
longer exists. The lean runner detects the shared-env case with ``_shared_env``
and drops the carry after every eval window.
"""

from __future__ import annotations

import gymnasium as gym

from myfpo.myfpo import _shared_env, make_train_env


def test_a_single_wrapper_env_shares_the_sub_env_with_the_evaluator():
    env = gym.make("Pendulum-v1")
    train_env = make_train_env(env, "Pendulum-v1", 1)
    try:
        assert _shared_env(train_env, env) is True
    finally:
        env.close()


def test_independent_vec_envs_do_not_share():
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(4)],
                                    autoreset_mode="SameStep")
    other = gym.make("Pendulum-v1")
    try:
        assert _shared_env(envs, other) is False
        assert _shared_env(envs, envs.envs[0]) is True
    finally:
        envs.close()
        other.close()


def test_a_plain_object_env_is_its_own_shared_env():
    env = object()
    assert _shared_env(env, env) is True
    assert _shared_env(env, object()) is False
