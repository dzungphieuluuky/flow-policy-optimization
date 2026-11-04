"""The collector's cached env state must not survive an eval on the same env.

A single-env run evaluates on the env the collector drives. The evaluator resets
it and runs whole episodes, so the carry (windows and autoreset flags) describes
a state that no longer exists.
"""

from __future__ import annotations

from myfpo.training import invalidate_shared_env_carry


class Buffers:
    def __init__(self):
        self.carry = None


class Policy:
    def __init__(self):
        self.rollout_buffers = Buffers()


class Evaluator:
    def __init__(self, env):
        self.env = env


def test_the_carry_is_dropped_when_eval_ran_on_the_training_env():
    train_env = object()
    policy = Policy()
    policy.rollout_buffers.carry = object()

    assert invalidate_shared_env_carry(policy, Evaluator(train_env), train_env) is True
    assert policy.rollout_buffers.carry is None


def test_a_separate_eval_env_keeps_the_carry():
    carry = object()
    policy = Policy()
    policy.rollout_buffers.carry = carry

    assert invalidate_shared_env_carry(policy, Evaluator(object()), object()) is False
    assert policy.rollout_buffers.carry is carry


def test_a_policy_without_rollout_buffers_is_left_alone():
    train_env = object()
    assert invalidate_shared_env_carry(object(), Evaluator(train_env), train_env) is False
