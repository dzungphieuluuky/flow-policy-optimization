"""The collector's cached env state must not survive an eval on the same env.

A single-env run evaluates on the env the collector drives. The evaluator resets
it and runs whole episodes, so the carry (windows and autoreset flags) describes
a state that no longer exists.

The carry lives on the host rollout buffer, the object ``collect_chunks`` reads
it from. ``RolloutBuffers`` used to hold a second copy that nothing read, so the
invalidation cleared the copy and the collector kept the stale one.
"""

from __future__ import annotations

import dataclasses

import torch

from myfpo.rollout import EnvCarry, RolloutBuffer, RolloutBuffers
from myfpo.training import invalidate_shared_env_carry


def make_host() -> RolloutBuffer:
    """A real host buffer: the object the collector reads the carry from."""
    return RolloutBuffer(
        max_chunks=4, max_steps=8, obs_steps=2, obs_dim=3,
        action_horizon=2, action_dim=2,
    )


class Policy:
    def __init__(self, host=None):
        self.rollout_buffers = RolloutBuffers(host=host)


class Evaluator:
    def __init__(self, env):
        self.env = env


def make_carry() -> EnvCarry:
    return EnvCarry(windows=torch.ones(1, 2, 3), awaiting=[True])


def test_the_carry_is_dropped_when_eval_ran_on_the_training_env():
    train_env = object()
    host = make_host()
    host.carry = make_carry()

    assert invalidate_shared_env_carry(
        Policy(host), Evaluator(train_env), train_env
    ) is True
    assert host.carry is None


def test_a_separate_eval_env_keeps_the_carry():
    host = make_host()
    carry = make_carry()
    host.carry = carry

    assert invalidate_shared_env_carry(
        Policy(host), Evaluator(object()), object()
    ) is False
    assert host.carry is carry


def test_a_policy_without_rollout_buffers_is_left_alone():
    train_env = object()
    assert invalidate_shared_env_carry(object(), Evaluator(train_env), train_env) is False


def test_a_policy_that_has_not_collected_yet_is_left_alone():
    train_env = object()
    assert invalidate_shared_env_carry(
        Policy(None), Evaluator(train_env), train_env
    ) is False


def test_the_buffers_keep_no_second_copy_of_the_carry():
    """One holder. A field here is a trap: nothing reads it."""
    names = {f.name for f in dataclasses.fields(RolloutBuffers)}
    assert "carry" not in names
