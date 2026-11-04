"""The random baseline goes through the evaluator's own random policy.

The old code sampled ``action_space`` directly and guessed at
``single_action_space``/``unwrapped``. That breaks the batched torch path, whose
policy is a raw chunk, not one action per step.
"""

from __future__ import annotations

from myfpo.training import sample_random_baseline


class StubEvaluator:
    """Records what ``sample_random_baseline`` asks it to do."""

    def __init__(self, env, returns, random):
        self.env = env
        self.returns = returns
        self.random = random
        self.log: dict[str, object] = {}

    def clone(self, max_episodes: int):
        self.log["max_episodes"] = max_episodes
        return self

    def random_policy(self, env):
        self.log["random_env"] = env
        return self.random

    def evaluate(self, policy, *, out_dir=None):
        self.log["policy"] = policy
        self.log["out_dir"] = out_dir
        return [{"return": value} for value in self.returns]


def test_the_baseline_uses_the_evaluator_random_policy():
    env = object()
    random = object()
    evaluator = StubEvaluator(env, [1.0, 3.0], random)

    assert sample_random_baseline(evaluator, max_episodes=5) == 2.0
    assert evaluator.log["policy"] is random
    assert evaluator.log["random_env"] is env
    assert evaluator.log["max_episodes"] == 5
    assert evaluator.log["out_dir"] is None


def test_an_empty_baseline_scores_zero():
    evaluator = StubEvaluator(object(), [], object())
    assert sample_random_baseline(evaluator) == 0.0
