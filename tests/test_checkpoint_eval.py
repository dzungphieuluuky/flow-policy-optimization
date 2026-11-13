"""Checkpoint payload and evaluator random-baseline plumbing."""

import pytest
import torch

from myfpo.checkpointing import (
    CHECKPOINT_FILE,
    ResumeError,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from myfpo.modeling import make_optimizer
from myfpo.training import sample_random_baseline
from fixtures import make_test_policy


# ---- Checkpoint round trip (from test_checkpoint_roundtrip) ----

def test_a_round_trip_restores_weights_counters_and_ema(tmp_path):
    policy = make_test_policy()
    policy.update_ema(0)
    optimizer = make_optimizer(policy)
    policy.step_count = 7
    save_policy_checkpoint(
        tmp_path,
        policy=policy,
        optimizer=optimizer,
        global_step=42,
        ema_state=policy.ema_state_dict(),
    )

    loaded, loaded_optimizer, counters = load_policy_checkpoint(tmp_path)
    assert counters == {"global_step": 42, "step_count": 7}
    for name, value in policy.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert len(loaded_optimizer.param_groups) == len(optimizer.param_groups)


def test_an_old_style_directory_says_what_is_missing(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ResumeError, match=CHECKPOINT_FILE):
        load_policy_checkpoint(tmp_path)


def test_a_policy_only_checkpoint_loads_without_an_optimizer(tmp_path):
    policy = make_test_policy()
    save_policy_checkpoint(tmp_path, policy=policy, optimizer=None, global_step=3)

    loaded, optimizer, counters = load_policy_checkpoint(tmp_path)
    assert optimizer is None
    assert counters["global_step"] == 3
    for name, value in policy.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name


# ---- Random baseline (from test_random_baseline) ----

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
