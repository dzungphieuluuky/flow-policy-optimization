"""Checkpoint payload and evaluator random-baseline plumbing."""

import pytest
import torch

from myfpo.myfpo import load_checkpoint, make_optimizer, sample_random_baseline, save_checkpoint
from fixtures import make_test_policy


# ---- Checkpoint round trip ----

def test_a_round_trip_restores_weights_the_optimizer_and_the_step(tmp_path):
    policy = make_test_policy()
    optimizer = make_optimizer(policy)

    # one real step so the optimizer carries state
    cfg = policy.config
    batch = {
        "observation.state": torch.randn(2, cfg.num_obs_steps, 5),
        "action": torch.tanh(torch.randn(2, cfg.action_horizon, 2)),
        "action_is_pad": torch.zeros(2, cfg.action_horizon, dtype=torch.bool),
        "normalized_advantages": torch.tensor([1.0, -1.0]),
        "raw_returns": torch.tensor([2.0, 0.5]),
        "mc_t": torch.rand(2, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(2, cfg.num_mc_samples, cfg.action_horizon, 2),
        "behavior_cfm_loss": torch.rand(2, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.rand(2, cfg.num_mc_samples, cfg.action_horizon),
    }
    from myfpo.myfpo import compute_loss

    loss, _ = compute_loss(policy, batch, torch.device("cpu"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    step = 42
    save_checkpoint(tmp_path, "best", policy, optimizer, step)

    fresh = make_test_policy()
    fresh_optimizer = make_optimizer(fresh)
    loaded_step = load_checkpoint(tmp_path / "checkpoints" / "best", fresh, fresh_optimizer)
    assert loaded_step == step
    for name, value in policy.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name]), name
    assert len(fresh_optimizer.param_groups) == len(optimizer.param_groups)


def test_a_missing_checkpoint_says_what_is_missing(tmp_path):
    policy = make_test_policy()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "checkpoints" / "nope", policy)


def test_a_policy_only_checkpoint_loads_without_an_optimizer(tmp_path):
    policy = make_test_policy()
    save_checkpoint(tmp_path, "step_3", policy, optimizer=None, global_step=3)

    fresh = make_test_policy()
    step = load_checkpoint(tmp_path / "checkpoints" / "step_3", fresh, optimizer=None)
    assert step == 3
    for name, value in policy.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name]), name


# ---- Random baseline ----

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
