"""End-to-end wiring smoke: run 2 training steps on CPU and check the artifacts.

This pins no algorithm; it exists so a broken import, a bad config path or a
missing artifact fails loudly instead of greeting the user at 50k steps.
"""

import csv

import gymnasium as gym
import pytest

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.train import run_flow_policy_optimization_training


@pytest.mark.parametrize("seed", [0])
def test_two_steps_writes_checkpoint_and_csv(tmp_path, seed, monkeypatch):
    monkeypatch.setenv("FLOW_POLICY_OPTIMIZATION_DEVICE", "cpu")
    env = gym.make("Pendulum-v1")
    try:
        out = run_flow_policy_optimization_training(
            env,
            tmp_path,
            cfg=FlowPolicyOptimizationConfig(
                num_envs=4,
                chunks_per_update=8,
                chunks_per_minibatch=8,
                num_learning_epochs=1,
                num_mc_samples=2,
                num_steps_noise_start=2,
                hidden_dim=16,
                critic_hidden_dim=32,
                action_horizon=2,
                num_action_steps=2,
                num_obs_steps=2,
                eval_max_episodes=1,
                report_to="none",
            ),
            seed=seed,
            total_steps=2,
            eval_steps=10**9,
        )
    finally:
        env.close()

    assert out["global_step"] >= 2
    assert (tmp_path / "checkpoints" / "final" / "checkpoint.pt").is_file()
    csv_path = tmp_path / "eval_history.csv"
    assert csv_path.is_file()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows and all(float(r["return"]) == float(r["return"]) for r in rows)  # not NaN