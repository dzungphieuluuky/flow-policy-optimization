"""The rollout budget is chunks, not steps: one rollout collects chunks_per_update chunks."""

import pytest

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, _rollout_budget


def _cfg(num_action_steps: int, chunks_per_update: int = 2048) -> FlowPolicyOptimizationConfig:
    return FlowPolicyOptimizationConfig(
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=num_action_steps,
        chunks_per_update=chunks_per_update,
    )


@pytest.mark.parametrize("num_action_steps", [1, 2, 4])
def test_one_rollout_collects_the_planned_number_of_chunks(num_action_steps):
    cfg = _cfg(num_action_steps)
    n_envs = 64
    full_wave_steps = cfg.chunks_per_update * num_action_steps
    chunks_per_env = _rollout_budget(cfg, n_envs, full_wave_steps)
    assert chunks_per_env * n_envs == cfg.chunks_per_update


def test_a_final_partial_rollout_covers_the_remaining_steps_in_one_wave():
    cfg = _cfg(4)
    n_envs = 64
    # 2000 steps remain; one chunk wave is 64 * 4 = 256 steps.
    chunks_per_env = _rollout_budget(cfg, n_envs, 2000)
    assert chunks_per_env == 8  # ceil(2000 / 256)
    assert chunks_per_env * n_envs * cfg.num_action_steps >= 2000


def test_a_tiny_remainder_still_collects_a_chunk():
    cfg = _cfg(4)
    assert _rollout_budget(cfg, 64, 1) == 1
