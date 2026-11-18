"""flow_policy_optimization: lean, single-file FPO-style training (lean reformulation).

The public surface is small on purpose. ``FlowPolicyOptimizationConfig``, ``FlowPolicyOptimizationPolicy``,
``run_flow_policy_optimization_training`` and the env registry; the PPO baseline entry and its
config live in ``flow_policy_optimization.ppo``.
"""

from flow_policy_optimization.flow_policy_optimization import (
    KNOWN_ENVS,
    FlowPolicyOptimizationConfig,
    FlowPolicyOptimizationPolicy,
    Evaluator,
    get_observation_action_dim,
    run_flow_policy_optimization_training,
)
from flow_policy_optimization.ppo import run_ppo_training

__all__ = [
    "KNOWN_ENVS",
    "Evaluator",
    "FlowPolicyOptimizationConfig",
    "FlowPolicyOptimizationPolicy",
    "get_observation_action_dim",
    "run_flow_policy_optimization_training",
    "run_ppo_training",
]
