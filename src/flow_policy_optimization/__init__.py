"""flow_policy_optimization: lean, single-file FPO-style training (CleanRL / RSL-RL style).

The public surface is small on purpose. ``FlowPolicyOptimizationConfig``, ``FlowPolicyOptimizationPolicy``,
``run_flow_policy_optimization_training`` and the PPO baseline entry in ``flow_policy_optimization.ppo``.
"""

from flow_policy_optimization.flow_policy_optimization import (
    FlowPolicyOptimizationConfig,
    FlowPolicyOptimizationPolicy,
    run_flow_policy_optimization_training,
)
from flow_policy_optimization.ppo import run_ppo_training

__all__ = [
    "FlowPolicyOptimizationConfig",
    "FlowPolicyOptimizationPolicy",
    "run_flow_policy_optimization_training",
    "run_ppo_training",
]
