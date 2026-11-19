"""flow_policy_optimization: minimal CleanRL / RSL-RL-style FPO training.

Files: ``config`` (config, normalization, bounds), ``networks`` (modules),
``objective`` (CFM / ratio / chunk-GAE math), ``policy``, ``rollout``
(buffer, SameStep collector, rollout driver), ``update`` (optimizer,
minibatch update), ``train`` (evaluators, checkpoints, the training loop).
The plain PPO baseline lives in ``flow_policy_optimization.ppo``.
"""

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy
from flow_policy_optimization.train import run_flow_policy_optimization_training
from flow_policy_optimization.ppo import run_ppo_training

__all__ = [
    "FlowPolicyOptimizationConfig",
    "FlowPolicyOptimizationPolicy",
    "run_flow_policy_optimization_training",
    "run_ppo_training",
]
