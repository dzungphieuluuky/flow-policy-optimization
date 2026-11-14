"""myfpo: lean, single-file FPO-style training (lean reformulation).

The public surface is small on purpose. ``MyfpoConfig``, ``MyfpoPolicy``,
``run_myfpo_training`` and the env registry; the PPO baseline entry and its
config live in ``myfpo.ppo``.
"""

from myfpo.myfpo import (
    KNOWN_ENVS,
    MyfpoConfig,
    MyfpoPolicy,
    Evaluator,
    get_observation_action_dim,
    run_myfpo_training,
)
from myfpo.ppo import run_ppo_training

__all__ = [
    "KNOWN_ENVS",
    "Evaluator",
    "MyfpoConfig",
    "MyfpoPolicy",
    "get_observation_action_dim",
    "run_myfpo_training",
    "run_ppo_training",
]
