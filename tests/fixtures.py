"""Builders shared by several test modules.

The policy factory and the training batch were copy-pasted into three files
each. They live here once so a change to the batch keys reaches every test
that builds one.
"""

from __future__ import annotations

import torch

from myfpo.configuration import MyfpoConfig
from myfpo.modeling import MyfpoPolicy

OBS_DIM = 5
ACT_DIM = 2


def cfg(
    *, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, **kw
) -> MyfpoConfig:
    """A config with the dims the suite uses. ``kw`` goes to the fields."""
    return MyfpoConfig(obs_dim=obs_dim, action_dim=act_dim, **kw)


def make_test_policy(**kw) -> MyfpoPolicy:
    """A policy with the feature specs the suite uses. ``kw`` goes to the config."""
    return MyfpoPolicy(cfg(**kw))

def make_batch(
    policy: MyfpoPolicy,
    *,
    obs_dim: int,
    action_dim: int,
    horizon: int,
    n: int = 2,
) -> dict[str, torch.Tensor]:
    """One training batch with the paired MC draws the policy expects."""
    mc = policy.config.num_mc_samples
    return {
        "observation.state": torch.zeros(n, policy.config.num_obs_steps, obs_dim),
        "action": torch.zeros(n, horizon, action_dim),
        "action_is_pad": torch.zeros(n, horizon, dtype=torch.bool),
        "chunk_truncated": torch.zeros(n, dtype=torch.bool),
        "normalized_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, mc, horizon, 1),
        "mc_eps": torch.randn(n, mc, horizon, action_dim),
        "behavior_cfm_loss": torch.ones(n, mc),
        "behavior_cfm_loss_token": torch.ones(n, mc, horizon),
    }
