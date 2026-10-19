"""Builders shared by several test modules.

The policy factory and the training batch were copy-pasted into three files
each. They live here once so a change to the batch keys reaches every test
that builds one.
"""

from __future__ import annotations

import dataclasses

import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import SJFPOConfig
from myfpo.modeling import SJFPOPolicy

OBS_DIM = 5
ACT_DIM = 2


def make_policy(**kw) -> SJFPOPolicy:
    """A policy with the feature specs the suite uses. ``kw`` goes to the config."""
    cfg = dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS_DIM,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT_DIM,))},
    )
    return SJFPOPolicy(cfg)


def make_batch(
    policy: SJFPOPolicy,
    *,
    obs_dim: int,
    action_dim: int,
    horizon: int,
    n: int = 2,
) -> dict[str, torch.Tensor]:
    """One training batch with the paired MC draws the policy expects."""
    mc = policy.config.num_mc_samples
    return {
        "observation.state": torch.zeros(n, policy.config.n_obs_steps, obs_dim),
        "action": torch.zeros(n, horizon, action_dim),
        "action_is_pad": torch.zeros(n, horizon, dtype=torch.bool),
        "normalised_advantages": torch.ones(n),
        "raw_returns": torch.ones(n),
        "mc_t": torch.ones(n, mc, horizon, 1),
        "mc_eps": torch.randn(n, mc, horizon, action_dim),
        "behavior_cfm_loss": torch.ones(n, mc),
        "behavior_cfm_loss_token": torch.ones(n, mc, horizon),
    }
