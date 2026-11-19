"""Builders shared by several test modules.

The policy factory and the training batch were copy-pasted into three files
each. They live here once so a change to the batch keys reaches every test
that builds one.
"""

from __future__ import annotations

import torch

import numpy as np

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, FlowPolicyOptimizationPolicy

OBS_DIM = 5
ACT_DIM = 2


def cfg(**kw) -> FlowPolicyOptimizationConfig:
    return FlowPolicyOptimizationConfig(**kw)


def make_test_policy(**kw) -> FlowPolicyOptimizationPolicy:
    """A policy with the feature specs the suite uses. ``kw`` goes to the config."""
    kw_dims = {k: kw.pop(k) for k in ("obs_dim", "act_dim") if k in kw}
    obs_dim = kw_dims.get("obs_dim", OBS_DIM)
    act_dim = kw_dims.get("act_dim", ACT_DIM)
    low = np.full(act_dim, -1.0, dtype=np.float32)
    high = np.full(act_dim, 1.0, dtype=np.float32)
    return FlowPolicyOptimizationPolicy(cfg(**kw), obs_dim, act_dim, low, high)

def make_batch(
    policy: FlowPolicyOptimizationPolicy,
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
