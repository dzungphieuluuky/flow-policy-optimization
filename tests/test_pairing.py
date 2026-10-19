"""Old and new CFM losses must share one (t, eps) pair batch."""

import dataclasses

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import SJFPOConfig
from myfpo.fpo_core import sample_timesteps
from myfpo.training import build_policy

OBS, ACT, HORIZON, MC = 6, 3, 4, 4


def _cfg(**kw):
    return dataclasses.replace(
        SJFPOConfig(
            action_horizon=HORIZON,
            num_mc_samples=MC,
            hidden_dim=32,
            time_embed_dim=16,
            dit_depth=1,
            dit_heads=2,
            n_obs_steps=2,
            compile=False,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )


def _policy():
    return build_policy(
        OBS, ACT, _cfg(), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def _pairs(policy, n):
    torch.manual_seed(0)
    t = sample_timesteps((n, MC, HORIZON), policy.config, "cpu", None).unsqueeze(-1)
    eps = torch.randn(n, MC, HORIZON, ACT)
    return t, eps


def _batch(policy, states, actions, t, eps):
    pre = policy.preprocessor({"observation.state": states})["observation.state"]
    pad = torch.zeros(states.shape[0], HORIZON, dtype=torch.bool)
    old, old_token = policy.compute_behavior_cfm_losses(pre, actions, t, eps, pad)
    return {
        "observation.state": states,
        "action": actions,
        "action_is_pad": pad,
        "normalised_advantages": torch.ones(states.shape[0]),
        "raw_returns": torch.ones(states.shape[0]),
        "mc_t": t,
        "mc_eps": eps,
        "behavior_cfm_loss": old,
        "behavior_cfm_loss_token": old_token,
    }


def test_same_weights_and_same_pairs_give_ratio_one():
    """Fresh snapshots agree, so any delta left is mispairing, not learning."""
    policy = _policy()
    n = 2
    torch.manual_seed(1)
    states = torch.randn(n, policy.config.n_obs_steps, OBS)
    actions = torch.randn(n, HORIZON, ACT)
    t, eps = _pairs(policy, n)
    _, info = policy(_batch(policy, states, actions, t, eps), compute_info=True)
    assert info["mean_log_ratio"] == 0.0
    assert info["log_ratio_max"] == 0.0


def test_swapped_pairs_move_the_ratio():
    """The identity above is sensitive: wrong pairs must break it."""
    policy = _policy()
    n = 2
    torch.manual_seed(1)
    states = torch.randn(n, policy.config.n_obs_steps, OBS)
    actions = torch.randn(n, HORIZON, ACT)
    t, eps = _pairs(policy, n)
    batch = _batch(policy, states, actions, t, eps)
    batch["mc_t"] = torch.rand_like(t)
    batch["mc_eps"] = torch.randn_like(eps)
    _, info = policy(batch, compute_info=True)
    assert info["log_ratio_max"] > 0.0


def test_a_misshapen_stored_timestep_raises():
    """The stored branch runs, so a bad stored shape cannot fall through."""
    policy = _policy()
    n = 2
    states = torch.zeros(n, policy.config.n_obs_steps, OBS)
    actions = torch.zeros(n, HORIZON, ACT)
    t, eps = _pairs(policy, n)
    batch = _batch(policy, states, actions, t, eps)
    batch["mc_t"] = torch.ones(n, MC, HORIZON + 1, 1)
    with pytest.raises(ValueError, match="mc_t shape"):
        policy(batch, compute_info=False)


def test_the_frozen_pass_is_deterministic():
    """Two old-loss passes on one pair batch agree bit for bit."""
    policy = _policy()
    n = 2
    torch.manual_seed(2)
    states = torch.randn(n, policy.config.n_obs_steps, OBS)
    actions = torch.randn(n, HORIZON, ACT)
    pre = policy.preprocessor({"observation.state": states})["observation.state"]
    pad = torch.zeros(n, HORIZON, dtype=torch.bool)
    t, eps = _pairs(policy, n)
    first = policy.compute_behavior_cfm_losses(pre, actions, t, eps, pad)
    second = policy.compute_behavior_cfm_losses(pre, actions, t, eps, pad)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
