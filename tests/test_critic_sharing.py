"""``critic_share_encoder`` reproduces FPO++'s actor-critic sharing."""

import dataclasses

import pytest
import torch

from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import SJFPOConfig
from myfpo.critic import Critic
from myfpo.training import build_policy

OBS, ACT, HORIZON, HIDDEN = 6, 3, 8, 32


def _cfg(**kw):
    return dataclasses.replace(
        SJFPOConfig(
            action_horizon=HORIZON,
            hidden_dim=HIDDEN,
            critic_hidden_dim=None,
            dit_depth=1,
            dit_heads=2,
            n_obs_steps=2,
            num_mc_samples=2,
            compile=False,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **kw,
    )


def _policy(**kw):
    return build_policy(
        OBS, ACT, _cfg(**kw), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def test_the_shared_critic_takes_one_embedding_not_the_window():
    critic = Critic(OBS, HIDDEN, ACT, 2, HORIZON, 2, condition_dim=HIDDEN)
    cond = torch.randn(4, HIDDEN)
    actions = torch.randn(4, HORIZON, ACT)
    frac = torch.full((4,), 0.5)

    assert critic(cond, actions, frac).shape == (2, 4)
    assert critic.input_dim == HIDDEN + HORIZON * ACT + 1, (
        "the first layer must be the embedding width, not n_obs_steps * obs_dim"
    )


def test_the_shared_critic_rejects_an_obs_window():
    """Otherwise a caller could pass the window and get a silent shape error."""
    critic = Critic(OBS, HIDDEN, ACT, 2, HORIZON, 2, condition_dim=HIDDEN)
    with pytest.raises(ValueError, match="condition must have shape"):
        critic(torch.randn(4, 2, OBS), torch.randn(4, HORIZON, ACT), torch.full((4,), 0.5))


def test_the_default_critic_still_takes_the_window():
    policy = _policy()
    assert policy.config.critic_share_encoder is False
    assert policy.critic.condition_dim is None
    out = policy.critic(
        torch.randn(4, 2, OBS), torch.randn(4, HORIZON, ACT), torch.full((4,), 0.5)
    )
    assert out.shape == (policy.config.num_critics, 4)


def test_both_settings_pass_a_rollout_and_a_forward():
    """The flag only changes what the critic reads, so both must train."""
    for share in (False, True):
        policy = _policy(critic_share_encoder=share)
        batch = {
            "observation.state": torch.randn(4, 2, OBS),
            "action": torch.randn(4, HORIZON, ACT),
            "action_is_pad": torch.zeros(4, HORIZON, dtype=torch.bool),
            "normalised_advantages": torch.randn(4),
            "raw_returns": torch.randn(4),
            "mc_t": torch.rand(4, 2, HORIZON, 1),
            "mc_eps": torch.randn(4, 2, HORIZON, ACT),
        }
        states = policy.preprocessor({"observation.state": batch["observation.state"]})["observation.state"]
        old, old_tok = policy.compute_behavior_cfm_losses(
            states, batch["action"], batch["mc_t"], batch["mc_eps"], batch["action_is_pad"]
        )
        batch["behavior_cfm_loss"] = old
        batch["behavior_cfm_loss_token"] = old_tok

        loss, _ = policy(batch, compute_info=False, check_finite=False)
        loss.backward()  # already unpacked: forward returns (loss, info).
        assert torch.isfinite(loss), share
        assert policy.critic.condition_dim == (HIDDEN if share else None), share


def test_the_encoder_gets_gradient_from_the_critic_too():
    """Sharing is not a dead reference: the critic must reach the encoder."""
    policy = _policy(critic_share_encoder=True)
    batch = {
        "observation.state": torch.randn(4, 2, OBS),
        "action": torch.randn(4, HORIZON, ACT),
        "action_is_pad": torch.zeros(4, HORIZON, dtype=torch.bool),
        "normalised_advantages": torch.randn(4),
        "raw_returns": torch.randn(4),
        "mc_t": torch.rand(4, 2, HORIZON, 1),
        "mc_eps": torch.randn(4, 2, HORIZON, ACT),
    }
    states = policy.preprocessor({"observation.state": batch["observation.state"]})["observation.state"]
    old, old_tok = policy.compute_behavior_cfm_losses(
        states, batch["action"], batch["mc_t"], batch["mc_eps"], batch["action_is_pad"]
    )
    batch["behavior_cfm_loss"] = old
    batch["behavior_cfm_loss_token"] = old_tok

    policy(batch, compute_info=False, check_finite=False)[0].backward()
    grads = [p.grad for p in policy.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder gradient at all"
    assert any(float(g.abs().sum()) > 0 for g in grads), "encoder gradient is all zero"