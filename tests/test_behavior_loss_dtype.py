"""The behavior loss must read the action in fp32, like the training loss does.

The training pass reloads the action in fp32 and the noise and timestep in the
storage dtype. The rollout used to round the action to the storage dtype before
computing ``cfm_loss_old``, so on a bf16 storage device the ratio's denominator
came from a different flow point than its numerator.
"""

from __future__ import annotations

import dataclasses

import torch

import myfpo.modeling as modeling
from myfpo.configuration import MyfpoConfig
from myfpo.training import make_policy

OBS, ACT, HORIZON, N = 5, 2, 4, 1


def _policy():
    cfg = dataclasses.replace(
        MyfpoConfig(),
        obs_dim=OBS,
        action_dim=ACT,
        hidden_dim=32,
        dit_depth=1,
        dit_heads=2,
        num_obs_steps=4,
        action_horizon=HORIZON,
        num_action_steps=HORIZON,
    )
    return make_policy(
        OBS, ACT, cfg, device="cpu", action_low=[-1.0] * ACT, action_high=[1.0] * ACT
    )


def _inputs(action_value: float):
    policy = _policy()
    mc = policy.config.num_mc_samples
    states = torch.zeros(N, policy.config.num_obs_steps, OBS)
    actions = torch.zeros(N, HORIZON, ACT)
    actions[0, 0, 0] = action_value
    t = torch.full((N, mc, HORIZON, 1), 0.5)
    eps = torch.zeros(N, mc, HORIZON, ACT)
    pad = torch.zeros(N, HORIZON, dtype=torch.bool)
    return policy, states, actions, t, eps, pad


def test_the_action_is_not_rounded_to_the_storage_dtype(monkeypatch):
    """Two actions that round to the same bf16 must still give different losses."""
    monkeypatch.setattr(modeling, "resolve_storage_dtype", lambda *_: torch.bfloat16)
    first, states, actions, t, eps, pad = _inputs(1.0)
    rounded = actions.clone()
    rounded[0, 0, 0] = 1.0 + 1e-4  # bf16 rounds this back to 1.0
    assert rounded[0, 0, 0].to(torch.bfloat16).float().item() == 1.0

    loss_a, _ = first.compute_behavior_cfm_losses(states, actions, t, eps, pad)
    loss_b, _ = first.compute_behavior_cfm_losses(states, rounded, t, eps, pad)
    assert loss_a.dtype == torch.float32
    assert not torch.equal(loss_a, loss_b)


def test_the_behavior_loss_matches_the_training_pass_inputs():
    """A fp32 action gives an fp32 loss even when the storage dtype is bf16."""
    policy, states, actions, t, eps, pad = _inputs(1.0)
    loss, loss_token = policy.compute_behavior_cfm_losses(
        states, actions, t, eps, pad
    )
    assert loss.dtype == torch.float32
    assert loss_token.dtype == torch.float32
