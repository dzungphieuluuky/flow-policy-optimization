"""One advantage scale for one update.

The rollout standardizes the advantages over the whole batch, so every
minibatch in the update reads one scale. The rollout-level pass is the
only standardization in the update.
"""

from __future__ import annotations

import torch

from myfpo.config import MyfpoConfig
from myfpo.train import (
    make_optimizer,
    make_policy,
    policy_update_step,
    split_grad_params,
)

N = 2


def _cfg(**kw) -> MyfpoConfig:
    base = dict(
        obs_dim=5,
        action_dim=3,
        action_horizon=2,
        num_action_steps=2,
        num_obs_steps=2,
        hidden_dim=16,
        num_mc_samples=2,
        num_learning_epochs=1,
        chunks_per_update=2,
        chunks_per_minibatch=2,
    )
    base.update(kw)
    return MyfpoConfig(**base)


def _batch(cfg) -> dict:
    return {
        "observation.state": torch.zeros(N, cfg.num_obs_steps, 5),
        "action": torch.zeros(N, cfg.action_horizon, 3),
        "action_is_pad": torch.zeros(N, cfg.action_horizon, dtype=torch.bool),
        "chunk_truncated": torch.zeros(N, dtype=torch.bool),
        # The two differ on purpose: the actor must read one of them, and the
        # test tells them apart.
        "raw_advantages": torch.tensor([1.0, 2.0]),
        "normalized_advantages": torch.tensor([0.5, -0.5]),
        "raw_returns": torch.ones(N),
        "mc_t": torch.ones(N, cfg.num_mc_samples, cfg.action_horizon, 1),
        "mc_eps": torch.randn(N, cfg.num_mc_samples, cfg.action_horizon, 3),
        "behavior_cfm_loss": torch.ones(N, cfg.num_mc_samples),
        "behavior_cfm_loss_token": torch.ones(N, cfg.num_mc_samples, cfg.action_horizon),
    }


def _advantages_the_actor_reads(cfg):
    """The advantages the update hands to the loss, per minibatch."""
    policy = make_policy(5, 3, cfg, device="cpu")
    optimizer = make_optimizer(policy)
    seen: list[torch.Tensor] = []
    original = policy.compute_loss_and_probes

    def spy(minibatch, *args, **kwargs):
        seen.append(minibatch["normalized_advantages"].clone())
        return original(minibatch, *args, **kwargs)

    policy.compute_loss_and_probes = spy
    policy_update_step(
        policy, optimizer, _batch(cfg), [object()], cfg, split_grad_params(policy)
    )
    return seen


def test_the_rollout_normalization_reaches_the_actor_by_default():
    cfg = _cfg()
    seen = _advantages_the_actor_reads(cfg)
    assert seen, "the update read no minibatch"
    for advantages in seen:
        # The minibatch rows are shuffled, so the order does not matter.
        assert torch.equal(
            torch.sort(advantages).values, torch.tensor([-0.5, 0.5])
        )

