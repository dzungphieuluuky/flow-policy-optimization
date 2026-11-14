"""One advantage scale for one update.

The rollout standardizes the advantages over the whole batch, so every
minibatch in the update reads one scale. The rollout-level pass is the
only standardization in the update.
"""

from __future__ import annotations

import numpy as np
import torch

from myfpo.myfpo import MyfpoConfig, MyfpoPolicy, make_optimizer, update
import myfpo.myfpo as m

N = 2


def _cfg(**kw) -> MyfpoConfig:
    base = dict(
        hidden_dim=16,
        action_horizon=2,
        num_action_steps=2,
        num_obs_steps=2,
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


def test_the_rollout_normalization_reaches_the_actor_by_default(monkeypatch):
    cfg = _cfg()
    low = np.full(3, -1.0, dtype=np.float32)
    high = np.full(3, 1.0, dtype=np.float32)
    policy = MyfpoPolicy(cfg, 5, 3, low, high)
    optimizer = make_optimizer(policy)
    seen: list[torch.Tensor] = []

    original = m.compute_loss

    def spy(policy_, batch_, device):
        seen.append(batch_["normalized_advantages"].clone())
        return original(policy_, batch_, device)

    monkeypatch.setattr(m, "compute_loss", spy)
    update(policy, optimizer, _batch(cfg), cfg)
    assert seen, "the update read no minibatch"
    for advantages in seen:
        # The minibatch rows are shuffled, so the order does not matter.
        assert torch.equal(
            torch.sort(advantages).values, torch.tensor([-0.5, 0.5])
        )
