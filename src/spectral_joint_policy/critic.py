"""Ensemble chunk-Q critic."""

from __future__ import annotations

import torch
from torch import nn

from spectral_joint_policy.layers import CondMLP, modulate

class CriticMember(nn.Module):
    """One Q(s, a_{t:t+h}) head: FiLM from the chunk, so Q is not additive in
    (s, a). QC/DQC concatenate the chunk instead; this departure is deliberate."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_obs_steps: int,
        action_horizon: int,
    ):
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.state_enc = CondMLP(n_obs_steps * obs_dim, hidden_dim, hidden_dim)
        self.act_enc = CondMLP(action_horizon * action_dim, hidden_dim, 2 * hidden_dim)
        self.head = CondMLP(hidden_dim, hidden_dim, 1)
        # Zero the FiLM head so the chunk starts as identity (FiLM = adaLN-Zero):
        # scale==0, shift==0, and the state feature passes through unchanged.
        nn.init.constant_(self.act_enc.fc2.weight, 0.0)
        nn.init.constant_(self.act_enc.fc2.bias, 0.0)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``[B, T, obs]`` and ``[B, H, action_dim]`` -> ``[B]``."""
        h = self.state_enc(states.flatten(1))
        shift, scale = self.act_enc(actions.flatten(1)).chunk(2, dim=-1)
        # Bound the gain so FiLM cannot flip the sign of the state feature.
        scale = scale.clamp(-2.0, 2.0)
        return self.head(modulate(h, shift, scale)).squeeze(-1)


class Critic(nn.Module):
    """Q(s_t, a_{t:t+h}) over the full window; the chunk is the executed action
    sequence, so Q is the value the QC target backs up to. Returns
    ``[num_critics, B]``; the default ``num_critics=2`` is a real ensemble,
    so the caller must aggregate: members stay fully independent, and use
    ``min`` for the backup, ``mean`` for the value loss -- never ``min``
    inside the loss."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        n_obs_steps: int,
        action_horizon: int,
        num_critics: int = 2,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.num_critics = num_critics
        self.networks = nn.ModuleList(
            [
                CriticMember(
                    obs_dim, action_dim, hidden_dim, n_obs_steps, action_horizon
                )
                for _ in range(num_critics)
            ]
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """``[B, T, obs]``, ``[B, H, action_dim]`` -> ``[num_critics, B]``."""
        if states.dim() != 3 or states.shape[2] != self.obs_dim:
            raise ValueError(f"states must have shape [B, {self.n_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if states.shape[1] != self.n_obs_steps:
            raise ValueError(f"states has {states.shape[1]} time steps but the critic was built for n_obs_steps={self.n_obs_steps}")
        if actions.dim() != 3:
            raise ValueError(f"actions must have shape [B, H, action_dim], got {tuple(actions.shape)}")
        if actions.shape[2] != self.action_dim:
            raise ValueError(f"actions feature dimension must be {self.action_dim}, got {actions.shape[2]}")
        if actions.shape[0] != states.shape[0]:
            raise ValueError(f"actions batch {actions.shape[0]} != states batch {states.shape[0]}")
        if actions.shape[1] != self.action_horizon:
            raise ValueError(f"actions has {actions.shape[1]} steps but the critic was built for action_horizon={self.action_horizon}")
        return torch.stack([net(states, actions) for net in self.networks])
