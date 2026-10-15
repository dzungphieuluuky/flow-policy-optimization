"""Ensemble chunk-Q critic."""

from __future__ import annotations

import torch
from torch import nn



class CriticMember(nn.Module):
    """One Q(s, a_{t:t+h}) head: concat MLP, no FiLM modulation. Keeps the
    same input contracts but drops the deliberate non-additive conditioning.

    ``condition_on_exec_length`` appends the executed fraction of the chunk as a
    final input. Without it the head cannot distinguish two chunks that share an
    executed prefix but differ in the tail that was planned and never run, yet
    it is regressed onto the same return.

    ``condition_dim`` is the shared-encoder option. FPO++'s chunk-level critic
    consumes ``actor_module.model.encode_observations(...)`` -- the actor's own
    observation encoder (``finetune_online_rl.py:1177``) -- so with it set the
    head takes one ``[B, condition_dim]`` vector instead of an obs window.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_obs_steps: int,
        action_horizon: int,
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        condition_dim: int | None = None,
    ):
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        state_dim = (
            condition_dim if condition_dim is not None else n_obs_steps * obs_dim
        )
        if condition_dim is not None and n_obs_steps != 1:
            # The obs window is not read in this mode; refuse the ambiguity
            # rather than silently ignoring a caller that passes one.
            raise ValueError(
                f"condition_dim is set, so n_obs_steps must be 1 (unused), got "
                f"{n_obs_steps}"
            )
        self.input_dim = state_dim + action_horizon * action_dim + (
            1 if condition_on_exec_length else 0
        )
        in_dim = self.input_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, T, obs]`` (or ``[B, condition_dim]``) and ``[B, H, action_dim]`` -> ``[B]``."""
        if self.condition_dim is not None:
            if states.dim() != 2 or states.shape[1] != self.condition_dim:
                raise ValueError(
                    f"condition must have shape [B, {self.condition_dim}], got "
                    f"{tuple(states.shape)}"
                )
        elif states.dim() != 3 or states.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"states must have shape [B, {self.n_obs_steps}, {self.obs_dim}], "
                f"got {tuple(states.shape)}"
            )
        parts = [states.flatten(1), actions.flatten(1)]
        if self.condition_on_exec_length:
            if exec_length_frac is None:
                raise ValueError(
                    "exec_length_frac is required when condition_on_exec_length=True; it "
                    "tells the critic how much of the planned chunk was executed"
                )
            if exec_length_frac.dim() != 1 or exec_length_frac.shape[0] != states.shape[0]:
                raise ValueError(
                    f"exec_length_frac must be [B] matching states, got {tuple(exec_length_frac.shape)}"
                )
            parts.append(exec_length_frac.to(dtype=parts[0].dtype).unsqueeze(-1))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


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
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        condition_dim: int | None = None,
    ):
        super().__init__()
        if condition_dim is not None and condition_dim < 1:
            raise ValueError("condition_dim must be >= 1 when set")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}")
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.num_critics = num_critics
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        self.networks = nn.ModuleList(
            [
                CriticMember(
                    obs_dim,
                    action_dim,
                    hidden_dim,
                    1 if condition_dim is not None else n_obs_steps,
                    action_horizon,
                    condition_on_exec_length=condition_on_exec_length,
                    num_layers=num_layers,
                    condition_dim=condition_dim,
                )
                for _ in range(num_critics)
            ]
        )

    @property
    def input_dim(self) -> int:
        """First-layer input width. Shifts by 1 when the length feature is on,
        which is what makes old checkpoints unloadable without a migration."""
        return self.networks[0].input_dim

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, T, obs]`` (or ``[B, condition_dim]``), ``[B, H, action_dim]`` -> ``[num_critics, B]``."""
        if self.condition_dim is not None:
            if states.dim() != 2 or states.shape[1] != self.condition_dim:
                raise ValueError(
                    f"condition must have shape [B, {self.condition_dim}], got "
                    f"{tuple(states.shape)}"
                )
        else:
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
        return torch.stack(
            [net(states, actions, exec_length_frac) for net in self.networks]
        )
