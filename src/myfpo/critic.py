"""Ensemble chunk-Q critic."""

from __future__ import annotations

import torch
from torch import nn

from myfpo.layers import TemporalConv


class ActionAffine(nn.Module):
    """A learned per-element affine over the flattened action chunk.

    It computes no statistics from the chunk. A ``LayerNorm`` would subtract
    the chunk's mean and divide by its standard deviation, so every chunk
    would reach the critic with the same offset removed and the same scale.
    The critic would then be blind to a sustained push, and blind to the
    drifting raw magnitude that ``train/raw_action_abs_max`` monitors.

    The state dict matches ``nn.LayerNorm`` exactly: ``weight`` and ``bias``,
    the same shapes, the same initial values. A checkpoint written before this
    change loads into this module unchanged.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class CriticMember(nn.Module):
    """One Q(s, a_{t:t+h}) head. The network is a concat MLP. It has no FiLM
    conditioning; the executed-chunk fraction is concatenated instead.

    Every hidden layer is ``Linear -> LayerNorm -> Mish``, following TD-MPC2
    (arXiv:2310.16828). The norm is the part that matters: that paper reports
    that scaling an unnormalised critic does not improve performance while
    LayerNorm does. The output layer stays linear.

    ``dropout`` sits directly after the first ``Linear``, the only placement
    TD-MPC2 specifies ("1% Dropout after the first linear layer in each
    Q-function"). Its purpose here is to keep the ensemble from converging onto
    one function: every member regresses onto the same scalar target, so
    without a mask the only thing distinguishing them is initialisation.

    The action input is a learned affine, not a LayerNorm. See
    :class:`ActionAffine` for why the chunk's offset has to survive.
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
        dropout: float = 0.0,
        state_temporal_kernel: int = 0,
        chunk_temporal_kernel: int = 0,
        chunk_temporal_causal: bool = False,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
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
            raise ValueError(
                f"condition_dim is set, so n_obs_steps must be 1 (unused), got "
                f"{n_obs_steps}"
            )
        self.input_dim = state_dim + action_horizon * action_dim + (
            1 if condition_on_exec_length else 0
        )
        # Normalise state and action separately. One LayerNorm over the whole
        # concat would couple state dims to action dims, and they are unrelated
        # quantities. The state norm is skipped when the shared encoder already
        # produced a normalised vector, so critic_share_encoder keeps meaning
        # "raw shared-encoder vector".
        if condition_dim is None:
            self.norm_state = nn.LayerNorm(n_obs_steps * obs_dim)
        # The action gets a learned affine, not a LayerNorm. A LayerNorm over
        # the chunk subtracts the chunk's mean, so a chunk of all +5 and a
        # chunk of all 0 would reach the critic as the same input. The affine
        # keeps the name and the state dict of the LayerNorm it replaces.
        self.norm_action = ActionAffine(action_horizon * action_dim)
        # The filters sit before the flatten. They keep the width, so the
        # first layer sees the same input size as before. The state filter
        # needs the frames, so it only exists when the critic reads the
        # window and not the shared encoder's vector.
        self.frame_conv = (
            TemporalConv(obs_dim, state_temporal_kernel)
            if state_temporal_kernel and condition_dim is None
            else None
        )
        self.chunk_conv = (
            TemporalConv(action_dim, chunk_temporal_kernel, causal=chunk_temporal_causal)
            if chunk_temporal_kernel
            else None
        )
        in_dim = self.input_dim
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            if i == 0 and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.LayerNorm(hidden_dim), nn.Mish()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accept ``[B, T, obs]`` or ``[B, condition_dim]``, plus ``[B, H,
        action_dim]`` and the scalar exec length. Returns ``[B]``.
        """
        if self.condition_dim is None:
            frames = states.transpose(1, 2)
            if self.frame_conv is not None:
                frames = self.frame_conv(frames)
            state_part = frames.transpose(1, 2).flatten(1)
        else:
            state_part = states
        chunk = actions.transpose(1, 2)
        if self.chunk_conv is not None:
            chunk = self.chunk_conv(chunk)
        parts = [state_part, chunk.transpose(1, 2).flatten(1)]
        if self.condition_dim is None:
            parts[0] = self.norm_state(parts[0])
        parts[1] = self.norm_action(parts[1])
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
    """Q(s_t, a_{t:t+h}) over the full window. The chunk is the executed action
    sequence, and the executed-length fraction says how much of it ran.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        n_obs_steps: int,
        action_horizon: int,
        num_critics: int,
        *,
        condition_on_exec_length: bool = True,
        num_layers: int = 2,
        condition_dim: int | None = None,
        dropout: float = 0.0,
        state_temporal_kernel: int = 0,
        chunk_temporal_kernel: int = 0,
        chunk_temporal_causal: bool = False,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.num_critics = num_critics
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        self.dropout = dropout
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
                    dropout=dropout,
                    state_temporal_kernel=state_temporal_kernel,
                    chunk_temporal_kernel=chunk_temporal_kernel,
                    chunk_temporal_causal=chunk_temporal_causal,
                )
                for _ in range(num_critics)
            ]
        )

    @property
    def input_dim(self) -> int:
        """First-layer input width. Shifts by 1 with the exec-length input."""
        return self.networks[0].input_dim

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        exec_length_frac: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accept ``[B, T, obs]`` or ``[B, condition_dim]``, plus ``[B, H,
        action_dim]`` and the scalar exec length. Returns ``[B]``.
        """
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
