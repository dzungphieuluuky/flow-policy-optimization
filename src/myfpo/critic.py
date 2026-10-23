"""Owns: value critic + optimization. Key: Critic.__init__.
Ensemble chunk-Q critic.
"""

from __future__ import annotations

import torch
from torch import nn

from myfpo.layers import CrossAttention, SelfAttention, build_rms_norm
from myfpo.processor import StateEncoder


class ActionAffine(nn.Module):
    """Per-element affine over the chunk. No chunk statistics (no LayerNorm),
    so sustained pushes and raw magnitude drift survive. State dict matches
    ``nn.LayerNorm`` (weight/bias) for checkpoint compatibility."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias



class StateAffine(nn.Module):
    """Per-element affine over state window using RMS (not mean subtraction),
    so constant offsets survive. Matches ``nn.LayerNorm`` state dict."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps).to(x.dtype)
        return x * scale * self.weight + self.bias


def build_mlp_layers(in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
        if i == 0 and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.LayerNorm(hidden_dim), nn.Mish()]
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)

class CriticMember(nn.Module):
    """One Q(s, chunk) head: two token streams that condition each other.

    The state window enters as one token per frame, the action chunk as one
    token per step. Each stream queries the other in parallel, both read the
    raw streams, then the last state token and the mean action token go to
    ``Linear -> Dropout -> LayerNorm -> Mish``. The output layer stays linear.

    Inside the chunk one causal ``SelfAttention`` layer mixes the steps, and
    its RoPE codes carry the step position. Token ``i`` reads the prefix only,
    the same rule the actor's chunk attention uses. The cross blocks stay
    bidirectional: the critic reads a chunk it has already executed.
    The action tokens carry no norm of the chunk: a chunk of all +5 and a chunk
    of all 0 must reach the critic different. State and action keep the same
    affine normalisers the concat head used, for the same reason.
# INIT
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
        state_hidden: int | None = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.action_horizon = action_horizon
        self.condition_on_exec_length = condition_on_exec_length
        self.condition_dim = condition_dim
        state_dim = (
            condition_dim
            if condition_dim is not None
            else (state_hidden if state_hidden is not None else hidden_dim)
        )
        # Four heads on the attention width. The actor's window encoder already
        # requires a width that splits over its two heads with an even head
        # width, so state_dim divides by four on every reachable config.
        attn_heads = 4
        self.attn_heads = attn_heads

        if condition_dim is not None and n_obs_steps != 1:
            raise ValueError(
                f"condition_dim is set, so n_obs_steps must be 1 (unused), got "
                f"{n_obs_steps}"
            )
        # The pool is 2 * state_dim. The first MLP layer does the resize when
        # state_dim != hidden_dim.
        self.input_dim = 2 * state_dim + (
            1 if condition_on_exec_length else 0
        )
        # Normalise state and action separately. One norm over the whole
        # concat would couple state dims to action dims, and they are unrelated
        # quantities. The state norm divides by the RMS and keeps the mean, so
        # the critic still sees how high the state sits. It also puts the raw
        # window on a fixed scale before the first linear. The norm is skipped
        # when the shared encoder already produced a normalised vector, so
        # critic_share_encoder keeps meaning "raw shared-encoder vector".
        self.norm_state = (
            StateAffine(n_obs_steps * obs_dim) if condition_dim is None else None
        )
        # The window trunk is the actor's: one shared per-frame projection and
        # one causal attention pass, then the last frame. The old first layer
        # read the flat window, 5568 -> 512 on Humanoid, and that one layer was
        # 5.1M of the 6.4M critic. Now the first layer reads state_hidden.
        self.state_encoder = (
            StateEncoder(obs_dim, state_dim, n_obs_steps)
            if condition_dim is None
            else None
        )
        # The action gets a learned affine, not a LayerNorm. A LayerNorm over
        # the chunk subtracts the chunk's mean, so a chunk of all +5 and a
        # chunk of all 0 would reach the critic as the same input. The affine
        # keeps the name and the state dict of the LayerNorm it replaces.
        self.norm_action = ActionAffine(action_horizon * action_dim)
        # One token per action step, projected to the stream width. The affine
        # above already ran on the flat chunk, so a constant chunk stays
        # constant and the tokens keep its offset.
        self.act_in = nn.Linear(action_dim, state_dim)
        # The chunk is pooled by its mean, which is order-blind, and the cross
        # pass is position-wise. A causal self-attention layer over the steps
        # restores the order: it carries RoPE, so token i knows its step, and it
        # reads the prefix only. The state side needs none: its trunk carries
        # RoPE already.
        self.act_attn = SelfAttention(state_dim, attn_heads)
        self.act_attn_norm = build_rms_norm(state_dim)
        # Both directions read the same input streams, so neither pass sees the
        # other's output. Chaining them would make the second pass read a
        # half-mixed stream.
        self.obs_to_act = CrossAttention(state_dim, attn_heads)
        self.act_to_obs = CrossAttention(state_dim, attn_heads)
        self.cross_norm_obs = build_rms_norm(state_dim)
        self.cross_norm_act = build_rms_norm(state_dim)
        self.mlp = build_mlp_layers(self.input_dim, hidden_dim, num_layers, dropout)

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
            window = self.norm_state(states.flatten(1)).view(states.shape)
            # The encoder's output norm runs on every token, the same way it
            # runs on the pooled vector in StateEncoder.forward. Leaving it
            # out would keep a parameter that never sees a gradient.
            obs_tokens = self.state_encoder.out_norm(
                self.state_encoder.encode_tokens(window)
            )
        else:
            obs_tokens = states.unsqueeze(1)
        # The affine runs on the flat chunk in step-major order.
        chunk_flat = self.norm_action(actions.flatten(1))
        act_tokens = self.act_in(chunk_flat.view(actions.shape))
        act_tokens = act_tokens + self.act_attn(
            self.act_attn_norm(act_tokens)
        )
        obs = obs_tokens + self.obs_to_act(
            self.cross_norm_obs(obs_tokens), act_tokens
        )
        act = act_tokens + self.act_to_obs(
            self.cross_norm_act(act_tokens), obs_tokens
        )
        parts = [obs[:, -1], act.mean(dim=1)]
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
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


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
        state_hidden: int | None = None,
    ):
        super().__init__()
        if condition_dim is not None and condition_dim < 1:
            raise ValueError("condition_dim must be >= 1 when set")
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
                    state_hidden=state_hidden,
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
