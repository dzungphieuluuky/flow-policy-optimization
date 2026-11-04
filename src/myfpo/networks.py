"""The two networks that read the obs window: the actor flow model, and the
ensemble chunk-Q critic. Key: FlowModel, Critic.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from myfpo.layers import (
    AdaLNBlock,
    CrossAttention,
    DiTBlock,
    MIN_WIDTH,
    CondMLP,
    SelfAttention,
    SinusoidalEncoding,
    make_rms_norm,
    modulate,
    rms,
)
from myfpo.processor import StateEncoder

# ACTOR

def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """``exp`` spectral entropy of ``[n, hidden]``. The collapse probe."""
    matrix = features.contiguous().reshape(-1, features.shape[-1]).float()
    sigma = torch.linalg.svdvals(matrix)
    p = sigma / sigma.sum().clamp_min(1e-12)
    # A zero component contributes nothing to the entropy. Dropping it keeps
    # the logarithm finite, which is the whole point of not using
    # torch.special.entr.
    p = p[p > 0]
    if p.numel() == 0:
        # A zero spectrum has zero entropy, and exp of zero is one.
        return torch.ones((), device=features.device, dtype=torch.float32)
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


class FlowActor(nn.Module):
    """The stem both actors share: embed, project, condition, decode."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_embed_dim: int,
        num_blocks: int,
        block_cls: type[nn.Module],
        block_kwargs: dict,
    ):
        super().__init__()
        if action_dim < 1 or hidden_dim < MIN_WIDTH:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if time_embed_dim < MIN_WIDTH:
            raise ValueError(f"time_embed_dim={time_embed_dim} must be at least 2")
        if num_blocks < 1:
            raise ValueError(f"an actor needs >= 1 block, got {num_blocks}")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.num_blocks = num_blocks
        self.timestep_embedder = SinusoidalEncoding(self.time_embed_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            block_cls(**block_kwargs) for _ in range(self.num_blocks)
        )
        self.norm_out = make_rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    @torch.inference_mode()
    def weight_magnitude(self) -> torch.Tensor:
        """Mean absolute weight. The correlate of Dohare. One scalar per module."""
        total = None
        count = 0
        for param in self.parameters():
            s = param.detach().abs().sum(dtype=torch.float32)
            total = s if total is None else total + s
            count += param.numel()
        return total / max(count, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity from the token actions and the per-token times."""
        out, _ = self.forward_core(x_t, t, condition, collect_stats=False)
        return out

    def forward_with_stats(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The same forward, plus the per-block and conditioning diagnostics."""
        return self.forward_core(x_t, t, condition, collect_stats=True)

    def forward_core(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
        *,
        collect_stats: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Embed ``t``, condition the tokens, run the blocks, and decode. ``collect_stats`` adds the per-block diagnostics."""
        if x_t.dim() != 3 or x_t.shape[2] != self.action_dim:
            raise ValueError(f"x_t must have shape [B, K, {self.action_dim}], got {tuple(x_t.shape)}")
        if condition.dim() != 2 or condition.shape[0] != x_t.shape[0] or condition.shape[1] != self.hidden_dim:
            raise ValueError(f"condition must have shape [{x_t.shape[0]}, {self.hidden_dim}], got {tuple(condition.shape)}")
        if x_t.shape[1] > self.action_horizon:
            raise ValueError(f"action token count {x_t.shape[1]} exceeds action_horizon {self.action_horizon}")

        t = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        cond = condition.to(device=x_t.device, dtype=x_t.dtype)
        token_axis = t.dim() in (2, 3) and tuple(t.shape[:2]) == tuple(x_t.shape[:2])
        if token_axis:
            if t.dim() == 3:
                if t.shape[-1] != 1:
                    raise ValueError(f"t must end in a singleton action-dimension axis, got {tuple(t.shape)}")
                t = t.squeeze(-1)
            temb = self.timestep_embedder.encode_times(t.reshape(-1))
            temb = temb.reshape(x_t.shape[0], x_t.shape[1], -1).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond[:, None, :]
        elif t.dim() <= 1:
            t = t.reshape(-1)
            if t.numel() not in (1, x_t.shape[0]):
                raise ValueError(f"t must contain 1 or {x_t.shape[0]} values, got {t.numel()}")
            temb = self.timestep_embedder.encode_times(t).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond
        else:
            raise ValueError(f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}")

        h = self.in_proj(x_t)

        def decode(hidden: torch.Tensor) -> torch.Tensor:
            """Modulate the final hidden state and apply the output projection."""
            shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
            if not token_axis:
                shift, scale = shift[:, None, :], scale[:, None, :]
            return self.out(modulate(self.norm_out(hidden), shift, scale))

        if not collect_stats:
            for block in self.blocks:
                h = block(h, c)
            return decode(h), None

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block.forward_with_stats(h, c)
            block_stats.append(s)
        stats = self.collect_stats(temb, condition, c, h, block_stats)
        return decode(h), stats

    @torch.compiler.disable
    def collect_stats(
        self,
        temb: torch.Tensor,
        condition: torch.Tensor,
        c: torch.Tensor,
        h: torch.Tensor,
        block_stats: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Diagnostics, deliberately outside any compiled graph."""
        with torch.inference_mode():
            return {
                "temb": rms(temb),
                "fused": rms(condition),
                "c": rms(c),
                **{
                    k: torch.stack([s[k] for s in block_stats]).mean()
                    for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
                },
                "effective_rank": effective_rank(h),
                "weight_magnitude": self.weight_magnitude(),
            }


class FlowModel(FlowActor):
    """The DiT actor. One block is self-attention, one MishFFN, and adaLN."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        dit_depth: int,
        dit_heads: int,
        dit_mlp_ratio: float,
        time_embed_dim: int,
    ):
        if dit_depth < 1 or dit_heads < 1:
            raise ValueError(
                f"dit_depth={dit_depth} and dit_heads={dit_heads} must be >= 1"
            )
        if (
            not math.isfinite(dit_mlp_ratio)
            or dit_mlp_ratio <= 0
            or int(hidden_dim * dit_mlp_ratio) < 1
        ):
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            dit_depth,
            DiTBlock,
            {"hidden_dim": hidden_dim, "n_heads": dit_heads,
             "mlp_ratio": dit_mlp_ratio},
        )


class ResidualMLP(nn.Module):
    """The residual unit of Wang et al. (2025), arXiv:2503.14858, Section 3."""

    def __init__(self, hidden_dim: int, hidden_mult: float, num_units: int):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        if num_units < 1:
            raise ValueError(f"num_units={num_units} must be >= 1")
        inner = int(hidden_dim * hidden_mult)
        if inner < 1:
            raise ValueError("resmlp_hidden_mult must produce a positive width")
        self.denses = nn.ModuleList(
            nn.Linear(hidden_dim if i == 0 else inner, inner)
            for i in range(num_units)
        )
        self.norms = nn.ModuleList(make_rms_norm(inner) for _ in range(num_units))
        self.out_norm = make_rms_norm(inner)
        self.out = nn.Linear(inner, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x + residual(x)``. The residual is zero at init, so this is the identity."""
        h = x
        for dense, norm in zip(self.denses, self.norms, strict=True):
            h = norm(dense(h))
            h = F.silu(h)
        return x + self.out(self.out_norm(h))


class ResMLPBlock(AdaLNBlock):
    """The ``resmlp`` block: attention, then the paper's residual MLP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        hidden_mult: float,
        num_units: int,
    ):
        super().__init__(
            hidden_dim,
            n_heads,
            ResidualMLP(hidden_dim, hidden_mult, num_units),
        )


class FlowResMLP(FlowActor):
    """The ``resmlp`` actor. DiT's attention and adaLN, the paper's MLP unit."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        resmlp_blocks: int,
        resmlp_units: int,
        resmlp_hidden_mult: float,
        n_heads: int,
        time_embed_dim: int,
    ):
        if resmlp_blocks < 1:
            raise ValueError(f"resmlp_blocks={resmlp_blocks} must be >= 1")
        if resmlp_units < 1:
            raise ValueError(f"resmlp_units={resmlp_units} must be >= 1")
        if n_heads < 1:
            raise ValueError(f"n_heads={n_heads} must be >= 1")
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            resmlp_blocks,
            ResMLPBlock,
            {"hidden_dim": hidden_dim, "n_heads": n_heads,
             "hidden_mult": resmlp_hidden_mult, "num_units": resmlp_units},
        )

# CRITIC


class ActionAffine(nn.Module):
    """Per-element affine over the chunk. No chunk statistics (no LayerNorm),
    so sustained pushes and raw magnitude drift survive. State dict matches
    ``nn.LayerNorm`` (weight/bias) for checkpoint compatibility."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned per-element scale and shift."""
        return x * self.weight + self.bias


class StateAffine(nn.Module):
    """One RMS scale over the whole window, no mean subtraction, so constant
    offsets survive. The weight and bias stay per element. The caller has
    already z-scored each dim through RunningMeanStd, so this adds a window
    level scale, not the first normalization. Matches ``nn.LayerNorm`` state
    dict."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the window RMS scale, then the learned per-element scale and shift."""
        scale = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps).to(x.dtype)
        return x * scale * self.weight + self.bias


def make_mlp_layers(
    in_dim: int, hidden_dim: int, num_layers: int, dropout: float
) -> nn.Module:
    """The critic trunk. Each layer is ``Linear -> LayerNorm -> Mish``, and
    the first layer adds ``Dropout`` when ``dropout > 0``. One final linear
    maps to the scalar."""
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
    the trunk. The trunk is ``Linear -> LayerNorm -> Mish`` per layer, and the
    first layer adds ``Dropout`` when ``dropout > 0``. The output layer stays
    linear.

    Inside the chunk one causal ``SelfAttention`` layer mixes the steps, and
    its RoPE codes carry the step position. Token ``i`` reads the prefix only,
    the same rule the actor's chunk attention uses. The cross blocks stay
    bidirectional: the critic reads a chunk it has already executed.
    The action tokens carry no norm of the chunk: a chunk of all +5 and a chunk
    of all 0 must reach the critic different. State and action keep the same
    affine normalizers the concat head used, for the same reason.
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
        # Four heads on the attention width. StateEncoder requires width % 4,
        # which gives an even head width for two heads. validate_critic also
        # requires width % 8, so four even heads always fit.
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
        # quantities. The state norm divides by the window RMS and keeps the
        # mean, so the critic still sees how high the state sits. The obs
        # normalizer already z-scores each dim, so this adds one per-window
        # scale on top, not the first fixed scale. When ``condition_dim``
        # is set, the norm is skipped: the caller already encoded the window.
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
        self.act_attn_norm = make_rms_norm(state_dim)
        # Both directions read the same input streams, so neither pass sees the
        # other's output. Chaining them would make the second pass read a
        # half-mixed stream.
        self.obs_to_act = CrossAttention(state_dim, attn_heads)
        self.act_to_obs = CrossAttention(state_dim, attn_heads)
        self.cross_norm_obs = make_rms_norm(state_dim)
        self.cross_norm_act = make_rms_norm(state_dim)
        self.mlp = make_mlp_layers(self.input_dim, hidden_dim, num_layers, dropout)

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
