"""Conditional flow-matching velocity model over action tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from myfpo.attention import DiTBlock, SelfAttention
from myfpo.layers import CondMLP, SinusoidalEncoding, build_rms_norm, modulate
from myfpo.utils import rms

def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """``exp`` spectral entropy of ``[n, hidden]``. This is the collapse probe"""
    matrix = features.reshape(-1, features.shape[-1]).float()
    sigma = torch.linalg.svdvals(matrix)
    p = sigma / sigma.sum().clamp_min(1e-12)
    return torch.exp(torch.special.entr(p).sum())


class _FlowActorBase(nn.Module):
    """The stem both actors share: embed, project, condition, decode."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_embed_dim: int,
        block_cls: type[nn.Module],
        block_kwargs: dict,
    ):
        super().__init__()
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if time_embed_dim < 2:
            raise ValueError(f"time_embed_dim={time_embed_dim} must be at least 2")
        if self.num_blocks < 1:
            raise ValueError(f"an actor needs >= 1 block, got {self.num_blocks}")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.timestep_embedder = SinusoidalEncoding(self.time_embed_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            block_cls(**block_kwargs) for _ in range(self.num_blocks)
        )
        self.norm_out = build_rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    @torch.inference_mode()
    def weight_magnitude(self) -> torch.Tensor:
        """Mean absolute weight. This is the correlate of Dohare. The magnitudes"""
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
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the velocity from the token actions and the per-token times."""
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
            shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
            if not token_axis:
                shift, scale = shift[:, None, :], scale[:, None, :]
            return self.out(modulate(self.norm_out(hidden), shift, scale))

        if not return_stats:
            for block in self.blocks:
                h = block(h, c)
            return decode(h)

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block(h, c, return_stats=True)
            block_stats.append(s)
        stats = self._collect_stats(temb, condition, c, h, block_stats)
        return decode(h), stats

    def _collect_stats(
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



class SJFPOFlowModel(_FlowActorBase):
    """The DiT actor. One block is self-attention, one SwiGLU, and adaLN."""

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
        self.num_blocks = dit_depth
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            DiTBlock,
            {"hidden_dim": hidden_dim, "n_heads": dit_heads,
             "mlp_ratio": dit_mlp_ratio},
        )


class ResidualMLP(nn.Module):
    """The residual unit of Wang et al. (2025), arXiv:2503.14858, Section 3."""

    def __init__(self, hidden_dim: int, hidden_mult: float, num_units: int):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if num_units < 1:
            raise ValueError(f"num_units={num_units} must be >= 1")
        inner = int(hidden_dim * hidden_mult)
        if inner < 1:
            raise ValueError("resmlp_hidden_mult must produce a positive width")
        self.inner = inner
        self.num_units = num_units
        self.denses = nn.ModuleList(
            nn.Linear(hidden_dim if i == 0 else inner, inner)
            for i in range(num_units)
        )
        self.norms = nn.ModuleList(build_rms_norm(inner) for _ in range(num_units))
        self.out = nn.Linear(inner, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x + residual(x)``. The residual is zero at init, so this is the"""
        h = x
        for dense, norm in zip(self.denses, self.norms, strict=True):
            h = norm(dense(h))
            h = F.silu(h)
        return x + self.out(h)


class ResMLPBlock(nn.Module):
    """One ``resmlp`` block: attention, then the paper's residual MLP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        hidden_mult: float,
        num_units: int,
    ):
        super().__init__()
        self.norm1 = build_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = build_rms_norm(hidden_dim)
        self.mlp = ResidualMLP(hidden_dim, hidden_mult, num_units)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``[B, K, H]`` in, ``[B, K, H]`` out. ``c`` is ``[B, H]`` or"""
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, H], got {tuple(x.shape)}")
        b = x.shape[0]
        if c.shape[0] != b:
            raise ValueError(f"c has batch {c.shape[0]} but x has {b}")
        if c.dim() == 3:
            if c.shape[1] != x.shape[1]:
                raise ValueError(
                    f"c has {c.shape[1]} conditioning tokens but x has {x.shape[1]}"
                )
        else:
            if c.dim() != 2 or c.shape[1] != self.adaLN.fc1.in_features:
                raise ValueError(
                    f"c must have shape [{b}, {self.adaLN.fc1.in_features}], "
                    f"got {tuple(c.shape)}"
                )
            c = c[:, None, :]

        parts = self.adaLN(c).view(b, c.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts.unbind(
            dim=2
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        if not return_stats:
            return out
        with torch.inference_mode():
            stats = {
                "gate_msa": rms(gate_msa),
                "gate_mlp": rms(gate_mlp),
                "scale_msa": rms(1.0 + scale_msa),
                "scale_mlp": rms(1.0 + scale_mlp),
            }
        return out, stats


class SJFPOFlowResMLP(_FlowActorBase):
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
        self.num_blocks = resmlp_blocks
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            ResMLPBlock,
            {"hidden_dim": hidden_dim, "n_heads": n_heads,
             "hidden_mult": resmlp_hidden_mult, "num_units": resmlp_units},
        )
