"""Self-attention and the adaLN-Zero DiT block."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from myfpo.layers import MIN_WIDTH, CondMLP, SwiGLU, build_rms_norm, modulate
from myfpo.utils import rms

def _rope(x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    pos = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    freqs = torch.einsum("k,d->kd", pos, inv_freq)
    cos, sin = freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    """SDPA self-attention over the action tokens, causally masked.

    Token ``k`` sees only tokens ``0..k``. This is a deliberate deviation from
    ACT and Diffusion Policy, which attend bidirectionally over the chunk. It
    buys prefix consistency: the first ``n_action_steps`` tokens a chunk will
    execute do not depend on the tail that will be discarded.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        head_dim = hidden_dim // n_heads
        if head_dim % 2 != 0:
            # RoPE rotates (even, odd) pairs. An odd head_dim leaves a trailing
            # element unpaired and the broadcast fails deep inside the forward.
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(hidden_dim // (2 * n_heads)).float() / (hidden_dim // (2 * n_heads)))),
            persistent=False,
        )
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the ``L`` action tokens. The shape is preserved."""
        b, seq, _ = x.shape
        h = self.n_heads
        d = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, seq, h, d).transpose(1, 2)
        k = k.view(b, seq, h, d).transpose(1, 2)
        v = v.view(b, seq, h, d).transpose(1, 2)

        q = _rope(q, self.inv_freq)
        k = _rope(k, self.inv_freq)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, seq, h * d)
        return self.out(y)


class DiTBlock(nn.Module):
    """Pre-norm DiT block, adaLN-Zero. ``return_stats=True`` adds a diagnostics dict."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.norm1 = build_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = build_rms_norm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``[B, K, H]`` action tokens, with adaLN conditioning on ``c``. ``c`` is
        ``[B, H]`` for a whole chunk or ``[B, K, H]`` for per-token time.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, H], got {tuple(x.shape)}")
        b = x.shape[0]
        if c.shape[0] != b:
            raise ValueError(f"c has batch {c.shape[0]} but x has {b}")
        token_axis = c.dim() == 3
        if token_axis:
            k = x.shape[1]
            if c.shape[1] != k:
                raise ValueError(f"c has {c.shape[1]} conditioning tokens but x has {k}")
        else:
            if c.dim() != 2 or c.shape[1] != self.adaLN.fc1.in_features:
                raise ValueError(f"c must have shape [{b}, {self.adaLN.fc1.in_features}], got {tuple(c.shape)}")
            c = c[:, None, :]

        parts = self.adaLN(c).view(b, c.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts.unbind(
            dim=2
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

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

