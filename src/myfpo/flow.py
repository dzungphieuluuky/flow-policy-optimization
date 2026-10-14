"""Conditional flow-matching velocity model over action tokens."""

from __future__ import annotations

import math

import torch
from torch import nn

from myfpo.attention import DiTBlock
from myfpo.layers import CondMLP, SinusoidalEncoding, build_rms_norm, modulate
from myfpo.utils import rms

def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """``exp`` spectral entropy of ``[n, hidden]``. Dohare et al. (2306.13812)
    collapse probe: falls as backprop settles on a low-rank solution. In
    ``[1, min(n, hidden)]``."""
    matrix = features.reshape(-1, features.shape[-1]).float()
    sigma = torch.linalg.svdvals(matrix)
    p = sigma / sigma.sum().clamp_min(1e-12)
    # entr(0) is 0 by convention, so the rank-deficient case needs no clamp.
    return torch.exp(torch.special.entr(p).sum())


class SJFPOFlowModel(nn.Module):
    """DiT velocity model over noisy action tokens, adaLN conditioned.

    ``return_stats=True`` gives ``(velocity, stats)`` plus per-block gate and
    scale RMS reductions. Default path returns velocity only, nothing
    reduced. The stats path also runs an SVD (``effective_rank``) and a
    full parameter sweep (``weight_magnitude``); call it at eval cadence,
    not per optimizer step.
    """

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
        super().__init__()
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if dit_depth < 1 or dit_heads < 1 or time_embed_dim < 2:
            raise ValueError("dit_depth, dit_heads, and time_embed_dim must be positive")
        if not math.isfinite(dit_mlp_ratio) or dit_mlp_ratio <= 0 or int(hidden_dim * dit_mlp_ratio) < 1:
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.timestep_embedder = SinusoidalEncoding(self.time_embed_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.token_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.token_pos, std=0.02)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
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
        """Mean absolute weight. Dohare's correlate: magnitudes grow under
        continual training, before the rank drop. Per-param sums, not one
        accumulator: a float scalar would land on CPU."""
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
        """Predict velocity from token actions and per-token times. ``t``:
        scalar, ``[B]``, or ``[B, K, 1]``, one time per token shared across its
        action dims. Last layer zero-init, so untrained predicts near-zero
        velocity. Returns ``[B, K, action_dim]``, plus stats if asked."""
        if x_t.dim() != 3 or x_t.shape[2] != self.action_dim:
            raise ValueError(f"x_t must have shape [B, K, {self.action_dim}], got {tuple(x_t.shape)}")
        if condition.dim() != 2 or condition.shape[0] != x_t.shape[0] or condition.shape[1] != self.hidden_dim:
            raise ValueError(f"condition must have shape [{x_t.shape[0]}, {self.hidden_dim}], got {tuple(condition.shape)}")
        if x_t.shape[1] > self.action_horizon:
            raise ValueError(f"action token count {x_t.shape[1]} exceeds action_horizon {self.action_horizon}")

        t = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
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
            c = self.time_mlp(temb) + condition.to(
                device=x_t.device, dtype=x_t.dtype
            )[:, None, :]
            token_pos = self.token_pos[:, : x_t.shape[1], :].to(
                device=x_t.device, dtype=x_t.dtype
            )
            h = self.in_proj(x_t) + token_pos
        elif t.dim() <= 1:
            t = t.reshape(-1)
            if t.numel() not in (1, x_t.shape[0]):
                raise ValueError(f"t must contain 1 or {x_t.shape[0]} values, got {t.numel()}")
            temb = self.timestep_embedder.encode_times(t).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + condition.to(
                device=x_t.device, dtype=x_t.dtype
            )
            token_pos = self.token_pos[:, : x_t.shape[1], :].to(
                device=x_t.device, dtype=x_t.dtype
            )
            h = self.in_proj(x_t) + token_pos
        else:
            raise ValueError(f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}")

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

    @torch.compiler.disable
    def _collect_stats(
        self,
        temb: torch.Tensor,
        condition: torch.Tensor,
        c: torch.Tensor,
        h: torch.Tensor,
        block_stats: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Diagnostics, deliberately outside any compiled graph.

        ``torch.compiler.disable`` is load-bearing here, not an optimisation.
        Under ``torch.compile`` this block used to die with::

            BackendCompilerFailed: backend='inductor' raised:
            RuntimeError: <weakref ... to 'torch.storage.UntypedStorage'>
            While executing %reshape_5 ... features.reshape(-1, hidden)

        Two independent problems compound:

        1. ``torch.inference_mode()`` inside a traced region mints inference
           tensors, whose version counter inductor cannot reason about; the
           storage is released while the lowering still holds a weakref to it.
        2. ``svdvals`` lowers to a LAPACK call inductor does not own the
           lifetime of, so the same premature release hits it too.

        Graph-breaking once per stats pass sidesteps both: dynamo stops tracing
        at the decorator, so nothing downstream is captured. The default path
        (``return_stats=False``) never enters here and stays fully compiled,
        which is the path that matters for throughput.
        """
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

