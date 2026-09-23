"""Networks: window encoders, state conditioner (raw + spectral), flow heads, V(s) critic.

Changes in this revision (addresses review feedback):
* ``RunningMeanStd.forward`` pins the output dtype to the input dtype, so
  half-precision callers don't get silently promoted to fp32.
* ``RunningMeanStd.count`` is fp64 (avoids fp32 saturation at 2^24 frames);
  mean/var stay fp32.
* ``RunningMeanStd._uninitialized`` is registered with ``persistent=False``
  so pre-change checkpoints still load.
* Added ``RunningMeanStd.reset()`` to clear the one-way latch.
* Added a shape assertion in ``RunningMeanStd.update``.
* ``_RMSNorm`` uses the ``dtype=`` kwarg at construction instead of mutating
  ``.data`` (meta-device / ``to_empty`` friendly).
* ``condition.to(...)`` now moves device as well as dtype.
* ``temb`` broadcast replaces the explicit ``expand`` (scalar ``t`` runs the
  time MLP once, not B times).
* ``SelfAttention`` mask handling is shape-robust (``[:, None, None, :]``)
  and the stale "mask caching" docstring is gone.
* ``StateConditioner`` forwards ``n_heads`` to its inner ``WindowEncoder``.
* Type hints on the dual-return forwards and ``resolve_config_value``.

Performance notes (carried from the previous revision):
* Attention uses ``jax.nn.dot_product_attention``-equivalent SDPA on CUDA.
* Sinusoidal frequency tables are cached per-module per-device.
* ``SwiGLU`` fuses gate and up into a single GEMM.
* ``DiTBlock`` / ``SJFPOFlowModel`` return diagnostics only when
  ``return_stats=True`` — the default training path has no extra reductions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.dct_utils import dct
from sjfpo.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()

# Lazy config: only constructed on first ``resolve_config_value`` call.
_DEFAULTS: SJFPOConfig | None = None


def _defaults() -> SJFPOConfig:
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = SJFPOConfig()
    return _DEFAULTS


def resolve_config_value(name: str, override: Any) -> Any:
    """Policy default for ``name`` unless ``override`` is not None."""
    return override if override is not None else getattr(_defaults(), name)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


class _RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` initialized in ``STORAGE_DTYPE``.

    ``dtype=`` at construction is preferable to mutating ``.data`` post-hoc:
    it plays correctly with meta-device init and ``to_empty``.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps=eps, dtype=STORAGE_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)
        return super().forward(x)


def _rms_norm(hidden_dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    return _RMSNorm(hidden_dim, eps=eps)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    Gate and up are fused into a single ``gate_up`` projection (one GEMM
    instead of two); torch's weight layout is ``[out, in]`` so the fused
    weight is ``cat([gate.weight, up.weight], dim=0)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class CondMLP(nn.Module):
    """SiLU MLP for conditioning regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal codes with cached frequency tables.

    Cache is a per-module dict keyed by device string. Under ``torch.compile``
    the first miss can cause a graph break; if that bites, switch to a
    non-persistent buffer and force ``.float()`` inside ``_freqs`` so a
    ``model.to(bf16)`` doesn't demote the table.
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.dim = dim
        self.scale = 1.0 if scale is None else float(scale)
        self.half = dim // 2
        self._freq_cache: dict[str, torch.Tensor] = {}

    def _freqs(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._freq_cache.get(key)
        if cached is None:
            i = torch.arange(self.half, device=device, dtype=torch.float32)
            cached = torch.exp(-math.log(10000.0) * i / max(self.half - 1, 1))
            self._freq_cache[key] = cached
        return cached

    def add_to(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add codes for ``positions`` ([..., L]) to ``x`` ([..., L, D])."""
        ang = (positions.to(dtype=torch.float32) * self.scale).unsqueeze(
            -1
        ) * self._freqs(x.device)
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if pe.shape[-1] < self.dim:  # odd dim: pad last column with zeros
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        return x + pe.to(dtype=x.dtype)

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding for continuous timestep values."""
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float32)
        t = t.to(dtype=torch.float32) * self.scale
        ang = t.unsqueeze(-1) * self._freqs(t.device)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def positions(self, length: int, device) -> torch.Tensor:
        """0..L-1 index vector for slot/frame codes."""
        return torch.arange(length, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position codes by slot index: [..., L, D] -> [..., L, D]."""
        return self.add_to(x, self.positions(x.shape[-2], x.device))


class SelfAttention(nn.Module):
    """Multi-head self-attention on SDPA.

    ``key_mask`` uses PyTorch's ``True = ignore`` convention; a fully masked
    query row is forced to attend to key 0 to avoid SDPA's NaN behavior.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[B, L, D] -> [B, L, D]. ``key_mask`` uses True = ignore."""
        b, l, _ = x.shape
        h = self.n_heads
        d = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, L, H*D] -> [B, H, L, D] via view+transpose.
        q = q.view(b, l, h, d).transpose(1, 2)
        k = k.view(b, l, h, d).transpose(1, 2)
        v = v.view(b, l, h, d).transpose(1, 2)

        mask = None
        if key_mask is not None:
            # True = keep. Force at least one keep per query to avoid NaN.
            keep = ~torch.as_tensor(key_mask, dtype=torch.bool, device=x.device)
            keep = keep.clone()
            keep[..., 0] = True
            mask = keep[:, None, None, :]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(b, l, h * d)
        return self.out(y)


class WindowEncoder(nn.Module):
    """Pre-norm residual block over a time window."""

    def __init__(
        self,
        in_dim: int,
        n_obs_steps: int | None = None,
        hidden_dim: int | None = None,
        n_heads: int = 4,
        mlp_ratio: float | None = None,
    ):
        super().__init__()
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        mlp_ratio = float(resolve_config_value("dit_mlp_ratio", mlp_ratio))
        # n_obs_steps accepted for call-site compat; not used by this block.
        del n_obs_steps
        self.frame_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = SinusoidalEncoding(hidden_dim)
        self.norm1 = _rms_norm(hidden_dim, eps=1e-6)
        self.attn1 = SelfAttention(hidden_dim, n_heads)
        self.norm2 = _rms_norm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp1 = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[B, L, in_dim] -> [B, L, hidden_dim]; pad rows masked as keys when given."""
        h = self.pos_encoding(self.frame_proj(x))
        h = h + self.attn1(self.norm1(h), key_mask)
        return h + self.mlp1(self.norm2(h))


# ---------------------------------------------------------------------------
# Running stats
# ---------------------------------------------------------------------------


class RunningMeanStd(nn.Module):
    """Per-dim running mean/variance over raw obs frames.

    ``mean``/``var`` stay fp32; ``count`` is fp64 so it doesn't saturate at
    2^24 frames on long runs. ``forward`` preserves the input dtype so
    half-precision callers don't get silently promoted.
    """

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        # Non-persistent so pre-change checkpoints load without this key.
        self.register_buffer(
            "_uninitialized", torch.tensor(True, dtype=torch.bool), persistent=False
        )
        self.eps = eps
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running stats (any device, no grad)."""
        if x.shape[-1] != self.mean.numel():
            raise ValueError(
                f"update got last dim {x.shape[-1]}, expected {self.mean.numel()}"
            )
        batch = (
            x.detach()
            .reshape(-1, x.shape[-1])
            .to(self.mean.device, dtype=torch.float32)
        )
        b = batch.shape[0]
        if b == 0:
            return
        b_mean = batch.mean(dim=0)
        b_var = batch.var(dim=0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + b  # fp64
        self.mean.add_((delta * b / total).to(self.mean.dtype))
        # Chan et al. parallel variance merge; result cast back to fp32 by copy_.
        self.var.copy_(
            (
                self.count * self.var
                + b * b_var
                + delta.square() * (self.count * b / total)
            )
            / total
        )
        self.count.copy_(total)
        self._uninitialized.fill_(False)

    @torch.no_grad()
    def reset(self) -> None:
        """Clear the one-way latch and reset stats to their initial values."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()
        self._uninitialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with frozen stats; identity until the first update."""
        norm = (x - self.mean.to(x.device)) * torch.rsqrt(
            self.var.to(x.device) + self.eps
        )
        norm = norm.clamp(-self.clip, self.clip).to(x.dtype)
        return torch.where(self._uninitialized.to(x.device), x, norm)


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    ``return_stats=True`` returns ``(out, stats_dict)``; the default path
    returns the tensor only, so no diagnostic reductions run in training.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = _rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = _rms_norm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        # Zero-init modulation head: block starts as identity (adaLN-Zero).
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """([B, H, D], [B, D]) -> [B, H, D] (or a tuple when stats are requested)."""
        b = c.shape[0]
        # reshape+unbind avoids building a [B, 1, 6H] intermediate.
        parts = self.adaLN(c).view(b, 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts.unbind(
            dim=1
        )

        h = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        x = x + gate_msa.unsqueeze(1) * self.attn(h)

        out = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        )

        if not return_stats:
            return out

        with torch.no_grad():
            stats = {
                "gate_msa": gate_msa.square().mean().sqrt(),
                "gate_mlp": gate_mlp.square().mean().sqrt(),
                "scale_msa": (1.0 + scale_msa).square().mean().sqrt(),
                "scale_mlp": (1.0 + scale_mlp).square().mean().sqrt(),
            }
        return out, stats


class StateConditioner(nn.Module):
    """Fuse raw state window + DCT bins into ``[B, D]``.

    The window is never padded in this codebase; ``key_mask`` is not
    forwarded to the inner encoder.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = int(resolve_config_value("hidden_dim", hidden_dim))
        self.state_encoder = WindowEncoder(
            obs_dim, hidden_dim=hidden_dim, n_heads=n_heads
        )
        self.bin_norm = _rms_norm(obs_dim)
        self.bin_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = nn.Linear(obs_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """[B, T, obs] -> [B, D]."""
        raw = self.state_encoder(states).mean(dim=1)
        spec = self.key_proj(self.bin_encoding(self.bin_norm(dct(states, dim=1)))).mean(
            dim=1
        )
        return self.fuse(torch.cat([raw, spec], dim=-1))


class SJFPOFlowModel(nn.Module):
    """DiT velocity model over noisy-action tokens, conditioned via adaLN.

    ``return_stats=True`` returns ``(velocity, stats_dict)`` and computes
    per-block gate/scale RMS reductions; the default path returns only the
    velocity, with no diagnostic reductions in the graph.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int | None = None,
        dit_depth: int | None = None,
        dit_heads: int | None = None,
        dit_mlp_ratio: float | None = None,
        dit_time_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = int(resolve_config_value("hidden_dim", hidden_dim))
        dit_depth = int(resolve_config_value("dit_depth", dit_depth))
        dit_heads = int(resolve_config_value("dit_heads", dit_heads))
        dit_mlp_ratio = float(resolve_config_value("dit_mlp_ratio", dit_mlp_ratio))
        dit_time_dim = int(resolve_config_value("dit_time_dim", dit_time_dim))
        self.time_dim = dit_time_dim
        self.timestep_embedder = SinusoidalEncoding(
            self.time_dim, scale=float(resolve_config_value("timestep_scale", None))
        )
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.bin_encoding = SinusoidalEncoding(hidden_dim)
        self.time_mlp = CondMLP(dit_time_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = _rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        # Zero final layer: untrained policy predicts ~zero velocity.
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict velocity: ([B, Ka, D], t, [B, D]) -> [B, Ka, D] (or tuple).

        ``t`` may be a scalar or a ``[B]`` tensor. When scalar, the time MLP
        runs on an unbatched embedding and broadcasting handles the batch.
        """
        temb = self.timestep_embedder.encode_times(t).to(
            device=x_t.device, dtype=x_t.dtype
        )
        # No explicit expand: nn.Linear and the add below both broadcast.
        c = self.time_mlp(temb) + condition.to(device=x_t.device, dtype=x_t.dtype)

        h = self.bin_encoding(self.in_proj(x_t))

        if not return_stats:
            for block in self.blocks:
                h = block(h, c)
            shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
            return self.out(modulate(self.norm_out(h), shift_out, scale_out))

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block(h, c, return_stats=True)
            block_stats.append(s)
        with torch.no_grad():
            stats = {
                "temb": temb.square().mean().sqrt(),
                "fused": condition.square().mean().sqrt(),
                "c": c.square().mean().sqrt(),
                **{
                    k: torch.stack([s[k] for s in block_stats]).mean()
                    for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
                },
            }
        shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
        out = self.out(modulate(self.norm_out(h), shift_out, scale_out))
        return out, stats


class Critic(nn.Module):
    """State-value baseline V(s): MLP on current frame."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int | None = None,
        n_obs_steps: int | None = None,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        del n_obs_steps  # accepted for call-site compat; only the last frame scores
        self.mlp = CondMLP(obs_dim, hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """[B, T, obs] -> [B]."""
        return self.mlp(states[:, -1, :]).squeeze(-1)
