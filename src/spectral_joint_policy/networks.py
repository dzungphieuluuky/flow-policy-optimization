"""Networks: window encoders, conditioner, flow head, critic.

Key facts:
* ``RunningMeanStd``: count fp64 (fp32 dies at 2^24 frames), mean/var fp32,
  ``_uninitialised`` non-persistent so old checkpoints load.
* ``RunningMeanStd.forward``: out dtype = in dtype. No silent fp32 promote.
* ``_RMSNorm``: reduce in fp32, take ``dtype=`` at build, never touch ``.data``.
* ``StateConditioner``: per-frame MLP + DCT branch. No window attention.
* ``SelfAttention``: ``[B, L]`` key-padding mask, 1 = keep. Inverts for SDPA,
  whose bool mask is the opposite of HF. Empty row is forced to keep key 0,
  else NaN.
* ``temb`` broadcast, not ``expand``: a scalar ``t`` runs the time MLP once.

Speed:
* Attention on SDPA (CUDA).
* Sinusoidal freq tables cached per device.
* ``SwiGLU`` fuses gate+up into one GEMM.
* ``DiTBlock`` / flow return stats only when asked. No reductions in training.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from spectral_joint_policy.dct_utils import extract_dct_coeffs
from spectral_joint_policy.utils import rms


class _RMSNorm(nn.RMSNorm):
    """RMSNorm computed in FP32, out dtype = in dtype.

    Params start FP32. Activations may be FP16/BF16 under autocast, but the
    result goes back to the input dtype. Norm must not change the model-wide
    precision policy.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps=eps, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm trailing dims with the learned scale.

        Reduce in fp32, cast back on exit, so a module moved to a lower param
        dtype by ``.to()`` still norms right. Returns normalised, input dtype.
        """
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.rms_norm(
                x.to(dtype=torch.float32),
                self.normalized_shape,
                self.weight.to(dtype=torch.float32),
                self.eps,
            )
        return y.to(dtype=input_dtype)


def _rms_norm(hidden_dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    """fp32-safe RMS norm over the given width."""
    return _RMSNorm(hidden_dim, eps=eps)


class SwiGLU(nn.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    Gate+up fused into one ``gate_up`` GEMM. torch weight is ``[out, in]``,
    so fused = ``cat([gate.weight, up.weight], dim=0)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated GELU. ``[..., in_dim]`` -> ``[..., hidden_dim]``."""
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class CondMLP(nn.Module):
    """SiLU MLP for conditioning regression."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("CondMLP dimensions must be positive")
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-layer SiLU-gated MLP. ``[..., in_dim]`` -> ``[..., out_dim]``."""
        return self.fc2(F.silu(self.fc1(x)))


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal codes. Freq tables cached per device."""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.scale = 1.0 if scale is None else float(scale)
        self.half = dim // 2
        self._freq_cache: OrderedDict[torch.device, torch.Tensor] = OrderedDict()

    def _freqs(self, device: torch.device) -> torch.Tensor:
        """Cached ``[half]`` freq table for a device.

        Holds 4 devices max, drops the LRU past that.
        """
        device = torch.device(device)
        cached = self._freq_cache.get(device)
        if cached is not None:
            self._freq_cache.move_to_end(device)
            return cached
        i = torch.arange(self.half, device=device, dtype=torch.float32)
        cached = torch.exp(-math.log(10000.0) * i / max(self.half - 1, 1))
        self._freq_cache[device] = cached
        if len(self._freq_cache) > 4:
            self._freq_cache.popitem(last=False)
        return cached

    def add_to(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal codes at ``positions``, growing ``[..., L, D]``."""
        ang = (positions.to(dtype=torch.float32) * self.scale).unsqueeze(
            -1
        ) * self._freqs(x.device)
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if pe.shape[-1] < self.dim:
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        return x + pe.to(dtype=x.dtype)

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embed of continuous times. ``[N]`` -> ``[N, 2 * half]``."""
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float32)
        t = t.to(dtype=torch.float32) * self.scale
        ang = t.unsqueeze(-1) * self._freqs(t.device)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def positions(self, length: int, device) -> torch.Tensor:
        """``[length]`` index holding ``0..length-1``, for slot or frame codes."""
        return torch.arange(length, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add slot-indexed position codes to ``[..., L, D]``."""
        return self.add_to(x, self.positions(x.shape[-2], x.device))


class SelfAttention(nn.Module):
    """Multi-head self-attention on SDPA.

    ``attention_mask`` is ``[B, L]`` key-padding. 1/True = keep,
    0/False = ignore. Each row is forced to keep key 0 when empty, else
    SDPA NaNs. Arbitrary query/key masks not supported.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must split over {n_heads} heads")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head self attention.

        ``attention_mask``: ``[B, L]``, 1 = keep.

        - HF inverts SDPA's mask, so invert first
        - all-masked row emits NaN, so force one key
        - fused ``[B, L, H*D]`` proj uses view + transpose, not permute

        ``ValueError`` if ``x`` is not 3D, or mask length != seq length.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, L, D], got {tuple(x.shape)}")
        b, l, _ = x.shape
        if b < 1 or l < 1:
            raise ValueError(
                f"x must have positive batch and sequence sizes, got {(b, l)}"
            )
        h = self.n_heads
        d = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, h, d).transpose(1, 2)
        k = k.view(b, l, h, d).transpose(1, 2)
        v = v.view(b, l, h, d).transpose(1, 2)

        mask = None
        if attention_mask is not None:
            keep = torch.as_tensor(
                attention_mask, dtype=torch.bool, device=x.device
            ).clone()
            if keep.shape != (b, l):
                raise ValueError(
                    f"attention_mask must have shape [B, L] = {(b, l)}, "
                    f"got {tuple(keep.shape)}"
                )
            keep[:, 0] = True
            mask = keep[:, None, None, :]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(b, l, h * d)
        return self.out(y)


class RunningMeanStd(nn.Module):
    """Per-dim running mean/var over raw obs frames.

    ``mean``/``var`` fp32. ``count`` fp64, else dies at 2^24 frames.
    ``forward`` keeps the input dtype, so half-precision callers are not
    silently promoted.

    The warm-up latch is a plain Python bool, not a buffer. It never had to be
    a buffer: it is non-persistent, so it is absent from every ``state_dict``
    and re-derived on load. As a tensor it cost a full-size ``torch.where`` on
    every forward plus three redundant ``.to(x.device)`` casts on buffers whose
    device is fixed by ``.to()`` on the module. Reading a buffer as a Python
    bool would be worse -- ``bool(tensor)`` is a dispatch *and* a D2H sync on
    CUDA -- so it has to be mirrored in Python at the three places that set it.
    """

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self._uninitialised = True
        self.eps = eps
        self.clip = clip

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running stats.

        Chan et al. parallel variance merge in fp64, cast back to fp32, so
        incremental updates do not drift. Latch is non-persistent, so
        pre-change checkpoints load. ``ValueError`` if the trailing dim
        != tracked obs size.
        """
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
        # One pass for both moments; mean() then var() walked the batch twice.
        b_var, b_mean = torch.var_mean(batch, dim=0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + b
        self.mean.add_((delta * b / total).to(self.mean.dtype))
        self.var.copy_(
            (
                self.count * self.var
                + b * b_var
                + delta.square() * (self.count * b / total)
            )
            / total
        )
        self.count.copy_(total)
        self._uninitialised = False

    @torch.inference_mode()
    def reset(self) -> None:
        """Clear the latch, reset stats."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()
        self._uninitialised = True

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state, re-derive the latch.

        The latch is not persisted, so ``count == 0`` means pass frames through
        unchanged. Positional and keyword args go to ``nn.Module``.
        """
        super()._load_from_state_dict(*args, **kwargs)
        self._uninitialised = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm with frozen stats. Returns ``x`` unchanged until first update.

        ``x`` must already be on the module's device: ``mean`` and ``var`` are
        buffers that ``.to()`` moves once, and the per-call casts this used to
        do were no-ops whenever that held.
        """
        if self._uninitialised:
            return x
        norm = (x - self.mean) * torch.rsqrt(self.var + self.eps)
        return norm.clamp(-self.clip, self.clip).to(x.dtype)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: ``x * (1 + scale) + shift``.

    ``torch.addcmul(shift, x, 1 + scale)`` is the same value in one kernel
    instead of three. Twelve call sites per training microbatch, on tensors
    that are the widest thing in the DiT, so the saved pass over memory is the
    point; the multiply is exact either way.
    """
    return torch.addcmul(shift, x, 1.0 + scale)


class DiTBlock(nn.Module):
    """Pre-norm DiT block, adaLN-Zero conditioning.

    ``return_stats=True`` gives ``(out, stats)``. Default path runs no
    diagnostic reductions, so training pays nothing.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if (
            not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or int(hidden_dim * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.norm1 = _rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = _rms_norm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the DiT stack over the action tokens.

        - modulation head zero-init (adaLN-Zero), so the stack starts as
          identity, a no-op
        - layout ``[B, K, D_a, H]``: attention runs over the K action tokens
          per action dim. Only the batch/attn axis is reshaped.
        - modulation params use reshape + unbind, no ``[B, 1, 6H]`` intermediate

        ``c``: ``[B, D]`` or ``[B, K, D]``. Returns tokens, plus stats if
        ``return_stats``.
        """
        coordinate_axis = x.dim() == 4
        token_axis = not coordinate_axis and c.dim() == 3
        b = c.shape[0]
        if coordinate_axis:
            _, k, action_axis, _ = x.shape
            parts = self.adaLN(c).view(b, k, action_axis, 6, -1)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = parts.unbind(dim=3)
            h = modulate(self.norm1(x), shift_msa, scale_msa)
            h_attn = h.permute(0, 2, 1, 3).reshape(
                b * action_axis, k, h.shape[-1]
            )
            attn_out = self.attn(h_attn)
            attn_out = attn_out.reshape(b, action_axis, k, -1).permute(0, 2, 1, 3)
            x = x + gate_msa * attn_out
            out = x + gate_mlp * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        elif token_axis:
            _, k, _ = x.shape
            parts = self.adaLN(c).view(b, k, 6, -1)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = parts.unbind(dim=2)
            h = modulate(self.norm1(x), shift_msa, scale_msa)
            x = x + gate_msa * self.attn(h)
            out = x + gate_mlp * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        else:
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

        with torch.inference_mode():
            stats = {
                "gate_msa": rms(gate_msa),
                "gate_mlp": rms(gate_mlp),
                "scale_msa": rms(1.0 + scale_msa),
                "scale_mlp": rms(1.0 + scale_mlp),
            }
        return out, stats


class StateConditioner(nn.Module):
    """Fuse a mean-pooled per-frame state with DCT-frequency features.

    Raw path is per-frame MLP + mean-pool, not attention: the spectral
    branch already encodes time, and the caller mean-pools anyway, so
    attention is redundant. ``freq_norm`` norms each DCT frequency across
    the obs-feature axis before coding and projection.

    Known limit: ``freq_encoding`` runs on the feature axis, then
    mean-pools over frequency, so bins get no identity. See REPORT.md
    item 11.

    ``frame_norm`` normalizes the frame path. The critic's entire trunk is
    ``num_critics`` copies of this module, and it was the one unnormalized
    trunk in the model. Lyle et al. (2024) rank normalization above resets and
    Shrink-and-Perturb for plasticity. Also normalizes the actor's conditioner.

    A checkpoint from before this has no ``frame_norm.weight`` and will not
    resume; the failure surfaces as an optimizer group-size error.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        if obs_dim < 2:
            raise ValueError("StateConditioner obs_dim must be at least 2")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.obs_dim = obs_dim
        self.frame_proj = nn.Linear(obs_dim, hidden_dim)
        self.frame_norm = _rms_norm(hidden_dim)
        self.frame_mlp = SwiGLU(hidden_dim, 4 * hidden_dim, hidden_dim)
        self.freq_norm = _rms_norm(obs_dim)
        self.freq_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = nn.Linear(obs_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Window ``[B, T, obs]`` -> one ``[B, D]`` conditioning vector.

        ``ValueError`` if rank != 3, batch or time dim < 1, or ``obs_dim``
        disagrees with the config.
        """
        if states.dim() != 3:
            raise ValueError(
                f"states must have shape [B, T, obs], got {tuple(states.shape)}"
            )   
        if states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("states must have positive batch and time dimensions")
        if states.shape[2] != self.obs_dim:
            raise ValueError(
                f"states feature dimension must be {self.obs_dim}, got {states.shape[2]}"
            )
        raw = self.frame_mlp(self.frame_norm(self.frame_proj(states))).mean(dim=1)
        # num_modes == T keeps every frequency, the whole orthonormal transform.
        coeffs = extract_dct_coeffs(states, num_modes=states.shape[1])
        spec = self.key_proj(self.freq_encoding(self.freq_norm(coeffs))).mean(dim=1)
        return self.fuse(torch.cat([raw, spec], dim=-1))


class SJFPOFlowModel(nn.Module):
    """DiT velocity model over noisy action tokens, adaLN conditioned.

    ``return_stats=True`` gives ``(velocity, stats)`` plus per-block gate and
    scale RMS reductions. Default path returns velocity only, nothing
    reduced.
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
        timestep_scale: float,
    ):
        super().__init__()
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if dit_depth < 1 or dit_heads < 1 or time_embed_dim < 2:
            raise ValueError("dit_depth, dit_heads, and time_embed_dim must be positive")
        if (
            not math.isfinite(dit_mlp_ratio)
            or dit_mlp_ratio <= 0
            or int(hidden_dim * dit_mlp_ratio) < 1
        ):
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        if not math.isfinite(timestep_scale) or timestep_scale <= 0:
            raise ValueError("timestep_scale must be finite and positive")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.timestep_embedder = SinusoidalEncoding(
            self.time_embed_dim, scale=timestep_scale
        )
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.token_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.token_pos, std=0.02)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = _rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict velocity from token actions and per-token times.

        ``t``: scalar, ``[B]``, or ``[B, K, 1]``. One time per token, shared
        across that token's action dims.

        Two input paths:

        - raw: per-frame MLP, then mean-pool
        - spectral: slice the learned positional table to the token count

        Last layer is zero-init, so untrained predicts near-zero velocity.

        Returns ``[B, K, action_dim]``, plus stats if ``return_stats``.
        ``ValueError`` if ``x_t`` is not ``[B, K, action_dim]`` with positive
        dims, if tokens exceed the horizon, or if ``condition`` mismatches
        batch size or hidden width.
        """
        if x_t.dim() != 3:
            raise ValueError(f"x_t must have shape [B, K, D], got {tuple(x_t.shape)}")
        if x_t.shape[0] < 1 or x_t.shape[1] < 1:
            raise ValueError("x_t must have positive batch and token dimensions")
        if x_t.shape[2] != self.action_dim:
            raise ValueError(
                f"x_t feature dimension must be {self.action_dim}, got {x_t.shape[2]}"
            )
        if condition.dim() != 2 or condition.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"condition must have shape [{x_t.shape[0]}, {self.hidden_dim}], "
                f"got {tuple(condition.shape)}"
            )
        if condition.shape[1] != self.hidden_dim:
            raise ValueError(
                f"condition feature dimension must be {self.hidden_dim}, "
                f"got {condition.shape[1]}"
            )
        if x_t.shape[1] > self.action_horizon:
            raise ValueError(
                f"action token count {x_t.shape[1]} exceeds action_horizon "
                f"{self.action_horizon}"
            )

        t = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        token_axis = t.dim() in (2, 3) and tuple(t.shape[:2]) == tuple(x_t.shape[:2])
        if token_axis:
            if t.dim() == 3:
                if t.shape[-1] != 1:
                    raise ValueError(
                        f"t must end in a singleton action-dimension axis, got {tuple(t.shape)}"
                    )
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
                raise ValueError(
                    f"t must contain 1 or {x_t.shape[0]} values, got {t.numel()}"
                )
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
            raise ValueError(
                f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}"
            )

        if not return_stats:
            for block in self.blocks:
                h = block(h, c)
            if token_axis:
                shift_out, scale_out = self.adaLN_out(c).chunk(2, dim=-1)
                return self.out(modulate(self.norm_out(h), shift_out, scale_out))
            shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
            return self.out(modulate(self.norm_out(h), shift_out, scale_out))

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block(h, c, return_stats=True)
            block_stats.append(s)
        with torch.inference_mode():
            stats = {
                "temb": rms(temb),
                "fused": rms(condition),
                "c": rms(c),
                **{
                    k: torch.stack([s[k] for s in block_stats]).mean()
                    for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
                },
            }
        if token_axis:
            shift_out, scale_out = self.adaLN_out(c).chunk(2, dim=-1)
            out = self.out(modulate(self.norm_out(h), shift_out, scale_out))
        else:
            shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
            out = self.out(modulate(self.norm_out(h), shift_out, scale_out))
        return out, stats


class Critic(nn.Module):
    """V(s) over the full window.

    Each member is a ``StateConditioner`` plus a linear head, so the critic
    sees the window the way the actor does: per-frame mean-pool concatenated
    with a full orthonormal DCT, fused to ``hidden_dim``. ``num_modes == T`` is
    the whole basis, so nothing is truncated.

    Members stay independent or the ensemble is worthless: a shared trunk
    leaves only a linear map differing, and ``min`` over that means nothing.

    Returns ``[num_critics, B]``, unaggregated. Caller picks ``min`` for the
    GAE backup, ``mean`` for the value loss. ``min`` inside a loss would only
    train the lowest member.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_obs_steps: int = 1,
        num_critics: int = 1,
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
        self.n_obs_steps = n_obs_steps
        self.num_critics = num_critics
        self.conditioners = nn.ModuleList(
            [StateConditioner(obs_dim, hidden_dim) for _ in range(num_critics)]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_critics)]
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """``[B, T, obs]`` -> ``[num_critics, B]``.

        ``ValueError`` if rank != 3, ``obs_dim`` mismatched, or
        ``T != n_obs_steps``.
        """
        if states.dim() != 3:
            raise ValueError(
                f"states must have shape [B, T, obs], got {tuple(states.shape)}"
            )
        if states.shape[2] != self.obs_dim:
            raise ValueError(
                f"states feature dimension must be {self.obs_dim}, got {states.shape[2]}"
            )
        if states.shape[1] != self.n_obs_steps:
            raise ValueError(
                f"states has {states.shape[1]} time steps but the critic was built "
                f"for n_obs_steps={self.n_obs_steps}; the collector builds windows "
                "to the declared length"
            )
        return torch.stack(
            [
                head(conditioner(states)).squeeze(-1)
                for conditioner, head in zip(self.conditioners, self.heads)
            ]
        )
