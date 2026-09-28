"""Networks: window encoders, state conditioner (raw + spectral), flow heads, V(s) critic.

Design notes:
* ``RunningMeanStd`` keeps ``count`` in fp64 (avoids fp32 saturation at 2^24
  frames) while mean/var stay fp32, and registers ``_uninitialized`` with
  ``persistent=False`` so older checkpoints still load.
* ``RunningMeanStd.forward`` pins the output dtype to the input dtype, so
  half-precision callers are not silently promoted to fp32.
* ``_RMSNorm`` runs its reduction in fp32 and uses the ``dtype=`` kwarg at
  construction instead of mutating ``.data`` (meta-device / ``to_empty``
  friendly).
* ``StateConditioner`` uses a per-frame raw-state MLP plus a temporal DCT
  branch; it does not instantiate an inner ``WindowEncoder``.
* ``SelfAttention`` takes an explicit ``[B, L]`` key-padding mask, validates
  its shape, and inverts it to reach SDPA (whose boolean mask means the
  opposite of the HF convention).
* ``temb`` broadcast replaces an explicit ``expand``, so a scalar ``t`` runs
  the time MLP once rather than B times.

Performance notes:
* Attention uses SDPA on CUDA.
* Sinusoidal frequency tables are cached per-module per-device.
* ``SwiGLU`` fuses gate and up into a single GEMM.
* ``DiTBlock`` / ``SJFPOFlowModel`` return diagnostics only when
  ``return_stats=True`` -- the default training path has no extra reductions.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from sjfpo.dct_utils import extract_dct_coeffs


class _RMSNorm(nn.RMSNorm):
    """RMSNorm computed in FP32 with the input dtype preserved on output.

    The normalization parameters are initialized in FP32. Activations may be
    FP16/BF16 under autocast, but the result is always cast back to the input
    activation dtype so normalization does not change the model-wide policy.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__(hidden_dim, eps=eps, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the trailing dimensions with the learned scale.

        The reduction always runs in fp32 and is cast back on exit, so a module
        converted to a lower parameter dtype with ``module.to`` still gets an accurate
        norm. Returns the normalised tensor, in the input dtype.
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
    """Build an fp32-safe RMS norm over the given width.
    """
    return _RMSNorm(hidden_dim, eps=eps)


class SwiGLU(nn.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    Gate and up are fused into a single ``gate_up`` projection (one GEMM
    instead of two); torch's weight layout is ``[out, in]`` so the fused
    weight is ``cat([gate.weight, up.weight], dim=0)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_up = nn.Linear(in_dim, 2 * hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated GELU activation; ``[..., in_dim]`` -> ``[..., hidden_dim]``.
        """
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
        """Apply the two-layer SiLU-gated MLP; ``[..., in_dim]`` -> ``[..., out_dim]``.
        """
        return self.fc2(F.silu(self.fc1(x)))


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal codes with cached frequency tables.

    Cache is a small per-module LRU keyed by ``torch.device``.
    """

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        if dim < 2:
            raise ValueError(f"SinusoidalEncoding dim must be >= 2, got {dim}")
        self.dim = dim
        self.scale = 1.0 if scale is None else float(scale)
        self.half = dim // 2
        self._freq_cache: OrderedDict[torch.device, torch.Tensor] = OrderedDict()

    def _freqs(self, device: torch.device) -> torch.Tensor:
        """Return the cached ``[half]`` frequency table for a device.

        At most four devices are cached; the least recently used is evicted beyond
        that.
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

    def clear_cache(self) -> None:
        """Drop the cached device-specific frequency tables.

        Needed after moving the module between devices, since the cache is
        keyed on the original ones.
        """
        self._freq_cache.clear()

    def add_to(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal codes for the given positions, growing ``[..., L, D]`` in place
        of the caller's base tensor.
        """
        ang = (positions.to(dtype=torch.float32) * self.scale).unsqueeze(
            -1
        ) * self._freqs(x.device)
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if pe.shape[-1] < self.dim:
            pe = F.pad(pe, (0, self.dim - pe.shape[-1]))
        return x + pe.to(dtype=x.dtype)

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        """Embed continuous timestep values sinusoidally; ``[N]`` -> ``[N, 2 * half]``.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float32)
        t = t.to(dtype=torch.float32) * self.scale
        ang = t.unsqueeze(-1) * self._freqs(t.device)
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def positions(self, length: int, device) -> torch.Tensor:
        """Build the ``[length]`` index vector holding ``0..length-1``, used for slot or
        frame codes.
        """
        return torch.arange(length, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position codes to a ``[..., L, D]`` tensor, indexed by slot.
        """
        return self.add_to(x, self.positions(x.shape[-2], x.device))


class SelfAttention(nn.Module):
    """Multi-head self-attention on SDPA.

    ``attention_mask`` is a key-padding mask with shape ``[B, L]``. A
    True/1 entry marks a valid key and False/0 marks an ignored key. Every
    batch item is forced to retain key 0 when its mask is otherwise empty so
    SDPA cannot produce NaNs; arbitrary query/key masks are not supported.
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
        """Run multi-head self attention.

        ``attention_mask`` is a boolean/0-1 key-padding mask of shape ``[B, L]``. Under
        the Hugging Face convention 1/True marks a valid key and 0/False is ignored, so
        the mask is inverted before it reaches SDPA, whose boolean mask means the
        opposite. Every batch item is forced to keep at least one key, because an
        all-masked row makes SDPA emit NaN. The fused ``[B, L, H*D]`` projection is
        reshaped with a view plus transpose rather than a permute.

        Raises ``ValueError`` if ``x`` is not 3D, or the mask length disagrees with
        the sequence length.
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


class WindowEncoder(nn.Module):
    """Pre-norm residual block over a time window."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        if in_dim < 1 or hidden_dim < 2:
            raise ValueError("WindowEncoder dimensions must be positive")
        if (
            not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or int(hidden_dim * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.frame_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = SinusoidalEncoding(hidden_dim)
        self.norm1 = _rms_norm(hidden_dim, eps=1e-6)
        self.attn1 = SelfAttention(hidden_dim, n_heads)
        self.norm2 = _rms_norm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp1 = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project an observation window and run self attention over it.

        ``attention_mask`` is a boolean/0-1 key-padding mask of shape ``[B, L]``;
        1/True marks a valid key and 0/False is ignored. Returns ``[B, L, hidden_dim]``.
        """
        h = self.pos_encoding(self.frame_proj(x))
        h = h + self.attn1(self.norm1(h), attention_mask)
        return h + self.mlp1(self.norm2(h))


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
        self.register_buffer(
            "_uninitialized", torch.tensor(True, dtype=torch.bool), persistent=False
        )
        self.eps = eps
        self.clip = clip

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running statistics.

        The merge follows Chan et al.'s parallel variance formula in fp64 and casts the
        result back to fp32, so incremental updates do not drift. The latch buffer is
        non-persistent so pre-change checkpoints load without it. Raises
        ``ValueError`` if the trailing dimension does not match the tracked
        observation size.
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
        b_mean = batch.mean(dim=0)
        b_var = batch.var(dim=0, unbiased=False)
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
        self._uninitialized.fill_(False)

    @torch.inference_mode()
    def reset(self) -> None:
        """Clear the one-way latch and reset stats to their initial values."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()
        self._uninitialized.fill_(True)

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state and re-derive the one-way initialisation latch.

        The latch is not persisted, so a freshly loaded module whose count is still zero
        must pass frames through unchanged. Positional and keyword arguments are
        forwarded to ``nn.Module``.
        """
        super()._load_from_state_dict(*args, **kwargs)
        self._uninitialized.copy_(self.count.eq(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise with the frozen running statistics.

        Returns ``x`` unchanged until the first update.
        """
        norm = (x - self.mean.to(x.device)) * torch.rsqrt(
            self.var.to(x.device) + self.eps
        )
        norm = norm.clamp(-self.clip, self.clip).to(x.dtype)
        return torch.where(self._uninitialized.to(x.device), x, norm)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer-norm modulation: ``x * (1 + scale) + shift``.
    """
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    ``return_stats=True`` returns ``(out, stats_dict)``; the default path
    returns the tensor only, so no diagnostic reductions run in training.
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

        The modulation head is zero-initialised, so the block starts as an identity
        mapping (adaLN-Zero) and the network begins as a no-op. On the
        coordinate-aware ``[B, K, D_a, H]`` layout, attention runs over the K action
        tokens independently for each action dimension, and only the batch/attention
        axis is reshaped. The modulation parameters are gathered with ``reshape`` plus
        ``unbind`` to avoid materialising a ``[B, 1, 6H]`` intermediate.

        ``c`` is the conditioning, ``[B, D]`` or ``[B, K, D]``. Returns the transformed
        tokens, or those plus a stats dict when ``return_stats`` is set.
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
                "gate_msa": gate_msa.square().mean().sqrt(),
                "gate_mlp": gate_mlp.square().mean().sqrt(),
                "scale_msa": (1.0 + scale_msa).square().mean().sqrt(),
                "scale_mlp": (1.0 + scale_mlp).square().mean().sqrt(),
            }
        return out, stats


class StateConditioner(nn.Module):
    """Fuse a mean-pooled per-frame state summary with DCT-frequency features.

    The raw path uses a per-frame MLP + mean-pool rather than an attention
    block: the spectral branch already encodes temporal structure explicitly,
    so the attention block is redundant for the chunk-level conditioning
    vector (the caller mean-pools the result anyway). ``freq_norm`` normalizes
    each DCT frequency across the observation-feature dimension before
    positional encoding and projection.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        if obs_dim < 2:
            raise ValueError("StateConditioner obs_dim must be at least 2")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.obs_dim = obs_dim
        self.frame_proj = nn.Linear(obs_dim, hidden_dim)
        self.frame_mlp = SwiGLU(hidden_dim, 4 * hidden_dim, hidden_dim)
        self.freq_norm = _rms_norm(obs_dim)
        self.freq_encoding = SinusoidalEncoding(obs_dim)
        self.key_proj = nn.Linear(obs_dim, hidden_dim)
        self.fuse = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Fuse the observation window into a single ``[B, D]`` conditioning vector.

        Raises ``ValueError`` if the rank is not 3, the batch or time dimension is not
        positive, or ``obs_dim`` disagrees with the configuration.
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
        raw = self.frame_mlp(self.frame_proj(states)).mean(dim=1)
        # M = T-1 keeps all T frequencies, so this is the full orthonormal
        # transform of the window; extract_dct_coeffs is the only public entry.
        coeffs = extract_dct_coeffs(states, M=states.shape[1] - 1)
        spec = self.key_proj(self.freq_encoding(self.freq_norm(coeffs))).mean(dim=1)
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

        ``t`` is either a scalar, ``[B]``, or ``[B, K, 1]``. The per-token form gives
        each action token its own time, shared across its action dimensions.

        The raw input path applies a per-frame MLP and mean-pools, replacing a full
        attention window over frames. The spectral path instead slices the learned
        positional table, which carries one row per configured action position, down to
        the reduced DCT token count. The final layer is zero-initialised so an
        untrained model predicts near-zero velocity.

        Returns a velocity of shape ``[B, K, action_dim]``, or that plus a stats dict
        when ``return_stats`` is set. Raises ``ValueError`` if ``x_t`` is not
        ``[B, K, action_dim]`` with positive batch and token dimensions, if it carries
        more tokens than the action horizon, or if ``condition`` disagrees with the
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
                "temb": temb.square().mean().sqrt(),
                "fused": condition.square().mean().sqrt(),
                "c": c.square().mean().sqrt(),
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
    """State-value baseline V(s): MLP on current frame."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.obs_dim = obs_dim
        self.mlp = CondMLP(obs_dim, hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Predict the value of the current frame in an observation window, ``[B]``.

        Raises ``ValueError`` if the rank is not 3, the batch or time dimension is not
        positive, or ``obs_dim`` disagrees with the configuration.
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
        return self.mlp(states[:, -1, :]).squeeze(-1)
