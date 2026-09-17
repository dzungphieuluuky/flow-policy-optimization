"""Networks: window encoders (for robot state and action), spectral tokens, flow heads, critic.

Tensor conventions: B batch, N chunk batch, T obs window, H action_horizon,
K spectral tokens, D hidden dim, action_dim env action dim. Params are fp32;
compute may autocast matmuls (see utils.autocast_context).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.dct_utils import dct

# Single source for constructor defaults (see SJFPOConfig).
_DEFAULTS = SJFPOConfig()


class SwiGLU(nn.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    Drop-in for ``Sequential(Linear(in_dim, hidden_dim), SiLU,
    Linear(hidden_dim, out_dim))`` with the same (in, hidden, out) widths.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, hidden_dim)
        self.up = nn.Linear(in_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_dim] -> [..., out_dim]."""
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class CondMLP(nn.Module):
    """SiLU MLP for conditioning regression (DiT canon: SiLU then Linear).

    Gating belongs in block FFNs, not in the gate-regression path; the
    modulation readout stays (near-)linear. Final layer is zero-initialized
    by callers that need identity-at-init (adaLN-Zero)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_dim] -> [..., out_dim]."""
        return self.fc2(self.act(self.fc1(x)))


def resolve_config_value(name: str, override):
    """Policy default for ``name`` unless ``override`` is not None."""
    return override if override is not None else getattr(_DEFAULTS, name)

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position codes (stateless: nothing in state_dict)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position codes: [..., T, D] -> [..., T, D]."""
        t, dim = x.shape[-2], x.shape[-1]
        pos = torch.arange(t, dtype=torch.float32, device=x.device).unsqueeze(1)
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / half
        )
        ang = pos * freqs
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if pe.shape[-1] < dim:  # odd dim: pad the missing column with zeros
            pe = torch.nn.functional.pad(pe, (0, dim - pe.shape[-1]))
        return x + pe[:, :dim].to(dtype=x.dtype)

class WindowEncoder(nn.Module):
    """Two pre-norm residual blocks over a time window -> Z[B, L, hidden_dim].

    Single class for state windows and action chunks: the only difference is
    the optional boolean key mask [B, L] (True = pad, ignore as attention
    key), which hides padded tails. Actor and critic each own their instances
    (no RL weight sharing); call sites keep the ``state_encoder`` /
    ``action_encoder`` attribute names, so old checkpoints still load.
    """

    def __init__(
        self,
        in_dim: int,
        n_obs_steps: int | None = None,
        hidden_dim: int | None = None,
        n_heads: int = 4,
        mlp_ratio: float | None = None,
    ):
        super().__init__()
        n_obs_steps = resolve_config_value("n_obs_steps", n_obs_steps)
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        mlp_ratio = float(resolve_config_value("dit_mlp_ratio", mlp_ratio))
        self.frame_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp1 = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm4 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.mlp2 = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B, L, in_dim] -> [B, L, hidden_dim]; pad rows masked as keys when given."""
        # ------- project frames into latent space + add position codes -------
        h = self.frame_proj(x)
        h = self.pos_encoding(h)
        if key_mask is not None:
            key_mask = torch.as_tensor(key_mask, dtype=torch.bool, device=h.device)
        # ------- self-attention block 1 (pad rows masked as keys) -------
        n = self.norm1(h)
        attn, _ = self.attn1(n, n, n, key_padding_mask=key_mask)
        h = h + attn
        h = h + self.mlp1(self.norm2(h))
        # ------- self-attention block 2 -------
        n = self.norm3(h)
        attn, _ = self.attn2(n, n, n, key_padding_mask=key_mask)
        h = h + attn
        h = h + self.mlp2(self.norm4(h))
        return h

class RunningMeanStd(nn.Module):
    """Per-dim running mean/variance over raw obs frames (buffers ride state_dict).

    update() folds raw frames in with parallel Welford; forward() normalizes
    with frozen stats. Fresh (count == 0) is the identity.
    """

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0))
        self.eps = eps
        self.clip = clip

    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running stats (any device, no grad)."""
        batch = x.detach().reshape(-1, x.shape[-1]).to(self.mean.device)
        b = batch.shape[0]
        if b == 0:
            return
        b_mean = batch.mean(dim=0)
        b_var = batch.var(dim=0, unbiased=False)
        delta = b_mean - self.mean
        total = self.count + b
        self.mean += delta * b / total
        self.var = (
            self.count * self.var + b * b_var + delta.square() * self.count * b / total
        ) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with frozen stats; identity until the first update."""
        if int(self.count) == 0:
            return x
        norm = (x - self.mean.to(x.device)) * torch.rsqrt(
            self.var.to(x.device) + self.eps
        )
        return norm.clamp(-self.clip, self.clip)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift (shift/scale broadcast over tokens)."""
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        time_dim: int,
        scale: float = None,
    ):
        super().__init__()
        assert time_dim % 2 == 0
        self.time_dim = time_dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute a sinusoidal embedding for continuous timestep values.

        ``t`` is scaled by ``timestep_scale`` before applying the sinusoidal
        frequency ladder. This is useful when ``t`` is normalized to [0, 1]
        but the model was designed around DDPM-like timestep magnitudes.

        Returns an embedding of shape ``t.shape + (dim,)``.
        """
        scale = float(resolve_config_value("timestep_scale", self.scale))
        t = torch.as_tensor(t, dtype=torch.float32) * scale

        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        angles = t.unsqueeze(-1) * freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.time_dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    x <- x + gate_msa * Attn(modulate(norm(x))); x <- x + gate_mlp * MLP(...).
    Zero-initialized modulation makes each block start as the identity.
    FFN is SwiGLU (DiT uses GELU); width still follows mlp_ratio.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)
        self.last_gate_norms: dict[str, torch.Tensor] | None = None  # detached RMS stash

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """([B, H, D], [B, D]) -> [B, H, D]."""
        # ------- regress adaLN-Zero modulation from condition -------
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)[
            :, None, :
        ].chunk(6, dim=-1)
        # ------- gated self-attention residual -------
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn, _ = self.attn(h, h, h)
        x = x + gate_msa * attn
        # ------- gated MLP residual -------
        out = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        self.last_gate_norms = {
            "gate_msa": gate_msa.detach().pow(2).mean().sqrt(),
            "gate_mlp": gate_mlp.detach().pow(2).mean().sqrt(),
            "scale_msa": (1.0 + scale_msa).detach().pow(2).mean().sqrt(),
            "scale_mlp": (1.0 + scale_mlp).detach().pow(2).mean().sqrt(),
        }
        return out


class StateConditioner(nn.Module):
    """Fuse raw state window + its DCT bins into one [B, D] vector (Flamingo-style).

    State tokens query spectral tokens in one cross-attention block
    (residual), then mean-pool. Bin order rides a sinusoidal code, so
    attention weights index frequency.
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
        self.state_encoder = WindowEncoder(obs_dim, hidden_dim=hidden_dim)
        self.bin_norm = nn.RMSNorm(obs_dim)
        self.bin_encoding = PositionalEncoding()  # bin index = frequency order
        self.key_proj = nn.Linear(obs_dim, hidden_dim)
        self.norm_q = nn.RMSNorm(hidden_dim)
        self.norm_kv = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """[B, T, obs] -> [B, D]."""
        # ------- encode raw window into state tokens -------
        q = self.norm_q(self.state_encoder(states))
        # ------- encode DCT bins into spectral tokens -------
        kv = self.norm_kv(
            self.key_proj(self.bin_encoding(self.bin_norm(dct(states, dim=1))))
        )
        # ------- fuse spectral into state by cross-attention + pool -------
        attn, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out_proj((q + attn).mean(dim=1))


class SJFPOFlowModel(nn.Module):
    """DiT velocity model over noisy-action tokens, conditioned via adaLN.

    Takes a pre-fused condition vector [B, D] (see StateConditioner);
    ``c = time + condition`` modulates every block (adaLN-Zero). The
    transformer never sees state tokens, so there is exactly one
    observation path and no competing in-context route. Only action
    positions exist, so no readout slice.
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
        self.timestep_embedder = TimestepEmbedder(time_dim=self.time_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.bin_encoding = PositionalEncoding()
        self.time_mlp = CondMLP(dit_time_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = nn.RMSNorm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        # Zero final layer: an untrained policy predicts ~zero velocity.
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)
        self.last_cond_stats: dict[str, torch.Tensor] | None = None

    def forward(self, x_t: torch.Tensor, t, condition: torch.Tensor) -> torch.Tensor:
        """Predict velocity: ([B, Ka, action_dim], t, [B, D]) -> [B, Ka, action_dim].

        Ka = top-p low-frequency action rows (<= H); condition is the
        pre-fused state vector (see StateConditioner)."""
        # ------- embed flow timestep -------
        timesteps_embedding = self.timestep_embedder(t).to(
            device=x_t.device, dtype=x_t.dtype
        )
        if timesteps_embedding.dim() == 1:
            timesteps_embedding = timesteps_embedding.expand(x_t.shape[0], -1)
        # ------- blend time + fused state into one adaLN vector -------
        c = self.time_mlp(timesteps_embedding) + condition.to(dtype=x_t.dtype)
        # ------- embed noisy action tokens + add slot codes -------
        h = self.bin_encoding(self.in_proj(x_t))
        # ------- denoise through DiT blocks -------
        for block in self.blocks:
            h = block(h, c)
        self.last_cond_stats = {
            "temb": timesteps_embedding.detach().pow(2).mean().sqrt(),
            "fused": condition.detach().pow(2).mean().sqrt(),
            "c": c.detach().pow(2).mean().sqrt(),
            **{
                k: torch.stack(
                    [b.last_gate_norms[k] for b in self.blocks]
                ).mean()
                for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
            },
        }
        # ------- read out velocity -------
        shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
        return self.out(modulate(self.norm_out(h), shift_out, scale_out))

class Critic(nn.Module):
    """State-value baseline V(s) (PPO standard): mean-pooled window encoding."""

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
        self.state_encoder = WindowEncoder(
            obs_dim, n_obs_steps=n_obs_steps, hidden_dim=hidden_dim
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """[B, T, obs] -> [B]."""
        return self.out(self.state_encoder(states).mean(dim=1)).squeeze(-1)
