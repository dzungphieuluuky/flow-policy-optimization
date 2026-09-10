"""Networks: context trunk, SigReg, spectral tokens, flow heads, critic, EMA projection.

Plan tasks 3-7. All modules are fp32-only, use ELU activations, and contain no
BatchNorm / Dropout(p>0) / batch-statistics normalization in inference paths.
CELLORDER: 5

Tensor conventions (used throughout this module):
    B: batch size
    N: context length (default 16)
    H: action-chunk horizon (default 8)
    K: low-K spectral tokens (default 5)
    D: fused latent dimension Z_dim (default 128)
    Dk: spectral projection dimension (default 64)
    Da: action dimension (default 7, env-dependent)
    Ds: state dimension (default D = 128)
    T: generic sequence length (often N or H)

All tensors are fp32. Run-matrix construction is MuJoCo-only (flat Box);
it does not affect forward-pass logic in this module.
"""

# %%
from __future__ import annotations

import math
import warnings

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from mfpo import config


def resolve_config_value(name: str, override):
    """Config default with explicit-argument override.

    Args:
        name: Attribute name on ``config.MFPO_CFG``.
        override: Explicit value to use if not ``None``.

    Returns:
        The override value if provided, otherwise ``getattr(config.MFPO_CFG, name)``.
    """
    return override if override is not None else getattr(config.MFPO_CFG, name)


# %% [markdown]
# ## Sinusoidal timestep embedding (8 dims: cos/sin of t*2^k)
#
# Convention from amazon-far/fpo-control `_embed_timestep` (arXiv 2602.02481
# III-A): the first half is sin(t*2^k), the second half cos(t*2^k),
# k = 0 .. dim/2-1.


# %%
def timestep_embedding(t, dim: int = 8) -> torch.Tensor:
    """Embed scalar-or-[B] timestep(s) as [..., dim] sinusoidal features.

    Convention from amazon-far/fpo-control `_embed_timestep` (arXiv 2602.02481
    III-A): the first half is sin(t*2^k), the second half cos(t*2^k),
    k = 0 .. dim/2-1.

    Args:
        t: Scalar or tensor of shape [..., B] or [B].
        dim: Embedding dimension (must be even). Default 8.

    Returns:
        Tensor of shape [..., dim] with sin/cos features.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    half = dim // 2
    freqs = 2.0 ** torch.arange(half, dtype=torch.float32)  # [half]
    ang = t.unsqueeze(-1) * freqs  # [..., half]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # [..., dim]


# %% [markdown]
# ## ContextTrunk — 16 stacked obs frames -> per-frame encoder -> attention
# fusion -> latent sequence Z[B, T, Z_dim]. Vision DiT dropped per user
# decision D2: single proprio-style modality, so "cross-attn fusion" is a
# residual self-attention block over the frame-token sequence.


# %%
class ContextTrunk(nn.Module):
    """Context trunk: stacked observation frames -> per-frame encoder -> attention fusion.

    Maps a sequence of stacked frames to a latent sequence via a frame-wise encoder
    followed by a residual self-attention block.

    Args:
        obs_dim: Input observation dimension per frame.
        context_len: Number of stacked frames (default from config.MFPO_CFG.context_len).
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        n_heads: Number of attention heads. Default 4.

    Attributes:
        frame_enc: Sequential(Linear(obs_dim -> z_dim), ELU()).
        fusion: MultiheadAttention(z_dim, n_heads, batch_first=True).
    """

    def __init__(
        self,
        obs_dim: int,
        context_len: int | None = None,
        z_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        context_len = resolve_config_value("context_len", context_len)
        z_dim = resolve_config_value("Z_dim", z_dim)
        self.frame_enc = nn.Sequential(nn.Linear(obs_dim, z_dim), nn.ELU())
        self.fusion = nn.MultiheadAttention(z_dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Stacked frames of shape [B, T, obs_dim] where T = context_len.

        Returns:
            Z: Latent sequence of shape [B, T, Z_dim].
        """
        h = self.frame_enc(x)  # [B, T, Z_dim]
        attn, _ = self.fusion(h, h, h)  # [B, T, Z_dim]
        return nn.functional.elu(h + attn)  # [B, T, Z_dim]


# %% [markdown]
# ## SIGReg — Sketched Isotropic Gaussian Regularizer (Epps-Pulley, LeWorldModel)
#
# Penalizes the empirical characteristic function (CF) of resampled random 1-D
# projections of Z toward the N(0,1) target CF phi(t)=exp(-t^2/2), via
# trapezoidal quadrature on [0,3].  Replicates the official implementation from
# lucas-maes/le-wm (arXiv 2603.19312, Algorithm 1).

# Module-level quadrature nodes / weights for the default knots=17 (cached once).
_DEFAULT_KNOTS = 17
_t_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_quadrature(knots: int, device: torch.device):
    """Return cached (t, weights_scaled, phi) for trapezoid on [0, 3].

    t:       [K]            quadrature nodes
    w_phi:   [K]            weights * phi (pre-multiplied, same dtype/device as inputs)
    phi:     [K]            N(0,1) CF target exp(-t^2/2)
    """
    if knots not in _t_cache:
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        phi = torch.exp(-t.square() / 2.0)
        _t_cache[knots] = (t, w * phi, phi)
    t, w_phi, phi = _t_cache[knots]
    # Move to requested device; cached tensors are CPU; re-cache if new device.
    if t.device != device:
        t = t.to(device)
        w_phi = w_phi.to(device)
        phi = phi.to(device)
    return t, w_phi, phi


# %%
def sigreg_losses(
    z: torch.Tensor,
    *,
    knots: int = 17,
    num_proj: int = 1024,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularizer (Epps-Pulley CF test).

    Penalizes the empirical CF of random 1-D projections of *z* toward the
    standard-normal CF phi(t) = exp(-t^2/2).  Returns a single 0-dim scalar
    loss (differentiable; gradients flow through the cosine/sine activations).

    Replicates the official LeWorldModel SIGReg.forward() (lucas-maes/le-wm,
    arXiv 2603.19312).

    Args:
        z: Tensor of shape [N, T, D] (project layout; internally permuted to
            the official (T, B=N, D) convention).
        knots: Number of quadrature nodes in the trapezoidal rule on [0, 3].
        num_proj: Number of random projections (columns of the sketch matrix).
        projection: Optional pre-computed projection matrix [D, num_proj] with
            unit-norm columns.  When None (default), a fresh matrix is sampled
            from N(0,1) and normalized each call (stochastic estimator).

    Returns:
        Single 0-dim scalar loss (differentiable).
    """
    zp = z.permute(1, 0, 2)  # (T, B=N, D) — official le-wm input layout
    T_count, B, D = zp.shape

    t, w_phi, phi = _get_quadrature(knots, zp.device)

    # Projection matrix: unit-norm columns
    if projection is None:
        A = torch.randn(D, num_proj, device=zp.device, dtype=zp.dtype)
        A = A / A.norm(dim=0, keepdim=True)
    else:
        A = projection

    # x_t = (zp @ A).unsqueeze(-1) * t  →  (T, B, M, K)
    x_t = (zp @ A).unsqueeze(-1) * t

    # Empirical CF error vs N(0,1): mean over B (dim 1)
    cos_mean = x_t.cos().mean(1)  # (T, M, K)
    sin_mean = x_t.sin().mean(1)
    err = (cos_mean - phi).square() + sin_mean.square()  # (T, M, K)

    # Trapezoidal integral + N scaling (B), then average over (T, M)
    statistic = (err @ w_phi) * B  # (T, M)
    return statistic.mean()


# %% [markdown]
# ## SpectralTokens — Hann-window rFFT low-K tokens with FROZEN stats
#
# Early design computed per-batch z-scored FFT tokens; frozen statistics
# replaced the batch statistics. Frozen mu/std live in
# Z-space ([Z_dim], matching the plan's `set_frozen_stats(zeros(128),
# ones(128))`): inputs are z-scored with the frozen stats BEFORE windowing,
# and the drift guard compares the raw batch-mean against them under no_grad.
# forward raises RuntimeError('frozen stats not initialized') until stats are
# set via set_frozen_stats(mu, std) or captured once via capture_stats(sample).


# %%
class SpectralTokens(nn.Module):
    """Hann-window rFFT tokens with FROZEN post-cat norm + learnable projector.

    Order of operations (plan binding):
        Z ⊙ hann → rFFT(dim=1) → cat(real, imag) → frozen norm → Linear(2D→Dk)

    Args:
        z_dim: Latent dimension Z_dim (default from config.MFPO_CFG.Z_dim).
        k_spectral: Number of low-frequency tokens K (default from config.MFPO_CFG.K_spectral).
        context_len: Context length N (default from config.MFPO_CFG.context_len).

    Attributes:
        z_dim: Latent dimension D.
        k: Effective K = min(k_spectral, context_len // 2 + 1).
        hann: Hann window buffer of shape [context_len].
        frozen_mu: Frozen mean buffer of shape [2 * z_dim].
        frozen_std: Frozen std buffer of shape [2 * z_dim].
        _has_stats: Boolean flag indicating if frozen stats are initialized.
        projector: Linear(2 * z_dim -> Dk).

    Raises:
        RuntimeError: If forward() called before frozen stats are initialized via
            set_frozen_stats() or capture_stats().
    """

    def __init__(
        self,
        z_dim: int | None = None,
        k_spectral: int | None = None,
        context_len: int | None = None,
    ):
        super().__init__()
        z_dim = int(resolve_config_value("Z_dim", z_dim))
        k_spectral = int(resolve_config_value("K_spectral", k_spectral))
        context_len = int(resolve_config_value("context_len", context_len))
        self.z_dim = z_dim
        self.k = min(k_spectral, context_len // 2 + 1)
        cat_dim = 2 * z_dim  # after cat(real, imag)
        self.register_buffer("hann", torch.hann_window(context_len))  # [context_len]
        self.register_buffer("frozen_mu", torch.zeros(cat_dim))  # [2*D]
        self.register_buffer("frozen_std", torch.ones(cat_dim))  # [2*D]
        self._has_stats = False
        dk = (
            int(resolve_config_value("Dk", 64))
            if hasattr(config, "MFPO_CFG") and hasattr(config.MFPO_CFG, "Dk")
            else 64
        )
        self.projector = nn.Linear(cat_dim, dk)  # [2*D -> Dk]

    def set_frozen_stats(self, mu: torch.Tensor, std: torch.Tensor) -> None:
        """Set frozen normalization statistics.

        Args:
            mu: Mean tensor of shape [2 * z_dim].
            std: Std tensor of shape [2 * z_dim], clamped to min 1e-4.

        Raises:
            ValueError: If mu or std have incorrect shape.
        """
        cat_dim = 2 * self.z_dim
        if mu.shape != (cat_dim,) or std.shape != (cat_dim,):
            raise ValueError(
                f"frozen stats must have shape ({cat_dim},), "
                f"got {tuple(mu.shape)} / {tuple(std.shape)}"
            )
        with torch.no_grad():
            self.frozen_mu.copy_(mu.detach().to(torch.float32))  # [2*D]
            self.frozen_std.copy_(
                std.detach().to(torch.float32).clamp_min(1e-4)
            )  # [2*D]
        self._has_stats = True

    def capture_stats(self, z_sample: torch.Tensor) -> None:
        """Capture frozen stats ONCE from a representative Z sample.

        Stats are computed AFTER rFFT+cat so they reflect the actual FFT
        feature distribution, not the raw Z-space distribution.

        Args:
            z_sample: Sample tensor of shape [B, T, Z_dim].
        """
        with torch.no_grad():
            zw = z_sample.detach() * self.hann.view(1, -1, 1)  # [B, T, D]
            fft = torch.fft.rfft(zw, dim=1)  # [B, T//2+1, D] complex
            feat = torch.cat([fft.real, fft.imag], dim=-1)  # [B, T//2+1, 2*D]
            flat = feat.reshape(-1, 2 * self.z_dim)  # [B*(T//2+1), 2*D]
            mu = flat.mean(dim=0)  # [2*D]
            std = flat.std(dim=0).clamp_min(1e-4)  # [2*D]
        self.set_frozen_stats(mu, std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: Z -> spectral tokens.

        Args:
            z: Latent sequence of shape [B, T, Z_dim] where T = context_len.

        Returns:
            tokens: Spectral tokens of shape [B, K, Dk].

        Raises:
            RuntimeError: If frozen stats not initialized.
        """
        if not self._has_stats:
            raise RuntimeError("frozen stats not initialized")
        with torch.no_grad():
            fft_raw = torch.fft.rfft(
                z * self.hann.view(1, -1, 1), dim=1
            )  # [B, T//2+1, D] complex
            feat_raw = torch.cat(
                [fft_raw.real, fft_raw.imag], dim=-1
            )  # [B, T//2+1, 2*D]
            drift = (
                feat_raw.reshape(-1, 2 * self.z_dim).mean(dim=0) - self.frozen_mu
            )  # [2*D]
            if bool((drift.abs() > 3 * self.frozen_std).any()):
                warnings.warn("spectral stats drift", UserWarning, stacklevel=2)
        zw = z * self.hann.view(1, -1, 1)  # [B, T, D]
        fft = torch.fft.rfft(zw, dim=1)  # [B, T//2+1, D] complex
        feat = torch.cat([fft.real, fft.imag], dim=-1)  # [B, T//2+1, 2*D]
        feat = (feat - self.frozen_mu) / self.frozen_std  # [B, T//2+1, 2*D]
        return self.projector(feat[:, : self.k, :])  # [B, K, Dk]


# %% [markdown]
# ## Flow heads — velocity prediction in orthonormal-DCT coefficient space
#
# Inputs: x_t[..., CHUNK, A] (DCT coefficients), t scalar-or-[B], conditioning
# token sources forming the plan's [Z; fft] sequence. Sources with last dim
# != Z_dim (e.g. SpectralTokens' [B, K, 2*Z_dim]) are projected to Z_dim by
# their own Linear before token-concatenation; a single [B, T_c, Z_dim]
# tensor passes through unchanged. Query = ELU(Linear(x_t ⊕ t_embed)),
# keys/values = cond via nn.MultiheadAttention(Z_dim, 4). Output velocity has
# the SAME shape as x_t.


# %%
class FlowHeadBase(nn.Module):
    """Base flow head: velocity prediction in DCT coefficient space.

    Inputs: x_t[..., CHUNK, A] (DCT coefficients), t scalar-or-[B], conditioning
    token sources forming the plan's [Z; fft] sequence. Sources with last dim
    != Z_dim (e.g. SpectralTokens' [B, K, 2*Z_dim]) are projected to Z_dim by
    their own Linear before token-concatenation; a single [B, T_c, Z_dim]
    tensor passes through unchanged. Query = ELU(Linear(x_t ⊕ t_embed)),
    keys/values = cond via nn.MultiheadAttention(Z_dim, 4). Output velocity has
    the SAME shape as x_t.

    Args:
        in_dim: Input dimension (action_dim or state_dim).
        out_dim: Output dimension (same as in_dim).
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        n_heads: Number of attention heads. Default 4.
        cond_dims: Tuple of conditioning source dimensions. Each source with
            dim != z_dim gets a Linear projection to z_dim. Default (z_dim,).

    Attributes:
        in_proj: Linear(in_dim + 8 -> z_dim) for query projection.
        cond_projs: ModuleList of Linear projections for conditioning sources.
        attn: MultiheadAttention(z_dim, n_heads, batch_first=True).
        out: Linear(z_dim -> out_dim) for velocity output.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        z_dim: int | None,
        n_heads: int = 4,
        cond_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()
        z_dim = int(resolve_config_value("Z_dim", z_dim))
        cond_dims = cond_dims if cond_dims is not None else (z_dim,)
        self.in_proj = nn.Linear(in_dim + 8, z_dim)
        self.cond_projs = nn.ModuleList(
            nn.Linear(d, z_dim) if d != z_dim else nn.Identity() for d in cond_dims
        )
        self.attn = nn.MultiheadAttention(z_dim, n_heads, batch_first=True)
        self.out = nn.Linear(z_dim, out_dim)

    def forward(self, x_t: torch.Tensor, t, cond) -> torch.Tensor:
        """Forward pass: predict velocity.

        Args:
            x_t: Noisy input of shape [B, CHUNK, in_dim] (DCT coefficients).
            t: Timestep scalar or tensor of shape [B].
            cond: Conditioning tensor(s) [B, T_c, D_c] or tuple thereof.

        Returns:
            Velocity prediction of shape [B, CHUNK, out_dim].
        """
        temb = timestep_embedding(t, dim=8).to(x_t.dtype)  # [B, 8] or [8]
        if temb.dim() == 1:
            temb = temb.expand(x_t.shape[0], -1)  # [B, 8]
        chunk_len = x_t.shape[-2]
        xt = torch.cat(
            [x_t, temb.unsqueeze(-2).expand(*x_t.shape[:-2], chunk_len, 8)],
            dim=-1,
        )  # [B, CHUNK, in_dim+8]
        q = nn.functional.elu(self.in_proj(xt))  # [B, CHUNK, z_dim]
        sources = (cond,) if isinstance(cond, torch.Tensor) else tuple(cond)
        if len(sources) != len(self.cond_projs):
            raise ValueError(
                f"expected {len(self.cond_projs)} conditioning source(s), "
                f"got {len(sources)}"
            )
        kv = torch.cat(
            [proj(c) for proj, c in zip(self.cond_projs, sources)], dim=1
        )  # [B, sum(T_c), z_dim]
        attn, _ = self.attn(q, kv, kv)  # [B, CHUNK, z_dim]
        return self.out(nn.functional.elu(q + attn))  # [B, CHUNK, out_dim]


class PolicyFlowHead(FlowHeadBase):
    """Policy velocity head over action-chunk DCT coefficients.

    Args:
        action_dim: Action dimension Da (default 7, env-dependent).
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        cond_dims: Conditioning source dimensions (default (z_dim,)).
    """

    def __init__(
        self,
        action_dim: int,
        z_dim: int | None = None,
        cond_dims: tuple[int, ...] | None = None,
    ):
        super().__init__(action_dim, action_dim, z_dim, cond_dims=cond_dims)


class AuxStateFlowHead(FlowHeadBase):
    """Auxiliary future-state velocity head over DCT(future 8 obs) space.

    Args:
        state_dim: State dimension (default from config.MFPO_CFG.Z_dim).
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        cond_dims: Conditioning source dimensions (default (z_dim,)).
    """

    def __init__(
        self,
        state_dim: int | None = None,
        z_dim: int | None = None,
        cond_dims: tuple[int, ...] | None = None,
    ):
        state_dim = int(resolve_config_value("Z_dim", state_dim))
        super().__init__(state_dim, state_dim, z_dim, cond_dims=cond_dims)


# %% [markdown]
# ## ValueCritic — meanpool Z over time -> MLP(256,256,256) ELU -> scalar V[B]


# %%
class ValueCritic(nn.Module):
    """Value critic: mean-pool Z over time -> MLP(256,256,256) ELU -> scalar V[B].

    Args:
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        hidden: Hidden layer size. Default 256.

    Attributes:
        net: Sequential MLP with 3 hidden layers of size `hidden` and ELU activations.
    """

    def __init__(self, z_dim: int | None = None, hidden: int = 256):
        super().__init__()
        z_dim = resolve_config_value("Z_dim", z_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent sequence of shape [B, T, Z_dim].

        Returns:
            V: Scalar value per batch of shape [B].
        """
        return self.net(z.mean(dim=1)).squeeze(-1)  # [B]


# %% [markdown]
# ## EMAProjection — online/target linear obs->latent via AveragedModel
#
# Internal: AveragedModel(self.online, multi_avg_fn=get_ema_multi_avg_fn(decay))
# target proxies averaged.module; decay=config.MFPO_CFG.ema_decay=0.99 via
# torch._foreach_lerp_; ema_update once per optimizer step (Metis #24);
# copy_from hard-sets target := online and resets n_averaged.


# %%
class EMAProjection(nn.Module):
    """EMA projection: online/target linear obs->latent via AveragedModel.

    Internal: AveragedModel(self.online, multi_avg_fn=get_ema_multi_avg_fn(decay))
    with target property -> averaged.module. Formula retained:
    theta_target <- decay*theta_target + (1-decay)*theta_online via
    torch._foreach_lerp_ (decay=config.MFPO_CFG.ema_decay=0.99), called once per
    optimizer step (Metis #24). copy_from hard-sets target := online and
    resets n_averaged so next ema_update hard-copies. state_dict keys:
    online.* + averaged.module.* + averaged.n_averaged.

    Args:
        obs_dim: Observation dimension.
        latent: Latent dimension (default from config.MFPO_CFG.Z_dim).
        ema_decay: EMA decay factor (default from config.MFPO_CFG.ema_decay).

    Attributes:
        decay: EMA decay factor.
        online: Online Linear(obs_dim -> latent).
        averaged: AveragedModel wrapping the target projection.
        target: Target Linear(obs_dim -> latent) with frozen parameters
            (property proxying averaged.module).
    """

    def __init__(
        self,
        obs_dim: int,
        latent: int | None = None,
        ema_decay: float | None = None,
    ):
        super().__init__()
        latent = resolve_config_value("Z_dim", latent)
        self.decay = float(resolve_config_value("ema_decay", ema_decay))
        self.online = nn.Linear(obs_dim, latent)
        self.averaged = AveragedModel(
            self.online, multi_avg_fn=get_ema_multi_avg_fn(self.decay)
        )
        for p in self.averaged.module.parameters():
            p.requires_grad_(False)
        self.copy_from(self.online)

    @property
    def target(self) -> nn.Module:
        """Target projection (property proxying averaged.module)."""
        return self.averaged.module

    @torch.no_grad()
    def ema_update(self, step: int | None = None) -> None:
        """Polyak blend toward the online copy; `step` is informational only.

        Args:
            step: Optional step counter (unused, for logging compatibility).
        """
        del step
        self.averaged.update_parameters(self.online)

    @torch.no_grad()
    def copy_from(self, source: nn.Module | None = None) -> None:
        """Hard-set target := source (defaults to the online projection).

        Sets n_averaged to 1 so the next ema_update performs Polyak
        blending (decay*target + (1-decay)*online) rather than hard-copy.

        Args:
            source: Module to copy from (default: self.online).
        """
        src = source if source is not None else self.online
        self.averaged.module.load_state_dict(src.state_dict())
        self.averaged.n_averaged.fill_(1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode obs with the ONLINE projection.

        Args:
            obs: Observation tensor of shape [B, obs_dim].

        Returns:
            Latent encoding of shape [B, latent].
        """
        return self.online(obs)  # [B, latent]


# %%
class JointFlowTransformer(nn.Module):
    """Velocity predictor for joint [action, state] targets.

    Predicts v_θ(x_t, t, z_context) where x_t ∈ R^{B×H×(Da+D)} is the
    noisy joint state at time t, and z_context ∈ R^{B×H×D} is the fused
    conditioning from the encoders.

    T-convention (plan binding): t=0 → noise (predict target), t=1 → data
    (predict noise). Flow: z_t = (1-t)·z_0 + t·X*, v* = X* - z_0.

    Args:
        da: Action dimension Da. Default 7.
        ds: State dimension Ds (default from config.MFPO_CFG.Z_dim).
        z_dim: Latent dimension (default from config.MFPO_CFG.Z_dim).
        n_heads: Number of attention heads. Default 4.

    Attributes:
        da: Action dimension.
        ds: State dimension.
        in_proj: Linear(Da+Ds+8 -> z_dim) for query projection.
        cond_proj: Identity (since z_dim == z_dim).
        attn: MultiheadAttention(z_dim, n_heads, batch_first=True).
        out: Linear(z_dim -> Da+Ds) for velocity output.

    Note:
        A ``FlowHeadBase`` drop-in subclass was considered (tier-b task 8) but
        deferred: the name ``FlowHeadBase`` is already taken by the MuJoCo
        base class at line ~316 (parent of ``PolicyFlowHead``/``AuxStateFlowHead``),
        and the base ``JointFlowTransformer`` has no ``cond_dims`` constructor
        param nor a ``T_FFT`` forward arg, so a same-named subclass would shadow
        the base class and diverge from the base signature. This class is the
        canonical joint flow head; no subclass is forced.
    """

    def __init__(
        self,
        da: int = 7,
        ds: int | None = None,
        z_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        ds = int(resolve_config_value("Z_dim", ds))
        z_dim = int(resolve_config_value("Z_dim", z_dim))
        self.da = da
        self.ds = ds
        joint_dim = da + ds
        self.in_proj = nn.Linear(joint_dim + 8, z_dim)  # +8 for timestep embedding
        self.cond_proj = nn.Identity()
        self.attn = nn.MultiheadAttention(z_dim, n_heads, batch_first=True)
        self.out = nn.Linear(z_dim, joint_dim)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t: Noisy joint state of shape [B, H, Da+Ds].
            t: Timestep tensor of shape [B].
            cond: Conditioning features of shape [B, H, D].

        Returns:
            Velocity prediction of shape [B, H, Da+Ds].
        """
        B, H, _ = x_t.shape
        temb = timestep_embedding(t, dim=8).to(x_t.dtype)  # [B, 8]
        xt = torch.cat(
            [x_t, temb.unsqueeze(1).expand(-1, H, -1)], dim=-1
        )  # [B, H, Da+Ds+8]
        q = nn.functional.elu(self.in_proj(xt))  # [B, H, z_dim]
        kv = self.cond_proj(cond)  # [B, H, z_dim]
        attn, _ = self.attn(q, kv, kv)  # [B, H, z_dim]
        return self.out(nn.functional.elu(q + attn))  # [B, H, Da+Ds]


# %%
class PerBranchFlowLoss(nn.Module):
    """Weighted MSE loss split by action and state branches.

    L = λ_A · MSE(v_pred[..., :Da], v_target[..., :Da])
      + λ_S · MSE(v_pred[..., Da:], v_target[..., Da:])

    Plan binding: λ_A = 1.0, λ_S = 0.5.

    Args:
        da: Action dimension Da. Default 7.
        ds: State dimension (unused, kept for API compatibility).
        lambda_a: Weight for action branch MSE. Default 1.0.
        lambda_s: Weight for state branch MSE. Default 0.5.

    Attributes:
        da: Action dimension.
        lambda_a: Action loss weight.
        lambda_s: State loss weight.
    """

    def __init__(
        self,
        da: int = 7,
        ds: int | None = None,
        lambda_a: float = 1.0,
        lambda_s: float = 0.5,
    ):
        super().__init__()
        self.da = da
        self.lambda_a = lambda_a
        self.lambda_s = lambda_s

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            v_pred: Predicted velocity of shape [B, H, Da+Ds].
            v_target: Target velocity of shape [B, H, Da+Ds].

        Returns:
            Scalar weighted MSE loss.
        """
        a_pred, s_pred = (
            v_pred[..., : self.da],
            v_pred[..., self.da :],
        )  # [B, H, Da], [B, H, Ds]
        a_targ, s_targ = (
            v_target[..., : self.da],
            v_target[..., self.da :],
        )  # [B, H, Da], [B, H, Ds]
        return self.lambda_a * nn.functional.mse_loss(
            a_pred, a_targ
        ) + self.lambda_s * nn.functional.mse_loss(s_pred, s_targ)

