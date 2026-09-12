"""Networks: state trunk, SigReg, spectral tokens, flow heads, critic.

Plan tasks 3-7. All modules are fp32-only, use SiLU activations, and contain no
BatchNorm / Dropout(p>0) / batch-statistics normalization in inference paths.
CELLORDER: 5

Tensor conventions (used throughout this module):
    B: batch size
    N: batch size (number of chunks)
    T: state window length (default 16)
    H: action-chunk horizon (default 8)
    K: low-K spectral tokens (default 5)
    D: fused latent dimension hidden_size (default 128)
    spectral_dim: spectral projection dimension (default 64)
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

from sjfpo import config
from sjfpo.dct_utils import dct


def resolve_config_value(name: str, override):
    """Config default with explicit-argument override.

    Args:
        name: Attribute name on ``config.SJFPO_CFG``.
        override: Explicit value to use if not ``None``.

    Returns:
        The override value if provided, otherwise ``getattr(config.SJFPO_CFG, name)``.
    """
    return override if override is not None else getattr(config.SJFPO_CFG, name)

    # %% [markdown]
    # ## Sinusoidal timestep embedding (8 dims: cos/sin of t*2^k)
    #
    # Convention from amazon-far/fpo-control `_embed_timestep` (arXiv 2602.02481
    # III-A): the first half is sin(t*2^k), the second half cos(t*2^k),
    # k = 0 .. dim/2-1.

    # %%


def get_timestep_embedding(t, dim: int = 8) -> torch.Tensor:
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
    freqs = 2.0 ** torch.arange(half, dtype=torch.float32, device=t.device)  # [half]
    ang = t.unsqueeze(-1) * freqs  # [..., half]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # [..., dim]

    # %% [markdown]
    # ## StateEncoder — 16 stacked obs frames -> per-frame encoder -> attention
    # fusion -> latent sequence Z[B, T, hidden_size]. Vision DiT dropped per user
    # decision D2: single proprio-style modality, so "cross-attn fusion" is a
    # residual self-attention block over the frame-token sequence.

    # %%


class StateEncoder(nn.Module):
    """State trunk: stacked observation frames -> per-frame encoder -> attention fusion.

    Maps a sequence of stacked frames to a latent sequence via a frame-wise encoder
    followed by a residual self-attention block.

    Args:
        obs_dim: Input observation dimension per frame.
        n_obs_steps: Number of stacked frames (default from config.SJFPO_CFG.n_obs_steps).
        hidden_size: Latent dimension (default from config.SJFPO_CFG.hidden_size).
        n_heads: Number of attention heads. Default 4.

    Attributes:
        frame_enc: Sequential(Linear(obs_dim -> hidden_size), SiLU()).
        fusion: MultiheadAttention(hidden_size, n_heads, batch_first=True).
    """

    def __init__(
        self,
        obs_dim: int,
        n_obs_steps: int | None = None,
        hidden_size: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        n_obs_steps = resolve_config_value("n_obs_steps", n_obs_steps)
        hidden_size = resolve_config_value("hidden_size", hidden_size)
        self.frame_enc = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.SiLU())
        self.fusion = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Stacked frames of shape [B, T, obs_dim] where T = n_obs_steps.

        Returns:
            Z: Latent sequence of shape [B, T, hidden_size].
        """
        h = self.frame_enc(x)  # [B, T, hidden_size]
        attn, _ = self.fusion(h, h, h)  # [B, T, hidden_size]
        return nn.functional.silu(h + attn)  # [B, T, hidden_size]

        # %% [markdown]
        # ## SIGReg — Sketched Isotropic Gaussian Regularizer (Epps-Pulley, LeWorldModel)
        #
        # Penalizes the empirical characteristic function (CF) of resampled random 1-D
        # projections of Z toward the N(0,1) target CF phi(t)=exp(-t^2/2), via
        # trapezoidal quadrature on [0,3].  Replicates the official implementation from
        # lucas-maes/le-wm (arXiv 2603.19312, Algorithm 1).

        # Module-level quadrature nodes / weights for the default knots=17 (cached once).


DEFAULT_KNOTS = 17
t_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def get_quadrature(knots: int, device: torch.device):
    """Return cached (t, weights_scaled, phi) for trapezoid on [0, 3].

    t:       [K]            quadrature nodes
    w_phi:   [K]            weights * phi (pre-multiplied, same dtype/device as inputs)
    phi:     [K]            N(0,1) CF target exp(-t^2/2)
    """
    if knots not in t_cache:
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        phi = torch.exp(-t.square() / 2.0)
        t_cache[knots] = (t, w * phi, phi)
    t, w_phi, phi = t_cache[knots]
    # Move to requested device; cached tensors are CPU; re-cache if new device.
    if t.device != device:
        t = t.to(device)
        w_phi = w_phi.to(device)
        phi = phi.to(device)
    return t, w_phi, phi

    # %%


def sigreg_loss(
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
            the official (T, B=N, D) convention). T=1 holds vector latents
            (e.g. the critic fusion unsqueezed).
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

    t, w_phi, phi = get_quadrature(knots, zp.device)

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
    # ## SpectralEncoder — DCT-II low-K tokens with FROZEN stats
    #
    # Early design computed per-batch z-scored FFT tokens; frozen statistics
    # replaced the batch statistics. Frozen mu/std live in
    # Z-space ([hidden_size], matching the plan's `set_frozen_stats(zeros(128),
    # ones(128))`): inputs are z-scored with the frozen stats BEFORE windowing,
    # and the drift guard compares the raw batch-mean against them under no_grad.
    # forward raises RuntimeError('frozen stats not initialized') until stats are
    # set via set_frozen_stats(mu, std) or captured once via capture_stats(sample).

    # %%


class SpectralEncoder(nn.Module):
    """DCT-II spectral encoder with FROZEN norm + learnable projector.

    Transform convention (repo-wide): observation.state and action use the
    orthonormal DCT-II (dct_utils.dct); the real FFT (torch.fft.rfft) is
    reserved for observation.image (vision path, absent in this MuJoCo-only
    build). DCT needs no window: its even extension handles boundaries, so
    unlike the old Hann+rFFT path there is no taper buffer here.

    Order of operations:
        Z → DCT-II(dim=1) → keep K low bins → frozen norm → Linear(D→spectral_dim)

    Args:
        hidden_size: Latent dimension hidden_size (default from config.SJFPO_CFG.hidden_size).
        num_spectral_tokens: Number of low-frequency tokens K (default from config.SJFPO_CFG.num_spectral_tokens).
        n_obs_steps: State window length T (default from config.SJFPO_CFG.n_obs_steps).

    Attributes:
        hidden_size: Latent dimension D.
        k: Effective K = min(num_spectral_tokens, n_obs_steps).
        frozen_mu: Frozen mean buffer of shape [hidden_size].
        frozen_std: Frozen std buffer of shape [hidden_size].
        has_stats: Boolean flag indicating if frozen stats are initialized.
        projector: Linear(hidden_size -> spectral_dim).

    Raises:
        RuntimeError: If forward() called before frozen stats are initialized via
            set_frozen_stats() or capture_stats().
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        num_spectral_tokens: int | None = None,
        n_obs_steps: int | None = None,
    ):
        super().__init__()
        hidden_size = int(resolve_config_value("hidden_size", hidden_size))
        num_spectral_tokens = int(
            resolve_config_value("num_spectral_tokens", num_spectral_tokens)
        )
        n_obs_steps = int(resolve_config_value("n_obs_steps", n_obs_steps))
        self.hidden_size = hidden_size
        self.k = min(num_spectral_tokens, n_obs_steps)  # DCT-II yields T bins
        self.register_buffer("frozen_mu", torch.zeros(hidden_size))  # [D]
        self.register_buffer("frozen_std", torch.ones(hidden_size))  # [D]
        self.has_stats = False
        spectral_dim = int(config.SJFPO_CFG.spectral_dim)
        self.projector = nn.Linear(hidden_size, spectral_dim)  # [D -> spectral_dim]

    def set_frozen_stats(self, mu: torch.Tensor, std: torch.Tensor) -> None:
        """Set frozen normalization statistics.

        Args:
            mu: Mean tensor of shape [hidden_size].
            std: Std tensor of shape [hidden_size], clamped to min 1e-4.

        Raises:
            ValueError: If mu or std have incorrect shape.
        """
        if mu.shape != (self.hidden_size,) or std.shape != (self.hidden_size,):
            raise ValueError(
                f"frozen stats must have shape ({self.hidden_size},), "
                f"got {tuple(mu.shape)} / {tuple(std.shape)}"
            )
        with torch.no_grad():
            self.frozen_mu.copy_(mu.detach().to(torch.float32))  # [D]
            self.frozen_std.copy_(std.detach().to(torch.float32).clamp_min(1e-4))  # [D]
        self.has_stats = True

    def capture_stats(self, z_sample: torch.Tensor) -> None:
        """Capture frozen stats ONCE from a representative Z sample.

        Stats are computed in DCT space so they reflect the actual encoder
        feature distribution, not the raw Z-space distribution.

        Args:
            z_sample: Sample tensor of shape [B, T, hidden_size].
        """
        with torch.no_grad():
            feat = dct(z_sample.detach(), dim=1)  # [B, T, D] DCT-II over time
            flat = feat.reshape(-1, self.hidden_size)  # [B*T, D]
            mu = flat.mean(dim=0)  # [D]
            std = flat.std(dim=0).clamp_min(1e-4)  # [D]
        self.set_frozen_stats(mu, std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: Z -> spectral tokens.

        Args:
            z: Latent sequence of shape [B, T, hidden_size] where T = n_obs_steps.

        Returns:
            tokens: Spectral tokens of shape [B, K, spectral_dim].

        Raises:
            RuntimeError: If frozen stats not initialized.
        """
        if not self.has_stats:
            raise RuntimeError("frozen stats not initialized")
        with torch.no_grad():
            feat_raw = dct(z, dim=1)  # [B, T, D] DCT-II over time
            drift = (
                feat_raw.reshape(-1, self.hidden_size).mean(dim=0) - self.frozen_mu
            )  # [D]
            if bool((drift.abs() > 3 * self.frozen_std).any()):
                warnings.warn("spectral stats drift", UserWarning, stacklevel=2)
        feat = dct(z, dim=1)  # [B, T, D] DCT-II over time
        feat = (feat - self.frozen_mu) / self.frozen_std  # [B, T, D]
        return self.projector(feat[:, : self.k, :])  # [B, K, spectral_dim]

        # %% [markdown]
        # ## Flow heads — velocity prediction in orthonormal-DCT coefficient space
        # %% [markdown]
        # ## Flow policy — DiT velocity model over action-chunk DCT coefficients
        #
        # The actor predicts ONLY action velocities v(x_t, t, cond) in orthonormal-DCT
        # space (no next-state prediction). DiT
        # (Peebles & Xie) with adaLN-Zero: `dit_depth` pre-norm self-attention blocks
        # over the noisy action tokens, conditioned by one vector (time-embed +
        # pooled spectral tokens). Output velocity has the SAME shape as x_t.

        # %%


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift (shift/scale broadcast over tokens)."""
    return x * (1.0 + scale) + shift  # [..., D]


def dit_timestep_embedding(t, dim: int = 256) -> torch.Tensor:
    """Diffusers-style sinusoidal timestep embedding with inverse-frequency ladder.

    Unlike get_timestep_embedding (2^k ladder, only valid for tiny dims), this
    uses exp(-log(10000) * k / (half - 1)) so dim=256 stays finite in fp32.
    First half sin, second half cos.

    Args:
        t: Scalar or tensor of shape [B].
        dim: Embedding dimension (must be even). Default 256.

    Returns:
        Tensor of shape [..., dim] with sin/cos features.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(math.log(10000.0) / max(half - 1, 1))
    )  # [half]
    ang = t.unsqueeze(-1) * freqs  # [..., half]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # [..., dim]


class DiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning (self-attention + MLP).

    x <- x + gate_msa * Attn(modulate(norm(x))); x <- x + gate_mlp * MLP(...).
    Zero-initialized modulation makes each block start as the identity.
    """

    def __init__(self, hidden_size: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.constant_(self.adaLN[-1].weight, 0.0)
        nn.init.constant_(self.adaLN[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy tokens of shape [B, H, D].
            c: Conditioning vector of shape [B, D].

        Returns:
            Tokens of shape [B, H, D].
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)[
            :, None, :
        ].chunk(6, dim=-1)  # each [B, 1, D]
        h = modulate(self.norm1(x), shift_msa, scale_msa)  # [B, H, D]
        attn, _ = self.attn(h, h, h)  # [B, H, D]
        x = x + gate_msa * attn  # [B, H, D]
        return x + gate_mlp * self.mlp(  # [B, H, D]
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention fusion block (queries × key/value streams).

    x <- x + Attn(norm(x), norm(kv)); x <- x + MLP(norm(x)). RMSNorm +
    residuals throughout for stability.
    """

    def __init__(self, hidden_size: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.RMSNorm(hidden_size)
        self.norm_kv = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm_mlp = nn.RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query tokens of shape [B, Hq, D].
            kv: Key/value tokens of shape [B, Hkv, D].

        Returns:
            Fused tokens of shape [B, Hq, D].
        """
        attn, _ = self.attn(self.norm_q(x), self.norm_kv(kv), self.norm_kv(kv))
        x = x + attn  # [B, Hq, D]
        return x + self.mlp(self.norm_mlp(x))  # [B, Hq, D]


class SJFOPolicyModel(nn.Module):
    """DiT velocity model over action-chunk DCT coefficients (actions ONLY).

    `dit_depth` pre-norm self-attention blocks with adaLN-Zero over the noisy
    action tokens; conditioning vector = time-embed + mean-pooled spectral
    tokens. Same (x_t, t, cond) call signature as before, so the trainer,
    collector, and Euler sampler are untouched.

    Args:
        action_dim: Action dimension Da (env-dependent).
        hidden_size: Latent dimension (default from config.SJFPO_CFG.hidden_size).
        cond_dim: Spectral-token dim (default from config.SJFPO_CFG.spectral_dim).
        dit_depth/dit_heads/dit_mlp_ratio/dit_time_dim: DiT hyperparams
            (defaults from config.SJFPO_CFG.dit_*).
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int | None = None,
        cond_dim: int | None = None,
        dit_depth: int | None = None,
        dit_heads: int | None = None,
        dit_mlp_ratio: float | None = None,
        dit_time_dim: int | None = None,
    ):
        super().__init__()
        hidden_size = int(resolve_config_value("hidden_size", hidden_size))
        cond_dim = int(resolve_config_value("spectral_dim", cond_dim))
        dit_depth = int(resolve_config_value("dit_depth", dit_depth))
        dit_heads = int(resolve_config_value("dit_heads", dit_heads))
        dit_mlp_ratio = float(resolve_config_value("dit_mlp_ratio", dit_mlp_ratio))
        dit_time_dim = int(resolve_config_value("dit_time_dim", dit_time_dim))
        self.time_dim = dit_time_dim
        self.in_proj = nn.Linear(action_dim, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(dit_time_dim, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_size)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(hidden_size, action_dim)
        # Zero final layer: an untrained policy predicts ~zero velocity.
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x_t: torch.Tensor, t, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict velocity.

        Args:
            x_t: Noisy action chunk (DCT coefficients) of shape [B, H, Da].
            t: Timestep scalar or tensor of shape [B].
            cond: Spectral tokens of shape [B, K, Dk].

        Returns:
            Velocity prediction of shape [B, H, Da].
        """
        temb = dit_timestep_embedding(t, dim=self.time_dim).to(
            device=x_t.device, dtype=x_t.dtype
        )  # [B, T] or [T]
        if temb.dim() == 1:
            temb = temb.expand(x_t.shape[0], -1)  # [B, T]
        c = self.time_mlp(temb) + self.cond_proj(cond.mean(dim=1))  # [B, D]
        h = self.in_proj(x_t)  # [B, H, D]
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.norm_out(h))  # [B, H, Da]

        # %% [markdown]
        # ## Critic — V(s, a): 2 cross-attention fusion blocks -> scalar V[B]

        # %%


class Critic(nn.Module):
    """State-action critic: cross-attention fusion blocks -> scalar V[B].

    Action tokens query the trunk state sequence (mirror of the actor's
    conditioning); each block is pre-norm RMSNorm + residual. V(s, a) keeps
    its TD bootstrap target V(s', a'). The action input stays DCT
    coefficients [B, H, Da], never raw actions.

    Args:
        hidden_size: Latent dimension (default from config.SJFPO_CFG.hidden_size).
        act_dim: Action dimension Da (env-dependent, required).
        depth: Cross-attention blocks. Default 2.
        n_heads: Attention heads per block. Default 4.
        mlp_ratio: MLP width multiplier per block. Default 4.0.
    """

    def __init__(
        self,
        hidden_size: int | None = None,
        act_dim: int = 0,
        depth: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if act_dim <= 0:
            raise ValueError(f"act_dim={act_dim} must be positive (env action dim)")
        hidden_size = resolve_config_value("hidden_size", hidden_size)
        self.act_embed = nn.Linear(act_dim, hidden_size)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(hidden_size, n_heads, mlp_ratio) for _ in range(depth)
        )
        self.out = nn.Linear(hidden_size, 1)

    def encode(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Fuse state/action into the critic latent (SigReg-compatible).

        Args:
            z: Trunk state sequence of shape [B, T, hidden_size].
            a: Action chunk (DCT coefficients) of shape [B, H, Da].

        Returns:
            Fused vector of shape [B, hidden_size] (unsqueeze(1) feeds sigreg_loss).
        """
        h = self.act_embed(a)  # [B, H, D]
        for blk in self.blocks:
            h = blk(h, z)
        return h.mean(dim=1)  # [B, D]

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Trunk state sequence of shape [B, T, hidden_size].
            a: Action chunk (DCT coefficients) of shape [B, H, Da].

        Returns:
            V: Scalar value per batch of shape [B].
        """
        return self.out(self.encode(z, a)).squeeze(-1)  # [B]
