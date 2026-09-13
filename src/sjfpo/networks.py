"""Networks: state encoders, spectral tokens, flow heads, critic.

Plan tasks 3-7. All modules are fp32-only, use SwiGLU activations, and contain no
BatchNorm / Dropout(p>0) / batch-statistics normalization in inference paths.
CELLORDER: 5

Tensor conventions (used throughout this module):
    B: batch size
    N: batch size (number of chunks)
    T: state window length (default 16)
    H: action-chunk horizon (default 8)
    K: low-K spectral tokens (default 5)
    D: fused latent dimension hidden_size (default 128)
    Da: action dimension (default 7, env-dependent)
    Ds: state dimension (default D = 128)
    T: generic sequence length (often N or H)

All tensors are fp32. Run-matrix construction is MuJoCo-only (flat Box);
it does not affect forward-pass logic in this module.
"""

# %%
from __future__ import annotations

import math

import torch
from torch import nn

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.dct_utils import dct

# Bare policy config: single source for constructor defaults (replaces the
# deleted legacy config.SJFPO_CFG global). One legacy name differs:
# networks predate the policy config (hidden_size <-> hidden_dim).
_DEFAULTS = SJFPOConfig()
_LEGACY_NAMES = {"hidden_size": "hidden_dim"}


class SwiGLU(nn.Module):
    """Gated SiLU feedforward: down(silu(gate(x)) * up(x)).

    Drop-in for ``Sequential(Linear(in_dim, hidden_dim), SiLU,
    Linear(hidden_dim, out_dim))`` with the same (in, hidden, out) widths.

    Args:
        in_dim: Input dimension.
        hidden_dim: Gated intermediate dimension.
        out_dim: Output dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gate = nn.Linear(in_dim, hidden_dim)
        self.up = nn.Linear(in_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [..., in_dim].

        Returns:
            Tensor of shape [..., out_dim].
        """
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


def resolve_config_value(name: str, override):
    """Config default with explicit-argument override.

    Args:
        name: Attribute name on the policy ``SJFPOConfig`` (one legacy
            alias: ``hidden_size`` reads ``hidden_dim``).
        override: Explicit value to use if not ``None``.

    Returns:
        The override value if provided, otherwise the policy default.
    """
    key = _LEGACY_NAMES.get(name, name)
    return override if override is not None else getattr(_DEFAULTS, key)

    # %% [markdown]
    # ## Sinusoidal timestep embedding (diffusers-style 10000-ladder)
    #
    # First half sin, second half cos; finite in fp32 at any width.

    # %%


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position codes added to token embeds.

    Stateless: codes are computed on the fly for the input length/dim/device,
    so nothing enters the state_dict (checkpoint-safe). Same 10000-ladder as
    ``get_timestep_embedding`` so encodings stay finite in fp32 at any width;
    first half sin, second half cos.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position codes.

        Args:
            x: Tokens of shape [..., T, D].

        Returns:
            Tokens of shape [..., T, D] with position codes added.
        """
        t, dim = x.shape[-2], x.shape[-1]
        pos = torch.arange(t, dtype=torch.float32, device=x.device).unsqueeze(
            1
        )  # [T, 1]
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32, device=x.device)
            * -(math.log(10000.0) / max(half - 1, 1))
        )  # [half]
        ang = pos * freqs  # [T, half]
        pe = torch.cat([ang.sin(), ang.cos()], dim=-1)  # [T, 2*half]
        return x + pe[:, :dim]  # [..., T, dim]

    # %% [markdown]
    # ## StateEncoder — frame projection -> two pre-norm residual blocks
    # (attention + SwiGLU, norm before every sublayer) -> Z[B, T, hidden_size].

    # %%


class StateEncoder(nn.Module):
    """State encoder: frame projection + positional encoding + two pre-norm blocks.

    Maps stacked frames to a latent sequence: a linear per-frame projection
    plus fixed sinusoidal position codes, then two blocks of split residuals
    (attention and SwiGLU each under their own residual, RMSNorm before
    every sublayer).

    Args:
        obs_dim: Input observation dimension per frame.
        n_obs_steps: Number of stacked frames (policy SJFPOConfig default).
        hidden_size: Latent dimension (policy SJFPOConfig default).
        n_heads: Number of attention heads. Default 4.

    Attributes:
        frame_proj: Linear(obs_dim -> hidden_size).
        norm1/norm3: RMSNorm before each attention block.
        attn1/attn2: MultiheadAttention(hidden_size, n_heads, batch_first=True).
        norm2/norm4: RMSNorm before each SwiGLU block.
        swiglu1/swiglu2: SwiGLU(hidden_size -> hidden_size -> hidden_size).
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
        self.frame_proj = nn.Linear(obs_dim, hidden_size)
        self.pos_encoding = PositionalEncoding()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.swiglu1 = SwiGLU(hidden_size, hidden_size, hidden_size)
        self.norm3 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm4 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.swiglu2 = SwiGLU(hidden_size, hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Stacked frames of shape [B, T, obs_dim] where T = n_obs_steps.

        Returns:
            Z: Latent sequence of shape [B, T, hidden_size].
        """
        h = self.frame_proj(x)  # [B, T, hidden_size]
        # Fixed sinusoidal position codes: attention is permutation-invariant,
        # so order must enter here before the blocks.
        h = self.pos_encoding(h)  # [B, T, hidden_size]
        n = self.norm1(h)  # [B, T, hidden_size]
        attn, _ = self.attn1(n, n, n)  # [B, T, hidden_size]
        h = h + attn  # [B, T, hidden_size]
        h = h + self.swiglu1(self.norm2(h))  # [B, T, hidden_size]
        n = self.norm3(h)  # [B, T, hidden_size]
        attn, _ = self.attn2(n, n, n)  # [B, T, hidden_size]
        h = h + attn  # [B, T, hidden_size]
        h = h + self.swiglu2(self.norm4(h))  # [B, T, hidden_size]
        return h


class ActionEncoder(nn.Module):
    """Action-sequence encoder: per-step projection + two pre-norm blocks.

    Deliberate duplicate of StateEncoder (not a subclass): the critic action
    path must read explicitly on its own. THE ONLY DIFFERENCE vs
    StateEncoder is the key-padding mask — ``real_action_lengths`` marks
    executed rows and tail (padded) positions are masked out as attention
    keys, so fake tails cannot influence real actions. Everything else
    (submodule names, shapes, math) is identical by construction, which the
    parity test locks via ``load_state_dict``.

    Args:
        act_dim: Input action dimension per step.
        n_obs_steps: Unused window hint (kept for signature parity with
            StateEncoder; positions come from the input length).
        hidden_size: Latent dimension (policy SJFPOConfig default).
        n_heads: Number of attention heads. Default 4.
    """

    def __init__(
        self,
        act_dim: int,
        n_obs_steps: int | None = None,
        hidden_size: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        n_obs_steps = resolve_config_value("n_obs_steps", n_obs_steps)
        hidden_size = resolve_config_value("hidden_size", hidden_size)
        self.frame_proj = nn.Linear(act_dim, hidden_size)
        self.pos_encoding = PositionalEncoding()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.swiglu1 = SwiGLU(hidden_size, hidden_size, hidden_size)
        self.norm3 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm4 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.swiglu2 = SwiGLU(hidden_size, hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, real_action_lengths=None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Action chunk of shape [B, H, Da].
            real_action_lengths: Executed steps per row ([B] ints). Tail
                positions are masked out as attention keys, so padded rows
                cannot influence real positions. None = full attention.

        Returns:
            Latent action sequence of shape [B, H, hidden_size].
        """
        h = self.frame_proj(x)  # [B, H, hidden_size]
        # Fixed sinusoidal position codes: attention is permutation-invariant,
        # so order must enter here before the blocks.
        h = self.pos_encoding(h)  # [B, H, hidden_size]
        # DIVERGENCE FROM StateEncoder: key-padding mask over executed rows.
        key_mask = None
        if real_action_lengths is not None:
            n_real = torch.as_tensor(list(real_action_lengths), device=h.device)  # [B]
            step = torch.arange(h.shape[1], device=h.device).unsqueeze(0)  # [1, H]
            key_mask = step >= n_real.unsqueeze(1)  # [B, H] True = pad, ignore as key
        n = self.norm1(h)  # [B, H, hidden_size]
        attn, _ = self.attn1(n, n, n, key_padding_mask=key_mask)  # [B, H, hidden_size]
        h = h + attn  # [B, H, hidden_size]
        h = h + self.swiglu1(self.norm2(h))  # [B, H, hidden_size]
        n = self.norm3(h)  # [B, H, hidden_size]
        attn, _ = self.attn2(n, n, n, key_padding_mask=key_mask)  # [B, H, hidden_size]
        h = h + attn  # [B, H, hidden_size]
        h = h + self.swiglu2(self.norm4(h))  # [B, H, hidden_size]
        return h

        # %% [markdown]
        # ## RunningMeanStd — per-dim running obs stats (SB3 VecNormalize-style)
        #
        # Buffers ride the module state_dict, so checkpointing stores them
        # alongside the model parameters. update() folds a batch in with
        # parallel Welford; forward() normalizes frozen (call update explicitly).

        # %%


class RunningMeanStd(nn.Module):
    """Per-dim running mean/variance over raw obs frames.

    Buffers (mean/var/count) ride the module state_dict, so checkpoints
    store them alongside the model parameters. update() folds raw frames in
    with parallel Welford; forward() normalizes with frozen stats (no update).
    A fresh module (count == 0) is the identity — no clipping of raw obs
    before stats exist.

    Args:
        dim: Raw observation dimension per frame (env-dependent, required).
        eps: Numerical floor inside the sqrt. Default 1e-8.
        clip: Symmetric clamp on normalized output. Default 10.0.
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
        norm = (x - self.mean.to(x.device)) / torch.sqrt(
            self.var.to(x.device) + self.eps
        )
        return norm.clamp(-self.clip, self.clip)

        # %% [markdown]
        # ## SIGReg — Sketched Isotropic Gaussian Regularizer (Epps-Pulley, LeWorldModel)
        #
        # Penalizes the empirical characteristic function (CF) of resampled random 1-D
        # projections of Z toward the N(0,1) target CF phi(t)=exp(-t^2/2), via
        # trapezoidal quadrature on [0,3].  Replicates the official implementation from
        # lucas-maes/le-wm (arXiv 2603.19312, Algorithm 1).
        #
        # Module-level quadrature nodes / weights for the default knots=17 (cached once).
        #
        #


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
    # retained: test-only public API — exercised directly by tests/test_networks.py
    # and tests/test_einx_migration.py; not called by the src train path.
    # Keep per fn-consolidation D1.


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
    # ## SpectralEncoder — raw DCT-II low-K tokens, no normalization
    #
    # Orthonormal DCT preserves energy (Parseval), so raw coefficients are
    # the tokens; no projector, no frozen stats, no calibration, no drift guard.

    # %%


class SpectralEncoder(nn.Module):
    """Parameter-free DCT-II spectral encoder over the RAW observation window.

    Transform convention (repo-wide): observation.state and action use the
    orthonormal DCT-II (dct_utils.dct); the real FFT (torch.fft.rfft) is
    reserved for observation.image (vision path, absent in this MuJoCo-only
    build).

    Order of operations:
        obs → DCT-II(dim=1) → keep K lowest low-frequency bins → tokens
    where K = max(1, min(T, round(top_p * T))). No projector: the last dim
    stays obs_dim, so tokens are raw frequency coefficients.

    Args:
        obs_dim: Raw observation dimension per frame (env-dependent, required).
        top_p_spectral_tokens: Kept low-frequency fraction (policy SJFPOConfig default).
        n_obs_steps: State window length T (policy SJFPOConfig default).

    Attributes:
        obs_dim: Raw observation dimension.
        k: Effective token count K.
    """

    def __init__(
        self,
        obs_dim: int,
        top_p_spectral_tokens: float | None = None,
        n_obs_steps: int | None = None,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} must be positive")
        self.obs_dim = obs_dim
        t = int(
            resolve_config_value("n_obs_steps", n_obs_steps)
        )  # DCT-II yields T bins
        p = float(resolve_config_value("top_p_spectral_tokens", top_p_spectral_tokens))
        self.k = max(1, min(t, round(p * t)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw obs window -> spectral tokens.

        Args:
            obs: Raw observation window of shape [B, T, obs_dim].

        Returns:
            tokens: Raw DCT bins of shape [B, K, obs_dim].
        """
        feat = dct(obs, dim=1)  # [B, T, obs_dim] DCT-II over time
        return feat[:, : self.k, :]  # [B, K, obs_dim]

        # %% [markdown]
        # ## Flow heads — velocity prediction in orthonormal-DCT coefficient space
        # %% [markdown]
        # ## Flow policy — DiT velocity model over action-chunk DCT coefficients
        #
        # The actor predicts ONLY action velocities v(x_t, t, condition) in orthonormal-DCT
        # space (no next-state prediction). DiT
        # (Peebles & Xie) with adaLN-Zero: `dit_depth` pre-norm self-attention blocks
        # over the noisy action tokens, conditioned by one vector (time-embed +
        # pooled spectral tokens). Output velocity has the SAME shape as x_t.

        # %%


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift (shift/scale broadcast over tokens)."""
    return x * (1.0 + scale) + shift  # [..., D]


def get_timestep_embedding(t, dim: int = 256) -> torch.Tensor:
    """Diffusers-style sinusoidal timestep embedding with inverse-frequency ladder.

    Uses exp(-log(10000) * k / (half - 1)) so any dim stays finite in fp32.
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
    Args:
        hidden_size: Latent dimension (policy SJFPOConfig default).
        n_heads: Attention heads per block. Default 4.
        mlp_ratio: MLP width multiplier. Default 4.0.
    Returns:
        Tokens of shape [B, H, D].

    x <- x + gate_msa * Attn(modulate(norm(x))); x <- x + gate_mlp * MLP(...).
    Zero-initialized modulation makes each block start as the identity.
    """

    def __init__(self, hidden_size: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, mlp_hidden, hidden_size)
        self.adaLN = SwiGLU(hidden_size, hidden_size, 6 * hidden_size)
        nn.init.constant_(self.adaLN.down.weight, 0.0)
        nn.init.constant_(self.adaLN.down.bias, 0.0)

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
        self.mlp = SwiGLU(hidden_size, mlp_hidden, hidden_size)

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
    """DiT velocity model: policy state encoder prefix tokens + noisy action tokens.

    `dit_depth` pre-norm self-attention blocks with adaLN-Zero over the
    concatenated sequence [state tokens; action tokens]; conditioning vector
    = time-embed + mean-pooled spectral tokens. Only the action positions
    are read out as velocity.

    Args:
        action_dim: Action dimension Da (env-dependent).
        hidden_size: Latent dimension (policy SJFPOConfig default).
        obs_dim: Raw obs dim of the spectral tokens (env-dependent, required).
        dit_depth/dit_heads/dit_mlp_ratio/dit_time_dim: DiT hyperparams
            (policy SJFPOConfig dit_* defaults).
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        hidden_size: int | None = None,
        dit_depth: int | None = None,
        dit_heads: int | None = None,
        dit_mlp_ratio: float | None = None,
        dit_time_dim: int | None = None,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} must be positive")
        hidden_size = int(resolve_config_value("hidden_size", hidden_size))
        dit_depth = int(resolve_config_value("dit_depth", dit_depth))
        dit_heads = int(resolve_config_value("dit_heads", dit_heads))
        dit_mlp_ratio = float(resolve_config_value("dit_mlp_ratio", dit_mlp_ratio))
        dit_time_dim = int(resolve_config_value("dit_time_dim", dit_time_dim))
        self.time_dim = dit_time_dim
        self.in_proj = nn.Linear(action_dim, hidden_size)
        self.state_encoder = StateEncoder(
            obs_dim, hidden_size=hidden_size
        )  # policy state encoder
        self.time_mlp = SwiGLU(dit_time_dim, 4 * hidden_size, hidden_size)
        self.cond_proj = nn.Linear(obs_dim, hidden_size)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(hidden_size, action_dim)
        # Zero final layer: an untrained policy predicts ~zero velocity.
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(
        self, x_t: torch.Tensor, t, condition: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: predict velocity.

        Args:
            x_t: Noisy action chunk (DCT coefficients) of shape [B, Ka, Da]
                where Ka = top-p low-frequency rows (<= H).
            t: Timestep scalar or tensor of shape [B].
            condition: Raw spectral tokens of shape [B, K, obs_dim].
            state: Raw obs window of shape [B, T, obs_dim] (prefix tokens).

        Returns:
            Velocity prediction of shape [B, Ka, Da].
        """
        temb = get_timestep_embedding(t, dim=self.time_dim).to(
            device=x_t.device, dtype=x_t.dtype
        )  # [B, T] or [T]
        if temb.dim() == 1:
            temb = temb.expand(x_t.shape[0], -1)  # [B, T]
        c = self.time_mlp(temb) + self.cond_proj(condition.mean(dim=1))  # [B, D]
        s = self.state_encoder(state)  # [B, T, D] policy state encoder prefix
        h = torch.cat([s, self.in_proj(x_t)], dim=1)  # [B, T+H, D]
        for block in self.blocks:
            h = block(h, c)
        return self.out(self.norm_out(h[:, -x_t.shape[1] :, :]))  # [B, H, Da]

        # %% [markdown]
        # ## Critic — V(s, a): 2 cross-attention fusion blocks -> scalar V[B]

        # %%


class Critic(nn.Module):
    """State-action critic: owned encoders + cross-attention fusion -> scalar V[B].

    Inputs are raw (never latents, never DCT): the obs window goes through
    the owned critic state encoder, the action chunk through the owned
    critic action encoder (same definition as the state encoder), then the
    encoded actions query the encoded states through ``num_attn_blocks``
    cross-attention blocks. V(s, a) keeps its TD bootstrap target V(s', a').

    Args:
        obs_dim: Raw obs dim per frame (env-dependent, required).
        act_dim: Action dimension Da (env-dependent, required).
        hidden_size: Latent dimension (policy SJFPOConfig default).
        n_obs_steps: State window length T (policy SJFPOConfig default).
        num_attn_blocks: Cross-attention blocks. Default 1.
        n_heads: Attention heads per block. Default 4.
        mlp_ratio: MLP width multiplier per block. Default 4.0.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int | None = None,
        n_obs_steps: int | None = None,
        num_attn_blocks: int = 1,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} must be positive")
        if act_dim <= 0:
            raise ValueError(f"act_dim={act_dim} must be positive (env action dim)")
        hidden_size = resolve_config_value("hidden_size", hidden_size)
        self.state_encoder = StateEncoder(
            obs_dim, n_obs_steps=n_obs_steps, hidden_size=hidden_size
        )  # critic state encoder
        self.action_encoder = ActionEncoder(
            act_dim, n_obs_steps=n_obs_steps, hidden_size=hidden_size
        )  # critic action encoder
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(hidden_size, n_heads, mlp_ratio)
            for _ in range(num_attn_blocks)
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(
        self, states: torch.Tensor, a: torch.Tensor, real_action_lengths=None
    ) -> torch.Tensor:
        """State-action value: encode states, fuse actions, pool to scalar V.

        Args:
            states: Raw obs window of shape [B, T, obs_dim].
            a: Real-space action chunk of shape [B, H, Da].
            real_action_lengths: Executed steps per row ([B] ints). Tails stay
                invisible as attention keys inside the action encoder and are
                dropped from the mean (None = plain mean over all rows).

        Returns:
            V: Scalar value per batch of shape [B].
        """
        z = self.state_encoder(states)  # [B, T, D] critic state encoder
        h = self.action_encoder(a, real_action_lengths)  # [B, H, D] tails invisible as keys
        for block in self.blocks:
            h = block(h, z)
        if real_action_lengths is None:
            pooled = h.mean(dim=1)  # [B, D]
        else:
            # Masked mean over executed rows: tails are already invisible as
            # attention keys inside the action encoder, so dropping their
            # outputs here excludes unexecuted steps exactly.
            n_real = torch.as_tensor(list(real_action_lengths), device=h.device)  # [B]
            exec_mask = (
                (torch.arange(h.shape[1], device=h.device).unsqueeze(0) < n_real.unsqueeze(1))
                .to(h.dtype)
                .unsqueeze(-1)
            )  # [B, H, 1] 1 = real row
            pooled = (h * exec_mask).sum(dim=1) / exec_mask.sum(dim=1)  # [B, D]
        return self.out(pooled).squeeze(-1)  # [B]
