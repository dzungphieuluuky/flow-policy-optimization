"""Networks: state encoders, spectral tokens, flow heads, critic.

SwiGLU activations everywhere; no BatchNorm / Dropout / batch-stat norms.
CELLORDER: 5

Tensor conventions: B batch, N chunk batch, T obs window, H action_horizon,
K spectral tokens, D hidden dim, action_dim env action dim. Params are fp32;
compute may autocast matmuls (see utils.autocast_context).
"""

# %%
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
        """[..., in_dim] -> [..., out_dim]."""
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


def resolve_config_value(name: str, override):
    """Policy default for ``name`` unless ``override`` is not None."""
    return override if override is not None else getattr(_DEFAULTS, name)

    # %% [markdown]
    # ## Sinusoidal timestep embedding (diffusers-style 10000-ladder)
    #
    # First half sin, second half cos; finite in fp32 at any width.

    # %%


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position codes (stateless: nothing in state_dict)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position codes: [..., T, D] -> [..., T, D]."""
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
    # (attention + SwiGLU, norm before every sublayer) -> Z[B, T, hidden_dim].

    # %%


class StateEncoder(nn.Module):
    """Frame projection + two pre-norm residual blocks -> Z[B, T, hidden_dim].

    Split residuals (attention and SwiGLU each under their own residual,
    RMSNorm before every sublayer). Position codes enter before the blocks
    because attention is permutation-invariant.
    """

    def __init__(
        self,
        obs_dim: int,
        n_obs_steps: int | None = None,
        hidden_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        n_obs_steps = resolve_config_value("n_obs_steps", n_obs_steps)
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        self.frame_proj = nn.Linear(obs_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.swiglu1 = SwiGLU(hidden_dim, hidden_dim, hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm4 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.swiglu2 = SwiGLU(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, obs_dim] -> [B, T, hidden_dim]."""
        h = self.frame_proj(x)
        h = self.pos_encoding(h)
        n = self.norm1(h)
        attn, _ = self.attn1(n, n, n)
        h = h + attn
        h = h + self.swiglu1(self.norm2(h))
        n = self.norm3(h)
        attn, _ = self.attn2(n, n, n)
        h = h + attn
        h = h + self.swiglu2(self.norm4(h))
        return h


class ActionEncoder(nn.Module):
    """Deliberate duplicate of StateEncoder (not a subclass).

    THE ONLY DIFFERENCE is the key-padding mask: ``executed_n_action_steps``
    marks executed rows so padded tails cannot influence real actions as
    attention keys. Locked by the parity test via ``load_state_dict``.
    """

    def __init__(
        self,
        act_dim: int,
        n_obs_steps: int | None = None,
        hidden_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        n_obs_steps = resolve_config_value("n_obs_steps", n_obs_steps)
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        self.frame_proj = nn.Linear(act_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.swiglu1 = SwiGLU(hidden_dim, hidden_dim, hidden_dim)
        self.norm3 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm4 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.swiglu2 = SwiGLU(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, executed_n_action_steps=None) -> torch.Tensor:
        """[B, H, action_dim] -> [B, H, hidden_dim]; tails masked as keys when given."""
        h = self.frame_proj(x)
        h = self.pos_encoding(h)
        key_mask = None
        if executed_n_action_steps is not None:
            if not isinstance(executed_n_action_steps, torch.Tensor):
                # Eager callers may pass python lists; dynamo cannot trace
                # list() unbinds, so coerce before the math below.
                executed_n_action_steps = torch.as_tensor(
                    list(executed_n_action_steps), device=h.device
                )
            n_real = executed_n_action_steps.to(h.device)  # [B]
            step = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
            key_mask = step >= n_real.unsqueeze(1)  # True = pad, ignore as key
        n = self.norm1(h)
        attn, _ = self.attn1(n, n, n, key_padding_mask=key_mask)
        h = h + attn
        h = h + self.swiglu1(self.norm2(h))
        n = self.norm3(h)
        attn, _ = self.attn2(n, n, n, key_padding_mask=key_mask)
        h = h + attn
        h = h + self.swiglu2(self.norm4(h))
        return h

        # %% [markdown]
        # ## RunningMeanStd — per-dim running obs stats (SB3 VecNormalize-style)
        #
        # Buffers ride the module state_dict, so checkpointing stores them
        # alongside the model parameters. update() folds a batch in with
        # parallel Welford; forward() normalizes frozen (call update explicitly).

        # %%


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
    """Cached (nodes t, weights*phi, phi) for trapezoid on [0, 3]."""
    if knots not in t_cache:
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        phi = torch.exp(-t.square() / 2.0)
        t_cache[knots] = (t, w * phi, phi)
    t, w_phi, phi = t_cache[knots]
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
    Returns a single 0-dim scalar loss (differentiable).

    Replicates the official LeWorldModel SIGReg (lucas-maes/le-wm).

    Args:
        z: [N, T, D] (T=1 holds vector latents).
        knots: Quadrature nodes on [0, 3].
        num_proj: Random projections (sketch columns).
        projection: Optional precomputed [D, num_proj] unit-norm matrix.

    Returns:
        0-dim scalar loss.
    """
    zp = z.permute(1, 0, 2)  # official le-wm (T, B=N, D) layout
    T_count, B, D = zp.shape

    t, w_phi, phi = get_quadrature(knots, zp.device)

    if projection is None:
        A = torch.randn(D, num_proj, device=zp.device, dtype=zp.dtype)
        A = A / A.norm(dim=0, keepdim=True)
    else:
        A = projection

    x_t = (zp @ A).unsqueeze(-1) * t  # (T, B, M, K)

    cos_mean = x_t.cos().mean(1)
    sin_mean = x_t.sin().mean(1)
    err = (cos_mean - phi).square() + sin_mean.square()

    statistic = (err @ w_phi) * B  # (T, M), trapezoid + N scaling
    return statistic.mean()

    # %% [markdown]
    # ## SpectralEncoder — raw DCT-II low-K tokens, no normalization
    #
    # Orthonormal DCT preserves energy (Parseval), so raw coefficients are
    # the tokens; no projector, no frozen stats, no calibration, no drift guard.

    # %%


class SpectralEncoder(nn.Module):
    """Parameter-free DCT-II over the RAW obs window; keep K lowest bins.

    K = max(1, min(T, round(top_p * T))). Last dim stays obs_dim: tokens are
    raw frequency coefficients (orthonormal DCT preserves energy). rFFT is
    reserved for observation.image (absent in this MuJoCo-only build).
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
        """[B, T, obs_dim] -> [B, K, obs_dim]."""
        feat = dct(obs, dim=1)
        return feat[:, : self.k, :]


class SpectralAttentionPool(nn.Module):
    """Order-aware pooling of raw DCT bins into one conditioning vector.

    Per-bin projection shared over K, query from the DiT timestep embedding.
    Mean blending cannot tell DC from harmonics; attention keeps which-bin
    information and exposes inspectable weights.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int | None = None,
        time_dim: int | None = None,
        n_heads: int | None = None,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} must be positive")
        hidden_dim = int(resolve_config_value("hidden_dim", hidden_dim))
        time_dim = int(resolve_config_value("dit_time_dim", time_dim))
        n_heads = int(resolve_config_value("dit_heads", n_heads))
        self.key_proj = nn.Linear(obs_dim, hidden_dim)
        self.query_proj = nn.Linear(time_dim, hidden_dim)
        self.norm_q = nn.RMSNorm(hidden_dim)
        self.norm_kv = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(
        self, condition: torch.Tensor, temb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """([B, K, obs_dim], [B, time_dim]) -> (pooled [B, D], weights [B, K])."""
        keys = self.key_proj(condition)
        query = self.query_proj(temb).unsqueeze(1)
        pooled, weights = self.attn(
            self.norm_q(query), self.norm_kv(keys), self.norm_kv(keys)
        )
        return pooled.squeeze(1), weights.squeeze(1)

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
    """Sinusoidal embedding with inverse-frequency ladder (finite in fp32)."""
    t = torch.as_tensor(t, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(math.log(10000.0) / max(half - 1, 1))
    )
    ang = t.unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class DiTBlock(nn.Module):
    """Pre-norm DiT block with adaLN-Zero conditioning.

    x <- x + gate_msa * Attn(modulate(norm(x))); x <- x + gate_mlp * MLP(...).
    Zero-initialized modulation makes each block start as the identity.
    """

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = SwiGLU(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.down.weight, 0.0)
        nn.init.constant_(self.adaLN.down.bias, 0.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """([B, H, D], [B, D]) -> [B, H, D]."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c)[
            :, None, :
        ].chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn, _ = self.attn(h, h, h)
        x = x + gate_msa * attn
        return x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention fusion: x <- x + Attn(norm(x), norm(kv)) + MLP."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.RMSNorm(hidden_dim)
        self.norm_kv = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm_mlp = nn.RMSNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """([B, Hq, D], [B, Hkv, D]) -> [B, Hq, D]."""
        attn, _ = self.attn(self.norm_q(x), self.norm_kv(kv), self.norm_kv(kv))
        x = x + attn
        return x + self.mlp(self.norm_mlp(x))


class SJFOPolicyModel(nn.Module):
    """DiT velocity model over [state prefix; noisy action] tokens.

    Conditioning vector = time-embed + pooled spectral tokens (mean or
    attention pool). Only action positions are read out as velocity.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        hidden_dim: int | None = None,
        dit_depth: int | None = None,
        dit_heads: int | None = None,
        dit_mlp_ratio: float | None = None,
        dit_time_dim: int | None = None,
        spectral_pool: str | None = None,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim={obs_dim} must be positive")
        hidden_dim = int(resolve_config_value("hidden_dim", hidden_dim))
        dit_depth = int(resolve_config_value("dit_depth", dit_depth))
        dit_heads = int(resolve_config_value("dit_heads", dit_heads))
        dit_mlp_ratio = float(resolve_config_value("dit_mlp_ratio", dit_mlp_ratio))
        dit_time_dim = int(resolve_config_value("dit_time_dim", dit_time_dim))
        if spectral_pool is None:
            spectral_pool = str(resolve_config_value("spectral_pool", spectral_pool))
        if spectral_pool not in ("mean", "attention"):
            raise ValueError(
                f"spectral_pool={spectral_pool!r} must be 'mean' or 'attention'"
            )
        self.spectral_pool_mode = spectral_pool
        self.last_pool_weights: torch.Tensor | None = None
        self.time_dim = dit_time_dim
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.state_encoder = StateEncoder(obs_dim, hidden_dim=hidden_dim)
        self.time_mlp = SwiGLU(dit_time_dim, 4 * hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(obs_dim, hidden_dim)
        self.spectral_pool_module = (
            SpectralAttentionPool(
                obs_dim,
                hidden_dim=hidden_dim,
                time_dim=dit_time_dim,
                n_heads=dit_heads,
            )
            if spectral_pool == "attention"
            else None
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim, dit_heads, dit_mlp_ratio) for _ in range(dit_depth)
        )
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(hidden_dim, action_dim)
        # Zero final layer: an untrained policy predicts ~zero velocity.
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(
        self, x_t: torch.Tensor, t, condition: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity: ([B, Ka, action_dim], t, [B, K, obs_dim], [B, T, obs_dim])
        -> [B, Ka, action_dim]. Ka = top-p low-frequency rows (<= H)."""
        temb = get_timestep_embedding(t, dim=self.time_dim).to(
            device=x_t.device, dtype=x_t.dtype
        )
        if temb.dim() == 1:
            temb = temb.expand(x_t.shape[0], -1)
        if self.spectral_pool_mode == "attention":
            pooled, weights = self.spectral_pool_module(condition, temb)
            self.last_pool_weights = weights.detach()
            c = self.time_mlp(temb) + pooled
        else:
            self.last_pool_weights = None
            c = self.time_mlp(temb) + self.cond_proj(condition.mean(dim=1))
        s = self.state_encoder(state)
        h = torch.cat([s, self.in_proj(x_t)], dim=1)
        for block in self.blocks:
            h = block(h, c)
        return self.out(self.norm_out(h[:, -x_t.shape[1] :, :]))

        # %% [markdown]
        # ## Critic — V(s, a): 2 cross-attention fusion blocks -> scalar V[B]

        # %%


class Critic(nn.Module):
    """State-action critic with owned encoders; V(s,a) keeps its TD target V(s',a').

    Inputs are raw (never latents, never DCT). Encoded actions query encoded
    states through cross-attention blocks.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int | None = None,
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
        hidden_dim = resolve_config_value("hidden_dim", hidden_dim)
        self.state_encoder = StateEncoder(
            obs_dim, n_obs_steps=n_obs_steps, hidden_dim=hidden_dim
        )
        self.action_encoder = ActionEncoder(
            act_dim, n_obs_steps=n_obs_steps, hidden_dim=hidden_dim
        )
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio)
            for _ in range(num_attn_blocks)
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self, states: torch.Tensor, a: torch.Tensor, executed_n_action_steps=None
    ) -> torch.Tensor:
        """([B, T, obs_dim], [B, H, action_dim]) -> V[B]; tails dropped when given."""
        z = self.state_encoder(states)
        h = self.action_encoder(a, executed_n_action_steps)
        for block in self.blocks:
            h = block(h, z)
        if executed_n_action_steps is None:
            pooled = h.mean(dim=1)
        else:
            if not isinstance(executed_n_action_steps, torch.Tensor):
                executed_n_action_steps = torch.as_tensor(
                    list(executed_n_action_steps), device=h.device
                )
            n_real = executed_n_action_steps.to(h.device)  # [B]
            exec_mask = (
                (
                    torch.arange(h.shape[1], device=h.device).unsqueeze(0)
                    < n_real.unsqueeze(1)
                )
                .to(h.dtype)
                .unsqueeze(-1)
            )  # 1 = executed row
            pooled = (h * exec_mask).sum(dim=1) / exec_mask.sum(dim=1)
        return self.out(pooled).squeeze(-1)
