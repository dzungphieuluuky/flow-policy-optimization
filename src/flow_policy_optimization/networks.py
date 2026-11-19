"""flow_policy_optimization.networks: encoder, flow actor and critic modules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

TIME_SCALE = 1000.0
ROPE_BASE = 10000.0

def _ortho_linear(in_dim: int, out_dim: int, std: float = math.sqrt(2.0)) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer
class RMSNorm(nn.RMSNorm):
    """RMSNorm in fp32 compute while preserving model dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps)
        return y.to(dtype=x.dtype)
class CondMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if min(in_dim, hidden_dim, out_dim) < 1:
            raise ValueError("CondMLP dimensions must be positive")
        self.fc1 = _ortho_linear(in_dim, hidden_dim)
        self.fc2 = _ortho_linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))
class SinusoidalEncoding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if max_period <= 0:
            raise ValueError(f"max_period must be > 0, got {max_period}")
        self.dim = dim
        self.half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(self.half, dtype=torch.float32) / self.half),
            persistent=False,
        )

    def encode_times(self, t: torch.Tensor) -> torch.Tensor:
        ang = (t.float().unsqueeze(-1) * TIME_SCALE) * self.freqs.to(t.device)
        emb = torch.cat([ang.cos(), ang.sin()], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 else emb
class SelfAttention(nn.Module):
    """Causal SDPA self-attention with RoPE."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1 or n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(f"invalid SelfAttention(hidden_dim={hidden_dim}, n_heads={n_heads})")
        head_dim = hidden_dim // n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.n_heads = n_heads
        self.head_dim = head_dim
        half = head_dim // 2
        self._rope_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self.register_buffer("inv_freq", 1.0 / (ROPE_BASE ** (torch.arange(half).float() / half)), persistent=False)
        self.qkv = _ortho_linear(hidden_dim, 3 * hidden_dim)
        self.out = _ortho_linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        h, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, seq, h, d).transpose(1, 2)
        k = k.view(b, seq, h, d).transpose(1, 2)
        v = v.view(b, seq, h, d).transpose(1, 2)
        key = (seq, x.device)
        cached = self._rope_cache.get(key)
        if cached is None:
            # Build outside inference mode so cached tensors may enter autograd graphs.
            with torch.inference_mode(False):
                freqs = torch.outer(torch.arange(seq, device=x.device, dtype=torch.float32), self.inv_freq)
                cached = (freqs.cos()[None, None].clone(), freqs.sin()[None, None].clone())
            self._rope_cache[key] = cached
        cos, sin = cached
        q = torch.stack([q[..., 0::2] * cos - q[..., 1::2] * sin, q[..., 0::2] * sin + q[..., 1::2] * cos], dim=-1).flatten(-2)
        k = torch.stack([k[..., 0::2] * cos - k[..., 1::2] * sin, k[..., 0::2] * sin + k[..., 1::2] * cos], dim=-1).flatten(-2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, seq, h * d))
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim < 1 or n_heads < 1 or hidden_dim % n_heads != 0:
            raise ValueError(f"invalid CrossAttention(hidden_dim={hidden_dim}, n_heads={n_heads})")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = _ortho_linear(hidden_dim, hidden_dim)
        self.kv_proj = _ortho_linear(hidden_dim, 2 * hidden_dim)
        self.out = _ortho_linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, lq, _ = query.shape
        lk, h, d = context.shape[1], self.n_heads, self.head_dim
        q = self.q_proj(query).view(b, lq, h, d).transpose(1, 2)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(b, lk, h, d).transpose(1, 2)
        v = v.view(b, lk, h, d).transpose(1, 2)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, lq, h * d))
class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, max_obs_steps: int, n_heads: int = 2):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if max_obs_steps < 1:
            raise ValueError(f"max_obs_steps must be >= 1, got {max_obs_steps}")
        self.obs_dim = obs_dim
        self.max_obs_steps = max_obs_steps
        self.n_heads = n_heads
        self.frame = CondMLP(obs_dim, hidden_dim, hidden_dim)
        self.attn_norm = RMSNorm(hidden_dim, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.out_norm = RMSNorm(hidden_dim, eps=1e-6)

    def encode_tokens(self, states: torch.Tensor) -> torch.Tensor:
        h = self.frame(states)
        return h + self.attn(self.attn_norm(h))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.encode_tokens(states)[:, -1])
class FlowMLP(nn.Module):
    """Flat MLP flow velocity head over ``[x_flat | t_flat | cond]``."""

    def __init__(self, action_dim: int, action_horizon: int, hidden_dim: int, time_embed_dim: int, num_layers: int = 4):
        super().__init__()
        if action_dim < 1 or action_horizon < 1 or hidden_dim < 2 or time_embed_dim < 2 or num_layers < 1:
            raise ValueError("invalid FlowMLP dimensions")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.num_layers = num_layers
        flat = action_horizon * action_dim
        self.timestep_embedder = SinusoidalEncoding(time_embed_dim)
        self.in_block = CondMLP(flat + action_horizon * time_embed_dim + hidden_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(CondMLP(hidden_dim, 4 * hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norm_out = RMSNorm(hidden_dim, eps=1e-6)
        self.out = nn.Linear(hidden_dim, flat)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _embed_times(self, t: torch.Tensor, b: int, k: int) -> torch.Tensor:
        """Embed ``t`` given as ``[B, K]``, ``[B, K, 1]`` grid, a scalar, or ``[B]``."""
        if t.dim() == 3:
            t = t.squeeze(-1)
        if t.dim() == 2 and tuple(t.shape) == (b, k):
            grid = t
        else:
            flat = t.reshape(-1)
            grid = flat.expand(b, k) if flat.numel() == 1 else flat[:, None].expand(b, k)
        return self.timestep_embedder.encode_times(grid.reshape(-1)).reshape(b, k, -1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor | float, condition: torch.Tensor) -> torch.Tensor:
        b, k, _ = x_t.shape
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        temb = self._embed_times(t_tensor, b, k).reshape(b, -1).to(dtype=x_t.dtype)
        h = self.in_block(torch.cat([x_t.reshape(b, -1), temb, condition.to(device=x_t.device, dtype=x_t.dtype)], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm_out(h)).reshape_as(x_t)
class ActionAffine(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias
class StateAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * scale * self.weight + self.bias
class Critic(nn.Module):
    """Cross-attention Q(s_t, a_{t:t+h}) critic."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        num_obs_steps: int,
        action_horizon: int,
        *,
        num_layers: int = 2,
        state_hidden: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_steps = num_obs_steps
        self.action_horizon = action_horizon
        state_dim = state_hidden if state_hidden is not None else hidden_dim
        self.input_dim = 2 * state_dim + 1

        self.norm_state = StateAffine(num_obs_steps * obs_dim)
        self.state_encoder = StateEncoder(obs_dim, state_dim, num_obs_steps)
        self.norm_action = ActionAffine(action_horizon * action_dim)
        self.act_in = _ortho_linear(action_dim, state_dim)
        self.act_attn = SelfAttention(state_dim, 4)
        self.act_attn_norm = RMSNorm(state_dim, eps=1e-6)
        self.obs_to_act = CrossAttention(state_dim, 4)
        self.act_to_obs = CrossAttention(state_dim, 4)
        self.cross_norm_obs = RMSNorm(state_dim, eps=1e-6)
        self.cross_norm_act = RMSNorm(state_dim, eps=1e-6)
        self.v_out = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.v_out.weight)
        nn.init.zeros_(self.v_out.bias)

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.extend([_ortho_linear(self.input_dim if i == 0 else hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()])
        final = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, exec_length_frac: torch.Tensor | None = None) -> torch.Tensor:
        obs_tokens = self.state_encoder.out_norm(
            self.state_encoder.encode_tokens(self.norm_state(states.flatten(1)).view(states.shape))
        )
        act_tokens = self.act_in(self.norm_action(actions.flatten(1)).view(actions.shape))
        act_tokens = act_tokens + self.act_attn(self.act_attn_norm(act_tokens))
        obs = obs_tokens + self.obs_to_act(self.cross_norm_obs(obs_tokens), act_tokens)
        act = act_tokens + self.act_to_obs(self.cross_norm_act(act_tokens), obs_tokens)

        parts = [obs[:, -1], act.mean(dim=1)]
        if exec_length_frac is None or exec_length_frac.shape != (states.shape[0],):
            raise ValueError("exec_length_frac of shape [B] is required")
        parts.append(exec_length_frac.to(dtype=parts[0].dtype).unsqueeze(-1))
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)

    def forward_v(self, states: torch.Tensor) -> torch.Tensor:
        """State value V(s); same encoder, no action tokens. [B]"""
        obs_tokens = self.state_encoder.out_norm(
            self.state_encoder.encode_tokens(self.norm_state(states.flatten(1)).view(states.shape))
        )
        return self.v_out(obs_tokens[:, -1]).squeeze(-1)


# ---------------------------------------------------------------------------
# Flow & RL objective math
