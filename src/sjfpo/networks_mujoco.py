"""MLP-only actor for the SJFPO policy, for MuJoCo state-based runs.

A drop-in replacement for :class:`~sjfpo.networks.SJFPOFlowModel` that drops
self-attention from the actor stack. Everything the surrounding policy, loss,
collector, and checkpointing code depends on is reused verbatim rather than
copied:

* ``RunningMeanStd``, ``StateConditioner``, and ``Critic`` are imported from
  ``networks``. The conditioner and critic are already plain MLPs, so there is
  nothing to simplify and duplicating them would only create drift.
* ``SwiGLU``, ``CondMLP``, ``SinusoidalEncoding``, ``_rms_norm`` and
  ``modulate`` are shared building blocks, imported so there is exactly one
  copy of the numerics.

What actually changes is the actor block. ``networks.DiTBlock`` spends a
sub-layer on multi-head self-attention over the K action tokens; ``MLPFlowBlock``
spends it on a learned ``Linear`` across the same fixed K-token axis instead.
That removes ``dit_heads``, SDPA, the key-padding mask, and the six-way adaLN
output, and lets the dead four-dimensional "coordinate-aware" branch go: ``x_t``
is always ``[B, K, action_dim]`` and the conditioner always returns ``[B, D]``,
so that permute/reshape path was never taken.

The FiLM module is deliberately named ``adaLN`` even though it emits only
shift and scale, because ``SJFPOPolicy.get_actor_optim_params`` excludes
parameters matching ``"adaLN"`` from weight decay. Keeping the name means the
optimizer grouping is identical to the DiT run.

One intentional divergence: the ``return_stats`` dict reports ``gate_mix``,
``gate_mlp`` and ``scale_mlp``. The DiT actor reports
``gate_msa``/``scale_msa``/``gate_mlp``/``scale_mlp``. There is no ``msa``
gate because there is no attention half, and no key is invented to make the
wandb series line up; an MLP run's ``cond_*`` metrics are simply not
name-comparable with a DiT run's.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.networks import (
    CondMLP,
    Critic,
    RunningMeanStd,
    SinusoidalEncoding,
    StateConditioner,
    SwiGLU,
    _rms_norm,
    modulate,
)

__all__ = [
    "Critic",
    "MLPFlowBlock",
    "RunningMeanStd",
    "SJFPOFlowModelMLP",
    "SJFPOPolicyMLP",
    "StateConditioner",
]


class MLPFlowBlock(nn.Module):
    """Pre-norm MLP-mixer block with FiLM conditioning.

    Two residual halves: a learned mix across the fixed token axis, then a
    SwiGLU channel MLP gated by the conditioning. The modulation head is
    zero-initialised, so the block starts as an identity mapping and the whole
    actor begins as a no-op, matching the adaLN-Zero behaviour of the DiT
    block.
    """

    def __init__(self, hidden_dim: int, n_tokens: int, mlp_ratio: float):
        """Build one mixer block.

        Args:
            hidden_dim: Token width.
            n_tokens: Number of action tokens, fixed for the run.
            mlp_ratio: Channel-MLP expansion factor.

        Raises:
            ValueError: If ``hidden_dim`` is below two, if ``n_tokens`` is
                below one, or if ``mlp_ratio`` does not give a positive MLP
                width.
        """
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive")
        mlp_hidden = int(hidden_dim * mlp_ratio)
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or mlp_hidden < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        self.norm1 = _rms_norm(hidden_dim)
        self.token_mix = nn.Linear(n_tokens, n_tokens)
        self.norm2 = _rms_norm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, mlp_hidden, hidden_dim)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Mix tokens, then mix channels, both conditioned on ``c``.

        Args:
            x: Tokens of shape ``[B, K, D]``.
            c: Conditioning of shape ``[B, D]`` or ``[B, K, D]`` when each
                token carries its own flow time.
            return_stats: Whether to also return the modulation statistics.

        Returns:
            The transformed tokens, or a tuple of those tokens and a stats dict
            when ``return_stats`` is set.
        """
        token_axis = c.dim() == 3
        b = c.shape[0]
        if token_axis:
            shift, scale = self.adaLN(c).view(b, c.shape[1], 2, -1).unbind(dim=2)
        else:
            shift, scale = self.adaLN(c).view(b, 2, -1).unbind(dim=1)
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        mixed = self.token_mix(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        mlp_out = self.mlp(modulate(self.norm2(x), shift, scale))
        out = x + mlp_out

        if not return_stats:
            return out

        with torch.inference_mode():
            stats = {
                "gate_mix": mixed.square().mean().sqrt(),
                "gate_mlp": mlp_out.square().mean().sqrt(),
                "scale_mlp": (1.0 + scale).square().mean().sqrt(),
            }
        return out, stats


class SJFPOFlowModelMLP(nn.Module):
    """MLP-mixer velocity model over noisy-action tokens.

    Constructor and ``forward`` match
    :class:`~sjfpo.networks.SJFPOFlowModel` exactly, so this is a drop-in for
    ``SJFPOPolicy.actor``. Every validation in the DiT model's forward is
    reproduced so a shape error still fails loudly at the same place.
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
        """Build the actor.

        Args:
            action_dim: Action dimension.
            action_horizon: Number of action tokens in a full chunk.
            hidden_dim: Token width.
            dit_depth: Number of mixer blocks.
            dit_heads: Accepted and ignored, so the signature matches the DiT
                model and the same config drives both.
            dit_mlp_ratio: Channel-MLP expansion factor.
            time_embed_dim: Width of the sinusoidal time embedding.
            timestep_scale: Scale applied to the flow time before embedding.

        Raises:
            ValueError: If the dimensions are out of range, if ``dit_mlp_ratio``
                does not give a positive MLP width, or if ``timestep_scale`` is
                not finite and positive.
        """
        super().__init__()
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if dit_depth < 1 or time_embed_dim < 2:
            raise ValueError("dit_depth and time_embed_dim must be positive")
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
            time_embed_dim, scale=timestep_scale
        )
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.token_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.token_pos, std=0.02)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            MLPFlowBlock(hidden_dim, action_horizon, dit_mlp_ratio)
            for _ in range(dit_depth)
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

        ``t`` is either a scalar, ``[B]``, or ``[B, K, 1]``. The per-token form
        gives each action token its own time, shared across its action
        dimensions. The final layer is zero-initialised, so an untrained model
        predicts near-zero velocity.

        Args:
            x_t: Noisy action tokens of shape ``[B, K, action_dim]``.
            t: Flow time, as a scalar, ``[B]``, or ``[B, K, 1]``.
            condition: Conditioning of shape ``[B, hidden_dim]``.
            return_stats: Whether to also return the modulation statistics.

        Returns:
            Predicted velocity of shape ``[B, K, action_dim]``, or a tuple of
            that tensor and a stats dict when ``return_stats`` is set.

        Raises:
            ValueError: If ``x_t`` is not ``[B, K, action_dim]`` with positive
                batch and token dimensions, if it carries more tokens than the
                action horizon, or if ``condition`` disagrees with the batch
                size or hidden width.
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
            shift_out, scale_out = self._out_modulation(c, token_axis)
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
                    for k in ("gate_mix", "gate_mlp", "scale_mlp")
                },
            }
        shift_out, scale_out = self._out_modulation(c, token_axis)
        out = self.out(modulate(self.norm_out(h), shift_out, scale_out))
        return out, stats

    def _out_modulation(
        self, c: torch.Tensor, token_axis: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the output-layer FiLM into shift and scale.

        Args:
            c: Fused conditioning.
            token_axis: Whether ``c`` carries a per-token axis.

        Returns:
            The shift and scale, broadcast to ``[B, 1, D]`` or ``[B, K, D]``.
        """
        if token_axis:
            shift_out, scale_out = self.adaLN_out(c).chunk(2, dim=-1)
            return shift_out, scale_out
        shift_out, scale_out = self.adaLN_out(c)[:, None, :].chunk(2, dim=-1)
        return shift_out, scale_out


class SJFPOPolicyMLP(SJFPOPolicy):
    """:class:`SJFPOPolicy` with the actor swapped for the MLP model.

    Everything else — the state conditioner, the critic, the frozen behavior
    snapshot, the EMA, and the optimizer param groups — is inherited unchanged,
    so the training loop, the loss, and the collector need no changes.
    """

    def __init__(self, config: SJFPOConfig, dataset_stats: dict | None = None):
        """Build the policy, then replace the DiT actor with the MLP actor.

        ``SJFPOPolicy.__init__`` deep-copies the actor it just built into
        ``old_actor`` for the frozen behavior snapshot, so that copy has to be
        redone here; otherwise the snapshot would still hold DiT weights and
        ``snapshot_behavior_policy`` would fail on a key mismatch.

        Args:
            config: Active configuration, including the feature shapes.
            dataset_stats: Optional dataset statistics, forwarded to the base
                class.
        """
        super().__init__(config, dataset_stats)
        self.actor = SJFPOFlowModelMLP(
            config.action_feature.shape[0],
            config.action_horizon,
            config.hidden_dim,
            config.dit_depth,
            config.dit_heads,
            config.dit_mlp_ratio,
            config.time_embed_dim,
            config.timestep_scale,
        )
        self.old_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
