"""Conditional flow-matching velocity model over action tokens.

Two actors live here and both keep the action tokens on separate axes:

``SJFPOFlowModel``
    The DiT actor. One block is self-attention, one SwiGLU, and adaLN.
``SJFPOFlowResMLP``
    The ``resmlp`` actor. The self-attention and the adaLN conditioning are
    kept, because this repo needs both. What changes is the MLP: it becomes the
    residual unit of Wang et al. (2025), arXiv:2503.14858.

Both take ``(action_dim, action_horizon, hidden_dim, ...)``, so ``actor_arch``
swaps the trunk and nothing else.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from myfpo.attention import DiTBlock, SelfAttention
from myfpo.layers import CondMLP, SinusoidalEncoding, build_rms_norm, modulate
from myfpo.utils import rms

def effective_rank(features: torch.Tensor) -> torch.Tensor:
    """``exp`` spectral entropy of ``[n, hidden]``. This is the collapse probe
    of Dohare et al. (2306.13812). It falls as backprop settles on a low-rank
    solution.

    The result is 0-dim. The input reshapes to ``[n, hidden]``. The singular
    values number ``min(n, hidden)``. The function returns one scalar.
    """
    matrix = features.reshape(-1, features.shape[-1]).float()
    sigma = torch.linalg.svdvals(matrix)
    p = sigma / sigma.sum().clamp_min(1e-12)
    # entr(0) is 0 by convention, so the rank-deficient case needs no clamp.
    return torch.exp(torch.special.entr(p).sum())


class _FlowActorBase(nn.Module):
    """The stem both actors share: embed, project, condition, decode.

    Every part of this class is actor-independent. The action tokens stay on
    their own axis throughout, because the CFM draw gives each token its own
    ``t``. A subclass supplies only the block list.

    ``return_stats=True`` gives ``(velocity, stats)`` plus per-block gate and
    scale RMS reductions. The default path returns the velocity only. It
    reduces nothing.

    The stats path is expensive. It runs an SVD (``effective_rank``) and a full
    parameter sweep (``weight_magnitude``). ``SJFPOPolicy.forward`` calls it on
    every minibatch forward if ``compute_info=True``, and the trainer passes
    ``compute_info=tracker.enabled``. So if ``report_to`` is wandb or
    tensorboard, it runs once per microbatch per optimizer step. For throughput
    runs, use ``--sjfpo.report_to none``. That flag turns the stats path off,
    along with every other metric.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        time_embed_dim: int,
        block_cls: type[nn.Module],
        block_kwargs: dict,
    ):
        super().__init__()
        if action_dim < 1 or hidden_dim < 2:
            raise ValueError("action_dim must be positive and hidden_dim at least 2")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if time_embed_dim < 2:
            raise ValueError(f"time_embed_dim={time_embed_dim} must be at least 2")
        if self.num_blocks < 1:
            raise ValueError(f"an actor needs >= 1 block, got {self.num_blocks}")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        self.time_embed_dim = time_embed_dim
        self.timestep_embedder = SinusoidalEncoding(self.time_embed_dim)
        self.in_proj = nn.Linear(action_dim, hidden_dim)
        self.token_pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.normal_(self.token_pos, std=0.02)
        self.time_mlp = CondMLP(time_embed_dim, 4 * hidden_dim, hidden_dim)
        # The blocks are built at this exact point. Parameter construction order
        # fixes the torch RNG stream, so building them elsewhere would redraw
        # every weight in the actor and silently change a seeded run.
        self.blocks = nn.ModuleList(
            block_cls(**block_kwargs) for _ in range(self.num_blocks)
        )
        self.norm_out = build_rms_norm(hidden_dim)
        self.adaLN_out = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.constant_(self.adaLN_out.weight, 0.0)
        nn.init.constant_(self.adaLN_out.bias, 0.0)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.out.weight, 0.0)
        nn.init.constant_(self.out.bias, 0.0)

    @torch.inference_mode()
    def weight_magnitude(self) -> torch.Tensor:
        """Mean absolute weight. This is the correlate of Dohare. The magnitudes
        grow under continual training, before the rank drop.

        The function sums per parameter. It does not use one accumulator. A
        float scalar would land on CPU."""
        total = None
        count = 0
        for param in self.parameters():
            s = param.detach().abs().sum(dtype=torch.float32)
            total = s if total is None else total + s
            count += param.numel()
        return total / max(count, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        condition: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the velocity from the token actions and the per-token times.

        ``t`` is a scalar, ``[B]`` or ``[B, K, 1]``. One time belongs to each
        token, and that time is shared across the action dims of the token. The
        last layer is zero-init, so an untrained model predicts a near-zero
        velocity. The function returns ``[B, K, action_dim]``, plus stats if
        asked."""
        if x_t.dim() != 3 or x_t.shape[2] != self.action_dim:
            raise ValueError(f"x_t must have shape [B, K, {self.action_dim}], got {tuple(x_t.shape)}")
        if condition.dim() != 2 or condition.shape[0] != x_t.shape[0] or condition.shape[1] != self.hidden_dim:
            raise ValueError(f"condition must have shape [{x_t.shape[0]}, {self.hidden_dim}], got {tuple(condition.shape)}")
        if x_t.shape[1] > self.action_horizon:
            raise ValueError(f"action token count {x_t.shape[1]} exceeds action_horizon {self.action_horizon}")

        t = torch.as_tensor(t, dtype=torch.float32, device=x_t.device)
        cond = condition.to(device=x_t.device, dtype=x_t.dtype)
        token_axis = t.dim() in (2, 3) and tuple(t.shape[:2]) == tuple(x_t.shape[:2])
        if token_axis:
            if t.dim() == 3:
                if t.shape[-1] != 1:
                    raise ValueError(f"t must end in a singleton action-dimension axis, got {tuple(t.shape)}")
                t = t.squeeze(-1)
            temb = self.timestep_embedder.encode_times(t.reshape(-1))
            temb = temb.reshape(x_t.shape[0], x_t.shape[1], -1).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond[:, None, :]
        elif t.dim() <= 1:
            t = t.reshape(-1)
            if t.numel() not in (1, x_t.shape[0]):
                raise ValueError(f"t must contain 1 or {x_t.shape[0]} values, got {t.numel()}")
            temb = self.timestep_embedder.encode_times(t).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c = self.time_mlp(temb) + cond
        else:
            raise ValueError(f"t must be scalar, [B], or [B, K, 1], got {tuple(t.shape)}")

        # Both branches reach the same projection. Only temb's rank differed.
        token_pos = self.token_pos[:, : x_t.shape[1], :].to(
            device=x_t.device, dtype=x_t.dtype
        )
        h = self.in_proj(x_t) + token_pos

        def decode(hidden: torch.Tensor) -> torch.Tensor:
            shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
            if not token_axis:
                shift, scale = shift[:, None, :], scale[:, None, :]
            return self.out(modulate(self.norm_out(hidden), shift, scale))

        if not return_stats:
            for block in self.blocks:
                h = block(h, c)
            return decode(h)

        block_stats: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            h, s = block(h, c, return_stats=True)
            block_stats.append(s)
        stats = self._collect_stats(temb, condition, c, h, block_stats)
        return decode(h), stats

    @torch.compiler.disable
    def _collect_stats(
        self,
        temb: torch.Tensor,
        condition: torch.Tensor,
        c: torch.Tensor,
        h: torch.Tensor,
        block_stats: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Diagnostics, deliberately outside any compiled graph.

        The decorator ``torch.compiler.disable`` holds this code up. It does not
        make the code faster. Under ``torch.compile``, this block died with::

            BackendCompilerFailed: backend='inductor' raised:
            RuntimeError: <weakref ... to 'torch.storage.UntypedStorage'>
            While executing %reshape_5 ... features.reshape(-1, hidden)

        Two independent problems compound:

        1. A call to ``torch.inference_mode()`` inside a traced region mints inference
           tensors. The version counter of those tensors confuses inductor. The
           storage is released while the lowering still holds a weak reference
           to it.
        2. A call to ``svdvals`` lowers to a LAPACK call. Inductor does not own
           the lifetime of that call. So the same early release reaches it.

        Breaking the graph once per statistics pass avoids both problems.
        Dynamo stops tracing at the decorator. So nothing downstream of it is
        captured.

        The default path sets ``return_stats=False``. That path never enters
        this function. So it stays fully compiled. That is the path that matters
        for throughput.
        """
        with torch.inference_mode():
            return {
                "temb": rms(temb),
                "fused": rms(condition),
                "c": rms(c),
                **{
                    k: torch.stack([s[k] for s in block_stats]).mean()
                    for k in ("gate_msa", "gate_mlp", "scale_msa", "scale_mlp")
                },
                "effective_rank": effective_rank(h),
                "weight_magnitude": self.weight_magnitude(),
            }



class SJFPOFlowModel(_FlowActorBase):
    """The DiT actor. One block is self-attention, one SwiGLU, and adaLN.

    ``dit_depth`` counts blocks. Each block holds six Dense layers, so the
    `` Dense-layer depth of Wang et al. (2025)`` is ``6 * dit_depth``. At the
    default ``dit_depth=3`` that is 18, against the depth-4 baseline that paper
    scales from.

    The block itself already contains every component that paper credits:
    residual connections, a norm, and a Swish-based activation. So this actor is
    that paper's building block plus attention and a wider modulation.
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
    ):
        if dit_depth < 1 or dit_heads < 1:
            raise ValueError(
                f"dit_depth={dit_depth} and dit_heads={dit_heads} must be >= 1"
            )
        if (
            not math.isfinite(dit_mlp_ratio)
            or dit_mlp_ratio <= 0
            or int(hidden_dim * dit_mlp_ratio) < 1
        ):
            raise ValueError("dit_mlp_ratio must produce a positive hidden width")
        self.num_blocks = dit_depth
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            DiTBlock,
            {"hidden_dim": hidden_dim, "n_heads": dit_heads,
             "mlp_ratio": dit_mlp_ratio},
        )


class ResidualMLP(nn.Module):
    """The residual unit of Wang et al. (2025), arXiv:2503.14858, Section 3.

    The paper's block is ``num_units`` repeats of Dense -> LayerNorm -> Swish,
    with one residual skip added immediately after the final activation. The
    paper puts the skip there and not after every unit, so the gradient path is
    ``num_units`` layers long however deep the stack.

    Two departures from the paper, both deliberate and both recorded here.

    The norm is RMSNorm, not LayerNorm. The paper's other citations and the
    DiT lineage this repo descends from both use it, and the repo already has
    one norm implementation.

    The activation is ``F.silu``, which is Swish. The paper's Swish is the same
    function. The DiT block here gates it into SwiGLU; this unit does not, so
    the two actors differ in the MLP as well as in the residual layout.
    """

    def __init__(self, hidden_dim: int, hidden_mult: float, num_units: int):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if num_units < 1:
            raise ValueError(f"num_units={num_units} must be >= 1")
        inner = int(hidden_dim * hidden_mult)
        if inner < 1:
            raise ValueError("resmlp_hidden_mult must produce a positive width")
        self.inner = inner
        self.num_units = num_units
        self.denses = nn.ModuleList(
            nn.Linear(hidden_dim if i == 0 else inner, inner)
            for i in range(num_units)
        )
        self.norms = nn.ModuleList(build_rms_norm(inner) for _ in range(num_units))
        self.out = nn.Linear(inner, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x + residual(x)``. The residual is zero at init, so this is the
        identity on a fresh actor."""
        h = x
        for dense, norm in zip(self.denses, self.norms, strict=True):
            h = norm(dense(h))
            h = F.silu(h)
        return x + self.out(h)


class ResMLPBlock(nn.Module):
    """One ``resmlp`` block: attention, then the paper's residual MLP.

    The self-attention and the adaLN conditioning are the DiT parts, and they
    stay. Two reasons, both measured rather than assumed.

    The attention keeps the chunk as a sequence. The CFM loss is separable
    across tokens, so per-token mixing would train, but the ASPO trust region
    and the per-chunk advantage still couple the tokens. Both reference networks
    for chunked action prediction mix across action steps: Diffusion Policy with
    a temporal ``Conv1d`` (``kernel_size=5``) and ACT with self-attention over
    the chunk.

    The adaLN keeps ``t`` available at every depth. The 1000-layer paper has no
    time input, so it can condition once at the input. A flow model cannot, and
    the paper has no evidence about that case.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        hidden_mult: float,
        num_units: int,
    ):
        super().__init__()
        self.norm1 = build_rms_norm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, n_heads)
        self.norm2 = build_rms_norm(hidden_dim)
        self.mlp = ResidualMLP(hidden_dim, hidden_mult, num_units)
        self.adaLN = CondMLP(hidden_dim, hidden_dim, 6 * hidden_dim)
        nn.init.constant_(self.adaLN.fc2.weight, 0.0)
        nn.init.constant_(self.adaLN.fc2.bias, 0.0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``[B, K, H]`` in, ``[B, K, H]`` out. ``c`` is ``[B, H]`` or
        ``[B, K, H]``. The modulation head is zero-init, so a fresh stack is the
        identity."""
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, K, H], got {tuple(x.shape)}")
        b = x.shape[0]
        if c.shape[0] != b:
            raise ValueError(f"c has batch {c.shape[0]} but x has {b}")
        if c.dim() == 3:
            if c.shape[1] != x.shape[1]:
                raise ValueError(
                    f"c has {c.shape[1]} conditioning tokens but x has {x.shape[1]}"
                )
        else:
            if c.dim() != 2 or c.shape[1] != self.adaLN.fc1.in_features:
                raise ValueError(
                    f"c must have shape [{b}, {self.adaLN.fc1.in_features}], "
                    f"got {tuple(c.shape)}"
                )
            c = c[:, None, :]

        parts = self.adaLN(c).view(b, c.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts.unbind(
            dim=2
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(h)
        out = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

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


class SJFPOFlowResMLP(_FlowActorBase):
    """The ``resmlp`` actor. DiT's attention and adaLN, the paper's MLP unit.

    ``resmlp_blocks`` counts blocks and ``resmlp_units`` counts the Dense layers
    inside one block's residual unit. So the paper's depth metric, which counts
    Dense layers, is ``resmlp_blocks * resmlp_units`` for the MLP part alone.
    The paper fixes the unit at four.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        resmlp_blocks: int,
        resmlp_units: int,
        resmlp_hidden_mult: float,
        n_heads: int,
        time_embed_dim: int,
    ):
        if resmlp_blocks < 1:
            raise ValueError(f"resmlp_blocks={resmlp_blocks} must be >= 1")
        if resmlp_units < 1:
            raise ValueError(f"resmlp_units={resmlp_units} must be >= 1")
        if n_heads < 1:
            raise ValueError(f"n_heads={n_heads} must be >= 1")
        self.num_blocks = resmlp_blocks
        super().__init__(
            action_dim,
            action_horizon,
            hidden_dim,
            time_embed_dim,
            ResMLPBlock,
            {"hidden_dim": hidden_dim, "n_heads": n_heads,
             "hidden_mult": resmlp_hidden_mult, "num_units": resmlp_units},
        )
