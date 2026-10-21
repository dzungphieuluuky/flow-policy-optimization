"""myfpo pre/post processors (LeRobot style). The pair normalises the obs and
squashes the policy output into the env's action bounds.
"""

from __future__ import annotations
import warnings
import torch
from torch import nn
from myfpo.layers import MIN_WIDTH, CondMLP, TemporalConv, build_rms_norm
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

# ---- from encoder.py ----
class RunningMeanStd(nn.Module):
    """Per-dimension running mean and variance over raw obs frames."""

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim={dim} must be positive")
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))
        self.uninitialised = True
        self.passthrough_warned = False
        self.eps = eps
        self.clip = clip

    @torch.inference_mode()
    def update(self, x: torch.Tensor) -> None:
        """Fold raw frames into the running statistics."""
        if x.shape[-1] != self.mean.numel():
            raise ValueError(f"update got last dim {x.shape[-1]}, expected {self.mean.numel()}")
        batch = (
            x.detach()
            .reshape(-1, x.shape[-1])
            .to(self.mean.device, dtype=torch.float32)
        )
        b = batch.shape[0]
        if b == 0:
            return
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
        self.uninitialised = False

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state. Derive the latch again."""
        super()._load_from_state_dict(*args, **kwargs)
        self.uninitialised = bool(self.count.item() == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise with the frozen statistics."""
        if self.uninitialised:
            if not self.passthrough_warned:
                warnings.warn(
                    f"{type(self).__name__}: forward before the first update(); "
                    "returning raw input", stacklevel=2,
                )
                self.passthrough_warned = True
            return x
        norm = (x - self.mean) * torch.rsqrt(self.var + self.eps)
        return norm.clamp(-self.clip, self.clip).to(x.dtype)

class StateEncoder(nn.Module):
    """Flatten the window. Pass it to one SiLU MLP. Normalise. Return it.

    The norm is the point. This was the only unnormalised path in the stack:
    every observation reaches the actor through here and then through an
    unnormalised adaLN projection. RMSNorm over LayerNorm, because it does not
    subtract the mean and it matches the actor's norm1/norm2/norm_out.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        max_obs_steps: int,
        temporal_kernel: int = 0,
    ):
        super().__init__()
        if hidden_dim < MIN_WIDTH:
            raise ValueError("hidden_dim must be at least 2")
        if max_obs_steps < 1:
            raise ValueError(f"max_obs_steps must be >= 1, got {max_obs_steps}")
        self.obs_dim = obs_dim
        self.max_obs_steps = max_obs_steps
        self.net = CondMLP(max_obs_steps * obs_dim, hidden_dim, hidden_dim)
        self.out_norm = build_rms_norm(hidden_dim)
        # The filter is on the frame axis. It keeps the width, so the flatten
        # below and every shape after it stay the same.
        self.frame_conv = (
            TemporalConv(obs_dim, temporal_kernel) if temporal_kernel else None
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Accept the window ``[B, T, obs]``. Return one normalised conditioning
        vector of width ``hidden_dim``.
        """
        if states.dim() != 3 or states.shape[2] != self.obs_dim:
            raise ValueError(f"states must have shape [B, {self.max_obs_steps}, {self.obs_dim}], got {tuple(states.shape)}")
        if states.shape[1] != self.max_obs_steps:
            raise ValueError(f"states has {states.shape[1]} frames, built for max_obs_steps={self.max_obs_steps}")
        frames = states.transpose(1, 2)
        if self.frame_conv is not None:
            frames = self.frame_conv(frames)
        return self.out_norm(self.net(frames.transpose(1, 2).flatten(1)))


def env_squash_terms(
    low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scale and bias of the env's action bounds.

    ``scale = (high - low) / 2`` and ``bias = low + scale``. The bias is
    the low end plus the half width, so the two terms come from one
    difference and cannot drift apart.
    """
    scale = 0.5 * (high - low)
    return scale, low + scale


def squash_to_env(
    raw: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Map a raw policy output into the env's action bounds.

    ``bias + scale * tanh(raw)``. The ``tanh`` is the only thing that bounds
    the action, and it is applied exactly once, on the way to the env: the
    flow is trained on the raw value, not on the squash of its own output.

    ``scale`` and ``bias`` are the affine terms of the env's bounds, from
    :func:`env_squash_terms`.
    """
    scale = scale.to(device=raw.device, dtype=raw.dtype)
    bias = bias.to(device=raw.device, dtype=raw.dtype)
    return torch.addcmul(bias, scale, torch.tanh(raw))


def squash_to_env_bounds(
    raw: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Map a raw policy output into the env's action bounds.

    One place computes the affine terms, so the collector, the training
    diagnostics and the postprocessor cannot drift apart.
    """
    scale, bias = env_squash_terms(low, high)
    return squash_to_env(raw, scale, bias)


class MyfpoObservationProcessorStep(ProcessorStep):
    """Hold a reference to the ``RunningMeanStd`` of the policy, not a copy."""

    def __init__(self, normalizer: RunningMeanStd):
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[OBS_STATE] = self.normalizer(batch[OBS_STATE])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through. The schema is unchanged."""
        return features


class MyfpoActionPostprocessorStep(ProcessorStep):
    """Map the actor's raw output into the env's action bounds with ``tanh``."""

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        self.set_action_bounds(action_low, action_high)

    def set_action_bounds(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        """Set the action bounds. Recomputes the cached scale and bias."""
        low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1).clone()
        high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1).clone()
        if low.shape != high.shape:
            raise ValueError(
                f"action_low shape {tuple(low.shape)} != "
                f"action_high shape {tuple(high.shape)}"
            )
        if not bool((high >= low).all()):
            raise ValueError("action_high must be elementwise >= action_low")

        self.action_low = low
        self.action_high = high

        self.scale_cpu, self.bias_cpu = env_squash_terms(low, high)

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw = batch[ACTION]
        out = dict(batch)
        out[ACTION] = squash_to_env(raw, self.scale_cpu, self.bias_cpu)
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through. The function does not change the schema."""
        return features


def make_pre_post_processors(
    policy,
) -> tuple[MyfpoObservationProcessorStep, MyfpoActionPostprocessorStep]:
    """Build the pre/post pair. The policy setter syncs action bounds."""
    return (
        MyfpoObservationProcessorStep(policy.obs_normalizer),
        MyfpoActionPostprocessorStep(policy.action_low, policy.action_high),
    )
