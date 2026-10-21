"""myfpo pre/post processors (LeRobot style). The pair normalises the obs and
squashes the policy output into the env's action bounds.
"""

from __future__ import annotations

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.encoder import RunningMeanStd


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
