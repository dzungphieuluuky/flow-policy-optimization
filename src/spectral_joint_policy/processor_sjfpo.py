"""SJFPO pre/post processors (LeRobot style): norm obs, unscale action.

``ProcessorStep`` is not an ``nn.Module``, so bounds stay plain attributes with
a per-``(device, dtype)`` cache and ``unnormalize`` needs no per-call ``.to()``.
"""

from __future__ import annotations

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from spectral_joint_policy.encoder import RunningMeanStd


class SJFPOObservationProcessorStep(ProcessorStep):
    """Holds a reference to the policy's ``RunningMeanStd``, not a copy, so stat
    updates fold into the buffers the policy reads at forward time."""

    def __init__(self, normalizer: RunningMeanStd):
        self.normalizer = normalizer

    @torch.no_grad()
    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Norm a raw obs window. Identity until the first stats update."""
        return self.normalizer(states)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[OBS_STATE] = self.normalize(batch[OBS_STATE])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through unchanged; the schema is untouched."""
        return features


class SJFPOActionPostprocessorStep(ProcessorStep):
    """Unscale action chunks from ``[-1, 1]`` to env bounds.
    ``set_action_bounds`` clears both caches."""

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        self.set_action_bounds(action_low, action_high)

    def set_action_bounds(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        """Set the action bounds, invalidate cached coefficients."""
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

        self._scale_cpu = 0.5 * (high - low)
        self._bias_cpu = 0.5 * (high + low)

    @torch.inference_mode()
    def unnormalize(self, scaled: torch.Tensor) -> torch.Tensor:
        """Unscale a chunk into real env bounds. Direct, no cache."""
        scale = self._scale_cpu.to(device=scaled.device, dtype=scaled.dtype)
        bias = self._bias_cpu.to(device=scaled.device, dtype=scaled.dtype)
        return torch.addcmul(bias, scale, scaled)

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[ACTION] = self.unnormalize(batch[ACTION])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through unchanged; the schema is untouched."""
        return features


def make_sjfpo_pre_post_processors(
    policy,
) -> tuple[SJFPOObservationProcessorStep, SJFPOActionPostprocessorStep]:
    """Build the pre/post pair. The policy setter syncs action bounds."""
    return (
        SJFPOObservationProcessorStep(policy.obs_normalizer),
        SJFPOActionPostprocessorStep(policy.action_low, policy.action_high),
    )
