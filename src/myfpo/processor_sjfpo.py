"""SJFPO pre/post processors (LeRobot style). The pair norms the obs and
unscales the action.

``ProcessorStep`` is not an ``nn.Module``, so the bounds stay plain CPU tensors.
The postprocessor casts them to the device and dtype of the caller on every
call. The cost is small: two tiny tensors, once per chunk.
"""

from __future__ import annotations

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.encoder import RunningMeanStd


class SJFPOObservationProcessorStep(ProcessorStep):
    """Hold a reference to the ``RunningMeanStd`` of the policy, not a copy.

    So a stat update folds into the buffers that the policy reads at forward
    time.
    """

    def __init__(self, normalizer: RunningMeanStd):
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        # Identity until the first stats update.
        out[OBS_STATE] = self.normalizer(batch[OBS_STATE])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through. The function does not change the
        schema.
        """
        return features


class SJFPOActionPostprocessorStep(ProcessorStep):
    """Unscale action chunks from ``[-1, 1]`` to env bounds.

    ``set_action_bounds`` recomputes the scale and bias. The object has no
    caches to clear. Only the two CPU coefficient tensors exist.
    """

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        self.set_action_bounds(action_low, action_high)

    def set_action_bounds(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        """Set the action bounds. The function recomputes the scale and bias
        coefficients from them."""
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
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Unscale into real env bounds. The coefficients cast per call.
        scaled = batch[ACTION]
        scale = self._scale_cpu.to(device=scaled.device, dtype=scaled.dtype)
        bias = self._bias_cpu.to(device=scaled.device, dtype=scaled.dtype)
        out = dict(batch)
        out[ACTION] = torch.addcmul(bias, scale, scaled)
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through. The function does not change the schema."""
        return features


def make_sjfpo_pre_post_processors(
    policy,
) -> tuple[SJFPOObservationProcessorStep, SJFPOActionPostprocessorStep]:
    """Build the pre/post pair. The policy setter syncs action bounds."""
    return (
        SJFPOObservationProcessorStep(policy.obs_normalizer),
        SJFPOActionPostprocessorStep(policy.action_low, policy.action_high),
    )
