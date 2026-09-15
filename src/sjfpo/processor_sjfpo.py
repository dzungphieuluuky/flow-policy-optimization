"""SJFPO pre/post processors (LeRobot fashion): obs normalize, action unscale.

Mirrors other policies' ``processor_<name>.py``: named ``ProcessorStep``s plus a
``make_sjfpo_pre_post_processors`` pair factory. Two SJFPO deviations from the
LeRobot default:
- stats are ONLINE (RunningMeanStd updated every training batch), not frozen
  dataset stats, so the obs step shares the policy's live normalizer object
  and exposes ``update_stats`` (no LeRobot equivalent);
- steps operate on the policy's native batch dicts (``{"observation.state"}`` /
  ``{"action"}``) instead of EnvTransitions; shapes pass through unchanged.
"""

from __future__ import annotations

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.networks import RunningMeanStd


@ProcessorStepRegistry.register(name="sjfpo_observation_normalizer")
class SJFPOObservationProcessorStep(ProcessorStep):
    """Normalize ``observation.state`` with the shared live RunningMeanStd."""

    def __init__(self, normalizer: RunningMeanStd):
        self.normalizer = normalizer

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Raw window -> normalized window (identity until the first update)."""
        return self.normalizer(states)

    def update_stats(self, states: torch.Tensor) -> None:
        """Fold raw training frames into the running stats (train batches only)."""
        self.normalizer.update(states)

    def __call__(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        batch[OBS_STATE] = self.normalize(batch[OBS_STATE])
        return batch

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="sjfpo_action_unnormalizer")
class SJFPOActionPostprocessorStep(ProcessorStep):
    """Unscale policy action chunks from [-1, 1] to real env bounds."""

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        self.action_low = action_low
        self.action_high = action_high

    def unnormalize(self, scaled: torch.Tensor) -> torch.Tensor:
        """Scaled chunk -> real env actions (identity when bounds are ±1)."""
        low = self.action_low.to(scaled.device)
        high = self.action_high.to(scaled.device)
        return low + 0.5 * (scaled + 1.0) * (high - low)

    def __call__(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        batch = dict(batch)
        batch[ACTION] = self.unnormalize(batch[ACTION])
        return batch

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_sjfpo_pre_post_processors(
    policy,
) -> tuple[SJFPOObservationProcessorStep, SJFPOActionPostprocessorStep]:
    """Build the (preprocessor, postprocessor) pair sharing the policy's live state.

    Steps hold references (not copies): training stat updates and
    ``set_action_bounds`` stay visible, and ``.to(device)`` keeps working
    because the tensors remain owned by the policy module.
    """
    return (
        SJFPOObservationProcessorStep(policy.obs_normalizer),
        SJFPOActionPostprocessorStep(policy.action_low, policy.action_high),
    )
