"""SJFPO pre/post processors (LeRobot fashion): obs normalize, action unscale.

Note: ``ProcessorStep`` is not a ``torch.nn.Module`` — no ``register_buffer``.
Bounds are stored as plain attributes with a small per-(device, dtype) cache
so ``unnormalize`` avoids a per-call ``.to()``.
"""

from __future__ import annotations

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.networks import RunningMeanStd


@ProcessorStepRegistry.register(name="sjfpo_observation_normalizer")
class SJFPOObservationProcessorStep(ProcessorStep):
    """Normalize ``observation.state`` with the shared live ``RunningMeanStd``.

    Holds a **reference** to the policy's normalizer, not a copy, so
    ``update_stats`` folds into the same buffers the policy reads at forward
    time.
    """

    def __init__(self, normalizer: RunningMeanStd):
        """Bind to the policy's live normalizer.

        Args:
            normalizer: Shared running-statistics module the policy reads at
                forward time.
        """
        self.normalizer = normalizer

    @torch.no_grad()
    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize a raw observation window.

        Args:
            states: Raw observation window.

        Returns:
            The normalized window; an unchanged identity until the first stats
            update.
        """
        return self.normalizer(states)

    def update_stats(self, states: torch.Tensor) -> None:
        """Fold raw training frames into the running stats.

        Args:
            states: Raw observation window from a training batch.
        """
        self.normalizer.update(states)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[OBS_STATE] = self.normalize(batch[OBS_STATE])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through unchanged.

        Args:
            features: Incoming pipeline feature spec.

        Returns:
            The same feature spec; this step does not change the schema.
        """
        return features


@ProcessorStepRegistry.register(name="sjfpo_action_unnormalizer")
class SJFPOActionPostprocessorStep(ProcessorStep):
    """Unscale policy action chunks from [-1, 1] to real env bounds.

    Bounds are updated atomically by ``set_action_bounds()``. A tiny
    per-``(device, dtype)`` cache keeps the pre-computed affine coefficients
    ``_scale`` / ``_bias`` local to whatever device the incoming chunk is on,
    so ``unnormalize`` is a single FMA with no per-call ``.to()``. Updating
    bounds clears both caches.
    """

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        self._scale_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._bias_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.set_action_bounds(action_low, action_high)

    def set_action_bounds(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        """Replace the action bounds and invalidate cached coefficients.

        Args:
            action_low: Per-dimension lower bounds of the env action space.
            action_high: Per-dimension upper bounds of the env action space.

        Raises:
            ValueError: If the bounds disagree in shape, or if any high bound
                is below its low bound.
        """
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
        self._scale_cache.clear()
        self._bias_cache.clear()

    def _coefs_for(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up the cached affine coefficients for a reference tensor.

        Args:
            ref: Tensor whose device and dtype the coefficients must match.

        Returns:
            The cached ``(scale, bias)`` pair, populating the cache on first
            use for that device and dtype.
        """
        key = (str(ref.device), ref.dtype)
        scale = self._scale_cache.get(key)
        if scale is None:
            scale = self._scale_cpu.to(device=ref.device, dtype=ref.dtype)
            bias = self._bias_cpu.to(device=ref.device, dtype=ref.dtype)
            self._scale_cache[key] = scale
            self._bias_cache[key] = bias
        return scale, self._bias_cache[key]

    def to(self, device: torch.device | str) -> "SJFPOActionPostprocessorStep":
        """Optional eager move: pre-populate the cache for ``device``.

        Not required (``unnormalize`` will populate it on first call), but
        useful when you want the transfer to happen outside the hot path.

        Args:
            device: Device the processor is moved to.

        Returns:
            This processor, with its bounds and cache primed for ``device``.
        """
        device = torch.device(device)
        self._scale_cpu.to(device=device)
        self._bias_cpu.to(device=device)
        key = (str(device), torch.float32)
        self._scale_cache.setdefault(
            key, self._scale_cpu.to(device=device, dtype=torch.float32)
        )
        self._bias_cache.setdefault(
            key, self._bias_cpu.to(device=device, dtype=torch.float32)
        )
        self.action_low = self.action_low.to(device=device)
        self.action_high = self.action_high.to(device=device)
        return self

    @torch.inference_mode()
    def unnormalize(self, scaled: torch.Tensor) -> torch.Tensor:
        """Unscale an action chunk into real env bounds.

        Args:
            scaled: Action chunk in the policy's ``[-1, 1]`` space.

        Returns:
            The chunk in real env action units; an identity transform when the
            bounds are ±1.
        """
        scale, bias = self._coefs_for(scaled)
        return torch.addcmul(bias, scale, scaled)

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[ACTION] = self.unnormalize(batch[ACTION])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass the pipeline features through unchanged.

        Args:
            features: Incoming pipeline feature spec.

        Returns:
            The same feature spec; this step does not change the schema.
        """
        return features


def make_sjfpo_pre_post_processors(
    policy,
) -> tuple[SJFPOObservationProcessorStep, SJFPOActionPostprocessorStep]:
    """Build the pre/post pair; policy setter synchronizes action bounds.

    The observation step reads and updates the policy's ``obs_normalizer``
    buffers in place. The action step owns its validated affine cache, and
    ``SJFPOPolicy.set_action_bounds()`` updates it atomically.

    Args:
        policy: Policy supplying the normalizer and current action bounds.

    Returns:
        The pre-processor and post-processor pair.
    """
    return (
        SJFPOObservationProcessorStep(policy.obs_normalizer),
        SJFPOActionPostprocessorStep(policy.action_low, policy.action_high),
    )
