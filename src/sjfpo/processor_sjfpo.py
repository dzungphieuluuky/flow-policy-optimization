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
        # ProcessorStep is not nn.Module; do not call super().__init__().
        self.normalizer = normalizer

    @torch.no_grad()
    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Raw window -> normalized window (identity until the first update)."""
        return self.normalizer(states)

    def update_stats(self, states: torch.Tensor) -> None:
        """Fold raw training frames into the running stats (train batches only)."""
        self.normalizer.update(states)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[OBS_STATE] = self.normalize(batch[OBS_STATE])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="sjfpo_action_unnormalizer")
class SJFPOActionPostprocessorStep(ProcessorStep):
    """Unscale policy action chunks from [-1, 1] to real env bounds.

    Bounds are stored as plain tensors (``ProcessorStep`` is not a
    ``torch.nn.Module``, so ``register_buffer`` is unavailable). A tiny
    per-``(device, dtype)`` cache keeps the pre-computed affine coefficients
    ``_scale`` / ``_bias`` local to whatever device the incoming chunk is on,
    so ``unnormalize`` is a single FMA with no per-call ``.to()``.
    """

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
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

        # Affine form: y = _scale * s + _bias
        #   = 0.5 (high - low) * s + 0.5 (high + low)
        scale_cpu = 0.5 * (high - low)
        bias_cpu = 0.5 * (high + low)

        # Cache keyed by (device string, dtype); populated lazily on first use
        # per target. Since there is typically one device per run, the cache
        # has exactly one entry in steady state.
        self._scale_cpu = scale_cpu
        self._bias_cpu = bias_cpu
        self._scale_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._bias_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def _coefs_for(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        """
        device = torch.device(device)
        self._scale_cpu.to(device=device)
        self._bias_cpu.to(device=device)
        # Prime the cache for the default float dtype (unchanged if already primed).
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

    @torch.no_grad()
    def unnormalize(self, scaled: torch.Tensor) -> torch.Tensor:
        """Scaled chunk -> real env actions (identity when bounds are ±1)."""
        scale, bias = self._coefs_for(scaled)
        return torch.addcmul(bias, scale, scaled)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out[ACTION] = self.unnormalize(batch[ACTION])
        return out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_sjfpo_pre_post_processors(
    policy,
) -> tuple[SJFPOObservationProcessorStep, SJFPOActionPostprocessorStep]:
    """Build the (preprocessor, postprocessor) pair sharing the policy's live state.

    Both steps hold references, not copies: the observation step reads and
    updates the policy's ``obs_normalizer`` buffers in place, and the action
    step snapshots the policy's current action bounds as plain tensors.
    """
    return (
        SJFPOObservationProcessorStep(policy.obs_normalizer),
        SJFPOActionPostprocessorStep(policy.action_low, policy.action_high),
    )
