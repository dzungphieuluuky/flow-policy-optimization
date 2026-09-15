"""Tests for src/sjfpo/processor_sjfpo.py (LeRobot-fashion pre/post pair)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.processor.pipeline import ProcessorStepRegistry  # noqa: E402

from sjfpo.processor_sjfpo import (  # noqa: E402
    SJFPOActionPostprocessorStep,
    SJFPOObservationProcessorStep,
    make_sjfpo_pre_post_processors,
)


def test_registry_names():
    assert ProcessorStepRegistry.get("sjfpo_observation_normalizer") is (
        SJFPOObservationProcessorStep
    )
    assert ProcessorStepRegistry.get("sjfpo_action_unnormalizer") is (
        SJFPOActionPostprocessorStep
    )


def test_action_unnormalize_math():
    post = SJFPOActionPostprocessorStep(
        torch.full((2,), -3.0), torch.full((2,), 3.0)
    )
    out = post.unnormalize(torch.tensor([[-1.0, 0.0], [0.5, 1.0]]))
    assert torch.allclose(out, torch.tensor([[-3.0, 0.0], [1.5, 3.0]]))
    batch = post({"action": torch.zeros(2, 4, 2)})
    assert torch.equal(batch["action"], torch.zeros(2, 4, 2))  # ±3 midpoint is 0


def test_obs_normalize_identity_then_stats():
    from sjfpo.networks import RunningMeanStd

    normalizer = RunningMeanStd(3)
    pre = SJFPOObservationProcessorStep(normalizer)
    x = torch.randn(2, 5, 3)
    assert torch.equal(pre.normalize(x), x)  # count == 0 -> identity
    pre.update_stats(torch.ones(4, 3))
    assert not torch.equal(pre.normalize(torch.ones(2, 3)), torch.ones(2, 3))
    batch = pre({"observation.state": x})
    assert batch["observation.state"].shape == x.shape


def test_policy_pair_shares_live_state():
    import dataclasses

    from lerobot.configs.types import FeatureType, PolicyFeature

    from sjfpo.configuration_sjfpo import SJFPOConfig
    from sjfpo.modeling_sjfpo import SJFPOPolicy

    cfg = SJFPOConfig(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    cfg = dataclasses.replace(cfg, num_mc_samples=2, dit_depth=2)
    policy = SJFPOPolicy(cfg)
    pre, post = make_sjfpo_pre_post_processors(policy)
    assert pre.normalizer is policy.obs_normalizer
    assert post.action_low is policy.action_low  # bounds stay visible to the step
    assert policy.preprocessor.normalizer is policy.obs_normalizer
    # Custom bounds flow through predict without touching the policy.
    policy.set_action_bounds(
        torch.full((2,), -2.0), torch.full((2,), 2.0)
    )
    chunk = policy.predict_action_chunk(
        {"observation.state": torch.randn(2, 16, 4)}
    )
    assert chunk.shape == (2, 16, 2)
    assert (chunk <= 2.0).all() and (chunk >= -2.0).all()
