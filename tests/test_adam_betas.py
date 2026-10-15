"""``adam_betas`` has to reach the optimizer, not just validate."""

import dataclasses

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.modeling_sjfpo import SJFPOPolicy, build_optimizer
from myfpo.train_sjfpo import build_policy

OBS, ACT = 11, 2


def _config(**overrides):
    return dataclasses.replace(
        SJFPOConfig(
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **overrides,
    )


def test_the_default_overrides_the_preset():
    """The field exists to depart from ``AdamWConfig``, so this is a real override."""
    preset = _config().get_optimizer_preset()
    assert preset.betas == (0.9, 0.999)
    assert _config().adam_betas == (0.9, 0.99)
    assert all(
        group["betas"] == (0.9, 0.99)
        for group in build_optimizer(SJFPOPolicy(_config())).param_groups
    )


def test_any_pair_reaches_every_param_group():
    """All four groups, or the critic keeps a different decay than the actor."""
    cfg = _config(adam_betas=(0.95, 0.95))
    groups = build_optimizer(SJFPOPolicy(cfg)).param_groups
    assert len(groups) == 4
    assert all(group["betas"] == (0.95, 0.95) for group in groups)


def test_the_actor_output_layer_skips_weight_decay_when_compiled():
    """``torch.compile`` renames params, so the ``out.`` test must still match."""
    policy = build_policy(
        OBS, ACT, _config(compile=True), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    assert hasattr(policy.actor, "_orig_mod"), "the actor must be compiled"
    no_decay = {id(p) for p in policy.get_actor_optim_params()["params"][1]["params"]}
    out = policy.actor._orig_mod.out
    assert id(out.weight) in no_decay
    assert id(out.bias) in no_decay


@pytest.mark.parametrize("betas", [(0.0, 0.999), (0.999, 1.0), (1.5, 0.9)])
def test_a_beta_outside_the_open_unit_interval_raises(betas):
    """0.0 and 1.0 both make the running average degenerate, silently."""
    with pytest.raises(ValueError, match="adam_betas"):
        _config(adam_betas=betas).validate_optimizer()