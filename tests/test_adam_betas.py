"""``adam_betas`` has to reach the optimizer, not just validate.

Adam's two moment decay rates are the plasticity knob (Dohare et al.,
arXiv:2306.13812): he reports Adam exacerbating loss of plasticity, and policy
collapse being mitigated by using the same value for both moments. The default is
therefore the symmetric ``(0.999, 0.999)``, overriding the asymmetric
``(0.9, 0.999)`` that ``AdamWConfig`` hands over. Without the override line in
``build_optimizer`` the preset wins and the run trains with the very betas this
field exists to avoid, with nothing in the logs to say so.
"""

import dataclasses

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy, build_optimizer

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
    """The field exists to depart from ``AdamWConfig``, so this is a real override.

    ``build_optimizer`` used to pass the preset through verbatim, which made the
    asymmetric ``(0.9, 0.999)`` pair unavoidable. The default is now the
    symmetric one, so if the override line is dropped the preset wins and the
    run silently trains with the asymmetric betas this field exists to avoid.
    """
    preset = _config().get_optimizer_preset()
    assert preset.betas == (0.9, 0.999)
    assert _config().adam_betas == (0.999, 0.999)
    assert all(
        group["betas"] == (0.999, 0.999)
        for group in build_optimizer(SJFPOPolicy(_config())).param_groups
    )


def test_any_pair_reaches_every_param_group():
    """All four groups, or the critic keeps a different decay than the actor."""
    cfg = _config(adam_betas=(0.95, 0.95))
    groups = build_optimizer(SJFPOPolicy(cfg)).param_groups
    assert len(groups) == 4
    assert all(group["betas"] == (0.95, 0.95) for group in groups)


@pytest.mark.parametrize("betas", [(0.0, 0.999), (0.999, 1.0), (1.5, 0.9)])
def test_a_beta_outside_the_open_unit_interval_raises(betas):
    """0.0 and 1.0 both make the running average degenerate, silently."""
    with pytest.raises(ValueError, match="adam_betas"):
        _config(adam_betas=betas).validate_optimizer()