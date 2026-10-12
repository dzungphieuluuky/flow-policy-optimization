"""``ratio_granularity`` must select the trust region the papers name.

The three values were once two, and the surviving names did not match what they
computed: ``per_action`` exponentiated each (tau, eps) pair separately, which is
FPO++ Eq 10, so a file claiming to reproduce FPO's Eq 9 was running FPO++'s
first contribution. Nothing in the loss said so -- all three granularities agree
exactly when every ratio is 1, which is every on-policy step.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PolicyFeature

import spectral_joint_policy.modeling_sjfpo as M
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy
from spectral_joint_policy.train_sjfpo import build_policy

OBS, ACT, N, MC, HORIZON = 11, 2, 6, 4, 16
ROOT = Path(__file__).resolve().parents[1]
LOW = np.array([-2.0, -2.0], dtype=np.float32)
HIGH = np.array([2.0, 2.0], dtype=np.float32)


def _cfg(**overrides):
    return dataclasses.replace(
        SJFPOConfig(
            num_mc_samples=MC,
            action_horizon=HORIZON,
            value_loss_coef=0.0,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
        ),
        **overrides,
    )


def _policy_and_batch(cfg, seed=0):
    """A batch like ``generate_rollout``'s, with the behaviour losses filled in.

    ``out`` is zero-initialised, which would make ``old_actor`` and ``actor``
    agree and collapse every ratio to 1, so it is perturbed first.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    policy = build_policy(OBS, ACT, cfg, device="cpu", action_low=LOW, action_high=HIGH)
    with torch.no_grad():
        policy.actor.out.weight.add_(0.1 * torch.randn_like(policy.actor.out.weight))

    batch = {
        M.OBS_STATE: torch.from_numpy(
            rng.standard_normal((N, cfg.n_obs_steps, OBS), dtype=np.float32)
        ),
        M.ACTION: torch.from_numpy(
            rng.standard_normal((N, HORIZON, ACT), dtype=np.float32)
        ),
        "action_is_pad": torch.zeros(N, HORIZON, dtype=torch.bool),
        "normalised_advantages": torch.from_numpy(
            rng.standard_normal(N, dtype=np.float32)
        ),
        "raw_returns": torch.from_numpy(
            (rng.standard_normal(N, dtype=np.float32) * 10.0)
        ),
        "mc_tau": torch.from_numpy(
            rng.uniform(0.05, 0.95, size=(N, MC, cfg.action_horizon, 1)).astype(
                np.float32
            )
        ),
        "mc_eps": torch.from_numpy(
            rng.standard_normal(
                (N, MC, cfg.action_horizon, ACT), dtype=np.float32
            )
        ),
    }
    normalised = policy.preprocessor({M.OBS_STATE: batch[M.OBS_STATE]})[M.OBS_STATE]
    l_old, l_old_token = policy.compute_behavior_cfm_losses(
        normalised, batch[M.ACTION], batch["mc_tau"], batch["mc_eps"],
        batch["action_is_pad"],
    )
    batch["behavior_cfm_loss"] = l_old
    batch["behavior_cfm_loss_token"] = l_old_token
    return policy, batch


def _recorded(cfg):
    """What the surrogate was handed, and what each ratio was built from."""
    policy, batch = _policy_and_batch(cfg)
    ratios, surrogate_inputs = [], []
    real_ratio, real_surrogate = M.compute_policy_ratio, M.compute_aspo_surrogate

    def ratio_spy(l_old, l_new, *args, **kwargs):
        ratios.append(l_old.detach().clone())
        return real_ratio(l_old, l_new, *args, **kwargs)

    def surrogate_spy(ratio, advantage, *args, **kwargs):
        surrogate_inputs.append(tuple(ratio.shape))
        return real_surrogate(ratio, advantage, *args, **kwargs)

    M.compute_policy_ratio, M.compute_aspo_surrogate = ratio_spy, surrogate_spy
    try:
        # compute_info=True: the metric pass reads the same delta, so a branch
        # that forgets to bind it only fails once the tracker is on.
        policy(batch, compute_info=True, check_finite=False)
    finally:
        M.compute_policy_ratio = real_ratio
        M.compute_aspo_surrogate = real_surrogate
    return ratios, surrogate_inputs, batch


def test_per_chunk_averages_the_mc_losses_before_exponentiating():
    """FPO Eq 9: exp(mean_i l_i - mean_i l'_i), so one ratio per chunk."""
    ratios, surrogate_inputs, batch = _recorded(_cfg(ratio_granularity="per_chunk"))
    assert surrogate_inputs == [(N,)]
    averaged = [call for call in ratios if call.dim() == 1]
    assert len(averaged) == 1
    assert torch.allclose(averaged[0], batch["behavior_cfm_loss"].mean(dim=1))


def test_per_action_keeps_one_ratio_per_mc_pair():
    """FPO++ Eq 10: each (tau, eps) pair clips on its own."""
    ratios, surrogate_inputs, batch = _recorded(_cfg(ratio_granularity="per_action"))
    assert surrogate_inputs == [(N, MC)]
    per_pair = [call for call in ratios if call.dim() == 2]
    assert len(per_pair) == 1
    assert torch.allclose(per_pair[0], batch["behavior_cfm_loss"])


def test_per_sample_clips_per_token():
    """Finer than either paper: one ratio per (chunk, pair, token)."""
    _, surrogate_inputs, _ = _recorded(_cfg(ratio_granularity="per_sample"))
    assert surrogate_inputs == [(N, MC, HORIZON)]


def test_per_chunk_and_per_action_disagree_once_the_actor_moves():
    """The regression: identical on-policy, different the moment l_old != l_new."""
    losses = []
    for granularity in ("per_chunk", "per_action"):
        policy, batch = _policy_and_batch(_cfg(ratio_granularity=granularity))
        per_chunk_loss, _ = policy(
            batch, reduction="none", compute_info=False, check_finite=False
        )
        losses.append(per_chunk_loss)
    assert not torch.allclose(losses[0], losses[1])


def test_an_unknown_granularity_is_rejected():
    with pytest.raises(ValueError, match="ratio_granularity"):
        _cfg(ratio_granularity="per_action_chunk").validate_loss_params()


def test_the_baselines_select_the_trust_regions_they_name():
    """A baseline that names an equation must select the ratio implementing it.

    Checked through baselines.apply_baseline, which is now the only source of
    baseline config. The configs/baselines/*.yaml files are gone; they disagreed
    with this registry (max_grad_norm 48.0 vs 1.0), so keeping them would have
    left two ways to run "the same" baseline with different gradient clipping.
    """
    from spectral_joint_policy.baselines import apply_baseline

    expected = {
        "fpo": ("ppo", "per_chunk"),
        "fpo++": ("aspo", "per_action"),
    }
    for name, (mode, granularity) in expected.items():
        cfg, _ = apply_baseline(SJFPOConfig(), name)
        assert (cfg.trust_region_mode, cfg.ratio_granularity) == (mode, granularity), name
        # Eq 8's loss is a sum over the chunk, so the paper's eps_clip only means
        # what it says with a sum reduction.
        assert cfg.cfm_loss_reduction == "sum", name


def test_a_baseline_cannot_inherit_the_default_it_is_been_compared_against():
    """The reason fpo++ got explicit overrides.

    Every baseline pins its identifying fields, so moving a default cannot
    silently redefine what the baseline is.
    """
    from spectral_joint_policy.baselines import BASELINES, _OVERRIDES

    for name in BASELINES:
        assert _OVERRIDES[name], f"{name} inherits the default config verbatim"
