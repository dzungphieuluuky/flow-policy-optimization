"""``ratio_granularity`` must select the trust region the papers name."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PolicyFeature

import myfpo.modeling_sjfpo as M
from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.train_sjfpo import build_policy

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
    """A batch like ``generate_rollout``'s, with the behaviour losses filled in."""
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
    cfm_loss_old, cfm_loss_old_token = policy.compute_behavior_cfm_losses(
        normalised, batch[M.ACTION], batch["mc_tau"], batch["mc_eps"],
        batch["action_is_pad"],
    )
    batch["behavior_cfm_loss"] = cfm_loss_old
    batch["behavior_cfm_loss_token"] = cfm_loss_old_token
    return policy, batch


def _recorded(cfg):
    """What the surrogate was handed, and what each ratio was built from."""
    policy, batch = _policy_and_batch(cfg)
    ratios, surrogate_inputs = [], []
    real_ratio, real_surrogate = M.compute_policy_ratio, M.compute_aspo_surrogate

    def ratio_spy(cfm_loss_old, cfm_loss_new, *args, **kwargs):
        ratios.append(cfm_loss_old.detach().clone())
        return real_ratio(cfm_loss_old, cfm_loss_new, *args, **kwargs)

    def surrogate_spy(ratio, advantage, *args, **kwargs):
        surrogate_inputs.append(tuple(ratio.shape))
        return real_surrogate(ratio, advantage, *args, **kwargs)

    M.compute_policy_ratio, M.compute_aspo_surrogate = ratio_spy, surrogate_spy
    try:
        # compute_info=True: the metric pass reads the same delta, so a branch.
        policy(batch, compute_info=True, check_finite=False)
    finally:
        M.compute_policy_ratio = real_ratio
        M.compute_aspo_surrogate = real_surrogate
    return ratios, surrogate_inputs, batch


def per_chunk(**kw):
    """FPO Eq 9: tokens averaged, then all MC pairs in one group."""
    return _cfg(do_average_cfm_loss_in_chunk=True, cfm_average_group_size=-1, **kw)


def per_action(**kw):
    """FPO++ Eq 10: tokens averaged, one ratio per (tau, eps) pair."""
    return _cfg(do_average_cfm_loss_in_chunk=True, cfm_average_group_size=1, **kw)


def per_sample(**kw):
    """Finer than either paper: one ratio per (chunk, pair, token)."""
    return _cfg(do_average_cfm_loss_in_chunk=False, cfm_average_group_size=1, **kw)


def test_per_chunk_averages_the_mc_losses_before_exponentiating():
    """FPO Eq 9: exp(mean_i l_i - mean_i l'_i), so one ratio per chunk."""
    ratios, surrogate_inputs, batch = _recorded(per_chunk())
    assert surrogate_inputs == [(N, 1)]
    averaged = [call for call in ratios if call.dim() == 2 and call.shape[1] == 1]
    assert len(averaged) == 1
    assert torch.allclose(
        averaged[0][:, 0], batch["behavior_cfm_loss"].mean(dim=1), atol=1e-6
    )


def test_per_action_keeps_one_ratio_per_mc_pair():
    """FPO++ Eq 10: each (tau, eps) pair clips on its own."""
    ratios, surrogate_inputs, batch = _recorded(per_action())
    assert surrogate_inputs == [(N, MC)]
    per_pair = [call for call in ratios if call.dim() == 2]
    assert len(per_pair) == 1
    assert torch.allclose(per_pair[0], batch["behavior_cfm_loss"])


def test_per_sample_clips_per_token():
    """Finer than either paper: one ratio per (chunk, pair, token)."""
    _, surrogate_inputs, _ = _recorded(per_sample())
    assert surrogate_inputs == [(N, MC, HORIZON)]


def _padded_per_sample_batch(pad_from=HORIZON // 2):
    """A ``per_sample`` batch whose second row is padded from ``pad_from`` on."""
    cfg = per_sample()
    policy, batch = _policy_and_batch(cfg)
    batch["action_is_pad"][1, pad_from:] = True
    normalised = policy.preprocessor({M.OBS_STATE: batch[M.OBS_STATE]})[M.OBS_STATE]
    old, old_token = policy.compute_behavior_cfm_losses(
        normalised, batch[M.ACTION], batch["mc_tau"], batch["mc_eps"],
        batch["action_is_pad"],
    )
    batch["behavior_cfm_loss"] = old
    batch["behavior_cfm_loss_token"] = old_token
    return policy, batch


def _per_sample_surrogate(policy, batch):
    """The per-token surrogate ``forward`` built, captured at the call."""
    captured = []
    real = M.compute_aspo_surrogate

    def spy(ratio, advantage, *args, **kwargs):
        out = real(ratio, advantage, *args, **kwargs)
        captured.append(out.detach().clone())
        return out

    M.compute_aspo_surrogate = spy
    try:
        losses, _ = policy(batch, reduction="none", compute_info=False, check_finite=False)
    finally:
        M.compute_aspo_surrogate = real
    return losses, captured[-1]


def test_per_sample_is_unchanged_when_every_token_ran():
    """No padding means no mask and no rescale: the old mean was already right."""
    policy, batch = _padded_per_sample_batch(pad_from=HORIZON)
    losses, surrogate = _per_sample_surrogate(policy, batch)
    want = -surrogate.mean(dim=(1, 2))[0]
    assert torch.allclose(losses[0], want, atol=0.0, rtol=1e-6), (
        f"{losses[0].item()} != {want.item()}: a full chunk must reduce over every token"
    )


def test_per_sample_masks_padded_tokens_before_the_mean():
    """A short chunk reduces over its executed tokens only."""
    pad_from = HORIZON // 2
    policy, batch = _padded_per_sample_batch(pad_from=pad_from)
    losses, surrogate = _per_sample_surrogate(policy, batch)

    want = -surrogate[1, :, :pad_from].mean()
    assert torch.allclose(losses[1], want, atol=0.0, rtol=1e-6), (
        f"{losses[1].item()} != {want.item()}"
    )
    unmasked = -surrogate[1].mean()
    assert not torch.allclose(losses[1], unmasked), "padded tokens are still in the mean"


def test_padded_tokens_evaluate_to_the_advantage_not_to_zero():
    """Why the value needs masking and not just the divisor."""
    pad_from = HORIZON // 2
    policy, batch = _padded_per_sample_batch(pad_from=pad_from)
    _, surrogate = _per_sample_surrogate(policy, batch)
    adv = batch["normalised_advantages"].float()[1]
    padded = surrogate[1, :, pad_from:]
    assert torch.equal(padded, adv.expand_as(padded)), (
        "padded tokens are no longer exactly the advantage; if this changed, "
        "re-check whether they need masking at all"
    )


def test_per_chunk_and_per_action_disagree_once_the_actor_moves():
    """The regression: identical on-policy, different the moment cfm_loss_old != cfm_loss_new."""
    losses = []
    for make_cfg in (per_chunk, per_action):
        policy, batch = _policy_and_batch(make_cfg())
        per_chunk_loss, _ = policy(
            batch, reduction="none", compute_info=False, check_finite=False
        )
        losses.append(per_chunk_loss)
    assert not torch.allclose(losses[0], losses[1])


def test_an_unknown_granularity_is_rejected():
    with pytest.raises(ValueError, match="cfm_average_group_size"):
        _cfg(cfm_average_group_size=0).validate()
    with pytest.raises(ValueError, match="divide evenly"):
        _cfg(num_mc_samples=6, cfm_average_group_size=4).validate()


def test_the_baselines_select_the_trust_regions_they_name():
    """A baseline that names an equation must select the ratio implementing it."""
    from myfpo.baselines import apply_baseline

    # (trust region, token axis folded, MC group size).
    expected = {
        "fpo": ("ppo", True, -1),
        "fpo++": ("aspo", True, 1),
    }
    for name, (mode, token_avg, group) in expected.items():
        cfg, _ = apply_baseline(SJFPOConfig(), name)
        assert cfg.trust_region_mode == mode, name
        assert cfg.do_average_cfm_loss_in_chunk is token_avg, name
        assert cfg.cfm_average_group_size == group, name
        # FPO's Eq 8 is a sum over the chunk. FPO++'s manipulation variant.
        assert cfg.cfm_loss_reduction == ("sum" if name == "fpo" else "mean"), name


def test_a_baseline_cannot_inherit_the_default_it_is_been_compared_against():
    """The reason fpo++ got explicit overrides."""
    from myfpo.baselines import BASELINES, _OVERRIDES

    for name in BASELINES:
        assert _OVERRIDES[name], f"{name} inherits the default config verbatim"


def test_every_baseline_note_belongs_to_an_override_it_can_be_printed_for():
    """A provenance note keyed to nothing is documentation no reader sees."""
    from myfpo.baselines import _NOTES, _OVERRIDES

    for name, notes in _NOTES.items():
        orphans = sorted(set(notes) - set(_OVERRIDES[name]))
        assert not orphans, f"{name}: notes no report will print: {orphans}"
