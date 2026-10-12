"""Regression tests for the metric audit.

Before the fix these asserted the DEFECTIVE behaviour. They now assert the
corrected behaviour, so they fail if a defect is reintroduced.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature

from spectral_joint_policy.collector import resolve_action_bounds, scale_and_clip_chunk
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.evaluator import (
    compute_action_jerk,
    compute_mean_chunk_jerk,
    compute_seam_jerk,
)
from spectral_joint_policy.modeling_sjfpo import (
    compute_ratio_metrics,
    compute_value_metrics,
)
from spectral_joint_policy.tracking.base import compute_eval_summary_metrics
from spectral_joint_policy.train_sjfpo import compute_rollout_diagnostics


def _cfg(**kw):
    base = dict(
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (17,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        report_to="none", compile=False,
    )
    base.update(kw)
    return SJFPOConfig(**base)


# --- A. saturation is measured by clip count, not by post-clip magnitude ---
def test_stored_actions_never_exceed_unit_abs_so_a_magnitude_guard_is_useless():
    """Why the >1.05 guard was dead: clipping happens before storage."""
    lo, rng = resolve_action_bounds([-2.0, -1.0, -0.3], [2.0, 1.0, 0.3])
    _, scaled, n_clip = scale_and_clip_chunk(torch.tensor([[9.0, -9.0, 5.0]]), lo, rng)
    assert int(n_clip) == 3, "the proposal WAS out of bounds"
    assert scaled.abs().max().item() == pytest.approx(1.0)
    _, scaled2, _ = scale_and_clip_chunk(torch.tensor([[1e9, -1e9, 1e9]]), lo, rng)
    assert scaled2.abs().max().item() == pytest.approx(1.0)


def test_action_clip_rate_reports_real_saturation():
    from spectral_joint_policy.train_sjfpo import compute_action_clip_rate

    batch = {
        "action": torch.zeros(4, 8, 3),
        "action_is_pad": torch.zeros(4, 8, dtype=torch.bool),
        "chunk_clip_counts": torch.tensor([0, 12, 6, 0]),
    }
    out = compute_action_clip_rate(batch)
    assert out["action_clip_count"] == pytest.approx(18.0)
    # 4 chunks * 8 steps * 3 dims = 96 proposed coordinates.
    assert out["action_clip_frac"] == pytest.approx(18 / 96)


def test_action_clip_rate_is_absent_when_no_clip_counts_were_collected():
    from spectral_joint_policy.train_sjfpo import compute_action_clip_rate

    assert compute_action_clip_rate(
        {"action": torch.zeros(1, 2, 2), "action_is_pad": torch.zeros(1, 2, dtype=torch.bool)}
    ) == {}


# --- B. action_magnitude excludes unexecuted rows ----------------------
def test_rollout_diagnostics_reports_executed_rows_only():
    """action_magnitude must not be diluted by rows the robot never ran."""
    from spectral_joint_policy.train_sjfpo import compute_rollout_diagnostics

    a = torch.zeros(2, 8, 3)
    a[0, :3] = 1.0
    a[1, :] = 1.0
    pad = torch.zeros(2, 8, dtype=torch.bool)
    pad[0, 3:] = True
    out = compute_rollout_diagnostics({"action": a, "action_is_pad": pad})
    # The legacy key keeps its all-rows meaning; the executed-only figure,
    # which is what the robot actually did, is the new one.
    assert out["action_magnitude"] == pytest.approx(0.6875)
    assert out["action_magnitude_executed"] == pytest.approx(1.0)
    assert out["action_abs_max"] == pytest.approx(1.0)
    assert out["action_abs_max_executed"] == pytest.approx(1.0)
    assert out["step_delta"] == 11.0


def test_executed_only_metrics_differ_when_padding_carries_real_values():
    """Padding holds real proposed actions, not zeros, so the split matters."""
    a = torch.full((2, 8, 3), 0.5)
    a[0, :3] = 1.0
    a[1, :] = 1.0
    pad = torch.zeros(2, 8, dtype=torch.bool)
    pad[0, 3:] = True
    out = compute_rollout_diagnostics({"action": a, "action_is_pad": pad})
    assert out["action_magnitude_executed"] == pytest.approx(1.0)
    assert out["action_magnitude"] < 1.0


# --- C. action_jerk is not scale-free -----------------------------------


# --- C. action_jerk is not scale-free -----------------------------------
def test_action_jerk_scales_with_amplitude_squared():
    """It is NOT scale-free; the docstring now says so. Compare within a run."""
    rng = np.random.default_rng(0)
    t = np.arange(60, dtype=np.float32)
    sig = np.stack([t + rng.normal(0, 0.3, 60).astype(np.float32) for _ in range(2)], 1)
    base = compute_action_jerk(sig)
    assert compute_action_jerk(sig * 10.0) == pytest.approx(base * 100.0, rel=1e-4)
    assert compute_action_jerk(sig * 0.01) == pytest.approx(base * 1e-4, rel=1e-3)


def test_action_jerk_is_zero_for_constant_and_constant_velocity():
    t = np.arange(60, dtype=np.float32)
    assert compute_action_jerk(np.ones((60, 2), np.float32)) == 0.0
    assert compute_action_jerk(np.stack([t, 2 * t], 1)) == 0.0


# --- D. chunk jerk is blind to inter-chunk seams ------------------------
def test_chunk_jerk_is_still_chunk_internal_and_does_not_claim_otherwise():
    """The two jerk metrics are complementary, not duplicates."""
    H, D = 8, 2
    chunk = np.tile(np.linspace(0, 1, H, dtype=np.float32)[:, None], (1, D))
    shifted = chunk - 5.0
    acts = np.concatenate([chunk[None], shifted[None]], axis=0)
    pad = np.zeros((2, H), dtype=bool)
    assert compute_mean_chunk_jerk(acts, pad) == pytest.approx(0.0)
    stitched = np.concatenate([chunk, shifted[0:1]], axis=0)
    assert compute_action_jerk(stitched) > 1.0


def test_chunk_jerk_does_see_a_jump_inside_one_chunk():
    H, D = 8, 2
    chunk = np.tile(np.linspace(0, 1, H, dtype=np.float32)[:, None], (1, D))
    jumpy = chunk.copy()
    jumpy[H // 2:] -= 5.0
    acts = np.concatenate([chunk[None], jumpy[None]], axis=0)
    pad = np.zeros((2, H), dtype=bool)
    assert compute_mean_chunk_jerk(acts, pad) > 1.0


def test_seam_jerk_catches_the_discontinuity_chunk_jerk_misses():
    H, D = 8, 2
    chunk = np.tile(np.linspace(0, 1, H, dtype=np.float32)[:, None], (1, D))
    shifted = chunk - 5.0
    acts = np.concatenate([chunk[None], shifted[None]], axis=0)
    pad = np.zeros((2, H), dtype=bool)
    assert compute_mean_chunk_jerk(acts, pad) == pytest.approx(0.0)
    seam = compute_seam_jerk(acts, pad)
    assert seam is not None and seam > 1.0, "the whole point of the new metric"


def test_seam_jerk_is_zero_for_a_continuous_rollout():
    H, D = 8, 2
    t = np.linspace(0, 1, 2 * H, dtype=np.float32)
    acts = np.stack([t[:H], t[H:]], axis=0)[:, :, None].repeat(D, axis=2)
    pad = np.zeros((2, H), dtype=bool)
    assert compute_seam_jerk(acts, pad) == pytest.approx(0.0)


def test_seam_jerk_returns_none_when_nothing_qualifies():
    chunk = np.zeros((1, 8, 2), dtype=np.float32)
    assert compute_seam_jerk(chunk, np.zeros((1, 8), dtype=bool)) is None
    # A seam whose earlier chunk has a single executed step cannot form a
    # second difference, so it must not be counted as a clean seam.
    two = np.zeros((2, 8, 2), dtype=np.float32)
    pad = np.zeros((2, 8), dtype=bool)
    pad[0, 1:] = True
    assert compute_seam_jerk(two, pad) is None


def test_seam_jerk_skips_seams_whose_earlier_chunk_is_truncated():
    H, D = 4, 2
    a = np.zeros((2, H, D), dtype=np.float32)
    a[0, 0] = 10.0            # only one executed step before the seam
    a[1, 0] = 20.0
    pad = np.zeros((2, H), dtype=bool)
    pad[0, 1:] = True
    assert compute_seam_jerk(a, pad) is None


# --- E. explained_var vs the min the backup actually uses --------------
def test_explained_var_min_is_logged_and_is_never_optimistic():
    torch.manual_seed(0)
    C, n = 2, 64
    v = torch.randn(C, n) * 3.0 + 1.5
    R = torch.randn(n) * 5.0
    m = compute_value_metrics(
        raw_value_pred=v, raw_returns_flat=R,
        policy_per_chunk=torch.zeros(n), value_loss_per_chunk=torch.zeros(n),
    )
    ev_mean = m["explained_var"]
    ev_min = m["explained_var_min"]
    assert ev_mean == pytest.approx(
        1.0 - (R - v.mean(0)).var(correction=0) / R.var(correction=0)
    )
    assert ev_min == pytest.approx(
        1.0 - (R - v.amin(0)).var(correction=0) / R.var(correction=0)
    )
    assert ev_min <= ev_mean, "the backup's min must never look better"


def test_explained_var_min_coincides_with_mean_for_a_single_member():
    n = 32
    v = torch.randn(1, n)
    R = torch.randn(n)
    m = compute_value_metrics(
        raw_value_pred=v, raw_returns_flat=R,
        policy_per_chunk=torch.zeros(n), value_loss_per_chunk=torch.zeros(n),
    )
    assert m["explained_var_min"] == pytest.approx(m["explained_var"])


def test_value_pred_std_member_mean_is_within_member_spread():
    torch.manual_seed(1)
    C, n = 3, 20
    v = torch.randn(C, n) * 2.0
    R = torch.randn(n)
    m = compute_value_metrics(
        raw_value_pred=v, raw_returns_flat=R,
        policy_per_chunk=torch.zeros(n), value_loss_per_chunk=torch.zeros(n),
    )
    assert m["value_pred_std_member_mean"] == pytest.approx(
        v.std(dim=1, correction=0).mean().item(), rel=1e-5
    )
    # The pooled key mixes in between-member disagreement, so it is larger.
    assert m["value_pred_std"] >= m["value_pred_std_member_mean"]
    assert m["value_pred_member_gap"] == pytest.approx(
        (v.amax(0) - v.amin(0)).mean().item(), rel=1e-5
    )


# --- F. the token axis declares whether it is spectral ----------------
def _ratio_metrics(cfg, N=32, MC=8, TOK=8):
    ratio = torch.rand(N, MC, TOK) * 2.0
    return compute_ratio_metrics(
        ratio=ratio, ratio_token=ratio, delta=torch.zeros(N, MC),
        action_is_pad=torch.zeros(N, TOK, dtype=torch.bool), cfg=cfg,
    ), ratio


def test_the_token_axis_is_never_spectral():
    """The DCT is gone, so the flag is pinned at 0 and the spectral keys NaN.

    They stay in the dict on purpose: an old dashboard reading
    flow_ratio_token_spectral_low under the old name would otherwise silently
    pick up the positional split instead.
    """
    c = _cfg(action_horizon=8)
    m, _ = _ratio_metrics(c)
    assert m["token_axis_is_spectral"].item() == 0.0
    assert math.isnan(m["flow_ratio_token_spectral_low"])
    assert math.isnan(m["flow_ratio_token_spectral_high"])
    assert not math.isnan(m["flow_ratio_token_pos_low"])


def test_legacy_lowfreq_keys_survive_and_match_the_positional_split():
    """Old dashboards keep working; the new keys say what the split means.

    The lowfreq/highfreq names are now positional, not spectral. That is a
    deliberate lie kept for compatibility, which is why the *_spectral_* keys
    are pinned to NaN rather than also reporting the positional split.
    """
    c = _cfg(action_horizon=8)
    m, ratio = _ratio_metrics(c)
    per_token = ratio.mean(dim=(0, 1))
    assert m["flow_ratio_token_lowfreq"] == pytest.approx(per_token[:4].mean().item())
    assert m["flow_ratio_token_highfreq"] == pytest.approx(per_token[4:].mean().item())
    assert m["flow_ratio_token_pos_low"] == pytest.approx(
        m["flow_ratio_token_lowfreq"].item()
    )
    assert m["flow_ratio_token_pos_high"] == pytest.approx(
        m["flow_ratio_token_highfreq"].item()
    )


# --- G. reward_improvement_pct on negative returns ----------------------
def test_reward_improvement_pct_reads_wrong_on_a_negative_baseline():
    prev, cur = -1000.0, -500.0   # a real improvement of 500
    out = compute_eval_summary_metrics(
        mean_return=cur,
        eval_results=[
            {"return": r, "length": 10, "success": None, "jerk": None} for r in (cur,)
        ],
        all_eval_returns=[prev, cur],
        eval_count=2,
    )
    assert out["eval/reward_improvement"] == pytest.approx(500.0)  # correct
    assert out["eval/reward_improvement_pct"] == pytest.approx(0.5)  # reads as +50%


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))