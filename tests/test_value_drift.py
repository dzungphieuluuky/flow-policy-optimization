"""The two critic-drift metrics must be ratios, and the names must say so."""

import pytest
import torch

from myfpo.layers import rms

OBS_DIM, ACTION_DIM, N = 3, 1, 4


def _batch(start_value, return_value, n=N):
    return {
        "boot_observations": torch.zeros(n, 2, OBS_DIM),
        "boot_actions": torch.zeros(n, 4, ACTION_DIM),
        "boot_values_start": torch.full((n, 1), float(start_value)),
        "raw_returns": torch.full((n,), float(return_value)),
    }


class _FixedCritic:
    """Returns a constant, so the move is known without training anything."""

    def __init__(self, value):
        self.value = value
        self.training = True

    def eval(self):
        return self

    def train(self, _mode=True):
        return self

    def __call__(self, obs, _actions, _exec_length_frac):
        return torch.full((2, obs.shape[0]), float(self.value))


class _FixedConfig:
    num_action_steps = 4
    action_horizon = 4


class _FixedPolicy:
    def __init__(self, value):
        self.critic = _FixedCritic(value)
        self.config = _FixedConfig()
        self.action_low = torch.zeros(1)
        self.action_high = torch.ones(1)

    def exec_length_frac(self, executed_steps):
        return executed_steps.to(torch.float32)


def _measure(policy, batch):
    from myfpo.diagnostics import measure_value_drift

    return measure_value_drift(policy, batch, "cpu")


def test_rms_is_the_root_mean_square_not_the_norm():
    """Pin the scale both denominators use."""
    x = torch.tensor([3.0, 4.0])
    assert float(rms(x)) == pytest.approx((12.5) ** 0.5, abs=1e-6)
    # A constant tensor has rms equal to its own magnitude.
    assert float(rms(torch.full((5,), 10.0))) == pytest.approx(10.0, abs=1e-6)


def test_the_first_metric_is_a_ratio_against_the_bootstrap_value():
    """rms(move) / rms(bootstrap). A doubling reads 1.0, not 10.0."""
    drift_vs_boot, _ = _measure(_FixedPolicy(value=20.0), _batch(10.0, 0.0))
    # move = |20 - 10| = 10, rms(bootstrap) = 10 -> 1.0
    assert float(drift_vs_boot) == pytest.approx(1.0, abs=1e-6)


def test_the_second_metric_is_a_ratio_against_the_return():
    """The same move, scaled by the return RMS instead."""
    _, drift_vs_return = _measure(_FixedPolicy(value=20.0), _batch(10.0, 40.0))
    # move = 10, rms(returns) = 40 -> 0.25
    assert float(drift_vs_return) == pytest.approx(0.25, abs=1e-6)


def test_a_matching_critic_drifts_by_zero():
    a, b = _measure(_FixedPolicy(value=10.0), _batch(10.0, 5.0))
    assert float(a) == pytest.approx(0.0, abs=1e-6)
    assert float(b) == pytest.approx(0.0, abs=1e-6)


def test_the_logged_key_names_the_ratio_not_a_difference():
    """The old key read as V_new - V_old.

    A ratio of 1.0 means the live critic doubled the bootstrap value. The same
    number under a "delta" key reads as a move of one unit. The raw move here
    was 10.
    """
    from pathlib import Path

    # The whole package, not one file. The chain that logs this key moved into
    # the runner package, and a file-scoped search would have gone green on a
    # file that no longer held it.
    root = Path(__file__).resolve().parents[1] / "src" / "myfpo"
    src = "\n".join(p.read_text() for p in sorted(root.rglob("*.py")))
    # The docstring names the old key on purpose. The log call must not.
    assert '"train/boot_value_delta"' not in src, "the old key is still logged"
    assert "train/boot_value_relative_drift" in src

    drift_vs_boot, _ = _measure(_FixedPolicy(value=20.0), _batch(10.0, 0.0))
    assert float(drift_vs_boot) == pytest.approx(1.0, abs=1e-6)
    assert float(drift_vs_boot) != pytest.approx(10.0), (
        "the metric is a ratio; the raw move was 10"
    )


def test_the_two_metrics_differ_only_by_their_denominator():
    """One critic forward feeds both. Same numerator, two scales."""
    a, b = _measure(_FixedPolicy(value=20.0), _batch(10.0, 25.0))
    assert float(a) / float(b) == pytest.approx(25.0 / 10.0, abs=1e-4)