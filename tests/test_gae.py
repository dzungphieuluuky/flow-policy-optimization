"""Step-level GAE tests for executed action-chunk prefixes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.gae import compute_step_advantages_torch  # noqa: E402


def test_truncated_step_uses_reward_adjustment_and_stops_recursion():
    advantages, returns = compute_step_advantages_torch(
        rewards_1d=torch.tensor([1.0]),
        values=torch.tensor([2.0]),
        next_values=torch.tensor([10.0]),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        gamma=0.9,
        lam=1.0,
    )

    # r_adj = r + gamma * V(s_t); no V(s_{t+1}) bootstrap.
    expected = 1.0 + 0.9 * 2.0 - 2.0
    torch.testing.assert_close(advantages, torch.tensor([expected]))
    torch.testing.assert_close(returns, torch.tensor([2.0 + expected]))


def test_terminated_step_masks_bootstrap():
    advantages, returns = compute_step_advantages_torch(
        rewards_1d=torch.tensor([1.0]),
        values=torch.tensor([0.0]),
        next_values=torch.tensor([10.0]),
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        gamma=0.9,
        lam=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([1.0]))
    torch.testing.assert_close(returns, torch.tensor([1.0]))


def test_truncated_step_cuts_only_future_recursion():
    advantages, _ = compute_step_advantages_torch(
        rewards_1d=torch.tensor([1.0, 2.0, 3.0]),
        values=torch.tensor([0.0, 0.0, 0.0]),
        next_values=torch.tensor([10.0, 20.0, 30.0]),
        terminated=torch.tensor([False, False, False]),
        truncated=torch.tensor([False, True, False]),
        gamma=0.9,
        lam=0.5,
    )

    # The second step uses the FPO++ reward adjustment and cuts off the third
    # step; the first step still recurses into the second.
    torch.testing.assert_close(advantages, torch.tensor([10.9, 2.0, 30.0]))


def test_continuing_steps_share_one_recursion():
    advantages, returns = compute_step_advantages_torch(
        rewards_1d=torch.tensor([1.0, 2.0]),
        values=torch.tensor([0.0, 0.0]),
        next_values=torch.tensor([10.0, 20.0]),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        gamma=0.9,
        lam=0.5,
    )

    # A_1 = 2 + 0.9 * 20 = 20; A_0 = 1 + 0.9 * 10 + 0.45 * A_1.
    torch.testing.assert_close(advantages, torch.tensor([19.0, 20.0]))
    torch.testing.assert_close(returns, advantages)


def test_step_gae_validates_flat_stream_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_step_advantages_torch(
            rewards_1d=torch.tensor([1.0, 2.0]),
            values=torch.tensor([0.0]),
            next_values=torch.tensor([1.0, 2.0]),
            terminated=torch.tensor([False, False]),
        )


def test_empty_step_stream_returns_empty_targets():
    advantages, returns = compute_step_advantages_torch(
        rewards_1d=torch.empty(0),
        values=torch.empty(0),
        next_values=torch.empty(0),
        terminated=torch.empty(0, dtype=torch.bool),
    )

    assert advantages.numel() == 0
    assert returns.numel() == 0
