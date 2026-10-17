"""Tests for the batched rollout diagnostics (``myfpo.train_sjfpo``)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myfpo.train_sjfpo import compute_rollout_diagnostics

N, H, D = 8, 4, 3


def make_batch():
    """One rollout's action/mask, every chunk executing a different prefix."""
    g = torch.Generator().manual_seed(0)
    batch = {
        "action": torch.randn(N, H, D, generator=g),
        "action_is_pad": torch.zeros(N, H, dtype=torch.bool),
    }
    for i in range(N):
        batch["action_is_pad"][i, i % H + 1 :] = True
    return batch


def test_matches_the_unbatched_reductions():
    """One transfer, but the same numbers as the float()s it replaced."""
    batch = make_batch()
    out = compute_rollout_diagnostics(batch)
    action = batch["action"]
    assert out["executed_action_steps"] == int((~batch["action_is_pad"]).sum())
    assert out["action_abs_max"] == float(action.abs().max())
    assert out["action_abs_mean"] == float(action.abs().mean())


def test_executed_action_steps_counts_executed_steps():
    """The key drives global_step, so a padded row must contribute nothing.

    The key was named ``step_delta``. It is a count of steps, not a difference
    of anything, and the trainer only adds it to ``global_step``.
    """
    batch = make_batch()
    out = compute_rollout_diagnostics(batch)
    assert out["executed_action_steps"] == sum(
        int((~pad).sum()) for pad in batch["action_is_pad"]
    )


def test_executed_only_keys_use_the_executed_rows():
    """The legacy keys keep their all-rows meaning. The new ones mask first."""
    batch = make_batch()
    out = compute_rollout_diagnostics(batch)
    action = batch["action"]
    exec_action = action[~batch["action_is_pad"]]
    assert out["action_abs_mean"] == float(action.abs().mean())
    assert out["action_abs_mean_executed"] == float(exec_action.abs().mean())
    assert out["action_abs_max"] == float(action.abs().max())
    assert out["action_abs_max_executed"] == float(exec_action.abs().max())
    # With padding holding real values, the two means must genuinely differ.
    assert out["action_abs_mean"] != pytest.approx(out["action_abs_mean_executed"])