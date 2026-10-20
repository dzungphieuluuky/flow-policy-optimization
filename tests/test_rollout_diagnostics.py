"""Tests for the batched rollout diagnostics (``myfpo.training``)."""
import pytest
import torch

from myfpo.training import compute_rollout_diagnostics

N, H, D = 8, 4, 3
LOW = [-1.0] * D
HIGH = [1.0] * D


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


def diagnostics(batch):
    return compute_rollout_diagnostics(batch, action_low=LOW, action_high=HIGH)


def test_matches_the_unbatched_reductions():
    """One transfer, but the same numbers as the float()s it replaced."""
    batch = make_batch()
    out = diagnostics(batch)
    action = batch["action"]
    assert out["executed_action_steps"] == int((~batch["action_is_pad"]).sum())
    assert out["raw_action_abs_max"] == float(action.abs().max())
    assert out["raw_action_abs_mean"] == float(action.abs().mean())


def test_executed_action_steps_counts_executed_steps():
    """The key drives global_step, so a padded row must contribute nothing.

    The key was named ``step_delta``. It is a count of steps, not a difference
    of anything, and the trainer only adds it to ``global_step``.
    """
    batch = make_batch()
    out = diagnostics(batch)
    assert out["executed_action_steps"] == sum(
        int((~pad).sum()) for pad in batch["action_is_pad"]
    )


def test_executed_only_keys_use_the_executed_rows():
    """The raw keys cover every row. The env-facing ones mask first."""
    batch = make_batch()
    out = diagnostics(batch)
    action = batch["action"]
    exec_action = torch.tanh(action[~batch["action_is_pad"]])
    assert out["raw_action_abs_mean"] == float(action.abs().mean())
    assert out["executed_action_abs_mean"] == float(exec_action.abs().mean())
    assert out["raw_action_abs_max"] == float(action.abs().max())
    assert out["executed_action_abs_max"] == float(exec_action.abs().max())
    # With padding holding real values, the two means must genuinely differ.
    assert out["raw_action_abs_mean"] != pytest.approx(
        out["executed_action_abs_mean"]
    )


def test_the_env_facing_action_is_the_squash_of_the_raw_one():
    """The two action spaces, one number each.

    The stored chunk is the flow's raw output. The env received the squash of
    it, so the env-facing magnitude is the smaller one.
    """
    batch = make_batch()
    out = diagnostics(batch)
    assert out["executed_action_abs_mean"] < out["raw_action_abs_mean"]
    assert out["executed_action_abs_max"] <= out["raw_action_abs_max"]
