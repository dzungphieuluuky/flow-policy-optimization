"""The critic's regression target must survive the rollout buffer in fp32."""

import torch

from fixtures import make_test_policy
from myfpo.fpo_core import subsampled_min
from myfpo.modeling import MyfpoPolicy
from myfpo.rollout import run_rollout
from myfpo.training import (
    measure_value_drift,
)
from myfpo.layers import resolve_storage_dtype


def test_bf16_actually_destroys_a_large_target():
    """Why the dtype matters, independent of which device we are on."""
    x = torch.tensor([1234.5, 87.25, 41.0])
    assert not torch.equal(x.to(torch.bfloat16).to(torch.float32), x)
    # Relative error is ~2^-8, so it is invisible on O(1) and fatal on returns.
    small = torch.tensor([1.0, 0.5])
    assert torch.allclose(small.to(torch.bfloat16).to(torch.float32), small, atol=0.01)


def test_source_does_not_cast_the_value_target_to_storage_dtype():
    """Guard the fix itself."""
    import inspect

    # The value path lives in its own method now, so the guard reads it there.
    src = inspect.getsource(MyfpoPolicy.value_loss)
    assert "storage_dtype" not in src, (
        "the critic's value path must stay fp32; it regresses on raw returns"
    )
    # The promotion to fp32 must sit on the critic's own output line.
    critic_line = next(
        line for line in src.splitlines() if "raw_value_pred = " in line
    )
    assert critic_line.rstrip().endswith(".float()  # [num_critics, n]"), critic_line


def test_resolve_storage_dtype_is_fp32_on_cpu():
    assert resolve_storage_dtype(torch.device("cpu")) is torch.float32


def test_generate_rollout_does_not_cast_the_value_target():
    import inspect

    # The body moved to myfpo.rollout. The method on the policy is a wrapper,
    # so reading its source would find none of the casts and pass vacuously.
    # The casts live in the advantage step, so the guard reads both.
    from myfpo.rollout import chunk_advantages

    src = inspect.getsource(run_rollout) + inspect.getsource(chunk_advantages)
    for name in ("raw_returns", "raw_values"):
        # Both spellings matter. The batch is built as a dict, so the cast would
        # appear as "raw_returns": raw_returns.to(storage_dtype). A test that
        # only rejects the assignment form passes on the dict form, which is the
        # one the code actually uses.
        assert f"{name} = {name}.to(storage_dtype)" not in src, (
            f"{name} is the critic's regression target in raw reward units and "
            "must not be quantized by the storage dtype"
        )
        assert f'"{name}": {name}.to(storage_dtype)' not in src, (
            f"{name} is quantized by the storage dtype in the batch dict"
        )
        assert f"{name}.to(storage_dtype)" not in src, (
            f"{name} must not pass through the storage dtype in any form"
        )
    # The tensors that ride storage_dtype legitimately must still do so.
    assert "raw_advantages.to(storage_dtype)" in src
    assert "normalized_advantages.to(storage_dtype)" in src


def test_drift_diagnostics_take_the_bf16_observations_they_stored():
    """generate_rollout stores boot_observations/boot_actions in storage_dtype."""
    policy = make_test_policy()
    obs = torch.randn(3, policy.config.n_obs_steps, 5)
    actions = torch.randn(3, policy.config.action_horizon, 2)
    # Same fraction generate_rollout bootstrapped with: a full-horizon proposal that only runs for n_action_steps.
    boot_frac = policy.exec_length_frac(
        torch.full((3,), policy.config.n_action_steps, dtype=torch.long)
    )
    batch = {
        "boot_observations": obs.to(torch.bfloat16),
        "boot_actions": actions.to(torch.bfloat16),
        "boot_values_start": subsampled_min(
            policy.critic(obs, actions, boot_frac).float()
        ).unsqueeze(1),
        "raw_returns": torch.randn(3),
    }
    boot_delta, target_drift = measure_value_drift(
        policy, batch, torch.device("cpu")
    )
    assert torch.isfinite(boot_delta)
    assert torch.isfinite(target_drift)
