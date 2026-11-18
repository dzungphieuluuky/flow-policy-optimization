"""linear_rescale: stored actions live in the flow's unit box.

The flow regresses raw output; the stored chunk is clamped to the unit
hypercube, and the env-facing action is the affine unscale of the unit box
into the bounds the env declares. The env needs per-dimension bounds, and
they need not be symmetric.
"""

import numpy as np
import pytest
import torch

from flow_policy_optimization.flow_policy_optimization import FlowPolicyOptimizationConfig, FlowPolicyOptimizationPolicy, bound_for_storage, resolve_action_bounds, to_env_action


def _policy(**kw):
    cfg = FlowPolicyOptimizationConfig(
        hidden_dim=32,
        num_obs_steps=4,
        action_horizon=4,
        num_action_steps=4,
        **kw,
    )
    torch.manual_seed(0)
    low = np.full(2, -2.0, dtype=np.float32)
    high = np.full(2, 2.0, dtype=np.float32)
    return FlowPolicyOptimizationPolicy(cfg, 3, 2, low, high)


def test_to_env_action_unscales_the_unit_box():
    low = torch.tensor([-0.4, -3.0])
    high = torch.tensor([0.4, 3.0])
    x = torch.tensor([[-1.0, -1.0], [0.0, 0.5], [1.0, 1.0]])
    out = to_env_action(x, low, high)
    assert torch.allclose(out[0], low, atol=1e-6)
    assert torch.allclose(out[2], high, atol=1e-6)
    assert torch.allclose(out[1, 1], torch.tensor(1.5), atol=1e-6)


def test_bound_for_storage_clamps_outliers_only():
    x = torch.tensor([-50.0, -2.0, -1.0, 0.0, 1.0, 2.0, 50.0])
    got = bound_for_storage(x)
    assert torch.equal(got, x.clamp(-1.0, 1.0))
    assert torch.equal(got[3:5], x[3:5])


def test_the_policy_stores_only_bounded_actions():
    """Train mode, huge perturbation: the stored chunk must not exceed the unit box."""
    policy = _policy(action_perturb_std=50.0)
    policy.train()
    states = torch.randn(16, 4, 3) * 3.0
    for _ in range(8):
        chunk = policy.predict_action_chunk(states)
        assert float(chunk.abs().max()) <= 1.0


def test_infinite_or_nan_bounds_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        resolve_action_bounds([float("-inf"), -1.0], [1.0, float("inf")])
    with pytest.raises(ValueError, match="finite"):
        resolve_action_bounds([float("nan")], [1.0])
    with pytest.raises(ValueError, match="finite"):
        resolve_action_bounds([1.0], [float("inf")])


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="high > low"):
        resolve_action_bounds([1.0], [1.0])
    with pytest.raises(ValueError, match="high > low"):
        resolve_action_bounds([2.0, -1.0], [2.0, -1.0])
