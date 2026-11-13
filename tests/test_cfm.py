"""CFM math: shared token-loss prologue, MC grouping, standard-normal noise."""
import pytest
import torch

from fixtures import ACT_DIM, OBS_DIM, make_batch, make_test_policy

from myfpo.flow import group_cfm_loss, sample_noise
from myfpo.nn import autocast_context, resolve_storage_dtype


def test_normal_mode_is_plain_randn():
    """White == torch.randn, bit for bit, same generator."""
    shape = (4, 8, 3, 3)
    got = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    want = torch.randn(*shape, generator=g)
    assert torch.equal(got, want)


def test_generated_noise_is_reproducible():
    """Rollouts draw eps from a seeded generator. Twice must match."""
    shape = (2, 3, 8, 3)
    a = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(3))
    b = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(3))
    assert torch.equal(a, b)



# ---- MC grouping (from test_cfm_grouping) ----

def test_group_size_one_is_the_identity():
    loss = torch.randn(4, 8, 3)
    assert group_cfm_loss(loss, 1) is loss


def test_minus_one_puts_every_sample_in_one_group():
    loss = torch.randn(4, 8, 3)
    got = group_cfm_loss(loss, -1)
    assert got.shape == (4, 1, 3)
    assert torch.allclose(got[:, 0], loss.mean(dim=1), atol=1e-6)


@pytest.mark.parametrize("size,groups", [(2, 4), (4, 2), (8, 1)])
def test_partial_grouping_splits_the_sample_axis(size, groups):
    loss = torch.randn(3, 8)
    got = group_cfm_loss(loss, size)
    assert got.shape == (3, groups)
    for g in range(groups):
        assert torch.allclose(
            got[:, g], loss[:, g * size : (g + 1) * size].mean(dim=1), atol=1e-6
        )


def test_grouping_the_loss_equals_grouping_the_difference():
    """Why the reference can average the losses instead of the log-ratio."""
    old = torch.randn(2, 8)
    new = torch.randn(2, 8)
    grouped = group_cfm_loss(old, -1) - group_cfm_loss(new, -1)
    direct = group_cfm_loss(old - new, -1)
    assert torch.allclose(grouped, direct, atol=1e-6)


def test_a_group_size_that_does_not_divide_is_rejected():
    """The reference asserts divisibility. So does this, with the numbers."""
    with pytest.raises(ValueError, match="divide evenly"):
        group_cfm_loss(torch.randn(2, 6), 4)


def test_a_token_axis_survives_grouping():
    """Per-token ratios keep the token axis: ``[n, groups, H]``."""
    loss = torch.randn(2, 8, 5)
    assert group_cfm_loss(loss, -1).shape == (2, 1, 5)
    assert group_cfm_loss(loss, 4).shape == (2, 2, 5)


# ---- Shared token-loss prologue (from test_cfm_shared_loss) ----

def _policy_batch(n=3):
    torch.manual_seed(0)
    policy = make_test_policy()
    policy.eval()
    batch = make_batch(
        policy, obs_dim=OBS_DIM, action_dim=ACT_DIM,
        horizon=policy.config.action_horizon, n=n,
    )
    return policy, batch


def test_behavior_and_training_read_one_token_loss():
    policy, batch = _policy_batch()
    device = next(policy.parameters()).device
    b = policy.to_device_batch(
        batch, device, resolve_storage_dtype(device), grad_inputs=False
    )

    with autocast_context(device):
        v_new = policy.compute_predicted_velocity(
            b.x_t, b.flat_timesteps, b.condition_flat
        ).float()
    terms = policy.policy_loss(b, v_new, device)

    loss_b, token_b = policy.compute_behavior_cfm_losses(
        b.states, b.actions, b.timesteps, b.eps, b.action_is_pad
    )

    assert torch.equal(loss_b.detach(), terms.cfm_loss_new.detach())
    assert torch.isfinite(token_b).all()

