"""The behavior and training CFM reads share one token-loss prologue.

``compute_behavior_cfm_losses`` (rollout) and the training path
(``policy_loss``) score the same stored flow point through
``_token_cfm_loss``. The FPO ratio divides by the behavior loss, so the two
readings must stay identical; this pins that.
"""

import torch

from fixtures import ACT_DIM, OBS_DIM, make_batch, make_test_policy

from myfpo.layers import autocast_context, resolve_storage_dtype


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
            b.x_t, b.actor_t, b.condition_flat
        ).float()
    terms = policy.policy_loss(b, v_new, device)

    loss_b, token_b = policy.compute_behavior_cfm_losses(
        b.states, b.actions, b.t, b.eps, b.action_is_pad
    )

    assert torch.equal(loss_b.detach(), terms.cfm_loss_new.detach())
    assert torch.isfinite(token_b).all()
