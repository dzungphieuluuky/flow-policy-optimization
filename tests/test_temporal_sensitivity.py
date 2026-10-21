"""The per-position sensitivity probe. It measures. It does not change."""

import math

import torch
from fixtures import make_batch, make_policy

ACTION_KEY = "critic_action_grad_norm_pos"
FRAME_KEY = "obs_frame_grad_norm_pos"


def _policy(**kw):
    # Dropout off: the critic's 1% mask would make every call a new draw.
    return make_policy(critic_dropout=0.0, **kw)


def _batch(policy):
    return make_batch(
        policy,
        obs_dim=5,
        action_dim=2,
        horizon=policy.config.action_horizon,
    )


def _keys(info, prefix):
    return sorted(k for k in info if k.startswith(prefix))


def test_the_probe_is_off_by_default():
    policy = _policy()
    _, info = policy.loss_with_info(_batch(policy))
    assert _keys(info, ACTION_KEY) == []
    assert _keys(info, FRAME_KEY) == []


def test_the_probe_reports_one_value_per_position():
    policy = _policy(temporal_probe_every=1)
    _, info = policy.loss_with_info(_batch(policy))
    action = _keys(info, ACTION_KEY)
    frame = _keys(info, FRAME_KEY)
    assert len(action) == policy.config.action_horizon
    assert len(frame) == policy.config.n_obs_steps
    assert all(math.isfinite(info[k]) for k in action + frame)


def test_the_probe_runs_every_nth_call():
    policy = _policy(temporal_probe_every=2)
    _, first = policy.loss_with_info(_batch(policy))
    _, second = policy.loss_with_info(_batch(policy))
    assert _keys(first, ACTION_KEY) == []
    assert len(_keys(second, ACTION_KEY)) == policy.config.action_horizon


def test_the_policy_measures_its_own_step_times():
    policy = _policy()
    assert policy.actor_step_s == 0.0
    policy.loss_with_info(_batch(policy))
    assert policy.actor_step_s > 0.0
    assert policy.critic_step_s > 0.0


def test_the_probe_changes_neither_the_loss_nor_the_gradients():
    torch.manual_seed(0)
    off = _policy()
    torch.manual_seed(0)
    on = _policy(temporal_probe_every=1)
    torch.manual_seed(1)
    batch = _batch(off)

    loss_off, _ = off.loss_with_info(batch)
    loss_off.backward()
    loss_on, _ = on.loss_with_info(batch)
    loss_on.backward()

    torch.testing.assert_close(loss_off, loss_on, rtol=0, atol=0)
    for (name, a), (_, b) in zip(off.named_parameters(), on.named_parameters()):
        assert torch.equal(a.grad, b.grad), name
