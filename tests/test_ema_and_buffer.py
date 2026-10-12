"""Two reuse/initialisation defects that both failed silently.

The rollout buffer compared a 6-tuple against capacity_key()'s 7-tuple, so it
reallocated every rollout. The EMA warmup compared ==, against a step counter
that only ever lands on rollout multiples, so the shadow was never initialised
and ema_ready() was false forever. Neither raised; both just quietly did nothing.
"""

import dataclasses

import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from spectral_joint_policy.collector import RolloutBuffer
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy


def _policy(**kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    return SJFPOPolicy(cfg)


# --- #1 buffer reuse -------------------------------------------------------


def test_capacity_key_shape_is_what_the_reuse_check_compares():
    buf = RolloutBuffer(
        max_chunks=8, max_steps=8, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    assert len(buf.capacity_key()) == 7  # includes pin_memory
    assert len(buf.capacity_key()[:6]) == 6  # the shape generate_rollout builds


def test_reuse_condition_is_false_for_an_identical_key():
    """The exact expression generate_rollout uses, so a future 8th field is caught."""
    buf = RolloutBuffer(
        max_chunks=8, max_steps=8, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    buffer_key = (8, 8, 2, 3, 4, 2)
    # equal -> the reuse condition is false -> the existing buffer is kept
    assert buf.capacity_key()[:6] == buffer_key
    # unequal -> the reuse condition is true -> reallocate. This is what the
    # un-sliced comparison did on every rollout, because pin_memory made the
    # full tuple permanently different.
    assert buf.capacity_key() != buffer_key


def test_reuse_condition_is_true_when_capacity_grows():
    buf = RolloutBuffer(
        max_chunks=4, max_steps=4, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    assert buf.capacity_key()[:6] != (8, 8, 2, 3, 4, 2)


# --- #2 EMA warmup ---------------------------------------------------------


def test_ema_initialises_on_the_first_rollout_boundary():
    """ema_warmup_steps=500 but global_step lands on 4096, 8192, ...

    With == the initialisation never fired, so ema_ready() was false for the whole
    run and no checkpoint ever contained the EMA.
    """
    policy = _policy()
    steps = [4096, 8192, 12288]
    assert policy.config.ema_warmup_steps == 500
    assert 500 not in steps

    assert not policy.ema_ready(steps[0])
    for step in steps:
        policy.update_ema(step)
    assert policy._ema_initialised is True
    assert policy.ema_ready(steps[0]) is True


def test_ema_warmup_is_a_one_shot():
    """The gate must not re-hard-copy on every later call.

    Without `not _ema_initialised`, every call past warmup would reset the shadow
    and the running average would never form.
    """
    policy = _policy()
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(1.0)
    policy.update_ema(4096)
    seeded = {n: s.clone() for n, s in policy._ema_shadow.items()}

    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(10.0)
    policy.update_ema(8192)
    moved = any(
        not torch.equal(policy._ema_shadow[n], seeded[n]) for n in seeded
    )
    assert moved, "shadow should have averaged toward the new weights"

    # A third call must keep averaging, not reset to the live weights.
    averaged = {n: s.clone() for n, s in policy._ema_shadow.items()}
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(10.0)
    policy.update_ema(12288)
    assert any(
        not torch.equal(policy._ema_shadow[n], averaged[n]) for n in averaged
    )


def test_ema_is_inert_when_decay_is_zero():
    policy = _policy(ema_decay=0.0)
    policy.update_ema(4096)
    assert policy.ema_ready(4096) is False
    assert not policy._ema_shadow


# --- EMA swap --------------------------------------------------------------


def test_swap_in_ema_puts_the_average_in_the_actor():
    """The eval path depends on this: EMA is otherwise computed and discarded."""
    policy = _policy()
    policy.update_ema(4096)
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(10.0)
    policy.update_ema(8192)
    shadow = {n: s.clone() for n, s in policy._ema_shadow.items()}

    displaced = policy.swap_in_ema(8192)
    live = dict(policy.actor.named_parameters())
    assert set(displaced) == set(shadow)
    for name, value in shadow.items():
        assert torch.equal(live[name], value)


def test_swap_out_ema_restores_the_live_weights_exactly():
    """Training must never continue on the average."""
    policy = _policy()
    policy.update_ema(4096)
    with torch.no_grad():
        for p in policy.actor.parameters():
            p.add_(1.0)
    policy.update_ema(8192)
    before = {n: p.detach().clone() for n, p in policy.actor.named_parameters()}

    policy.swap_out_ema(policy.swap_in_ema(8192))
    after = dict(policy.actor.named_parameters())
    for name, value in before.items():
        assert torch.equal(after[name], value)


def test_swap_is_inert_before_the_shadow_is_ready():
    """Seeded at the first call, hard-copied at warmup: 499 is neither."""
    policy = _policy(ema_warmup_steps=500)
    policy.update_ema(499)
    assert policy.swap_in_ema(499) == {}
    policy.swap_out_ema({})


def test_swap_reads_the_env_step_clock_not_the_optimizer_step_counter():
    """The shadow warms on global_step; step_count runs far ahead of it.

    Reading step_count would call the average ready before the hard copy, so the
    first evals would score the seed weights.
    """
    policy = _policy(ema_warmup_steps=500)
    policy.update_ema(500)
    policy.step_count = 10_000
    assert policy.ema_ready(500) is False
    assert policy.swap_in_ema(500) == {}
    assert policy.swap_in_ema(501) != {}


# --- target critic ---------------------------------------------------------


def test_target_critic_hard_copies_on_the_first_update():
    policy = _policy()
    key = next(iter(policy.critic.state_dict()))
    assert torch.equal(
        policy.target_critic.state_dict()[key],
        policy.critic.state_dict()[key],
    )

    with torch.no_grad():
        for p in policy.critic.parameters():
            p.add_(1.0)
    policy.update_target_critic()
    # First call syncs exactly, so the teacher is not 0.5% behind at rollout 1.
    assert torch.equal(
        policy.target_critic.state_dict()[key], policy.critic.state_dict()[key]
    )


def test_target_critic_lags_by_exactly_the_decay():
    policy = _policy()
    key = next(iter(policy.critic.state_dict()))
    online = policy.critic.state_dict()[key].clone()
    policy.update_target_critic()

    with torch.no_grad():
        for p in policy.critic.parameters():
            p.add_(1.0)
    policy.update_target_critic()
    decay = policy.config.critic_ema_decay
    expected = online + decay
    assert torch.allclose(policy.target_critic.state_dict()[key], expected, atol=1e-6)


def test_target_critic_never_receives_gradients_or_training_mode():
    policy = _policy()
    assert all(not p.requires_grad for p in policy.target_critic.parameters())
    assert not policy.target_critic.training
    assert all(p.grad is None for p in policy.target_critic.parameters())
