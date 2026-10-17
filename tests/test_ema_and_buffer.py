"""Two reuse/initialisation defects that both failed silently."""

import dataclasses

import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.collector import RolloutBuffer
from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.modeling_sjfpo import SJFPOPolicy


def _policy(**kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    return SJFPOPolicy(cfg)


# --- #1 buffer reuse -------------------------------------------------------.


def test_shape_key_holds_the_six_shape_fields_only():
    """pin_memory is a device property. It cannot invalidate a reuse."""
    buf = RolloutBuffer(
        max_chunks=8, max_steps=8, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    assert len(buf.shape_key()) == 6, "the six fields generate_rollout builds"
    assert buf.shape_key() == (8, 8, 2, 3, 4, 2)
    # Two buffers that differ only in pinning share one key.
    pinned = RolloutBuffer(
        max_chunks=8,
        max_steps=8,
        obs_steps=2,
        obs_dim=3,
        action_horizon=4,
        action_dim=2,
        pin_memory=True,
    )
    assert pinned.shape_key() == buf.shape_key()
    assert pinned.pin_memory is True and buf.pin_memory is False


def test_reuse_condition_is_false_for_an_identical_key():
    """The exact expression generate_rollout uses, so a future field is caught."""
    buf = RolloutBuffer(
        max_chunks=8, max_steps=8, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    buffer_key = (8, 8, 2, 3, 4, 2)
    # equal -> the reuse condition is false -> the existing buffer is kept.
    assert buf.shape_key() == buffer_key


def test_reuse_condition_is_true_when_capacity_grows():
    buf = RolloutBuffer(
        max_chunks=4, max_steps=4, obs_steps=2, obs_dim=3, action_horizon=4, action_dim=2
    )
    assert buf.shape_key() != (8, 8, 2, 3, 4, 2)


def test_no_caller_has_to_slice_the_key():
    """No call site may drop a trailing field again.

    The old key carried pin_memory as a seventh element, so both call sites
    sliced [:6]. A slice here means the key holds something a caller does not
    want.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "myfpo"
    for rel in ("modeling_sjfpo.py", "collector.py"):
        src = (root / rel).read_text()
        assert not re.search(r"shape_key\(\)\s*\[", src), f"{rel} slices shape_key"
        assert not re.search(r"def\s+capacity_key\b", src), f"{rel} still defines capacity_key"
        assert ".capacity_key(" not in src, f"{rel} still calls capacity_key"


# --- #2 EMA warmup ---------------------------------------------------------.


def test_ema_initialises_on_the_first_rollout_boundary():
    """ema_warmup_steps=500 but global_step lands on 4096, 8192, ..."""
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
    """The gate must not re-hard-copy on every later call."""
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


# --- EMA swap --------------------------------------------------------------.


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
    live = policy._ema_named_params()
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
    before = {n: p.detach().clone() for n, p in policy._ema_named_params().items()}

    policy.swap_out_ema(policy.swap_in_ema(8192))
    after = policy._ema_named_params()
    for name, value in before.items():
        assert torch.equal(after[name], value)


def test_swap_is_inert_before_the_shadow_is_ready():
    """Seeded at the first call, hard-copied at warmup: 499 is neither."""
    policy = _policy(ema_warmup_steps=500)
    policy.update_ema(499)
    assert policy.swap_in_ema(499) == {}
    policy.swap_out_ema({})


def test_swap_reads_the_env_step_clock_not_the_optimizer_step_counter():
    """The shadow warms on global_step. Step_count runs far ahead of it."""
    policy = _policy(ema_warmup_steps=500)
    policy.update_ema(500)
    policy.step_count = 10_000
    assert policy.ema_ready(500) is False
    assert policy.swap_in_ema(500) == {}
    assert policy.swap_in_ema(501) != {}
