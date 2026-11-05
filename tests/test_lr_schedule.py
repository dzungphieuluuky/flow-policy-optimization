"""The LR schedule counts updates, not env steps.

``LambdaLR`` steps once per update, while ``total_steps`` and
``lr_warmup_steps`` count env steps. The lambda used to compare the two
directly, so a real run never left its warmup: at 300k env steps the LR was
still ~0.7% of ``policy_lr``.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch

from myfpo.configuration import MyfpoConfig
from myfpo.checkpointing import load_policy_checkpoint, save_policy_checkpoint
from myfpo.modeling import make_optimizer
from myfpo.training import make_scheduler, restore_scheduler

from fixtures import make_test_policy

LR = 0.1
# 8 chunks x 4 actions = 32 env steps per update.
CFG = dict(chunks_per_update=8, chunks_per_minibatch=4, action_horizon=4)
# 3200 env steps = 100 updates. 320 env steps = 10 updates of warmup.
TOTAL_STEPS = 3200
WARMUP_STEPS = 320


def _optimizer():
    param = torch.zeros(1, requires_grad=True)
    return torch.optim.SGD([param], lr=LR)


def _cfg(**kw):
    fields = dict(lr_schedule="cosine", lr_warmup_steps=WARMUP_STEPS, **CFG)
    fields.update(kw)
    return dataclasses.replace(MyfpoConfig(), **fields)


def _lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def test_the_lr_leaves_the_warmup_at_the_env_step_warmup_budget():
    """The regression: update 10 must already sit at the full LR."""
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    seen = []
    for _ in range(100):
        seen.append(_lr(optimizer))
        scheduler.step()
    # 10 updates of warmup, and update 0 already carries its share.
    assert seen[0] == pytest.approx(LR / 10)
    assert seen[10] == pytest.approx(LR)
    # The cosine tail is reached, and it decays from there.
    assert seen[-1] < 0.01 * LR
    assert seen[-1] < seen[50] < seen[10]


def test_a_partial_warmup_rounds_up_to_a_whole_update():
    """48 env steps of warmup at 32 steps per update is 1.5 updates, so 2.

    A floor made the warmup reach the full LR after one update, half a
    window early.
    """
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(lr_warmup_steps=48), TOTAL_STEPS)
    seen = []
    for _ in range(4):
        seen.append(_lr(optimizer))
        scheduler.step()
    assert seen[0] == pytest.approx(0.5 * LR)
    assert seen[1] == pytest.approx(LR)
    assert seen[2] == pytest.approx(LR)


def test_the_first_update_does_not_train_at_zero_lr():
    """``LambdaLR`` steps once in its constructor, so update 0 used to run at LR 0."""
    optimizer = _optimizer()
    make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    assert _lr(optimizer) == pytest.approx(LR / 10)


def test_the_decay_is_monotone_after_the_warmup():
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    seen = []
    for _ in range(100):
        seen.append(_lr(optimizer))
        scheduler.step()
    tail = seen[10:]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(tail))


def test_the_last_partial_update_keeps_a_positive_lr():
    """The loop runs ceil(steps / per_update) updates, so the schedule must too.

    With a floor, the last update sat past the cosine's end and got lr = 0.
    """
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS + 16)
    seen = []
    for _ in range(101):        # 3216 env steps = 100.5 updates
        seen.append(_lr(optimizer))
        scheduler.step()
    assert seen[-1] > 0.0
    assert seen[-2] > seen[-1]


def test_zero_warmup_starts_at_the_full_lr():
    optimizer = _optimizer()
    make_scheduler(optimizer, _cfg(lr_warmup_steps=0), TOTAL_STEPS)
    assert _lr(optimizer) == pytest.approx(LR)


@pytest.mark.parametrize("total_steps", [16, 200])
def test_a_run_too_short_for_its_warmup_keeps_a_constant_lr(total_steps):
    """A smoke run must not train at a near-zero LR."""
    optimizer = _optimizer()
    assert make_scheduler(optimizer, _cfg(), total_steps) is None


def test_the_schedule_is_off_when_the_config_says_none():
    optimizer = _optimizer()
    assert make_scheduler(optimizer, _cfg(lr_schedule="none"), TOTAL_STEPS) is None


def test_a_saved_scheduler_state_restores_the_same_lr():
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    for _ in range(20):
        scheduler.step()
    state = scheduler.state_dict()

    resumed_optimizer = _optimizer()
    resumed = make_scheduler(resumed_optimizer, _cfg(), TOTAL_STEPS)
    restore_scheduler(
        resumed, state, global_step=20 * 32, cfg=_cfg()
    )

    assert resumed.last_epoch == 20
    assert _lr(resumed_optimizer) == pytest.approx(float(scheduler.get_last_lr()[0]))
    # The next step advances both curves by the same amount.
    scheduler.step()
    resumed.step()
    assert resumed.get_last_lr()[0] == pytest.approx(scheduler.get_last_lr()[0])


def test_an_old_checkpoint_fast_forwards_from_global_step():
    """A checkpoint without scheduler state restarts the curve from the counter."""
    optimizer = _optimizer()
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    restore_scheduler(scheduler, None, global_step=20 * 32, cfg=_cfg())
    assert scheduler.last_epoch == 20
    expected = LR * scheduler.lr_lambdas[0](20)
    assert _lr(optimizer) == pytest.approx(expected)


def test_the_scheduler_state_round_trips_through_a_checkpoint(tmp_path):
    policy = make_test_policy()
    optimizer = make_optimizer(policy)
    scheduler = make_scheduler(optimizer, _cfg(), TOTAL_STEPS)
    for _ in range(7):
        scheduler.step()
    save_policy_checkpoint(
        tmp_path,
        policy=policy,
        optimizer=optimizer,
        global_step=7 * 32,
        scheduler_state=scheduler.state_dict(),
    )

    _, _, counters = load_policy_checkpoint(tmp_path)
    assert counters["global_step"] == 7 * 32
    assert counters["scheduler"]["last_epoch"] == 7


def test_a_checkpoint_without_scheduler_state_still_loads(tmp_path):
    """Old checkpoints are FORMAT_VERSION 1 without the optional field."""
    policy = make_test_policy()
    optimizer = make_optimizer(policy)
    save_policy_checkpoint(tmp_path, policy=policy, optimizer=optimizer, global_step=5)
    _, _, counters = load_policy_checkpoint(tmp_path)
    assert "scheduler" not in counters


@pytest.mark.parametrize("bad", ["constant", "cosine2", ""])
def test_an_unknown_schedule_is_rejected(bad):
    with pytest.raises(ValueError, match="lr_schedule"):
        dataclasses.replace(MyfpoConfig(), lr_schedule=bad)


def test_a_negative_warmup_is_rejected():
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        dataclasses.replace(MyfpoConfig(), lr_warmup_steps=-1)
