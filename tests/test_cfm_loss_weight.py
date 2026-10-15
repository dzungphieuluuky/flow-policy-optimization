"""``cfm_loss_weight_from_t`` must not invert when the time convention flips.

FPO++'s two schedules are named for their endpoints: ``1.0`` at ``t=0`` falling
to ``0.1`` or ``0.01`` at ``t=1`` (``flow_model.py:648-661``). In the reference
``t=0`` is the **action** end, because its flow runs ``t: 1 -> 0`` from noise.
This repo runs ``t: 0 -> 1``, so ``t=1`` is the action end. A verbatim copy of
the formula would therefore weight the *noise* end hardest, which is the exact
opposite of what the schedule is for.

So: full weight at the action end, minimum at the noise end, and the pins below
are written in terms of endpoints rather than ``t`` values so a later reader
cannot mistake which is which.
"""

import dataclasses

import pytest
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import cfm_loss_weight, compute_cfm_loss

ACTION_END, NOISE_END = 1.0, 0.0


def _cfg(**kw):
    return dataclasses.replace(SJFPOConfig(), cfm_loss_use_huber=False, **kw)


def test_constant_is_ones_everywhere():
    """The released default: no schedule, so every token counts the same."""
    cfg = _cfg(cfm_loss_weight_from_t="constant")
    t = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    assert torch.equal(cfm_loss_weight(t, cfg), torch.ones_like(t))


@pytest.mark.parametrize(
    "mode,low", [("linear_1_to_0.1", 0.1), ("linear_1_to_0.01", 0.01)]
)
def test_the_ramp_is_full_at_the_action_end_and_low_at_noise(mode, low):
    """The inversion pin. Read as endpoints: 1.0 at the action, ``low`` at noise."""
    cfg = _cfg(cfm_loss_weight_from_t=mode)
    t = torch.tensor([[ACTION_END], [NOISE_END]])

    weight = cfm_loss_weight(t, cfg)
    assert float(weight[0]) == pytest.approx(1.0), (
        f"{mode}: the action end must carry full weight, got {float(weight[0])}"
    )
    assert float(weight[1]) == pytest.approx(low), (
        f"{mode}: the noise end must carry {low}, got {float(weight[1])}"
    )


def test_a_naive_copy_of_the_reference_formula_would_fail():
    """Spell out the bug this guards, so the sign convention is on the record.

    The reference's own expression on this repo's ``t`` is ``1 - 0.9t``: 1.0 at
    the noise end, 0.1 at the action end.
    """
    cfg = _cfg(cfm_loss_weight_from_t="linear_1_to_0.1")
    t = torch.tensor([[ACTION_END], [NOISE_END]])
    naive = 1.0 - 0.9 * t

    weight = cfm_loss_weight(t, cfg)
    assert float(naive[0]) == pytest.approx(0.1)
    assert float(weight[0]) == pytest.approx(1.0)
    assert not torch.allclose(weight, naive)


def test_the_weight_scales_the_loss_in_proportion():
    """It multiplies the squared error before the reduction, so the ratio holds."""
    cfg = _cfg(cfm_loss_weight_from_t="linear_1_to_0.01")
    pred = torch.zeros(2, 1, 1, 2)
    target = torch.ones(2, 1, 1, 2)          # squared error 1 everywhere
    t = torch.tensor([[ACTION_END], [NOISE_END]])

    loss = compute_cfm_loss(
        pred, target.expand(2, 1, 1, 2), cfg, "scalar", weight=cfm_loss_weight(t, cfg)
    )
    # weights are 1.0 and 0.01, so the mean of 1.0 and 0.01*1.0.
    assert float(loss) == pytest.approx((1.0 + 0.01) / 2)


def test_padded_tokens_are_still_zero_after_weighting():
    """The weight multiplies the masked square, so padding stays padded."""
    cfg = _cfg(cfm_loss_weight_from_t="linear_1_to_0.1")
    pred = torch.zeros(1, 1, 2, 1)
    target = torch.ones(1, 1, 2, 1)
    pad = torch.tensor([[False, True]])
    t = torch.full((1, 1, 2, 1), 0.5)

    loss = compute_cfm_loss(
        pred,
        target,
        cfg,
        "scalar",
        action_is_pad=pad,
        weight=cfm_loss_weight(t, cfg),
    )
    # Only one of the two tokens is executed; weight 0.55 on it, squared err 1.
    assert float(loss) == pytest.approx(0.55)


def test_an_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="cfm_loss_weight_from_t"):
        _cfg(cfm_loss_weight_from_t="linear_1_to_0.5").validate()