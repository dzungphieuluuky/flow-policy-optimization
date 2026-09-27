"""The CFM loss must ignore action rows a chunk never executed, under the DCT."""

import torch

from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.dct_utils import extract_dct_coeffs, velocity_to_action_steps
from sjfpo.fpo_core import compute_cfm_loss
from sjfpo.modeling_sjfpo import SJFPOPolicy

HORIZON = 16
ACTION_DIM = 3
MC = 4


def _cfg(**kw):
    import dataclasses

    base = dict(
        use_dct_transform=True,
        num_kept_tokens=HORIZON,
        num_mc_samples=MC,
        action_horizon=HORIZON,
    )
    return dataclasses.replace(
        SJFPOConfig(
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (11,))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
        ),
        **{**base, **kw},
    )


def _mask(n, executed):
    return (torch.arange(HORIZON).unsqueeze(0) >= executed).expand(n, HORIZON)


def test_rows_preserve_norm_at_full_resolution():
    """K == H means the basis is orthonormal, so expansion is a rotation."""
    torch.manual_seed(0)
    coeffs = torch.randn(3, HORIZON, ACTION_DIM)
    assert torch.allclose(
        velocity_to_action_steps(coeffs, HORIZON).pow(2).sum(),
        coeffs.pow(2).sum(),
        rtol=1e-5,
    )


def test_row_mask_zeroes_the_unexecuted_tail():
    """A residual living only on the padded tail must cost nothing."""
    cfg = _cfg()
    n, executed = 2, 4
    pred = torch.zeros(n, MC, HORIZON, ACTION_DIM)
    target = torch.zeros(n, MC, HORIZON, ACTION_DIM)
    # Put the whole error on rows [executed:], which the chunk never ran.
    target[:, :, executed:, :] = 5.0
    loss = compute_cfm_loss(
        pred, target, cfg, "sample", _mask(n, executed).to(torch.bool)
    )
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_row_mask_keeps_the_executed_error():
    cfg = _cfg()
    n, executed = 2, 4
    pred = torch.zeros(n, MC, HORIZON, ACTION_DIM)
    target = torch.zeros(n, MC, HORIZON, ACTION_DIM)
    target[:, :, :executed, :] = 1.0
    loss = compute_cfm_loss(
        pred, target, cfg, "sample", _mask(n, executed).to(torch.bool)
    )
    assert torch.allclose(loss, torch.ones_like(loss))


def test_unchanged_when_nothing_is_truncated():
    """K == H and a full mask: expansion is a rotation, so the loss is identical.

    This is what makes the change safe. When no chunk is cut short the mask is a
    no-op and expansion preserves the squared norm, so the loss the ratio is built
    from is bit-for-bit the loss it was before.
    """
    cfg = _cfg()
    n, mc = 2, 1
    torch.manual_seed(1)
    coeff_pred = torch.randn(n, mc, HORIZON, ACTION_DIM)
    coeff_target = torch.randn(n, mc, HORIZON, ACTION_DIM)
    full = torch.zeros(n, HORIZON, dtype=torch.bool)  # nothing padded

    spectral = compute_cfm_loss(coeff_pred, coeff_target, cfg, "sample", full)
    rowed = compute_cfm_loss(
        velocity_to_action_steps(coeff_pred, HORIZON),
        velocity_to_action_steps(coeff_target, HORIZON),
        cfg,
        "sample",
        full,
    )
    assert torch.allclose(spectral, rowed, rtol=1e-5, atol=1e-6)


def test_row_mask_differs_from_coefficient_mask():
    """The mask is not rotation-equivariant, which is why the cut belongs in rows.

    A row-space mask and a coefficient-space mask select genuinely different
    energy, so this cannot be done by masking coefficients.
    """
    cfg = _cfg()
    n, mc, executed = 2, 1, 4
    torch.manual_seed(1)
    pred = torch.randn(n, mc, HORIZON, ACTION_DIM)
    target = torch.randn(n, mc, HORIZON, ACTION_DIM)
    mask = _mask(n, executed).to(torch.bool)
    in_rows = compute_cfm_loss(
        velocity_to_action_steps(pred, HORIZON),
        velocity_to_action_steps(target, HORIZON),
        cfg,
        "sample",
        mask,
    )
    in_coeffs = compute_cfm_loss(pred, target, cfg, "sample", mask)
    assert not torch.allclose(in_rows, in_coeffs, rtol=1e-3)


def test_dct_basis_rows_are_dense_so_the_tail_is_not_the_high_bins():
    """Every coefficient spreads over every row, so no bin is 'the tail'.

    Justifies the row-space cut: a single highest-frequency coefficient, which is
    the one the old code dropped under ``num_kept_tokens``, still puts energy on
    rows the chunk never executed.
    """
    residual = torch.zeros(1, 1, HORIZON, 1)
    residual[0, 0, -1, 0] = 1.0  # highest-frequency coefficient only
    rows = velocity_to_action_steps(residual, HORIZON)[0, 0, :, 0]
    executed = 4
    head, tail = rows[:executed].abs().sum(), rows[executed:].abs().sum()
    # The dropped 'high-frequency' bin puts most of its row-space energy on rows
    # the chunk never executed, so dropping the bin did not drop the tail.
    assert tail > 3 * head


def test_truncated_rows_are_the_only_thing_dropped():
    """Expanding then masking must not change the executed rows' contribution."""
    cfg = _cfg()
    n, executed = 2, 6
    torch.manual_seed(2)
    pred = torch.randn(n, MC, HORIZON, ACTION_DIM)
    target = torch.randn(n, MC, HORIZON, ACTION_DIM)
    rows = velocity_to_action_steps(pred - target, HORIZON)
    keep = (~_mask(n, executed).to(torch.bool)).float()[:, None, :, None]
    assert torch.allclose((rows * keep).sum(-1).sum(-1), rows[:, :, :executed].sum(-1).sum(-1))


def test_dct_off_path_is_untouched_by_the_helper():
    """Without the DCT the velocities already are the action steps, so the mask
    applies as-is and nothing is expanded."""
    cfg = _cfg(use_dct_transform=False)
    p = SJFPOPolicy(cfg)
    pred, target = torch.randn(2, HORIZON, ACTION_DIM), torch.randn(2, HORIZON, ACTION_DIM)
    is_pad = _mask(2, 4).to(torch.bool)
    mask, (out_pred, out_target) = p.prepare_cfm_loss_inputs(cfg, is_pad, pred, target)
    assert out_pred is pred and out_target is target
    assert mask is is_pad


def test_operands_helper_expands_with_dct():
    cfg = _cfg()
    p = SJFPOPolicy(cfg)
    pred = torch.randn(2, MC, HORIZON, ACTION_DIM)
    target = torch.randn(2, 1, HORIZON, ACTION_DIM)
    is_pad = _mask(2, 4).to(torch.bool)
    mask, (out_pred, out_target) = p.prepare_cfm_loss_inputs(cfg, is_pad, pred, target)
    assert out_pred.shape == (2, MC, HORIZON, ACTION_DIM)
    assert out_target.shape == (2, 1, HORIZON, ACTION_DIM)
    assert mask is is_pad
    # Linear, so expanding the operands matches expanding the difference.
    assert torch.allclose(
        out_pred - out_target,
        velocity_to_action_steps(pred - target, HORIZON),
        rtol=1e-5,
        atol=1e-6,
    )


def test_mask_and_layout_cannot_disagree():
    """DCT on with the mask off must return no mask, not a mask on coefficients.

    The two spaces share a shape at num_kept_tokens == action_horizon, so a mask
    paired with unexpanded velocities is applied to the frequency axis and the
    loss quietly means something else.
    """
    cfg = _cfg(cfm_mask_unexecuted=False)
    p = SJFPOPolicy(cfg)
    pred, target = torch.randn(2, MC, HORIZON, ACTION_DIM), torch.randn(2, 1, HORIZON, ACTION_DIM)
    mask, (out_pred, _) = p.prepare_cfm_loss_inputs(cfg, _mask(2, 4).to(torch.bool), pred, target)
    assert mask is None
    assert out_pred.shape == pred.shape  # left as coefficients
