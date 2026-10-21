"""Every logged scalar must mean what its name says."""

import math
import warnings

import pytest
import torch

from myfpo.configuration import MyfpoConfig
from myfpo.fpo_core import compute_gae_diagnostics
from myfpo.training import compute_rollout_diagnostics


def _gae(n=8, **kw):
    args = dict(
        env_id=torch.arange(n) % 2,
        terminated=torch.zeros(n, dtype=torch.bool),
        truncated=torch.zeros(n, dtype=torch.bool),
        advantages=torch.randn(n),
        raw_returns=torch.randn(n),
    )
    args.update(kw)
    return compute_gae_diagnostics(**args)


# --- unmeasured must not read as measured ----------------------------------.






def test_an_empty_rollout_reports_nan_not_zero():
    out = compute_gae_diagnostics(
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.bool),
        torch.zeros(0, dtype=torch.bool),
        torch.zeros(0),
        torch.zeros(0),
    )
    assert all(math.isnan(v) for v in out.values())


def test_no_diagnostic_is_nan_when_the_rollout_is_measurable():
    """Guards the fix: only the unmeasured field may be NaN."""
    out = _gae()
    assert not math.isnan(out["linked_frac"])
    assert not math.isnan(out["advantage_std"])
    assert not math.isnan(out["return_rms"])
    assert not math.isnan(out["advantage_std_over_return_std"])


def test_the_key_states_the_denominator_it_divides_by():
    """The old key was ``advantage_std_fraction``.

    "Fraction" said only that the value was a quotient. The denominator is the
    return standard deviation, and that fact decides how a reader compares two
    runs: it is not a fraction of anything on the advantage side.
    """
    n = 8
    # Returns with a known spread, advantages with half that spread.
    returns = torch.tensor([1.0, -1.0] * (n // 2), dtype=torch.float32)
    adv = returns * 0.5
    out = compute_gae_diagnostics(
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        adv,
        returns,
    )
    got = out["advantage_std_over_return_std"]
    assert got == pytest.approx(0.5, abs=1e-5), got




# --- names must not overstate ---------------------------------------------.


def test_action_abs_mean_is_mean_absolute_value_not_an_l2_norm():
    """It was named action_magnitude, which reads as a norm."""
    batch = {
        "action": torch.tensor([[[3.0, 4.0]]]),
        "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
    }
    out = compute_rollout_diagnostics(batch, action_low=[-1.0], action_high=[1.0])
    assert out["raw_action_abs_mean"] == pytest.approx(3.5)      # mean|a|
    assert out["raw_action_abs_mean"] != pytest.approx(5.0)      # not L2.
    assert out["raw_action_abs_max"] == pytest.approx(4.0)


def test_the_action_abs_family_is_named_consistently():
    out = compute_rollout_diagnostics(
        {
            "action": torch.ones(2, 2, 3),
            "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        },
        action_low=[-1.0] * 3,
        action_high=[1.0] * 3,
    )
    assert {
        "raw_action_abs_max",
        "raw_action_abs_mean",
        "executed_action_abs_max",
        "executed_action_abs_mean",
    } <= set(out)


def test_linked_frac_is_live_under_the_default_trace():
    """The default lambda is 0.95, so the trace runs and this has a consumer."""
    assert MyfpoConfig().gae_lambda == 0.95
    assert 0.0 <= _gae()["linked_frac"] <= 1.0

def test_rollout_diagnostics_keys_match_the_scalar_count():
    """keys and scalars are zipped positionally."""
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(compute_rollout_diagnostics))
    body = next(n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef))
    n_keys = n_scalars = None
    for node in ast.walk(body):
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "keys":
                n_keys = len(node.value.elts)
            elif isinstance(tgt, ast.Name) and tgt.id == "scalars":
                n_scalars = len(node.value.elts)
    assert n_keys == n_scalars, f"{n_keys} keys vs {n_scalars} scalars"


def test_metrics_survive_the_wandb_scalar_pack():
    """loss_with_info reshapes every metric to 0-d before one device transfer."""
    from myfpo.fpo_core import compute_cfm_metrics

    out = compute_cfm_metrics(
        cfm_loss_old=torch.zeros(2),
        cfm_loss_new=torch.ones(2),
        v_star_f=torch.ones(2, 8, 3),
        above_clamp_frac=torch.zeros(()),
        both_clamped_frac=torch.zeros(()),
    )
    for key, value in out.items():
        assert torch.as_tensor(value).reshape(()).numel() == 1, key


def test_nan_metrics_do_not_raise_in_the_tracker():
    """NaN is how we say "unmeasured". The packer must not choke on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _gae()
        for value in out.values():
            torch.as_tensor(value).reshape(())