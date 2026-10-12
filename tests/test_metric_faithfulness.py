"""Every logged scalar must mean what its name says.

Two failure modes, both of which have actually happened here:

  * a key that reports 0.0 because nothing measured it, so a clean line reads
    as a clean measurement -- ``budget_truncated_chunk_frac`` did exactly this,
    because ``mark_incomplete`` is a no-op so no budget-cut flag exists;
  * a key whose name promises a quantity its expression does not compute --
    ``action_magnitude`` was mean|a|, which reads as an L2 norm.

The arithmetic is checked elsewhere. This file checks the naming contract and
the unmeasured case, which no numeric assertion can catch.
"""

import math
import warnings

import pytest
import torch

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import compute_gae_diagnostics
from spectral_joint_policy.train_sjfpo import compute_rollout_diagnostics


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


# --- unmeasured must not read as measured ----------------------------------


def test_budget_truncated_is_nan_when_the_collector_cannot_supply_it():
    """The metric that motivated this file.

    mark_incomplete is a no-op, so chunk_budget_truncated does not exist and
    this quantity is unmeasured. 0.0 would assert "no chunk was cut short by the
    budget", which is a claim about the data.
    """
    out = _gae()
    assert math.isnan(out["budget_truncated_chunk_frac"])


def test_budget_truncated_is_a_real_fraction_when_supplied():
    supplied = torch.tensor([True] * 4 + [False] * 4)
    out = _gae(budget_truncated=supplied)
    assert out["budget_truncated_chunk_frac"] == pytest.approx(0.5)


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
    assert not math.isnan(out["advantage_std_fraction"])


# --- names must not overstate ---------------------------------------------


def test_action_abs_mean_is_mean_absolute_value_not_an_l2_norm():
    """It was named action_magnitude, which reads as a norm."""
    batch = {
        "action": torch.tensor([[[3.0, 4.0]]]),
        "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
    }
    out = compute_rollout_diagnostics(batch)
    assert out["action_abs_mean"] == pytest.approx(3.5)          # mean|a|
    assert out["action_abs_mean"] != pytest.approx(5.0)          # not L2
    assert out["action_abs_max"] == pytest.approx(4.0)


def test_the_action_abs_family_is_named_consistently():
    out = compute_rollout_diagnostics(
        {
            "action": torch.ones(2, 2, 3),
            "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        }
    )
    assert {
        "action_abs_max",
        "action_abs_max_executed",
        "action_abs_mean",
        "action_abs_mean_executed",
    } <= set(out)


def test_linked_frac_is_live_under_the_default_trace():
    """The default lambda is 0.95, so the trace runs and this has a consumer."""
    assert SJFPOConfig().gae_lambda == 0.95
    assert 0.0 <= _gae()["linked_frac"] <= 1.0


# --- dead logging ---------------------------------------------------------


def test_the_spectral_energy_key_is_gone_from_the_batch():
    """d0f97c2 removed its source; a passthrough slot for it is dead weight."""
    import inspect

    import spectral_joint_policy.train_sjfpo as T

    src = inspect.getsource(T)
    assert "compute_spectral_energy_retained" not in src


def test_trainer_emits_no_key_it_cannot_compute():
    """A guarded assignment that never fires looks configured but logs nothing."""
    import ast
    import inspect

    import spectral_joint_policy.train_sjfpo as T

    tree = ast.parse(inspect.getsource(T))
    dead = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if not (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)):
                continue
            if t.value.id != "data":
                continue
            val = ast.unparse(node.value)
            # A key read straight out of the batch and never produced upstream.
            if isinstance(node.value, ast.Subscript) and "batch" in val:
                inner = val.split('"')[-1].split("'")[-1]
                if inner in {
                    "compute_spectral_energy_retained",
                }:
                    dead.append(inner)
    assert not dead, f"trainer logs keys nothing supplies: {dead}"


def test_rollout_diagnostics_keys_match_the_scalar_count():
    """keys and scalars are zipped positionally.

    A length mismatch does not raise -- zip truncates -- it silently drops or
    mislabels a metric. step_delta rides along here and is consumed by the
    global-step accumulator rather than logged, so it is expected.
    """
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
    """compute_info reshapes every metric to 0-d before one device transfer.

    A non-scalar that reaches the tracker fails the whole update.
    """
    from spectral_joint_policy.modeling_sjfpo import compute_cfm_metrics
    from spectral_joint_policy.fpo_core import sample_noise

    eps = sample_noise((2, 2, 8, 3), SJFPOConfig(), torch.device("cpu"), None)
    out = compute_cfm_metrics(
        l_old=torch.zeros(2),
        l_new=torch.ones(2),
        v_star_f=torch.ones(2, 8, 3),
        cfg=SJFPOConfig(),
        eps=eps,
    )
    for key, value in out.items():
        assert torch.as_tensor(value).reshape(()).numel() == 1, key


def test_nan_metrics_do_not_raise_in_the_tracker():
    """NaN is how we say "unmeasured"; the packer must not choke on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _gae()
        for value in out.values():
            torch.as_tensor(value).reshape(())