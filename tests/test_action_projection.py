"""The action projection, and the metric that could not see it.

``predict_action_chunk`` clamps the sampled chunk to [-1, 1] before the
postprocessor maps it onto the env bounds. That clamp is the projection the CFM
target is built on, so it enters the objective.

The tracker measured a different clamp. ``chunk_clip_counts`` comes from the
collector, which only ever receives in-bounds values, so it read zero. A metric
that always reads zero looks like evidence of health.

This file pins both facts.
"""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.processor_sjfpo import SJFPOActionPostprocessorStep

ROOT = Path(__file__).resolve().parents[1]


def _policy(action_dim: int = 3, **over):
    from myfpo.train_sjfpo import build_policy

    cfg = dataclasses.replace(SJFPOConfig(), compile=False, **over)
    return build_policy(
        11,
        action_dim,
        cfg,
        device="cpu",
        action_low=np.array([-2.0] * action_dim, dtype=np.float32),
        action_high=np.array([2.0] * action_dim, dtype=np.float32),
    )


# --- the dead metric ------------------------------------------------------


def test_the_postprocessor_maps_the_clamped_range_onto_the_env_bounds():
    """The step that makes the collector's count unreachable.

    The policy clamps to [-1, 1]. The postprocessor then computes
    ``bias + scale * x`` with ``scale = (high - low) / 2``. So a value inside
    [-1, 1] lands inside [low, high], with the endpoints exact.
    """
    low = torch.tensor([-2.0, -1.0])
    high = torch.tensor([2.0, 3.0])
    post = SJFPOActionPostprocessorStep(low, high)
    for x in (-1.0, 0.0, 1.0):
        out = post({"action": torch.tensor([[x, x]])})["action"]
        assert bool((out >= low).all()) and bool((out <= high).all()), (
            f"x={x} left the env bounds, so the collector could see a clip"
        )
    # The endpoints are exact, so no float slack lets a value escape.
    assert torch.allclose(post({"action": torch.ones(1, 2)})["action"], high)
    assert torch.allclose(post({"action": -torch.ones(1, 2)})["action"], low)


def test_the_collector_clip_count_is_structurally_zero():
    """The defect: an out-of-bounds value cannot reach ``scale_and_clip_chunk``."""
    from myfpo.collector import scale_and_clip_chunk

    low = torch.tensor([-2.0, -2.0])
    high = torch.tensor([2.0, 2.0])
    # ``scale_and_clip_chunk`` takes a range, so the argument is high - low.
    span = high - low
    # Anything the policy can emit lives in [-1, 1] before the postprocessor, and
    # the postprocessor output lands inside [low, high]. Feed the collector the
    # widest value that path can produce and check the count.
    scaled = torch.full((8, 2), 1.0)
    env_space = 0.5 * span * scaled + 0.5 * (high + low)
    assert bool((env_space >= low).all()) and bool((env_space <= high).all())
    _executed, _normalized, clip_count = scale_and_clip_chunk(env_space, low, span)
    assert int(clip_count) == 0, (
        "a value that reached the env bounds still got counted; the collector is "
        "reachable and the dead-metric claim no longer holds"
    )


def test_the_dead_metric_is_still_computed_but_must_read_zero():
    """``action_clip_frac`` stays for continuity. It cannot report a projection."""
    from myfpo.train_sjfpo import compute_action_clip_rate

    batch = {
        "chunk_clip_counts": torch.zeros(4, dtype=torch.int64),
        "action_is_pad": torch.zeros(4, 8, dtype=torch.bool),
        "action": torch.zeros(4, 8, 3),
    }
    rates = compute_action_clip_rate(batch)
    assert rates["action_clip_frac"] == 0.0
    assert rates["action_clip_count"] == 0.0


def test_the_real_metric_is_a_different_key():
    """The projection must not be read off the dead key."""
    from myfpo.train_sjfpo import measure_pre_clamp_projection

    rates = measure_pre_clamp_projection(_policy())
    assert "action_pre_clamp_frac" in rates
    assert "action_clip_frac" not in rates


# --- the real projection --------------------------------------------------


def test_the_policy_records_a_pre_clamp_fraction():
    policy = _policy()
    assert float(policy.action_pre_clamp_frac) == 0.0, "must start at zero"
    states = {"observation.state": torch.zeros(2, 16, 11)}
    policy.predict_action_chunk(states, start_from_zero=False)
    frac = float(policy.action_pre_clamp_frac)
    assert 0.0 <= frac <= 1.0


def test_an_untrained_policy_projects_a_third_of_its_coordinates():
    """The untrained floor: the velocity is zero, so the chunk is ``eps``.

    ``eps`` is a unit Gaussian, and a unit Gaussian exceeds 1 in absolute value
    on 31.7 percent of its mass. So the projection is large before training
    starts, not a corner case.
    """
    policy = _policy()
    torch.manual_seed(0)
    states = {"observation.state": torch.zeros(64, 16, 11)}
    policy.predict_action_chunk(states, start_from_zero=False)
    frac = float(policy.action_pre_clamp_frac)
    assert 0.25 < frac < 0.40, f"untrained projection fraction was {frac:.4f}"


def test_the_projection_grows_with_the_velocity():
    """A stronger velocity pushes the chunk further past the bounds."""
    policy = _policy()
    states = {"observation.state": torch.zeros(64, 16, 11)}
    policy.predict_action_chunk(states, start_from_zero=False)
    weak = float(policy.action_pre_clamp_frac)

    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        policy.actor.out.weight.add_(
            0.5 * torch.randn(policy.actor.out.weight.shape, generator=gen)
        )
        policy.actor.out.bias.add_(
            0.5 * torch.randn(policy.actor.out.bias.shape, generator=gen)
        )
    policy.predict_action_chunk(states, start_from_zero=False)
    strong = float(policy.action_pre_clamp_frac)
    assert strong > weak, f"{strong:.4f} did not exceed {weak:.4f}"
    assert strong > 0.7, f"a 0.5 perturbation only reached {strong:.4f}"


def test_a_zero_start_projects_less_than_a_sampled_start():
    """The zero start begins at the origin, so it lands inside the bounds."""
    policy = _policy()
    states = {"observation.state": torch.zeros(64, 16, 11)}
    policy.predict_action_chunk(states, start_from_zero=True)
    zero_start = float(policy.action_pre_clamp_frac)
    policy.predict_action_chunk(states, start_from_zero=False)
    sampled = float(policy.action_pre_clamp_frac)
    assert zero_start < sampled, f"zero {zero_start:.4f} !< sampled {sampled:.4f}"


def test_the_value_is_a_zero_dim_tensor():
    """The trainer stacks it with other scalars, so the rank must be zero."""
    policy = _policy()
    states = {"observation.state": torch.zeros(4, 16, 11)}
    policy.predict_action_chunk(states, start_from_zero=False)
    assert policy.action_pre_clamp_frac.dim() == 0
    assert torch.stack(
        [policy.action_pre_clamp_frac, policy.cfm_above_clamp_frac]
    ).shape == (2,)


def test_the_metric_survives_a_policy_without_the_attribute():
    """An old checkpoint or a stub must not break the trainer."""
    from myfpo.train_sjfpo import measure_pre_clamp_projection

    class _Stub:
        pass

    assert measure_pre_clamp_projection(_Stub()) == {}


# --- the trainer ----------------------------------------------------------


def test_the_trainer_logs_the_real_projection_key():
    spec = importlib.util.spec_from_file_location(
        "train_mod", ROOT / "src" / "myfpo" / "train_sjfpo.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect

    src = inspect.getsource(mod)
    assert "measure_pre_clamp_projection(policy)" in src
    assert "action_clip_rates.update(" in src


def test_the_probe_script_exists_and_names_the_real_clamp():
    src = (ROOT / "scripts" / "projection_probe.py").read_text()
    assert "x0.clamp(-1.0, 1.0)" in src
    assert "structurally zero" in src


def test_the_clamp_still_runs_on_the_policy_path():
    """The behaviour must not change. The measurement sits beside the clamp."""
    src = (ROOT / "src" / "myfpo" / "modeling_sjfpo.py").read_text()
    assert "scaled = action_chunk.clamp(-1.0, 1.0)" in src
    # The measurement must read the pre-clamp tensor, or it would always read
    # zero. The tensor was renamed from x0, which read as the chain start.
    assert "self.action_pre_clamp_frac = (action_chunk.abs() > 1.0)" in src


def test_the_projection_probe_reports_a_monotone_curve():
    """The probe is the tool for this item, so pin its shape."""
    spec = importlib.util.spec_from_file_location(
        "probe", ROOT / "scripts" / "projection_probe.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert np.isnan(probe.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])), (
        "a constant series must not rank as a trend"
    )
    assert probe.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert probe.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert np.isnan(probe.spearman([1.0], [2.0])), "a short series has no rank"

def test_the_perturbation_cannot_leave_the_unit_interval():
    """The second clamp. Without it, chunk_clip_counts is not structural.

    The noise is unbounded, so a large ``action_perturb_std`` used to push the
    returned action past the env bounds. The collector would then clip it, and
    the count that ``action_pre_clamp_frac`` was said to make structural would
    read non-zero.
    """
    policy = _policy(action_perturb_std=0.9)
    policy.train()
    states = {"observation.state": torch.randn(64, 16, 11) * 3.0}
    # The env bounds for this fixture are [-2, 2]. The returned action is in env
    # units, not the normalised unit interval, so the bound to check is the env
    # one. Exceeding it is what makes the collector clip, and a clip is what
    # ``chunk_clip_counts`` counts.
    for _ in range(8):
        out = policy.predict_action_chunk(states, start_from_zero=False)
        assert out.shape[-1] == 3
        assert float(out.abs().max()) <= 2.0 + 1e-6, (
            f"perturbed action left the env bounds: {float(out.abs().max())}"
        )


def test_the_projection_metric_still_measures_the_first_clamp():
    """Moving the clamp must not move the metric it names.

    ``action_pre_clamp_frac`` reads ``action_chunk``, before either clamp. The
    second clamp is invisible to it by design.
    """
    policy = _policy(action_perturb_std=0.0)
    policy.train()
    states = {"observation.state": torch.randn(64, 16, 11) * 3.0}
    policy.predict_action_chunk(states, start_from_zero=False)
    frac = float(policy.action_pre_clamp_frac)
    assert 0.0 < frac <= 1.0, frac
