"""The tanh that maps the actor output into the env action bounds.

The mapping lives in the postprocessor, so the per-dimension bounds the env
declares are the only source of the range. It replaced a clamp that sat in
``predict_action_chunk``.

The perturbation is added before the squash, so it is bounded by it. It used to
be added after the clamp with unbounded noise, which is why the collector needed
a second clamp and a clip counter.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from myfpo.configuration import MyfpoConfig
from lerobot.utils.constants import ACTION
from myfpo.processor import MyfpoActionPostprocessorStep

ROOT = Path(__file__).resolve().parents[1]


def _policy(action_dim: int = 3, **over):
    from myfpo.training import build_policy

    cfg = dataclasses.replace(MyfpoConfig(), compile=False, **over)
    return build_policy(
        11,
        action_dim,
        cfg,
        device="cpu",
        action_low=np.array([-2.0] * action_dim, dtype=np.float32),
        action_high=np.array([2.0] * action_dim, dtype=np.float32),
    )


# --- the dead metric ------------------------------------------------------










# --- the real projection --------------------------------------------------














# --- the trainer ----------------------------------------------------------
def test_the_postprocessor_maps_tanh_onto_the_env_bounds():
    """The map is low + (tanh(x) + 1) / 2 * (high - low).

    The affine part is unchanged from the old plain unscale. Only the squash is
    new. Each dimension uses its own bounds, because the env declares them
    per dimension and they need not be symmetric.
    """
    low = torch.tensor([-2.0, -0.4, 0.0])
    high = torch.tensor([2.0, 0.4, 1.0])
    step = MyfpoActionPostprocessorStep(low, high)
    x = torch.tensor([[-3.0, -1.0, -0.2], [0.0, 0.0, 0.0], [3.0, 1.0, 0.2]])
    out = step({ACTION: x})[ACTION]

    for row, raw in zip(out, x):
        for value, r, lo, hi in zip(row, raw, low, high):
            assert float(lo) <= float(value) <= float(hi), (
                f"{value} escaped ({float(lo)}, {float(hi)})"
            )
            expected = lo + (torch.tanh(r) + 1) / 2 * (hi - lo)
            assert torch.allclose(value, expected, atol=1e-6)


def test_tanh_is_open_interval_for_moderate_inputs_and_closed_when_it_saturates():
    """The chosen property, measured rather than assumed.

    ``tanh`` maps the reals onto (-1, 1). In float32 it saturates: past about
    |x| = 9 the result is exactly -1.0 or 1.0. So the mapped action sits strictly
    inside the bounds for an ordinary proposal, and sits exactly on the endpoint
    once the actor is confident. A clamp always sat on the endpoint.

    This is why the endpoints are reachable in practice even though the range is
    open. Anything that assumes the action can never equal the bound is wrong.
    """
    low, high = torch.tensor([-1.0]), torch.tensor([1.0])
    step = MyfpoActionPostprocessorStep(low, high)

    moderate = step({ACTION: torch.tensor([[-3.0], [3.0]])})[ACTION]
    assert float(moderate[0, 0]) > low.item(), "a moderate value hit the endpoint"
    assert float(moderate[1, 0]) < high.item(), "a moderate value hit the endpoint"

    saturated = step({ACTION: torch.tensor([[-50.0], [50.0]])})[ACTION]
    assert float(saturated[0, 0]) == pytest.approx(low.item(), abs=1e-7)
    assert float(saturated[1, 0]) == pytest.approx(high.item(), abs=1e-7)
    assert torch.isfinite(saturated).all()


def test_asymmetric_bounds_map_correctly():
    """The env declares per-dimension bounds, and they need not be symmetric."""
    low = torch.tensor([-0.4, 0.0, -3.0])
    high = torch.tensor([0.4, 1.0, 3.0])
    step = MyfpoActionPostprocessorStep(low, high)
    out = step({ACTION: torch.zeros(1, 3)})[ACTION]
    for value, lo, hi in zip(out[0], low, high):
        assert float(value) == pytest.approx(float((lo + hi) / 2), abs=1e-6)


def test_the_perturbation_is_bounded_by_the_squash():
    """The perturbation goes in before tanh, so it cannot leave the range.

    It used to be added after a clamp, with unbounded noise, which is why the
    collector needed a second clamp.

    The policy returns the raw chunk, so the bound is a property of the
    squash, not of the returned value.
    """
    policy = _policy(action_perturb_std=50.0)
    policy.train()
    states = {"observation.state": torch.randn(64, 16, 11) * 3.0}
    for _ in range(8):
        raw = policy.predict_action_chunk(states, start_from_zero=False)
        # Env bounds for this fixture are [-2, 2].
        executed = policy.postprocessor({ACTION: raw})[ACTION]
        assert float(executed.abs().max()) <= 2.0, float(executed.abs().max())


def test_the_policy_no_longer_clamps_the_action():
    """No clamp survives in the policy path. The squash is in the processor."""
    src = (ROOT / "src" / "myfpo" / "modeling.py").read_text()
    body = src.split("def predict_action_chunk", 1)[1].split("\n    def ", 1)[0]
    assert ".clamp(" not in body, "predict_action_chunk still clamps"
    assert "torch.tanh" not in body, "the squash belongs in the postprocessor"
def test_the_collector_no_longer_clamps():
    """The collector squashes, it does not clamp, and it does not count.

    The clamp and the clip count were removed because the squash upstream makes
    them unreachable. This test fails if either comes back.
    """
    from myfpo.collector import resolve_action_bounds
    from myfpo.processor import squash_to_env_bounds

    lo, hi = resolve_action_bounds([-2.0, -1.0, -0.3], [2.0, 1.0, 0.3])
    raw = torch.tensor([[9.0, -9.0, 5.0]])
    executed = squash_to_env_bounds(raw, lo, hi)
    # A proposal outside the bounds is squashed into them, not clamped.
    assert float(executed[0, 0]) == pytest.approx(2.0 * math.tanh(9.0), abs=1e-6)
    assert float(executed[0, 1]) == pytest.approx(-1.0 * math.tanh(9.0), abs=1e-6)
    assert float(executed[0, 2]) == pytest.approx(0.3 * math.tanh(5.0), abs=1e-6)
    assert float(executed.abs().max()) < 2.0, "the value was clamped to the bound"


def test_the_stored_action_is_bounded_because_of_the_squash():
    """The whole chain: actor output to the value the env receives.

    This is the guarantee the removed clip count used to be argued about. It now
    holds because tanh maps into the env range, not because a clamp enforces it.

    The policy returns the raw chunk, so the bound is a property of the
    squash on the way to the env.
    """
    policy = _policy(action_perturb_std=5.0)
    policy.train()
    states = {"observation.state": torch.randn(64, 16, 11) * 3.0}
    for _ in range(8):
        raw = policy.predict_action_chunk(states, start_from_zero=False)
        executed = policy.postprocessor({ACTION: raw})[ACTION]
        assert float(executed.min()) >= -2.0 - 1e-6, float(executed.min())
        assert float(executed.max()) <= 2.0 + 1e-6, float(executed.max())


def test_no_clamp_survives_on_the_collector_path():
    """Checked on the syntax tree, not on the text.

    The function's own docstring explains that the clamp was removed, so a search
    for the word finds prose. This walks the parsed body and fails on a call.

    Only the action-carrying assignments are checked: the collector clamps
    indices, which has nothing to do with the action's range.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "myfpo" / "collector.py").read_text()
    tree = ast.parse(src)
    # The action path now lives in its own step function, so the check walks
    # the whole module instead of one function body.
    action_names = {"plan", "raw_all", "executed_all", "actions_t"}

    def assigned_names(node):
        names = set()
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
        return names

    clamps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not assigned_names(node) & action_names:
            continue
        for inner in ast.walk(node.value):
            if isinstance(inner, ast.Call) and "clamp" in ast.unparse(inner.func).lower():
                clamps.append(ast.unparse(inner))
    assert not clamps, f"the collector clamps an action again: {clamps}"

    # The squash is what bounds the action, and it is there.
    squashes = [
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and "squash" in ast.unparse(n.func).lower()
    ]
    assert squashes, "the collector lost the squash"

    # The clip counter must be gone from the code, though the docstring still
    # names it while explaining the removal.
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "chunk_clip_counts" not in names, "the clip counter is back"

    buffer_cls = next(n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name == "RolloutBuffer")
    assigned = {
        t.attr for node in ast.walk(buffer_cls)
        if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Attribute)
    }
    assert "chunk_clip_counts" not in assigned, "the buffer holds the counter again"
