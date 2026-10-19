"""The flow time is ``t``. It must not be spelled ``tau`` again.

The package used both letters for the same quantity. ``cfm_loss_weight`` and
``velocity_to_target_param`` took ``t``, while ``broadcast_over_mc_axis``,
``compute_predicted_velocity`` and the batch key took ``tau``. A reader had to
check both spellings to know whether one argument or two was involved.

``eps`` keeps its own name. That is the letter the flow-matching paper uses for
the base sample, and this repo follows the paper there.
"""

import ast
import inspect
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "myfpo"
FILES = list(SRC.rglob("*.py"))


def _identifiers(path):
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, (ast.arg, ast.keyword)) and getattr(node, "arg", None):
            found.add(node.arg)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def test_no_module_binds_or_names_tau_anymore():
    """One quantity, one letter."""
    for path in FILES:
        names = _identifiers(path)
        offenders = {n for n in names if n == "tau" or n.endswith("_tau")}
        assert not offenders, f"{path.name} still uses {offenders}"


def test_the_batch_key_pairs_with_eps():
    """``mc_t`` next to ``mc_eps``, so the pair reads as one draw."""
    src = (SRC / "modeling.py").read_text()
    assert '"mc_t"' in src and '"mc_eps"' in src
    assert "mc_t" in src and "mc_eps" in src and "Mandatory paired reuse" in src


def test_broadcast_over_mc_axis_takes_action_eps_and_t():
    """The argument named ``a`` is the action. ``action`` says so."""
    import inspect

    from myfpo.fpo_core import broadcast_over_mc_axis

    params = list(inspect.signature(broadcast_over_mc_axis).parameters)
    assert params == ["action", "eps", "t"], params


def test_eps_keeps_its_paper_name():
    """The base sample stays ``eps``. This is the one name the rename keeps."""
    from myfpo.fpo_core import broadcast_over_mc_axis, sample_noise

    assert "eps" in _identifiers(SRC / "fpo_core.py")
    assert "eps" in inspect.signature(broadcast_over_mc_axis).parameters
    assert "shape" in inspect.signature(sample_noise).parameters


def test_the_two_sampling_helpers_are_named_for_what_they_return():
    """``sample_timesteps`` draws. ``sample_mc_timesteps`` shapes for the batch."""
    from myfpo.fpo_core import sample_timesteps
    from myfpo.metrics import sample_mc_timesteps

    assert sample_timesteps.__name__ == "sample_timesteps"
    assert sample_mc_timesteps.__name__ == "sample_mc_timesteps"
    # The wrapper adds the trailing axis and the MC axes.
    out = sample_mc_timesteps(
        __import__("myfpo.configuration", fromlist=["MyfpoConfig"]).MyfpoConfig(),
        2,
        "cpu",
    )
    assert out.dim() == 4 and out.shape[-1] == 1, out.shape


def test_the_only_tau_left_is_the_note_that_records_the_rename():
    """One mention survives, and it is the docstring that explains the change."""
    hits = []
    for path in FILES:
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "tau" in line and "was also spelled" not in line:
                hits.append(f"{path.name}:{i}: {line.strip()}")
    assert not hits, "unexpected tau mentions:\n" + "\n".join(hits)