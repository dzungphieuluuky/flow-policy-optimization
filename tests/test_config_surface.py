"""The dataclass surface itself, checked by parsing rather than by importing.

A field declared twice is invisible at runtime: __annotations__ collapses to one
key, dataclasses.fields() reports one field, the generated __init__ takes one
argument, and the YAML parity test sees one key. Every existing check passed
while noise_sampling_mode sat on lines 49 and 53, both reading "normal". Only
the source said otherwise, so that is what this reads.
"""

import ast
import collections
import dataclasses
from pathlib import Path

from myfpo.configuration_sjfpo import SJFPOConfig

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "myfpo"
    / "configuration_sjfpo.py"
)


def _annotated_assignments() -> list[tuple[str, int, str]]:
    """Every `name: type = value` at class-body level, in source order."""
    tree = ast.parse(CONFIG_PATH.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SJFPOConfig"
    )
    out = []
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(
                (
                    node.target.id,
                    node.lineno,
                    ast.unparse(node.annotation) if node.annotation else "",
                )
            )
    return out


def test_no_config_field_is_declared_twice():
    """The defect this file exists for: a bad merge can duplicate a field.

    A duplicate with the same default is a no-op; a duplicate with a different
    default silently wins by source order, and nothing else in the suite
    notices.
    """
    names = [name for name, _, _ in _annotated_assignments()]
    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dupes, f"declared more than once in SJFPOConfig: {sorted(dupes)}"


def test_every_declared_field_exists_on_the_dataclass():
    """Every annotation in the class body must become a real field.

    A duplicate collapses to one field, and a stray annotation would be
    silently absent from the generated __init__.
    """
    from lerobot.configs import PreTrainedConfig

    declared = {name for name, _, _ in _annotated_assignments()}
    actual = {f.name for f in dataclasses.fields(SJFPOConfig)}
    inherited = {f.name for f in dataclasses.fields(PreTrainedConfig)}

    assert declared <= actual, (
        f"annotated here but not a field: {sorted(declared - actual)}"
    )
    # Anything that is a field but not annotated here must be inherited, or
    # this file has drifted from the dataclass it claims to describe.
    unexplained = actual - declared - inherited
    assert not unexplained, f"fields with no declaration and no base: {sorted(unexplained)}"


def test_the_noise_marginal_is_normal_and_uniform_only():
    """One knob left: which distribution, not its spectrum."""
    cfg = SJFPOConfig()
    assert cfg.noise_sampling_mode == "normal"
    flipped = dataclasses.replace(cfg, noise_sampling_mode="uniform")
    assert flipped.noise_sampling_mode == "uniform"


def test_no_spectral_survives_in_the_config():
    """The prior is gone: no field, no validator, no comment inviting it back."""
    text = CONFIG_PATH.read_text()
    for gone in ("noise_scaling_mode", "noise_scaling_beta", "freq_weighted", "DCT"):
        assert gone not in text, f"{gone} still in configuration_sjfpo.py"