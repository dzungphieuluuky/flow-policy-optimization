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

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "spectral_joint_policy"
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


def test_the_noise_knobs_are_two_distinct_axes():
    """noise_sampling_mode is the marginal, noise_scaling_mode the spectrum.

    They look redundant next to each other and are not: one picks the
    distribution, the other its 1/f^beta profile.
    """
    cfg = SJFPOConfig()
    assert cfg.noise_sampling_mode == "normal"
    assert cfg.noise_scaling_mode == "white"
    # Setting one must not disturb the other.
    other = dataclasses.replace(cfg, noise_scaling_mode="freq_weighted_time")
    assert other.noise_sampling_mode == "normal"
    flipped = dataclasses.replace(cfg, noise_sampling_mode="uniform")
    assert flipped.noise_scaling_mode == "white"


def test_stale_dct_references_are_gone_from_the_field_block():
    """The DCT was removed; a comment inviting its use back is a live trap."""
    text = CONFIG_PATH.read_text()
    block = text.split("noise_sampling_mode: str", 1)[0]
    assert "DCT" not in block, "field-block comment still describes a removed DCT"
    # The one place that must still name it is the migration error itself.
    assert "was the DCT-coefficient mode" in text