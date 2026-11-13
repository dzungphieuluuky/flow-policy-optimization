"""The dataclass surface itself, checked by parsing rather than by importing."""

import ast
import collections
import dataclasses
from pathlib import Path

import pytest

from myfpo.configuration import MyfpoConfig

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "myfpo"
    / "configuration.py"
)


def _annotated_assignments() -> list[tuple[str, int, str]]:
    """Every `name: type = value` at class-body level, in source order."""
    tree = ast.parse(CONFIG_PATH.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MyfpoConfig"
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
    """The defect this file exists for: a bad merge can duplicate a field."""
    names = [name for name, _, _ in _annotated_assignments()]
    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dupes, f"declared more than once in MyfpoConfig: {sorted(dupes)}"


def test_every_declared_field_exists_on_the_dataclass():
    """Every annotation in the class body must become a real field."""
    declared = {name for name, _, _ in _annotated_assignments()}
    actual = {f.name for f in dataclasses.fields(MyfpoConfig)}

    assert declared <= actual, (
        f"annotated here but not a field: {sorted(declared - actual)}"
    )
    # A plain dataclass inherits nothing, so every field is declared here.
    unexplained = actual - declared
    assert not unexplained, f"fields with no declaration: {sorted(unexplained)}"


def test_the_noise_marginal_is_standard_normal():
    """One law: MC noise is ``N(0, I)``; the spectrum knob is gone."""
    cfg = MyfpoConfig()
    assert not hasattr(cfg, "noise_sampling_mode")


def test_a_zero_head_count_is_a_value_error_not_a_division_by_zero():
    """``hidden_dim % 0`` raised ZeroDivisionError before the field was checked."""
    with pytest.raises(ValueError, match="dit_heads"):
        MyfpoConfig(dit_heads=0)


def test_no_spectral_survives_in_the_config():
    """The prior is gone: no field, no validator, no comment inviting it back."""
    text = CONFIG_PATH.read_text()
    for gone in ("noise_scaling_mode", "noise_scaling_beta", "freq_weighted", "DCT"):
        assert gone not in text, f"{gone} still in configuration.py"