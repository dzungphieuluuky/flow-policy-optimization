"""Shared stateless utilities (no mfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import math

import torch


def safe_float_value(x: torch.Tensor | float) -> float:
    """Convert to float, replacing non-finite values with nan.

    Args:
        x: Tensor or float value to convert.

    Returns:
        Float value, or nan if input is non-finite.
    """
    if isinstance(x, torch.Tensor):
        v = float(x.detach())
    else:
        v = float(x)
    return v if math.isfinite(v) else float("nan")
