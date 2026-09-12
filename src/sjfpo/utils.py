"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch


def resolve_device() -> torch.device:
    """Default compute device: CUDA when available, else CPU.

    ``SJFPO_DEVICE`` overrides (``cuda``/``cpu``/``auto``); unknown values
    fall back to auto. An unavailable CUDA always falls back to CPU.
    """
    want = os.environ.get("SJFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_runtime_env() -> str:
    """Runtime environment: ``kaggle`` | ``colab`` | ``local``.

    Explicit env vars win over path probes, so tests and forced runs stay
    deterministic. No side effects (pure probing).
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def default_runs_base(env: str | None = None) -> Path:
    """Default per-run directory for the sweep matrix.

    Kaggle → ``/kaggle/working/runs``; Colab → Drive project dir when the
    drive is mounted (persistence across recycled runtimes), else ``/content/runs``;
    local → repo-relative ``.omo/evidence/runs``.
    """
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/runs")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/runs")
        return drive if Path("/content/drive").exists() else Path("/content/runs")
    return Path(".omo/evidence/runs")


def default_analysis_output(env: str | None = None) -> Path:
    """Default analysis output directory (mirrors default_runs_base)."""
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/analysis")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/analysis")
        return drive if Path("/content/drive").exists() else Path("/content/analysis")
    return Path(".omo/evidence/analysis")


def default_smoke_dir(env: str | None = None) -> Path:
    """Default smoke-gate output directory (mirrors default_runs_base)."""
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/smoke")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/smoke")
        return drive if Path("/content/drive").exists() else Path("/content/smoke")
    return Path(".omo/evidence/smoke")


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
