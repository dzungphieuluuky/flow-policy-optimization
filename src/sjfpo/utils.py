"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import contextlib
import math
import os
import sys
from pathlib import Path

import torch


def resolve_device() -> torch.device:
    """CUDA when available, else CPU (``SJFPO_DEVICE`` overrides)."""
    want = os.environ.get("SJFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_bf16_available() -> bool:
    """BF16 matmul support on the current CUDA device (False on CPU-only)."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_compute_dtype() -> torch.dtype:
    """Autocast matmul dtype (storage stays fp32; ``SJFPO_DTYPE`` overrides)."""
    want = os.environ.get("SJFPO_DTYPE", "auto").lower()
    if want == "fp32":
        return torch.float32
    if want == "bf16":
        return torch.bfloat16 if is_bf16_available() else torch.float32
    return torch.bfloat16 if is_bf16_available() else torch.float32


def autocast_context(device=None):
    """CUDA bf16 autocast when enabled and supported; else no-op."""
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    try:
        dev_type = device.type if isinstance(device, torch.device) else None
    except Exception:
        dev_type = None
    if dev_type is None:
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_type != "cuda" or not is_bf16_available():
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    """Compile an nn.Module with inductor after build, before train/infer.

    On by default; ``SJFPO_COMPILE=0`` opts out (e.g. driver-only CUDA
    without a toolkit, where inductor fails, or CPU-only debugging).
    Default dynamic-shape mode: batch shapes vary, and static-shape
    cudagraphs would waste the 8GB budget. Wrapping is lazy — the compile
    cost lands on the first train/infer call, not here. Params are untouched;
    checkpointing unwraps wrappers before saving, so artifacts hold plain
    eager keys whatever this flag was. Raises loudly (never silent
    fallback) when inductor itself fails — set ``SJFPO_COMPILE=0``.
    """
    if os.environ.get("SJFPO_COMPILE", "1") == "0":
        return module
    return torch.compile(module)


def detect_runtime_env() -> str:
    """``kaggle`` | ``colab`` | ``local`` (env vars win over path probes)."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def default_runs_base(env: str | None = None) -> Path:
    """Per-run sweep directory (kaggle / colab-drive-or-content / local .omo)."""
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/runs")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/runs")
        return drive if Path("/content/drive").exists() else Path("/content/runs")
    return Path(".omo/evidence/runs")


def default_analysis_output(env: str | None = None) -> Path:
    """Analysis output directory (mirrors default_runs_base)."""
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/analysis")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/analysis")
        return drive if Path("/content/drive").exists() else Path("/content/analysis")
    return Path(".omo/evidence/analysis")


def default_smoke_dir(env: str | None = None) -> Path:
    """Smoke-gate output directory (mirrors default_runs_base)."""
    env = env or detect_runtime_env()
    if env == "kaggle":
        return Path("/kaggle/working/smoke")
    if env == "colab":
        drive = Path("/content/drive/MyDrive/spectral-joint-policy/smoke")
        return drive if Path("/content/drive").exists() else Path("/content/smoke")
    return Path(".omo/evidence/smoke")


def safe_float_value(x: torch.Tensor | float) -> float:
    """float(x), or nan when non-finite."""
    if isinstance(x, torch.Tensor):
        v = float(x.detach())
    else:
        v = float(x)
    return v if math.isfinite(v) else float("nan")
