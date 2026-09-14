"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import contextlib
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


def is_bf16_available() -> bool:
    """BF16 matmul support on the current CUDA device (torch-native check).

    Same probe accelerate's ``is_bf16_available`` uses under the hood
    (``torch.cuda.is_bf16_supported``); kept torch-native so this repo
    gains no new dependency. False on CPU-only machines.
    """
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_compute_dtype() -> torch.dtype:
    """Compute dtype for autocast matmuls (storage stays fp32).

    ``SJFPO_DTYPE`` overrides: ``bf16`` forces bf16 (falls back to fp32
    without bf16-capable CUDA), ``fp32`` forces fp32, ``auto`` (default)
    picks bf16 when the GPU supports it, else fp32.

    Params, buffers, DCT bases, GAE, and losses stay fp32/fp64; autocast
    only casts compute-heavy matmuls on the fly (bf16 needs no loss
    scaler, unlike fp16). Full-bf16 storage is intentionally unsupported:
    DCT roundtrip (<1e-5) and ratio math need fp32 precision.
    """
    want = os.environ.get("SJFPO_DTYPE", "auto").lower()
    if want == "fp32":
        return torch.float32
    if want == "bf16":
        return torch.bfloat16 if is_bf16_available() else torch.float32
    return torch.bfloat16 if is_bf16_available() else torch.float32


def autocast_context(device=None):
    """No-op on CPU/fp32; CUDA bf16 autocast when enabled and supported.

    Args:
        device: torch device (or tensor-derived). Only ``cuda`` devices
            autocast; CPU stays fp32 (bf16 CPU kernels are slow and the
            numeric gates assume fp32).
    """
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
    """Compile an nn.Module with inductor before training (off by default).

    Opt-in via ``SJFPO_COMPILE=1``. Off by default because inductor needs a
    full CUDA toolkit at first call (this repo's machines are often
    driver-only, where it fails) and CPU dynamo overhead would slow the
    test suite. Default inductor mode: static-shape cudagraphs
    (``reduce-overhead``) would waste the 8GB budget and break on variable
    batch shapes, so the default dynamic-shape mode is used.
    Storage/dtype semantics unchanged: state_dict keys and params are
    preserved through the wrapper.
    """
    if os.environ.get("SJFPO_COMPILE", "0") != "1":
        return module
    return torch.compile(module)


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
