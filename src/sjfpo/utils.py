"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from lerobot.processor.converters import from_tensor_to_numpy, to_tensor


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Report whether CUDA is usable. Cached: the first call builds the context."""
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """Detect an NVIDIA RTX PRO 1000 Blackwell GPU.

    Cached, and False whenever CUDA is unavailable or the device name is
    unreadable.
    """
    if not cuda_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return False
    return "rtx pro 1000" in name and "blackwell" in name


@lru_cache(maxsize=1)
def _colab_drive_present() -> bool:
    """Report whether a mounted Colab Drive is present."""
    return Path("/content/drive").exists()


def resolve_device() -> torch.device:
    """Resolve the execution device from ``SJFPO_DEVICE``.

    Falls back to CPU when CUDA is requested but unavailable. Raises
    ``ValueError`` unless ``SJFPO_DEVICE`` is ``auto``, ``cpu``, or ``cuda``.
    """
    want = os.environ.get("SJFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        return torch.device("cuda") if cuda_available() else torch.device("cpu")
    if want != "auto":
        raise ValueError(
            f"SJFPO_DEVICE={want!r} unknown; expected 'auto', 'cpu', or 'cuda'"
        )
    return torch.device("cuda") if cuda_available() else torch.device("cpu")


def resolve_storage_dtype(
    device: str | torch.device | None = None,
) -> torch.dtype:
    """Resolve the storage dtype for the actual execution device.

    CPU always uses FP32. RTX PRO 1000 Blackwell uses BF16 storage when the
    execution device is CUDA; all other devices use FP32.
    """
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Resolve the matmul/autocast dtype, honouring ``SJFPO_DTYPE``.

    ``SJFPO_DTYPE`` is parsed before the Blackwell fast path so ``fp32`` is
    honored there as well. BF16 is only selected when CUDA supports it, so
    this never returns BF16 without CUDA support. Raises ``ValueError``
    unless ``SJFPO_DTYPE`` is ``auto``, ``fp32``, or ``bf16``.
    """
    want = os.environ.get("SJFPO_DTYPE", "auto").lower()
    if want not in {"auto", "fp32", "bf16"}:
        raise ValueError(
            f"SJFPO_DTYPE={want!r} unknown; expected 'auto', 'fp32', or 'bf16'"
        )
    if want == "fp32":
        return torch.float32
    if not cuda_available():
        return torch.float32
    if want == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if is_rtx_1000_pro_blackwell():
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def autocast_context(device=None):
    """Build a bf16 autocast context for CUDA, or a no-op elsewhere."""
    if not cuda_available():
        return contextlib.nullcontext()
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def detect_runtime_env() -> str:
    """Identify the current runtime as ``kaggle``, ``colab``, or ``local``.

    Environment variables win over path probes, so an explicit variable is
    never overridden by an incidental directory.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def _runtime_root(
    kaggle: str,
    colab_drive: str,
    colab_local: str,
    local: str,
    runtime: str | None = None,
) -> Path:
    """Resolve a runtime-specific output root.

    Branch logic shared by the ``default_*`` path helpers. On Colab the Drive
    path is used when a Drive is mounted, otherwise the local one.
    """
    runtime = runtime or detect_runtime_env()
    if runtime == "kaggle":
        return Path(kaggle)
    if runtime == "colab":
        drive = Path(colab_drive)
        return drive if _colab_drive_present() else Path(colab_local)
    return Path(local)


def default_runs_base(runtime: str | None = None) -> Path:
    """Resolve the per-run sweep directory for a runtime."""
    return _runtime_root(
        "/kaggle/working/runs",
        "/content/drive/MyDrive/spectral-joint-policy/runs",
        "/content/runs",
        ".omo/evidence/runs",
        runtime,
    )


def default_smoke_dir(runtime: str | None = None) -> Path:
    """Resolve the smoke-gate output directory for a runtime."""
    return _runtime_root(
        "/kaggle/working/smoke",
        "/content/drive/MyDrive/spectral-joint-policy/smoke",
        "/content/smoke",
        ".omo/evidence/smoke",
        runtime,
    )


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor to a device, overlapping the copy with CPU work on CUDA.

    Pinned staging plus ``non_blocking=True`` lets the DMA engine copy while
    Python keeps running. The caller is responsible for syncing before reading
    the result (any ``.item()`` / ``.cpu()`` / numpy conversion will sync
    implicitly).
    """
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def filter_low_freq(x: torch.Tensor, k: int, dim: int = 1) -> torch.Tensor:
    """Keep only the ``k`` lowest frequencies along ``dim``."""
    if k < 1 or k > x.shape[dim]:
        raise ValueError(f"k={k} out of range for dim {dim} with size {x.shape[dim]}")
    s = [slice(None)] * x.dim()
    s[dim] = slice(None, k)
    return x[tuple(s)]


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """Convert to numpy, guaranteeing at least one dimension.

    Singletons stay ``[1]`` and are never collapsed to a 0-d scalar.
    """
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert to a tensor, guaranteeing at least one dimension.

    Singletons stay ``[1]`` and are never collapsed to a 0-d scalar.
    """
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
