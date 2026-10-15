"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import contextlib
import math
import os
from functools import lru_cache

import numpy as np
import torch

from lerobot.processor.converters import from_tensor_to_numpy, to_tensor


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Report whether CUDA is usable. Cached: first call builds the context."""
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """Detect an NVIDIA RTX PRO 1000 Blackwell GPU.

    Cached. False whenever CUDA is unavailable or the name is unreadable.
    """
    if not cuda_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except (RuntimeError, ValueError):
        return False
    return "rtx pro 1000" in name and "blackwell" in name


@lru_cache(maxsize=1)
def detect_gpu() -> dict | None:
    """Return the device name, the compute capability, and what that
    architecture supports.

    The function returns ``None`` when CUDA is unusable.

    The result comes from the capability. It does not come from matching a
    name. So it stays correct for a GPU that ships tomorrow, and for any SKU
    that the host spells differently. That includes the ``machine_shape`` of
    Kaggle, a cloud provider, and a vGPU profile.

    The thresholds are these. FP16 tensor cores start at Volta, sm_70. Both bf16
    and TF32 start at Ampere, sm_80. See the NVIDIA architecture whitepapers.
    """
    if not cuda_available():
        return None
    try:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
    except (RuntimeError, ValueError, AttributeError):
        return None
    return {
        "device_name": name,
        "compute_capability": [major, minor],
        "supports": {
            "bf16": major >= 8,
            "tf32": major >= 8,
            "fp16_tensor_cores": major >= 7,
        },
    }


def resolve_device() -> torch.device:
    """Execution device from ``SJFPO_DEVICE``.

    Falls back to CPU when CUDA is asked for but unavailable. ``ValueError``
    unless ``SJFPO_DEVICE`` is ``auto``, ``cpu``, or ``cuda``.
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
    """Storage dtype for the real execution device.

    CPU is always FP32. RTX PRO 1000 Blackwell uses BF16 storage on CUDA.
    Everything else is FP32.
    """
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Matmul/autocast dtype, honouring ``SJFPO_DTYPE``. Parsed before the
    Blackwell fast path, so ``fp32`` is honoured there too. BF16 only with
    CUDA support."""
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
    """bf16 autocast context on CUDA, no-op elsewhere."""
    if not cuda_available():
        return contextlib.nullcontext()
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over every element. ``torch.linalg.vector_norm`` in one
    reduction, not ``square().mean().sqrt()``'s three passes. ``F.rms_norm`` is
    no substitute: it divides by the RMS and returns the input's shape."""
    return torch.linalg.vector_norm(x) / math.sqrt(x.numel())


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Move to device, overlapping the copy with CPU work on CUDA.

    Caller must sync before reading: any ``.item()``, ``.cpu()`` or numpy
    conversion syncs implicitly.
    """
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """To numpy, always at least 1-D. Singletons stay ``[1]``, never 0-d."""
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """To a tensor, always at least 1-D. Singletons stay ``[1]``, never 0-d."""
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
