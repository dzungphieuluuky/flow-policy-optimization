"""Shared stateless utilities. The module imports no ``myfpo.*`` name, so it
stays importable from any layer without a cycle.
"""

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
    """Report whether CUDA is usable. The result is cached on first call."""
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """Detect an NVIDIA RTX PRO 1000 Blackwell GPU."""
    if not cuda_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except (RuntimeError, ValueError):
        return False
    return "rtx pro 1000" in name and "blackwell" in name


@lru_cache(maxsize=1)
def detect_gpu() -> dict | None:
    """Device name, compute capability, and the precision flags they imply."""
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
    """Execution device from ``MYFPO_DEVICE``."""
    want = os.environ.get("MYFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want not in {"auto", "cuda"}:
        raise ValueError(
            f"MYFPO_DEVICE={want!r} unknown; expected 'auto', 'cpu', or 'cuda'"
        )
    return torch.device("cuda") if cuda_available() else torch.device("cpu")


def resolve_storage_dtype(
    device: str | torch.device | None = None,
) -> torch.dtype:
    """Storage dtype for the real execution device."""
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Matmul and autocast dtype, from ``MYFPO_DTYPE``. BF16 needs CUDA."""
    want = os.environ.get("MYFPO_DTYPE", "auto").lower()
    if want not in {"auto", "fp32", "bf16"}:
        raise ValueError(
            f"MYFPO_DTYPE={want!r} unknown; expected 'auto', 'fp32', or 'bf16'"
        )
    if want == "fp32" or not cuda_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def autocast_context(device=None):
    """A bf16 autocast context on CUDA. A no-op elsewhere."""
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def rms(x: torch.Tensor) -> torch.Tensor:
    """Root mean square over every element."""
    return torch.linalg.vector_norm(x) / math.sqrt(x.numel())


def cudagraph_safe(x: torch.Tensor) -> torch.Tensor:
    """A copy of a tensor that a cudagraph may still own.

    A module compiled with ``mode="reduce-overhead"`` runs on a cudagraph, and
    the tensor it returns is memory the cudagraph pool owns: the next
    invocation of that module can overwrite it, and the pool can retire the
    tensor while another compiled region is still holding it. Reading it then
    raises "accessing tensor output of CUDAGraphs that has been overwritten by
    a subsequent run".

    The copy leaves the pool, so the consumer of the tensor reads a stable
    value. Call it on the way out of a compiled module, before the tensor
    crosses into another one.
    """
    return x.clone()


def cudagraph_step_begin() -> None:
    """Tell the cudagraph manager that a new step starts here.

    A module compiled with ``mode="reduce-overhead"`` runs on a cudagraph, and
    the manager holds the memory of the finished step until it is told that
    the step is over. Marking the step lets it recycle that memory, and it
    keeps the manager from warning that it cannot hit the fast path.

    It is a no-op when nothing is compiled, so it can be called
    unconditionally.
    """
    torch.compiler.cudagraph_mark_step_begin()


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor to the device. On CUDA the copy overlaps CPU work."""
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """To numpy, always at least 1-D. A singleton stays ``[1]``, never ``()``."""
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """To a tensor, always at least 1-D. A singleton stays ``[1]``, never ``()``."""
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
