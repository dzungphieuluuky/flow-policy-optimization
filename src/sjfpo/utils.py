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
    """Report whether CUDA is usable.

    Cached, because the first call initialises the CUDA context.

    Returns:
        True when a CUDA device is available.
    """
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """Detect an NVIDIA RTX PRO 1000 Blackwell GPU.

    Cached, and returns False whenever CUDA is unavailable or the device name
    cannot be read.

    Returns:
        True when running on an RTX PRO 1000 Blackwell GPU.
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
    """Report whether a mounted Colab Drive is present.

    Returns:
        True when ``/content/drive`` exists.
    """
    return Path("/content/drive").exists()


def resolve_device() -> torch.device:
    """Resolve the execution device from ``SJFPO_DEVICE``.

    Args:
        None.

    Returns:
        The requested device, falling back to CPU when CUDA is requested but
        unavailable.

    Raises:
        ValueError: If ``SJFPO_DEVICE`` is not ``auto``, ``cpu``, or ``cuda``.
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

    Args:
        device: Device the dtype is resolved for, or None to use whatever the
            ambient default is.

    Returns:
        The storage dtype.
    """
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Resolve the matmul/autocast dtype, honouring ``SJFPO_DTYPE``.

    ``SJFPO_DTYPE`` is parsed before the Blackwell fast path so ``fp32`` is
    honored there as well. BF16 is only selected when CUDA supports it.

    Returns:
        The compute dtype, never BF16 without CUDA support.

    Raises:
        ValueError: If ``SJFPO_DTYPE`` is not ``auto``, ``fp32``, or ``bf16``.
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
    """Build a bf16 autocast context for CUDA, or a no-op elsewhere.

    Args:
        device: Device the context is opened for. Defaults to CUDA.

    Returns:
        A bf16 autocast context on CUDA, otherwise a null context.
    """
    if not cuda_available():
        return contextlib.nullcontext()
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def maybe_compile(
    module: torch.nn.Module, device: str | torch.device | None = None
) -> torch.nn.Module:
    """Compile a module with inductor, after build and before train/infer.

    On CUDA by default. ``SJFPO_COMPILE=0`` opts out (e.g., for debugging or
    when the module is called with a fresh shape per iteration and the
    recompile cost dominates).

    A symbolic batch dimension is used so train minibatches
    (``n_mb * num_mc``), remainder splits, and inference (``B=1``) all share
    one graph instead of recompiling per ``B``. ``capture_scalar_outputs``
    keeps guarded Python scalars (e.g. Critic's executed-step check) in the
    graph instead of forcing an eager fallback.

    Args:
        module: Module to compile.
        device: Device the module runs on. Read from its parameters when
            omitted.

    Returns:
        The compiled module, or the original one when compilation is
        disabled or not applicable.
    """
    if os.environ.get("SJFPO_COMPILE", "1") == "0":
        return module
    target = (
        next(module.parameters()).device
        if device is None
        else torch.device(device)
    )
    if target.type != "cuda" or not cuda_available():
        return module
    if not torch.cuda.is_bf16_supported() and not is_rtx_1000_pro_blackwell():
        return module
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled = torch.compile(module, dynamic=True)
    return compiled


def detect_runtime_env() -> str:
    """Identify the current runtime.

    Environment variables win over path probes, so an explicit variable is
    never overridden by an incidental directory.

    Returns:
        ``"kaggle"``, ``"colab"``, or ``"local"``.
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

    Shared branch logic behind the three ``default_*`` path helpers. On Colab
    the Drive path is used when a Drive is mounted, otherwise the local one.

    Args:
        kaggle: Root used on Kaggle.
        colab_drive: Root used on Colab with a mounted Drive.
        colab_local: Root used on Colab without a mounted Drive.
        local: Root used on a local machine.
        runtime: Runtime name; detected when omitted.

    Returns:
        The selected root path.
    """
    runtime = runtime or detect_runtime_env()
    if runtime == "kaggle":
        return Path(kaggle)
    if runtime == "colab":
        drive = Path(colab_drive)
        return drive if _colab_drive_present() else Path(colab_local)
    return Path(local)


def default_runs_base(runtime: str | None = None) -> Path:
    """Resolve the per-run sweep directory for a runtime.

    Args:
        runtime: Runtime name; detected when omitted.

    Returns:
        Root directory the sweep writes run subdirectories into.
    """
    return _runtime_root(
        "/kaggle/working/runs",
        "/content/drive/MyDrive/spectral-joint-policy/runs",
        "/content/runs",
        ".omo/evidence/runs",
        runtime,
    )


def default_smoke_dir(runtime: str | None = None) -> Path:
    """Resolve the smoke-gate output directory for a runtime.

    Args:
        runtime: Runtime name; detected when omitted.

    Returns:
        Root directory the smoke gate writes into.
    """
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

    Args:
        x: Tensor to move.
        device: Destination device.

    Returns:
        The tensor on the destination device.
    """
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def filter_low_freq(x: torch.Tensor, k: int, dim: int = 1) -> torch.Tensor:
    """Keep only the ``k`` lowest-frequency bins along ``dim``.

    Args:
        x: Tensor to slice.
        k: Number of leading bins to keep.
        dim: Dimension the bins live on.

    Returns:
        The leading slice of ``x`` along ``dim``.

    Raises:
        ValueError: If ``k`` is outside ``[1, x.shape[dim]]``.
    """
    if k < 1 or k > x.shape[dim]:
        raise ValueError(f"k={k} out of range for dim {dim} with size {x.shape[dim]}")
    s = [slice(None)] * x.dim()
    s[dim] = slice(None, k)
    return x[tuple(s)]


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """Convert to numpy, guaranteeing at least one dimension.

    Singletons stay ``[1]`` and are never collapsed to a 0-d scalar.

    Args:
        x: Tensor, array, or scalar to convert.

    Returns:
        A numpy array with at least one dimension.
    """
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert to a tensor, guaranteeing at least one dimension.

    Singletons stay ``[1]`` and are never collapsed to a 0-d scalar.

    Args:
        x: Tensor, array, or scalar to convert.
        dtype: Target dtype; fp32 by default.
        device: Target device; the tensor's current device when omitted.

    Returns:
        A tensor with at least one dimension.
    """
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
