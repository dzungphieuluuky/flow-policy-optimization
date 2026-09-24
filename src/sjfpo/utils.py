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


# ---------------------------------------------------------------------------
# Runtime probes (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Whether CUDA is usable. Cached — first call triggers CUDA init."""
    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """True when running on an NVIDIA RTX PRO 1000 Blackwell GPU."""
    if not cuda_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return False
    return "rtx pro 1000" in name and "blackwell" in name


@lru_cache(maxsize=1)
def _colab_drive_present() -> bool:
    return Path("/content/drive").exists()


# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------


def resolve_device() -> torch.device:
    """CUDA when available, else CPU. ``SJFPO_DEVICE`` ∈ {auto, cpu, cuda}."""
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
    """Storage dtype for the actual execution device.

    CPU always uses FP32. RTX PRO 1000 Blackwell uses BF16 storage when the
    execution device is CUDA; all other devices use FP32.
    """
    if device is not None and torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16 if is_rtx_1000_pro_blackwell() else torch.float32


def resolve_compute_dtype() -> torch.dtype:
    """Matmul/autocast dtype with explicit environment override.

    ``SJFPO_DTYPE`` is parsed before the Blackwell fast path so ``fp32`` is
    honored there as well. BF16 is only selected when CUDA supports it.
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
    """bf16 autocast on CUDA; no-op elsewhere (CPU, fp32, or non-CUDA device)."""
    if not cuda_available():
        return contextlib.nullcontext()
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    dev_type = device.type if isinstance(device, torch.device) else "cuda"
    if dev_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# torch.compile wrapper
# ---------------------------------------------------------------------------


def maybe_compile(
    module: torch.nn.Module, device: str | torch.device | None = None
) -> torch.nn.Module:
    """Compile with inductor after build, before train/infer.

    On CUDA by default. ``SJFPO_COMPILE=0`` opts out (e.g., for debugging or
    when the module is called with a fresh shape per iteration and the
    recompile cost dominates).
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
    # bf16 support is a proxy for "worth compiling"; Blackwell always qualifies.
    if not torch.cuda.is_bf16_supported() and not is_rtx_1000_pro_blackwell():
        return module
    # Symbolic batch dim: train minibatches (n_mb * num_mc), remainder splits,
    # and inference (B=1) share one graph instead of recompiling per B.
    # capture_scalar_outputs keeps guarded Python scalars (e.g. Critic's
    # executed-step check) in the graph instead of forcing an eager fallback.
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled = torch.compile(module, dynamic=True)
    return compiled


# ---------------------------------------------------------------------------
# Environment / path discovery
# ---------------------------------------------------------------------------


def detect_runtime_env() -> str:
    """``kaggle`` | ``colab`` | ``local`` (env vars win over path probes)."""
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
    """Shared branch logic for the three ``default_*`` path helpers."""
    runtime = runtime or detect_runtime_env()
    if runtime == "kaggle":
        return Path(kaggle)
    if runtime == "colab":
        drive = Path(colab_drive)
        return drive if _colab_drive_present() else Path(colab_local)
    return Path(local)


def default_runs_base(runtime: str | None = None) -> Path:
    """Per-run sweep directory (kaggle / colab-drive-or-content / local .omo)."""
    return _runtime_root(
        "/kaggle/working/runs",
        "/content/drive/MyDrive/spectral-joint-policy/runs",
        "/content/runs",
        ".omo/evidence/runs",
        runtime,
    )


def default_analysis_output(runtime: str | None = None) -> Path:
    """Analysis output directory (mirrors ``default_runs_base``)."""
    return _runtime_root(
        "/kaggle/working/analysis",
        "/content/drive/MyDrive/spectral-joint-policy/analysis",
        "/content/analysis",
        ".omo/evidence/analysis",
        runtime,
    )


def default_smoke_dir(runtime: str | None = None) -> Path:
    """Smoke-gate output directory (mirrors ``default_runs_base``)."""
    return _runtime_root(
        "/kaggle/working/smoke",
        "/content/drive/MyDrive/spectral-joint-policy/smoke",
        "/content/smoke",
        ".omo/evidence/smoke",
        runtime,
    )


# ---------------------------------------------------------------------------
# Tensor / array interop
# ---------------------------------------------------------------------------


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Host-to-device move that overlaps with CPU work on CUDA.

    Pinned staging + ``non_blocking=True`` lets the DMA engine copy while
    Python keeps running. The caller is responsible for syncing before
    reading the result (any ``.item()`` / ``.cpu()`` / numpy conversion
    will sync implicitly).
    """
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        if not x.is_pinned():
            x = x.pin_memory()
        return x.to(device, non_blocking=True)
    return x.to(device)


def filter_low_freq(x: torch.Tensor, k: int, dim: int = 1) -> torch.Tensor:
    """Keep only the ``k`` lowest-frequency bins along ``dim``."""
    if k < 1 or k > x.shape[dim]:
        raise ValueError(f"k={k} out of range for dim {dim} with size {x.shape[dim]}")
    s = [slice(None)] * x.dim()
    s[dim] = slice(None, k)
    return x[tuple(s)]


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """Numpy with ``atleast_1d``: singletons stay ``[1]``, never 0-d/scalar."""
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Tensor with ``atleast_1d``: singletons stay ``[1]``, never 0-d."""
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
