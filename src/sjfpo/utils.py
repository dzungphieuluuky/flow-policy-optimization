"""Shared stateless utilities (no sjfpo.* imports — circular-import prevention)."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.processor.converters import from_tensor_to_numpy, to_tensor

_CUDA_AVAILABLE = torch.cuda.is_available()


def resolve_device() -> torch.device:
    """CUDA when available, else CPU (``SJFPO_DEVICE`` overrides)."""
    want = os.environ.get("SJFPO_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if _CUDA_AVAILABLE:
        return torch.device("cuda")
    return torch.device("cpu")


def filter_low_freq(x: torch.Tensor, k: int, dim: int = 1) -> torch.Tensor:
    """Keep only the ``k`` lowest-frequency bins along ``dim`` (action only)."""
    s = [slice(None)] * x.dim()
    s[dim] = slice(None, k)
    return x[tuple(s)]


def resolve_compute_dtype() -> torch.dtype:
    """Autocast matmul dtype (storage stays fp32; ``SJFPO_DTYPE`` overrides)."""
    if not _CUDA_AVAILABLE:
        return torch.float32
    want = os.environ.get("SJFPO_DTYPE", "auto").lower()
    if want == "fp32":
        return torch.float32
    if want == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


def autocast_context(device=None):
    """CUDA bf16 autocast when enabled and supported; else no-op."""
    if not _CUDA_AVAILABLE:
        return contextlib.nullcontext()
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    try:
        dev_type = device.type if isinstance(device, torch.device) else None
    except Exception:
        dev_type = None
    if dev_type is None:
        dev_type = "cuda" if _CUDA_AVAILABLE else "cpu"
    if dev_type != "cuda" or not torch.cuda.is_bf16_supported():
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    """Compile an nn.Module with inductor after build, before train/infer.

    On by default; ``SJFPO_COMPILE=0`` opts out (e.g. driver-only CUDA
    without a toolkit, where inductor fails, or CPU-only debugging).
    CPU inductor hits a guard bug (``___from_numpy`` ADInplaceOrView
    DispatchKeySet mismatch on Colab) so it auto-disables when CUDA is
    unavailable — eager is numerically identical.
    Default dynamic-shape mode: batch shapes vary, and static-shape
    cudagraphs would waste the 8GB budget. Wrapping is lazy — the compile
    cost lands on the first train/infer call, not here. Params are untouched;
    checkpointing unwraps wrappers before saving, so artifacts hold plain
    eager keys whatever this flag was. Raises loudly (never silent
    fallback) when inductor itself fails — set ``SJFPO_COMPILE=0``.
    """
    if os.environ.get("SJFPO_COMPILE", "1") == "0":
        return module
    if not _CUDA_AVAILABLE:
        return module
    # Only compile when bf16 is available (GPU optimization tied to bf16 support).
    if not torch.cuda.is_bf16_supported():
        return module
    # Guarded scalar raises (e.g. Critic's executed-step check) stay in the
    # compiled graph instead of forcing an eager fallback for the whole module.
    torch._dynamo.config.capture_scalar_outputs = True
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


def to_device_async(x: torch.Tensor, device) -> torch.Tensor:
    """Host-to-device move that overlaps with CPU work on CUDA.

    Pinned staging + ``non_blocking=True`` lets DMA run while the collector
    stacks the next tensors; ordering stays safe on the default stream.
    Plain ``.to()`` on CPU.
    """
    device = torch.device(device)
    if device.type == "cuda" and x.device != device:
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)


def to_numpy_1d(x: torch.Tensor | np.ndarray | float) -> np.ndarray:
    """to numpy with atleast_1d: singletons stay [1], never 0-d/scalar."""
    return np.atleast_1d(from_tensor_to_numpy(x))


def to_tensor_1d(
    x: torch.Tensor | np.ndarray | float,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """to tensor with atleast_1d: singletons stay [1], never 0-d."""
    return torch.atleast_1d(to_tensor(x, dtype=dtype, device=device))
