"""Shared stateless utilities (JAX port). No sjfpo.* imports."""

from __future__ import annotations

import contextlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.processor.converters import from_tensor_to_numpy


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True when JAX has a GPU backend."""
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


@lru_cache(maxsize=1)
def is_rtx_1000_pro_blackwell() -> bool:
    """True on RTX PRO 1000 Blackwell (matches torch port)."""
    if not gpu_available():
        return False
    try:
        kind = jax.devices("gpu")[0].device_kind.lower()
    except Exception:
        return False
    return "rtx pro 1000" in kind and "blackwell" in kind


@lru_cache(maxsize=1)
def _colab_drive_present() -> bool:
    return Path("/content/drive").exists()


def resolve_device() -> jax.xla.Device:
    """GPU when available, else CPU. ``SJFPO_DEVICE`` ∈ {auto, cpu, cuda/gpu}."""
    want = os.environ.get("SJFPO_DEVICE", "auto").lower()
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if want == "cpu":
        return jax.devices("cpu")[0]
    if want in ("cuda", "gpu"):
        return gpus[0] if gpus else jax.devices("cpu")[0]
    if want != "auto":
        raise ValueError(f"SJFPO_DEVICE={want!r} unknown; expected 'auto', 'cpu', 'cuda'")
    return gpus[0] if gpus else jax.devices("cpu")[0]


def resolve_storage_dtype() -> jnp.dtype:
    """Param/checkpoint storage dtype; bf16 only on Blackwell."""
    return jnp.bfloat16 if is_rtx_1000_pro_blackwell() else jnp.float32


def resolve_compute_dtype() -> jnp.dtype:
    """bf16 matmuls when supported; ``SJFPO_DTYPE`` overrides."""
    if is_rtx_1000_pro_blackwell():
        return jnp.bfloat16
    want = os.environ.get("SJFPO_DTYPE", "auto").lower()
    if want == "fp32":
        return jnp.float32
    if want == "bf16":
        return jnp.bfloat16
    if want != "auto":
        raise ValueError(f"SJFPO_DTYPE={want!r} unknown; expected 'auto', 'fp32', 'bf16'")
    return jnp.bfloat16 if gpu_available() else jnp.float32


@contextlib.contextmanager
def autocast_context(device=None):
    """JAX has no autocast; matmuls take the input dtype. Kept for parity."""
    yield


def maybe_compile(fn, *, donate_argnums=()):
    """JIT a callable. ``SJFPO_COMPILE=0`` opts out (debugging)."""
    if os.environ.get("SJFPO_COMPILE", "1") == "0":
        return fn
    return jax.jit(fn, donate_argnums=donate_argnums)


def detect_runtime_env() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def _runtime_root(kaggle, colab_drive, colab_local, local, runtime=None) -> Path:
    runtime = runtime or detect_runtime_env()
    if runtime == "kaggle":
        return Path(kaggle)
    if runtime == "colab":
        drive = Path(colab_drive)
        return drive if _colab_drive_present() else Path(colab_local)
    return Path(local)


def default_runs_base(runtime: str | None = None) -> Path:
    return _runtime_root(
        "/kaggle/working/runs",
        "/content/drive/MyDrive/spectral-joint-policy/runs",
        "/content/runs",
        ".omo/evidence/runs",
        runtime,
    )


def default_analysis_output(runtime: str | None = None) -> Path:
    return _runtime_root(
        "/kaggle/working/analysis",
        "/content/drive/MyDrive/spectral-joint-policy/analysis",
        "/content/analysis",
        ".omo/evidence/analysis",
        runtime,
    )


def default_smoke_dir(runtime: str | None = None) -> Path:
    return _runtime_root(
        "/kaggle/working/smoke",
        "/content/drive/MyDrive/spectral-joint-policy/smoke",
        "/content/smoke",
        ".omo/evidence/smoke",
        runtime,
    )


def to_device(x, device=None):
    """Place ``x`` on ``device`` (defaults to ``resolve_device()``)."""
    device = device or resolve_device()
    return jax.device_put(jnp.asarray(x), device)


def filter_low_freq(x, k: int, dim: int = 1):
    if k < 1 or k > x.shape[dim]:
        raise ValueError(f"k={k} out of range for dim {dim} with size {x.shape[dim]}")
    s = [slice(None)] * x.ndim
    s[dim] = slice(None, k)
    return x[tuple(s)]


def to_numpy_1d(x) -> np.ndarray:
    return np.atleast_1d(from_tensor_to_numpy(np.asarray(x)))


def to_jnp_1d(x, dtype=None):
    if dtype is None:
        dtype = jnp.float32
    return jnp.atleast_1d(jnp.asarray(x, dtype=dtype))


def stop_gradient(tree):
    return jax.tree.map(jax.lax.stop_gradient, tree)