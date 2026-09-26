"""Shared stateless utilities (JAX port). No sjfpo.* imports."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp


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
















def stop_gradient(tree):
    return jax.tree.map(jax.lax.stop_gradient, tree)