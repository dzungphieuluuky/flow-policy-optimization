"""Storage/compute dtypes for the JAX path (no capability probes needed).

Target fleet is Blackwell-only, where bf16 is unconditional: bf16 params
when a Blackwell GPU is present, fp32 otherwise. Compute follows JAX
promotion from fp32 inputs (safe); bf16 compute is a follow-up after
Blackwell profiling. Never raises (falls back to fp32).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["is_blackwell", "resolve_param_dtype"]


def is_blackwell() -> bool:
    """True when a Blackwell GPU is visible to JAX."""
    try:
        if jax.default_backend() != "gpu":
            return False
        return any("blackwell" in d.device_kind.lower() for d in jax.devices())
    except Exception:
        return False


def resolve_param_dtype() -> jnp.dtype:
    """bf16 params on Blackwell, fp32 everywhere else (incl. CPU)."""
    return jnp.bfloat16 if is_blackwell() else jnp.float32
