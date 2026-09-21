"""Orbax checkpoints for the JAX policy (params + optimizer + counters).

Layout per checkpoint dir: ``arrays/`` (Orbax pytree: params, non-param
variables like normalizer stats, optimizer state) + ``meta.json``
(global_step, env, method). Overwrites any previous save at the path.
Hub/safetensors format is intentionally NOT reused (documented break).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

__all__ = ["save_checkpoint", "load_checkpoint"]


def _policy_states(policy) -> dict[str, Any]:
    graphdef, params, variables = nnx.split(policy, nnx.Param, nnx.Variable)
    del graphdef
    return {"params": params, "variables": variables}


def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    policy,
    optimizer,
    global_step: int,
    env_name: str = "unknown",
    method: str = "sjfpo_jax",
) -> dict[str, str]:
    """Persist params + optimizer + counters; returns artifact paths."""
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        **_policy_states(policy),
        "opt": nnx.state(optimizer),
    }
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(ckpt_dir / "arrays", state)
    meta = {"global_step": int(global_step), "env_name": env_name, "method": method}
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"dir": str(ckpt_dir), "meta": str(ckpt_dir / "meta.json")}


def load_checkpoint(
    ckpt_dir: str | Path, *, policy, optimizer
) -> dict[str, Any]:
    """Restore into live objects; returns the saved meta (KeyError if absent)."""
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    abstract = {
        **_policy_states(policy),
        "opt": nnx.state(optimizer),
    }
    with ocp.StandardCheckpointer() as ckptr:
        restored = ckptr.restore(ckpt_dir / "arrays", abstract)
    nnx.update(policy, restored["params"], restored["variables"])
    nnx.update(optimizer, restored["opt"])
    # Materialize on host (avoids stale-device reads in later steps).
    jax.block_until_ready(nnx.state(policy, nnx.Param))
    return meta
