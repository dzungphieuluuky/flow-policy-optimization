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
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "export_safetensors",
    "load_safetensors",
    "to_torch_dict",
]


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
    behavior=None,
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
    export_safetensors(
        policy, ckpt_dir / "model.safetensors",
        metadata={
            "global_step": str(int(global_step)),
            "env_name": env_name,
            "method": method,
        },
        behavior=behavior,
    )
    meta = {"global_step": int(global_step), "env_name": env_name, "method": method}
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "dir": str(ckpt_dir),
        "meta": str(ckpt_dir / "meta.json"),
        "model": str(ckpt_dir / "model.safetensors"),
    }


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


def to_torch_dict(policy, behavior=None) -> dict[str, np.ndarray]:
    """NNX weights as a torch-style state_dict (``SJFPOPolicy`` key names).

    Linear kernels transposed ([in, out] -> [out, in]); normalizer stats use
    the ``obs_normalizer.*`` buffer names; non-persistent action bounds are
    skipped. With ``behavior``, its actor/conditioner export as
    ``old_actor.*``/``old_conditioner.*`` so the file loads with strict keys.
    """
    from sjfpo_jax.policy import _path_names

    out: dict[str, np.ndarray] = {}

    def _emit(obj, prefix: str):
        graphdef, params, variables = nnx.split(obj, nnx.Param, nnx.Variable)
        del graphdef
        for tree in (params, variables):
            # flatten_mapping: same path convention as load_safetensors.
            for path, leaf in dict(nnx.traversals.flatten_mapping(tree)).items():
                names = [n for n in _path_names(path) if n]
                if names[0] in ("action_low", "action_high"):
                    continue  # non-persistent bounds, absent from torch dicts
                if prefix == "" and names[0] == "normalizer":
                    names = ["obs_normalizer"] + names[1:]
                arr = np.asarray(leaf[...])
                # Contiguous: safetensors serializes raw buffers, so an
                # F-order transpose view would land transposed on disk.
                if names[-1] == "kernel":
                    arr = np.ascontiguousarray(arr.T if arr.ndim == 2 else arr)
                    out[prefix + ".".join(names[:-1]) + ".weight"] = arr
                else:
                    out[prefix + ".".join(names)] = np.ascontiguousarray(arr)

    _emit(policy, "")
    if behavior is not None:
        _emit(behavior.actor, "old_actor.")
        _emit(behavior.conditioner, "old_conditioner.")
    return out


def export_safetensors(policy, path: str | Path, metadata: dict[str, str] | None = None,
                       behavior=None) -> str:
    """Write policy weights as ``model.safetensors`` (torch-loadable)."""
    from safetensors.flax import save_file

    path = str(path)
    save_file(to_torch_dict(policy, behavior), path, metadata=metadata or {})
    return path


def load_safetensors(policy, path: str | Path) -> list[str]:
    """Load a safetensors file (ours or torch's) into the NNX policy."""
    from safetensors.flax import load_file

    from sjfpo_jax.policy import _path_names

    flat = {k: np.asarray(v) for k, v in load_file(str(path)).items()}
    graphdef, params, variables = nnx.split(policy, nnx.Param, nnx.Variable)
    del graphdef
    loaded: list[str] = []

    def _take(tree, transpose_2d: bool):
        # flatten_mapping (not flatten_with_path): leaves stay assignable.
        for path, leaf in dict(nnx.traversals.flatten_mapping(tree)).items():
            names = [n for n in _path_names(path) if n]
            if names[0] in ("action_low", "action_high"):
                continue  # non-persistent bounds, absent from files
            key = ".".join(names[:-1]) + ".weight" if names[-1] == "kernel" else ".".join(names)
            if key.startswith("normalizer."):
                key = "obs_normalizer." + key[len("normalizer."):]
            if key not in flat:
                raise KeyError(f"safetensors file is missing expected tensor {key!r}")
            arr = flat[key]
            if names[-1] == "kernel" and transpose_2d and arr.ndim == 2:
                arr = arr.T
            # Scalar promotion: the flax binding stores 0-d as 1-d.
            leaf_arr = np.asarray(leaf[...])
            want, want_dtype = leaf_arr.shape, leaf_arr.dtype
            if arr.shape != want:
                if int(np.size(arr)) != int(np.size(np.zeros(want))):
                    raise ValueError(
                        f"shape mismatch for {key!r}: file {arr.shape} vs model {want}"
                    )
                arr = arr.reshape(want)
            leaf[...] = jnp.asarray(arr, dtype=want_dtype)  # keep leaf dtype
            loaded.append(key)

    _take(params, True)
    _take(variables, False)
    nnx.update(policy, params, variables)
    return loaded
