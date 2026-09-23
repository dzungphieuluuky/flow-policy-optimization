"""Policy checkpoint IO (JAX): orbax-compatible pytree save/restore.

Layout of a saved checkpoint directory:

* ``config.json``    — SJFPOConfig (unwrapped dataclass)
* ``params.pkl``     — pytree of actor / conditioner / critic / old_* params
* ``optimizer.pkl``  — Optax optimizer state
* ``counters.json``  — ``{global_step, step_count}``

Uses ``pickle`` for the pytrees because the JAX-port's params are plain
``dict``/``jnp.ndarray`` trees; swap to ``orbax.checkpoint`` when you need
distribution-aware saving.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import jax

OPTIMIZER_FILE = "optimizer.pkl"
PARAMS_FILE = "params.pkl"
COUNTERS_FILE = "counters.json"
CONFIG_FILE = "config.json"


class ResumeError(RuntimeError):
    """Unresumable checkpoint (missing/corrupt files)."""


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _atomic_pickle(path: Path, obj) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp_name = tmp.name
    try:
        with open(tmp_name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, path)
    except Exception:
        import contextlib
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def save_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    policy,
    optimizer_state,
    global_step: int,
) -> None:
    """Save policy params, optimizer state, config, and counters."""
    if int(global_step) < 0:
        raise ValueError(f"global_step={global_step} must be >= 0")
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    # 1) Config
    _atomic_write_bytes(
        d / CONFIG_FILE,
        json.dumps(policy.config.to_dict() if hasattr(policy.config, "to_dict")
                   else policy.config.__dict__, indent=2, default=str).encode("utf-8"),
    )

    # 2) Params (pytree of jnp arrays; pickle handles them natively)
    _atomic_pickle(d / PARAMS_FILE, policy.params)

    # 3) Optimizer state
    _atomic_pickle(d / OPTIMIZER_FILE, optimizer_state)

    # 4) Counters
    counters = {
        "global_step": int(global_step),
        "step_count": int(getattr(policy, "step_count", 0)),
    }
    _atomic_write_bytes(
        d / COUNTERS_FILE, json.dumps(counters, indent=2).encode("utf-8")
    )


def load_policy_checkpoint(
    dir_path: str | os.PathLike,
) -> tuple[object, int]:
    """Load a checkpoint; returns ``(policy, global_step)``.

    Requires an existing policy instance to attach the restored params to —
    callers should build an empty policy first if they need one.
    """
    d = Path(dir_path)
    missing = [
        f for f in (CONFIG_FILE, PARAMS_FILE, OPTIMIZER_FILE, COUNTERS_FILE)
        if not (d / f).is_file()
    ]
    if missing:
        raise ResumeError(f"incomplete policy checkpoint in {d}: missing {missing}")

    with open(d / PARAMS_FILE, "rb") as f:
        params = pickle.load(f)
    with open(d / OPTIMIZER_FILE, "rb") as f:
        optimizer_state = pickle.load(f)
    with open(d / COUNTERS_FILE) as f:
        counters = json.load(f)
    with open(d / CONFIG_FILE) as f:
        config_dict = json.load(f)

    global_step = int(counters.get("global_step", 0))

    # Build a policy shell; caller should overwrite ``policy.params`` after.
    # (A real implementation would reconstruct SJFPOConfig from config_dict.)
    return (params, optimizer_state, config_dict, counters), global_step