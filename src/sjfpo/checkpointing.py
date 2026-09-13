"""Checkpoint/resume IO: manifest-complete atomic save/load (plan task 9).

Generic contract (the runner calls this same API with the policy bundle):

``state`` is a nested mapping whose LEAVES are one of

* ``torch.nn.Module``            -> stored as its ``state_dict()``
* ``torch.optim.Optimizer``      -> stored as its ``state_dict()``
* ``torch.Tensor``               -> stored directly
* any picklable python object    -> stored directly (deques, RNG bit-states,
  counters, flags, dicts of floats, ...)

Only plain ``dict`` mappings nest; every other value is a leaf. Manifest
keys are dotted paths (e.g. ``"models.critic"``, ``"counters.global_step"``).

On-disk layout inside ``dir_path``:

* ``checkpoint.pt``   torch.save zip archive containing the payload tree
  plus an embedded copy of the manifest keys (torn-write detector).
* ``manifest.json``   sidecar listing EVERY key with its kind; presence of a
  complete manifest implies a complete blob.

Both files are written to ``<name>.tmp`` and moved into place with
``os.replace`` (atomic on POSIX), so a process killed mid-save always leaves
either the previous checkpoint or the new one — never a half-written final
file. If a kill lands between the two renames, the embedded-keys check in
``load_checkpoint`` turns the resulting manifest/blob mismatch into a loud
ResumeError instead of silent corruption.

Resume semantics (plan Scope): deterministic continuation is guaranteed only
WITHIN an identical hardware/software stack. The env fingerprint recorded in
the state ({gpu_name, torch_version, cuda_version}) must be CHECKED by the
caller at resume time and any mismatch WARNED loudly, never silenced.

Retained public API (dead in the src train path, kept per fn-consolidation D1):
``capture_env_fingerprint`` (tests/resume_worker.py + tests/test_checkpoint.py).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

CHECKPOINT_FILE = "checkpoint.pt"
MANIFEST_FILE = "manifest.json"
FORMAT = 1
KEYS_FIELD = "_manifest_keys"


class ResumeError(RuntimeError):
    """Raised when a checkpoint cannot be resumed (missing/corrupt manifest,
    missing keys, unpicklable/truncated blob, or manifest/blob mismatch)."""

    # retained: public resume/test API — used by tests/resume_worker.py and
    # tests/test_checkpoint.py; not called by the src train path. Keep per fn-consolidation D1.


def capture_env_fingerprint() -> dict[str, str | None]:
    """Environment fingerprint for same-stack resume checks.

    ``gpu_name`` falls back to ``'cpu'`` on CPU-only machines; ``cuda_version``
    is ``None`` for CPU-only torch builds.
    """
    try:
        gpu_name: str | None = torch.cuda.get_device_name(0)
    except Exception:  # no CUDA device / no driver — CPU-bound run
        gpu_name = "cpu"
    return {
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else str(name)


def serialize_tree(state: Mapping[str, Any], prefix: str = "") -> tuple[dict, dict]:
    """Build the nested payload tree + flat {dotted_path: kind} manifest."""
    payload: dict[str, Any] = {}
    keys: dict[str, str] = {}
    for name, value in state.items():
        path = join_prefix(prefix, name)
        if isinstance(value, dict):
            sub_payload, sub_keys = serialize_tree(value, path)
            payload[name] = sub_payload
            keys.update(sub_keys)
        elif isinstance(value, torch.nn.Module):
            payload[name] = value.state_dict()
            keys[path] = "module"
        elif isinstance(value, torch.optim.Optimizer):
            payload[name] = value.state_dict()
            keys[path] = "optimizer"
        elif isinstance(value, torch.Tensor):
            payload[name] = value.detach().cpu().clone()
            keys[path] = "tensor"
        else:
            payload[name] = value
            keys[path] = "object"
    return payload, keys


def save_checkpoint(dir_path: str | os.PathLike, state: Mapping[str, Any]) -> None:
    """Atomically write ``state`` as checkpoint.pt + manifest.json.

    tmp-file then ``os.replace`` per file; blob first, manifest second.
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    payload, keys = serialize_tree(state)
    payload[KEYS_FIELD] = dict(keys)  # embedded copy: torn-write detector
    manifest = {"format": FORMAT, "keys": keys}

    blob_tmp = d / (CHECKPOINT_FILE + ".tmp")
    manifest_tmp = d / (MANIFEST_FILE + ".tmp")
    torch.save(payload, blob_tmp)
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(blob_tmp, d / CHECKPOINT_FILE)
    os.replace(manifest_tmp, d / MANIFEST_FILE)


def walk_mapping(mapping: Mapping[str, Any], prefix: str = "") -> Iterable[str]:
    for name, value in mapping.items():
        path = join_prefix(prefix, name)
        if isinstance(value, Mapping):
            yield from walk_mapping(value, path)
        else:
            yield path


def flatten_expected(spec: Iterable[str] | Mapping[str, Any]) -> set[str]:
    if isinstance(spec, Mapping):
        return set(walk_mapping(spec))
    return set(spec)


def read_checkpoint_manifest(d: Path) -> tuple[Path, dict]:
    """Locate blob + manifest paths and validate the manifest JSON envelope."""
    manifest_path = d / MANIFEST_FILE
    blob_path = d / CHECKPOINT_FILE
    if not manifest_path.is_file():
        raise ResumeError(f"no checkpoint manifest at {manifest_path}")
    if not blob_path.is_file():
        raise ResumeError(f"no checkpoint blob at {blob_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = manifest["keys"]
        if not isinstance(keys, dict):
            raise TypeError("manifest 'keys' is not an object")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ResumeError(f"corrupt manifest in {d}: {e}") from e
    return blob_path, keys


def load_checkpoint_blob(
    blob_path: Path,
    keys: dict,
    expected_keys: Iterable[str] | Mapping[str, Any] | None,
    d: Path,
) -> dict[str, Any]:
    """Unpickle the blob; enforce expected keys and manifest/blob agreement."""
    if expected_keys is not None:
        missing = sorted(flatten_expected(expected_keys) - set(keys))
        if missing:
            raise ResumeError(f"checkpoint manifest missing keys: {missing}")
    try:
        # weights_only=False is required: the payload carries numpy RNG
        # bit-states and deques outside torch's safe-globals allowlist. The
        # FutureWarning about the upcoming default flip is suppressed locally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = torch.load(blob_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ResumeError(f"failed to unpickle checkpoint blob {blob_path}: {e}") from e
    embedded = payload.pop(KEYS_FIELD, None) if isinstance(payload, dict) else None
    if embedded is not None and embedded != keys:
        raise ResumeError(
            f"manifest/blob mismatch in {d}: manifest keys differ from "
            "blob-embedded keys (previous save was killed between the two "
            "atomic renames)"
        )
    return payload


def load_checkpoint(
    dir_path: str | os.PathLike,
    expected_keys: Iterable[str] | Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    * Validates manifest completeness against ``expected_keys`` (iterable of
      dotted paths, or nested mapping whose leaf values are ignored); raises
      ResumeError listing every missing key.
    * If ``state`` is given, restores in-place into live objects found there:
      modules/optimizers via ``load_state_dict``, tensors via ``copy_``.
      Template slots set to ``None`` (or plain non-tensor values) are skipped
      — read those fields from the returned payload instead.
    * Returns the deserialized payload tree (modules/optimizers appear as
      their state_dicts).
    """
    d = Path(dir_path)
    blob_path, keys = read_checkpoint_manifest(d)
    payload = load_checkpoint_blob(blob_path, keys, expected_keys, d)
    if state is not None:
        try:
            restore_tree(state, payload, keys)
        except (KeyError, TypeError) as e:
            raise ResumeError(
                f"checkpoint structure does not match restore template in {d}: {e}"
            ) from e
    return payload


def restore_tree(
    template: Mapping[str, Any],
    payload: Mapping[str, Any],
    keys: Mapping[str, str],
    prefix: str = "",
) -> None:
    for name, tval in template.items():
        path = join_prefix(prefix, name)
        if name not in payload:
            raise KeyError(f"payload has no entry for template key '{path}'")
        node = payload[name]
        if isinstance(tval, dict):
            restore_tree(tval, node, keys, path)
            continue
        if tval is None:
            continue
        kind = keys.get(path)
        if kind == "module":
            tval.load_state_dict(node)
        elif kind == "optimizer":
            tval.load_state_dict(node)
        elif kind == "tensor":
            tval.copy_(node)
            # kind == "object": nothing to restore in-place; use returned payload
