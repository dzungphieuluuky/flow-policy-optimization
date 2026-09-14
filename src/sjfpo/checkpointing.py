"""Checkpoint/resume IO: atomic save/load with manifest completeness.

``state`` is a nested mapping of modules / optimizers / tensors / picklable
objects (only plain dicts nest). On disk: ``checkpoint.pt`` (payload +
embedded manifest keys as torn-write detector) + ``manifest.json`` sidecar,
both via tmp-file + ``os.replace``. Same-stack resume only (see env
fingerprint); mismatches warn loudly, never silenced.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

CHECKPOINT_FILE = "checkpoint.pt"
MANIFEST_FILE = "manifest.json"
FORMAT = 1
KEYS_FIELD = "_manifest_keys"

# Policy-checkpoint files (HubMixin + training state).
OPTIMIZER_FILE = "optimizer.pt"
COUNTERS_FILE = "counters.json"


class ResumeError(RuntimeError):
    """Unresumable checkpoint (missing/corrupt manifest, blob mismatch)."""


def unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    """Return the eager module inside a torch.compile wrapper, else itself.

    The wrapper shares the inner module's tensors, so state_dicts differ
    only in the ``_orig_mod.`` key prefix — values are identical.
    """
    return getattr(module, "_orig_mod", module)


@contextlib.contextmanager
def unwrapped_policy(policy):
    """Temporarily expose a policy's eager submodules for checkpoint IO.

    Swap torch.compile wrappers for their inner modules (and swap back
    afterwards); params are shared, so training continues untouched and the
    artifact holds plain eager keys whatever SJFPO_COMPILE was.
    """
    swapped = []
    for name in ("actor", "old_actor", "critic"):
        module = getattr(policy, name, None)
        if getattr(module, "_orig_mod", None) is not None:
            setattr(policy, name, module._orig_mod)
            swapped.append((name, module))
    try:
        yield policy
    finally:
        for name, module in swapped:
            setattr(policy, name, module)


def capture_env_fingerprint() -> dict[str, str | None]:
    """Same-stack resume fingerprint (gpu/cpu, torch + cuda versions)."""
    try:
        gpu_name: str | None = torch.cuda.get_device_name(0)
    except Exception:
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
            payload[name] = unwrap_compiled(value).state_dict()
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
    """Atomically write ``state`` as checkpoint.pt + manifest.json."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    payload, keys = serialize_tree(state)
    payload[KEYS_FIELD] = dict(keys)
    manifest = {"format": FORMAT, "keys": keys}

    blob_tmp = d / (CHECKPOINT_FILE + ".tmp")
    manifest_tmp = d / (MANIFEST_FILE + ".tmp")
    torch.save(payload, blob_tmp)
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(blob_tmp, d / CHECKPOINT_FILE)
    os.replace(manifest_tmp, d / MANIFEST_FILE)


def save_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    policy,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> None:
    """Save through LeRobot's HubMixin (+ optimizer.pt, counters.json).

    ``model.safetensors`` holds unwrapped eager keys, so artifacts are
    compile-mode independent. ``config.json`` carries shapes + hypers.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: keeps this module import-light

    if not isinstance(policy, SJFPOPolicy):
        raise ValueError(
            f"save_policy_checkpoint needs an SJFPOPolicy, got {type(policy).__name__}"
        )
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    with unwrapped_policy(policy):
        policy.save_pretrained(d)
    torch.save(optimizer.state_dict(), d / OPTIMIZER_FILE)
    (d / COUNTERS_FILE).write_text(
        json.dumps(
            {"global_step": int(global_step), "step_count": int(policy.step_count)},
            indent=2,
        ),
        encoding="utf-8",
    )


def load_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    device: str | torch.device | None = None,
):
    """Load a checkpoint from :func:`save_policy_checkpoint`.

    Rebuilds components from ``config.json`` (no caller dims); policy left in
eval mode. Raises ResumeError on missing/unreadable files.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: keeps this module import-light

    d = Path(dir_path)
    missing = [
        f
        for f in (CONFIG_NAME, SAFETENSORS_SINGLE_FILE, OPTIMIZER_FILE, COUNTERS_FILE)
        if not (d / f).is_file()
    ]
    if missing:
        raise ResumeError(f"incomplete policy checkpoint in {d}: missing {missing}")
    policy = SJFPOPolicy.from_pretrained(d)
    if device is not None:
        policy.to(device)
    optimizer = policy.config.get_optimizer_preset().build(
        policy.get_optim_params()["params"]
    )
    try:
        opt_state = torch.load(d / OPTIMIZER_FILE, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ResumeError(f"failed to unpickle {d / OPTIMIZER_FILE}: {e}") from e
    optimizer.load_state_dict(opt_state)
    try:
        counters = json.loads((d / COUNTERS_FILE).read_text(encoding="utf-8"))
        policy.step_count = int(counters["step_count"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ResumeError(f"corrupt counters in {d}: {e}") from e
    return policy, optimizer, {"global_step": int(counters["global_step"]), "step_count": policy.step_count}


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
    """Locate blob + manifest paths and validate the manifest envelope."""
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
    """Unpickle the blob; enforce expected keys and manifest/blob agreement.

    weights_only=False: the payload carries numpy RNG bit-states and deques
    outside torch's safe-globals allowlist.
    """
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
    """Load a checkpoint from :func:`save_checkpoint` (validates + restores)."""
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
            # Load into the eager module: same tensors as any compile
            # wrapper, but plain keys (see save side above).
            unwrap_compiled(tval).load_state_dict(node)
        elif kind == "optimizer":
            tval.load_state_dict(node)
        elif kind == "tensor":
            tval.copy_(node)
            # kind == "object": nothing to restore in-place; use returned payload
