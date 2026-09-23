"""Policy checkpoint IO (HuggingFace style): save_pretrained/from_pretrained.

Layout of a saved checkpoint directory:

* ``config.json``                — LeRobot ``PreTrainedConfig`` (unwrapped keys)
* ``model.safetensors``          — policy weights (unwrapped keys)
* ``optimizer.pt``               — AdamW state_dict (pickled, weights-only)
* ``counters.json``              — ``{global_step, step_count}``
* ``rng.pt``                     — torch / cuda RNG state (optional, when
                                    ``save_rng=True``)

All file writes are atomic within the destination directory: temp files are
written first, then renamed into place, so a crash mid-save cannot leave a
half-written checkpoint that ``load_policy_checkpoint`` would accept.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from lerobot.utils.constants import CHECKPOINTS_DIR
from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

OPTIMIZER_FILE = "optimizer.pt"
COUNTERS_FILE = "counters.json"
RNG_FILE = "rng.pt"


class ResumeError(RuntimeError):
    """Unresumable checkpoint (missing/corrupt files)."""


# ---------------------------------------------------------------------------
# torch.compile unwrapping
# ---------------------------------------------------------------------------


def _iter_compiled_children(
    policy,
) -> list[tuple[str, torch.nn.Module, torch.nn.Module]]:
    """Return ``(name, compiled_module, inner_module)`` for every top-level
    submodule of ``policy`` wrapped by ``torch.compile``.

    Only top-level children are inspected; ``torch.compile`` is applied
    per-module via ``maybe_compile`` and never nested in this codebase.
    """
    out = []
    for name, child in policy.named_children():
        inner = getattr(child, "_orig_mod", None)
        if isinstance(inner, torch.nn.Module):
            out.append((name, child, inner))
    return out


@contextlib.contextmanager
def unwrapped_policy(policy):
    """Expose a compiled policy's eager submodules for checkpoint IO.

    Swaps every ``torch.compile`` wrapper (``X._orig_mod``) for its inner
    module so ``state_dict()`` emits unwrapped keys. Restores the wrappers on
    exit, even on exception.
    """
    swapped = _iter_compiled_children(policy)
    for name, _compiled, inner in swapped:
        setattr(policy, name, inner)
    try:
        yield policy
    finally:
        for name, compiled, _inner in swapped:
            setattr(policy, name, compiled)


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _atomic_torch_save(path: Path, obj) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp_name = tmp.name
    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    policy,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    save_rng: bool = True,
) -> None:
    """Save through LeRobot's ``HubMixin`` (+ optimizer.pt, counters.json).

    ``model.safetensors`` holds **unwrapped** eager keys (torch.compile
    wrappers are swapped out for the duration of ``save_pretrained``), so a
    fresh, uncompiled ``SJFPOFlowModel`` in :func:`load_policy_checkpoint`
    finds every key it expects.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

    if not isinstance(policy, SJFPOPolicy):
        raise ValueError(
            f"save_policy_checkpoint needs an SJFOPolicy, got {type(policy).__name__}"
        )
    if int(global_step) < 0:
        raise ValueError(f"global_step={global_step} must be >= 0")

    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    # 1) Weights (through LeRobot; unwrapped so keys match a fresh policy).
    with unwrapped_policy(policy):
        policy.save_pretrained(d)

    # 2) Optimizer state (pickled, weights-only safe on load).
    _atomic_torch_save(d / OPTIMIZER_FILE, optimizer.state_dict())

    # 3) Counters (atomic JSON).
    counters = {
        "global_step": int(global_step),
        "step_count": int(getattr(policy, "step_count", 0)),
    }
    _atomic_write_bytes(
        d / COUNTERS_FILE,
        json.dumps(counters, indent=2).encode("utf-8"),
    )

    # 4) RNG state (torch + cuda if available) — enables bit-reproducible resume.
    if save_rng:
        rng = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        _atomic_torch_save(d / RNG_FILE, rng)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _load_optimizer(policy, d: Path) -> torch.optim.Optimizer:
    """Rebuild the optimizer with the current config and load saved state.

    Fails loudly on any structural mismatch (wrong number of param groups,
    wrong key set) rather than relying on ``load_state_dict``'s opaque
    error messages.
    """
    groups = policy.get_optim_params()["params"]
    optimizer = policy.config.get_optimizer_preset().build(groups)

    try:
        opt_state = torch.load(
            d / OPTIMIZER_FILE, map_location="cpu", weights_only=True
        )
    except Exception as e:
        raise ResumeError(
            f"failed to unpickle {d / OPTIMIZER_FILE}: {e}. "
            "If the optimizer was subclassed with non-tensor state, re-save "
            "with weights_only=False at your own risk."
        ) from e

    saved_groups = opt_state.get("param_groups", [])
    if len(saved_groups) != len(optimizer.param_groups):
        raise ResumeError(
            f"optimizer param_groups mismatch: checkpoint has "
            f"{len(saved_groups)}, current config expects "
            f"{len(optimizer.param_groups)}. The config used to save differs "
            "from the current one — cannot resume safely."
        )
    try:
        optimizer.load_state_dict(opt_state)
    except Exception as e:
        raise ResumeError(f"optimizer.load_state_dict failed: {e}") from e
    return optimizer


def _load_rng(d: Path) -> None:
    """Restore torch/cuda RNG state if a saved RNG file exists. Best effort."""
    path = d / RNG_FILE
    if not path.is_file():
        return
    try:
        rng = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return  # corrupt RNG file shouldn't block resume
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def load_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    device: str | torch.device | None = None,
    restore_rng: bool = True,
):
    """Load a checkpoint written by :func:`save_policy_checkpoint`.

    Rebuilds every component from ``config.json`` — no caller-supplied dims.
    Returns ``(policy, optimizer, counters)``.
    """
    from sjfpo.modeling_sjfpo import SJFOPolicy  # lazy: import-light module

    d = Path(dir_path)
    missing = [
        f
        for f in (CONFIG_NAME, SAFETENSORS_SINGLE_FILE, OPTIMIZER_FILE, COUNTERS_FILE)
        if not (d / f).is_file()
    ]
    if missing:
        raise ResumeError(f"incomplete policy checkpoint in {d}: missing {missing}")

    # 1) Policy (fresh, uncompiled — state_dict keys are unwrapped by save).
    policy = SJFPOPolicy.from_pretrained(d)
    if device is not None:
        policy.to(device)

    # 2) Optimizer (built from current config, state from disk).
    optimizer = _load_optimizer(policy, d)

    # 3) Counters (validated).
    try:
        counters_raw = json.loads((d / COUNTERS_FILE).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ResumeError(f"corrupt {d / COUNTERS_FILE}: {e}") from e
    try:
        global_step = int(counters_raw["global_step"])
        step_count = int(counters_raw["step_count"])
    except (KeyError, TypeError, ValueError) as e:
        raise ResumeError(f"malformed counters in {d}: {e}") from e
    if global_step < 0 or step_count < 0:
        raise ResumeError(
            f"negative counters in {d}: global_step={global_step}, "
            f"step_count={step_count}"
        )
    policy.step_count = step_count

    # 4) RNG state (best-effort).
    if restore_rng:
        _load_rng(d)

    return policy, optimizer, {"global_step": global_step, "step_count": step_count}
