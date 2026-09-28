"""Policy checkpoint IO (HuggingFace style): save_pretrained/from_pretrained.

Layout of a saved checkpoint directory:

* ``config.json``                — LeRobot ``PreTrainedConfig`` (unwrapped keys)
* ``model.safetensors``          — policy weights (unwrapped keys)
* ``optimizer.pt``               — AdamW state_dict (pickled, weights-only)
* ``counters.json``              — ``{global_step, step_count}``
* ``rng.pt``                     — torch / cuda RNG state (optional, when
                                    ``save_rng=True``)

Optimizer, counters, and RNG files use temporary files followed by atomic
replacement. Policy model/config serialization follows the LeRobot/Hugging
Face ``save_pretrained`` path and is not claimed to be atomic as a complete
multi-file checkpoint generation.
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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a path atomically, durable across a crash."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _atomic_torch_save(path: Path, obj) -> None:
    """Pickle an object to a path atomically, cleaning up on failure.

    Re-raises whatever ``torch.save`` raised, after removing the temp file.
    """
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


def save_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    policy,
    optimizer: torch.optim.Optimizer | None = None,
    global_step: int,
    ema_state: dict | None = None,
    save_rng: bool = True,
) -> None:
    """Save through LeRobot's ``HubMixin`` (+ optimizer.pt, counters.json).

    ``model.safetensors`` holds eager keys, so a fresh ``SJFPOFlowModel`` in
    :func:`load_policy_checkpoint` finds every key it expects. The frozen behavior
    snapshots are runtime-only, so their duplicate parameters are dropped.

    ``optimizer`` keeps actor and critic in separate param groups, so one
    ``step`` drives both while grad clipping stays per role.

    Raises ``ValueError`` if ``policy`` is not an SJFPO policy, if
    ``global_step`` is negative, or if ``optimizer`` is missing.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

    if not isinstance(policy, SJFPOPolicy):
        raise ValueError(
            f"save_policy_checkpoint needs an SJFPOPolicy, got {type(policy).__name__}"
        )
    if int(global_step) < 0:
        raise ValueError(f"global_step={global_step} must be >= 0")
    if optimizer is None:
        raise ValueError("provide an optimizer")

    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor
        for name, tensor in policy.state_dict().items()
        if not name.startswith(("old_actor.", "old_conditioner."))
    }
    if ema_state is not None and policy.ema_ready(global_step):
        for name, shadow in ema_state["shadow_params"].items():
            full_name = f"actor.{name}"
            if full_name in state_dict:
                state_dict[full_name] = shadow.detach().clone()
    policy.save_pretrained(d, state_dict=state_dict)
    if ema_state is not None:
        _atomic_torch_save(d / "ema.pt", ema_state)

    _atomic_torch_save(d / OPTIMIZER_FILE, optimizer.state_dict())

    counters = {
        "global_step": int(global_step),
        "step_count": int(getattr(policy, "step_count", 0)),
    }
    _atomic_write_bytes(
        d / COUNTERS_FILE,
        json.dumps(counters, indent=2).encode("utf-8"),
    )

    if save_rng:
        rng = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        _atomic_torch_save(d / RNG_FILE, rng)


def _load_optimizer(policy, d: Path) -> torch.optim.Optimizer:
    """Rebuild the optimizer from the policy and load its state.

    Builds through :func:`sjfpo.modeling_sjfpo.build_optimizer`, the same
    construction the training loop uses, so the saved and rebuilt param groups
    cannot drift apart. Fails loudly on any structural mismatch (wrong number of
    param groups, wrong key set) rather than relying on ``load_state_dict``'s
    opaque error messages. Raises ``ResumeError`` if the file cannot be unpickled
    or the state does not match the current config.
    """
    from sjfpo.modeling_sjfpo import build_optimizer  # lazy: import-light module

    optimizer = build_optimizer(policy)

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

    if isinstance(opt_state, dict) and opt_state.get("format") == 2:
        raise ResumeError(
            f"{d / OPTIMIZER_FILE} holds separate actor/critic optimizers, which "
            "predate the merged single optimizer. That state cannot be loaded "
            "into one optimizer; resume from a checkpoint saved after the merge."
        )
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
    """Restore torch and CUDA RNG state, best effort.

    A missing or corrupt RNG file is ignored rather than raised on, since
    losing bit-reproducibility should not block a resume.
    """
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
    action_low=None,
    action_high=None,
):
    """Load a checkpoint written by :func:`save_policy_checkpoint`.

    Rebuilds every component from ``config.json`` — no caller-supplied dims.
    Environment action bounds are runtime state; pass both ``action_low`` and
    ``action_high`` to restore the current postprocessor mapping.
    Returns ``(policy, optimizer, counters)``.

    Raises ``ResumeError`` if the checkpoint is incomplete or any saved
    component is corrupt or structurally mismatched, and ``ValueError`` if
    only one of ``action_low`` and ``action_high`` is given.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

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
    policy.snapshot_behavior_policy()
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is not None:
        policy.set_action_bounds(action_low, action_high)

    optimizer = _load_optimizer(policy, d)

    ema_path = d / "ema.pt"
    if ema_path.is_file():
        policy.load_ema_state_dict(
            torch.load(ema_path, map_location="cpu", weights_only=True)
        )

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

    if restore_rng:
        _load_rng(d)

    counters = {"global_step": global_step, "step_count": step_count}
    return policy, optimizer, counters
