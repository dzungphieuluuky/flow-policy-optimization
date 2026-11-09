"""Policy checkpoint IO: one ``torch.save`` payload per checkpoint directory."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

import torch

from myfpo.configuration import MyfpoConfig

CHECKPOINT_FILE = "checkpoint.pt"

# The payload layout. Bump it when a key changes meaning.
FORMAT_VERSION = 1

# The action space the policy is trained in. "raw" means the flow's own output
# is the training target and the squash is applied on the way to the env.
# A checkpoint written before this change has a critic that was fit on the
# squashed actions, so its action space is a different quantity.
ACTION_SPACE_VERSION = "raw"


class ResumeError(RuntimeError):
    """Unresumable checkpoint: missing or corrupt files."""


def save_atomically(path: Path, write: Callable[[str], None]) -> None:
    """Write through a temp file beside the target, then replace it.

    A crash leaves the previous file intact: the target is only replaced once
    the write has finished.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        write(tmp_name)
        Path(tmp_name).replace(path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_name).unlink()
        raise


def save_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    policy,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    ema_state: dict | None = None,
    scheduler_state: dict | None = None,
) -> None:
    """Save the policy, the optimizer, the EMA shadow, the counters, the
    optional scheduler state, and the RNG."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()

    payload = {
        "format": FORMAT_VERSION,
        "action_space": ACTION_SPACE_VERSION,
        "config": dataclasses.asdict(policy.config),
        "state_dict": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "counters": {
            "global_step": int(global_step),
            "step_count": int(getattr(policy, "step_count", 0)),
        },
        "ema": ema_state,
        # Optional: a run that uses no schedule, and every checkpoint written
        # before this field existed, carries None. Both load as "restart the
        # curve from global_step".
        "scheduler": scheduler_state,
        "rng": rng,
    }
    save_atomically(
        d / CHECKPOINT_FILE, lambda name: torch.save(payload, name)
    )


def load_optimizer(policy, opt_state: dict) -> torch.optim.Optimizer:
    """Rebuild the optimizer from the policy, then load its state."""
    from myfpo.modeling import make_optimizer

    optimizer = make_optimizer(policy)
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
    except (ValueError, RuntimeError, KeyError) as e:
        raise ResumeError(f"optimizer.load_state_dict failed: {e}") from e
    return optimizer


def restore_rng(rng: dict | None) -> None:
    """Restore torch and CUDA RNG state, best effort."""
    if not rng:
        return
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def load_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    device: str | torch.device | None = None,
    action_low=None,
    action_high=None,
):
    """Load a checkpoint from :func:`save_policy_checkpoint`. Restores the policy,
    the optimizer, the EMA shadow, and the counters. A checkpoint that carries
    scheduler state adds it to ``counters["scheduler"]``. A policy-only
    checkpoint (``optimizer=None``) loads with ``optimizer=None``.
    """
    from myfpo.modeling import MyfpoPolicy

    d = Path(dir_path)
    path = d / CHECKPOINT_FILE
    if not path.is_file():
        raise ResumeError(
            f"{d} has no {CHECKPOINT_FILE}. The checkpoint format is one "
            "torch.save payload per directory; runs saved in the old "
            "HuggingFace layout cannot be resumed."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as e:
        raise ResumeError(f"failed to load {path}: {e}") from e
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ResumeError(
            f"{path} has format {found!r}; this code reads "
            f"format {FORMAT_VERSION}."
        )
    if payload.get("action_space") != ACTION_SPACE_VERSION:
        raise ResumeError(
            f"{path} was written with action space "
            f"{payload.get('action_space')!r}; this code trains with "
            f"{ACTION_SPACE_VERSION!r}. The critic's action input is a "
            "different quantity in the two, so the checkpoint cannot be "
            "resumed or evaluated."
        )

    try:
        cfg_payload = dict(payload["config"])
        policy = MyfpoPolicy(MyfpoConfig(**cfg_payload))
        policy.load_state_dict(payload["state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        raise ResumeError(f"cannot rebuild the policy from {path}: {e}") from e
    if device is not None:
        policy.to(device)
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is not None:
        policy.set_action_bounds(action_low, action_high)

    opt_state = payload.get("optimizer")
    # A policy-only checkpoint (saved with ``optimizer=None``) is still valid
    # for inference and for a determinism probe. A training resume must add its
    # own guard, because it needs the optimizer state.
    optimizer = load_optimizer(policy, opt_state) if opt_state is not None else None

    if payload.get("ema") is not None:
        restored = policy.load_ema_state_dict(payload["ema"])
        if not restored:
            warnings.warn(
                "the saved EMA shadow does not match the current EMA coverage; "
                "reseeding the shadow from live weights instead of restoring it",
                RuntimeWarning,
                stacklevel=2,
            )

    counters_raw = payload.get("counters")
    if not isinstance(counters_raw, dict):
        raise ResumeError(f"{path} has no counters")
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

    restore_rng(payload.get("rng"))

    counters = {"global_step": global_step, "step_count": step_count}
    if payload.get("scheduler") is not None:
        counters["scheduler"] = payload["scheduler"]
    return policy, optimizer, counters
