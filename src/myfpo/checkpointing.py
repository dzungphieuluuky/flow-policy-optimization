"""Policy checkpoint IO (HuggingFace style): save_pretrained/from_pretrained."""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

OPTIMIZER_FILE = "optimizer.pt"
COUNTERS_FILE = "counters.json"
RNG_FILE = "rng.pt"
ACTION_SPACE_FILE = "action_space.json"

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
    optimizer: torch.optim.Optimizer,
    global_step: int,
    ema_state: dict | None = None,
) -> None:
    """Save through the LeRobot ``HubMixin``. Then save ``optimizer.pt``, the EMA
    shadow, the counters, and the RNG state.
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(d, state_dict=policy.state_dict())
    if ema_state is not None:
        save_atomically(d / "ema.pt", lambda name: torch.save(ema_state, name))

    save_atomically(
        d / OPTIMIZER_FILE, lambda name: torch.save(optimizer.state_dict(), name)
    )

    counters = {
        "global_step": int(global_step),
        "step_count": int(getattr(policy, "step_count", 0)),
    }
    save_atomically(
        d / COUNTERS_FILE,
        lambda name: Path(name).write_bytes(
            json.dumps(counters, indent=2).encode("utf-8")
        ),
    )

    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    save_atomically(d / RNG_FILE, lambda name: torch.save(rng, name))

    save_atomically(
        d / ACTION_SPACE_FILE,
        lambda name: Path(name).write_text(
            json.dumps({"version": ACTION_SPACE_VERSION}, indent=2),
            encoding="utf-8",
        ),
    )


def load_optimizer(policy, d: Path) -> torch.optim.Optimizer:
    """Rebuild the optimizer from the policy, then load its state."""
    from myfpo.modeling import build_optimizer

    optimizer = build_optimizer(policy)

    try:
        opt_state = torch.load(
            d / OPTIMIZER_FILE, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as e:
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
    except (ValueError, RuntimeError, KeyError) as e:
        raise ResumeError(f"optimizer.load_state_dict failed: {e}") from e
    return optimizer


def load_rng(d: Path) -> None:
    """Restore torch and CUDA RNG state, best effort."""
    path = d / RNG_FILE
    if not path.is_file():
        return
    try:
        rng = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        warnings.warn(
            f"{path} is unreadable; the run resumes without the saved RNG state",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def check_action_space(d: Path) -> None:
    """Refuse a checkpoint whose action space is not this one.

    The critic reads the action chunk. A checkpoint written before the action
    space changed has a critic that was fit on the squashed chunk, and that
    is a different quantity from the raw chunk the critic reads now. Loading
    it would evaluate a policy against a value function that never saw its
    actions, so the load fails with the reason named.
    """
    path = d / ACTION_SPACE_FILE
    if not path.is_file():
        raise ResumeError(
            f"{d} has no {ACTION_SPACE_FILE}, so its action space is not "
            f"recorded. Checkpoints written before the action space changed "
            "were trained with the critic reading the squashed chunk, and "
            "the current critic reads the raw chunk; they are not "
            "interchangeable. Train a new run."
        )
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise ResumeError(f"corrupt {path}: {e}") from e
    version = recorded.get("version")
    if version != ACTION_SPACE_VERSION:
        raise ResumeError(
            f"{d} was written with action space {version!r}; this code trains "
            f"with {ACTION_SPACE_VERSION!r}. The critic's action input is a "
            "different quantity in the two, so the checkpoint cannot be "
            "resumed or evaluated."
        )


def load_policy_checkpoint(
    dir_path: str | os.PathLike,
    *,
    device: str | torch.device | None = None,
    action_low=None,
    action_high=None,
):
    """Load a checkpoint from :func:`save_policy_checkpoint`. Restores the policy,
    the optimizer, the EMA shadow, and the counters.
    """
    from myfpo.modeling import MyfpoPolicy

    d = Path(dir_path)
    missing = [
        f
        for f in (CONFIG_NAME, SAFETENSORS_SINGLE_FILE, OPTIMIZER_FILE, COUNTERS_FILE)
        if not (d / f).is_file()
    ]
    if missing:
        raise ResumeError(f"incomplete policy checkpoint in {d}: missing {missing}")

    check_action_space(d)

    policy = MyfpoPolicy.from_pretrained(d)
    if device is not None:
        policy.to(device)
    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is not None:
        policy.set_action_bounds(action_low, action_high)

    optimizer = load_optimizer(policy, d)

    ema_path = d / "ema.pt"
    if ema_path.is_file():
        restored = policy.load_ema_state_dict(
            torch.load(ema_path, map_location="cpu", weights_only=True)
        )
        if not restored:
            warnings.warn(
                f"{ema_path.name} does not match the current EMA coverage; "
                "reseeding the shadow from live weights instead of restoring it",
                RuntimeWarning,
                stacklevel=2,
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

    load_rng(d)

    counters = {"global_step": global_step, "step_count": step_count}
    return policy, optimizer, counters
