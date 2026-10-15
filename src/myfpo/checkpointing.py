"""Policy checkpoint IO (HuggingFace style): save_pretrained/from_pretrained.

Saved checkpoint dir:

* ``config.json``       — LeRobot ``PreTrainedConfig`` (unwrapped keys)
* ``model.safetensors`` — policy weights (unwrapped keys)
* ``optimizer.pt``      — AdamW state_dict (pickled, weights-only)
* ``counters.json``     — ``{global_step, step_count}``
* ``rng.pt``            — torch / cuda RNG state, when ``save_rng=True``

Optimizer, counters, and RNG go to a temp file then get replaced atomically.
Model and config follow the LeRobot/HuggingFace ``save_pretrained`` path and are
not claimed atomic as a whole multi-file generation.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
import tempfile
from pathlib import Path

import torch
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

OPTIMIZER_FILE = "optimizer.pt"
COUNTERS_FILE = "counters.json"
RNG_FILE = "rng.pt"


class ResumeError(RuntimeError):
    """Unresumable checkpoint: missing or corrupt files."""


def _reject_legacy_critic_checkpoint(d: Path) -> None:
    """Fail before construction if the checkpoint predates the exec-length input
    of the critic.

    Setting ``critic_condition_on_exec_length`` to True appends one input to the
    first layer of every critic member. The weight of that layer then has the
    shape ``[out, in+1]``.

    A checkpoint that predates the flag stores ``[out, in]``. Then
    ``from_pretrained`` fails with an opaque size mismatch. The worse case is
    that it loads non-strictly. The new column keeps its init value. The critic
    then ignores the feature that it was built for, and does so silently.

    The migration is deliberately NOT automatic.

    Copying an existing input column into the new slot would break the critic.
    A resumed critic would then assert "every chunk executed half the horizon"
    for the chunks that did not. The failure is easier to detect and to re-run
    than to get quietly wrong.
    """
    raw = (d / CONFIG_NAME).read_text(encoding="utf-8")
    cfg_keys = json.loads(raw).keys()
    if "critic_condition_on_exec_length" in cfg_keys:
        return
    raise ResumeError(
        f"checkpoint in {d} was written before critic_condition_on_exec_length "
        "existed, so its critic lacks the exec-length input and cannot be "
        "resumed into the current critic. Start a fresh run, or set "
        "critic_condition_on_exec_length: false to rebuild the old critic shape."
    )


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """Write bytes to a path atomically, durable across a crash."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def save_torch_atomically(path: Path, obj) -> None:
    """Pickle to a path atomically, cleaning up on failure.

    The function calls ``mkstemp``. It does not call a ``NamedTemporaryFile``
    block that exists only to mint a name. That block leaves an empty file
    behind for as long as ``torch.save`` takes to reopen the path.

    The function closes the file descriptor. ``torch.save`` writes to the path
    itself.

    Re-raises whatever ``torch.save`` raised, after removing the temp file.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
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
    """Save through the LeRobot ``HubMixin``. Then save ``optimizer.pt`` and
    ``counters.json``.

    The file ``model.safetensors`` holds the eager keys. So a fresh
    ``SJFPOFlowModel`` finds every key that it expects.

    Frozen behavior snapshots exist at run time only. The function drops their
    duplicate parameters.

    The argument ``optimizer`` keeps the actor and the critic in separate
    parameter groups. So one call to ``step`` drives both. The clipping stays
    per role.
    """
    # Deferred import: circular dependency checkpointing <-> modeling_sjfpo.
    from myfpo.modeling_sjfpo import SJFPOPolicy

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
        if not name.startswith(("old_actor.", "old_encoder."))
    }
    policy.save_pretrained(d, state_dict=state_dict)
    # EMA shadow is saved separately in ema.pt, NOT overlaid onto actor.* here:
    # the logical weights in model.safetensors stay the live policy, matching
    # FPO++ (finetune_online_rl.py) which resumes live and EMA independently.
    # Overwriting actor.* with the shadow made the restored live policy equal
    # the old EMA policy, silently changing the resumed training trajectory.
    if ema_state is not None:
        save_torch_atomically(d / "ema.pt", ema_state)

    save_torch_atomically(d / OPTIMIZER_FILE, optimizer.state_dict())

    counters = {
        "global_step": int(global_step),
        "step_count": int(getattr(policy, "step_count", 0)),
    }
    write_bytes_atomically(
        d / COUNTERS_FILE,
        json.dumps(counters, indent=2).encode("utf-8"),
    )

    if save_rng:
        rng = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        save_torch_atomically(d / RNG_FILE, rng)


def load_optimizer(policy, d: Path) -> torch.optim.Optimizer:
    """Rebuild the optimizer from the policy, then load its state.

    Builds through :func:`myfpo.modeling_sjfpo.build_optimizer`,
    the call the training loop makes, so saved and rebuilt param groups cannot
    drift. Fails loudly on a structural mismatch instead of relying on
    ``load_state_dict``'s opaque errors.
    """
    # Deferred import (circular: checkpointing uses modeling_sjfpo optimizer).
    from myfpo.modeling_sjfpo import build_optimizer

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
    """Restore torch and CUDA RNG state, best effort.

    A missing or corrupt file is ignored. Losing bit-reproducibility should not
    block a resume.
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
    """Load a checkpoint from :func:`save_policy_checkpoint` ->
    ``(policy, optimizer, counters)``.

    Rebuilds every component from ``config.json``, no caller-supplied dims.
    Action bounds are runtime state: pass both ``action_low`` and ``action_high``
    to restore the postprocessor mapping.
    """
    from myfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

    d = Path(dir_path)
    missing = [
        f
        for f in (CONFIG_NAME, SAFETENSORS_SINGLE_FILE, OPTIMIZER_FILE, COUNTERS_FILE)
        if not (d / f).is_file()
    ]
    if missing:
        raise ResumeError(f"incomplete policy checkpoint in {d}: missing {missing}")

    _reject_legacy_critic_checkpoint(d)

    policy = SJFPOPolicy.from_pretrained(d)
    if device is not None:
        policy.to(device)
    policy.snapshot_behavior_policy()
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
            # Pre-encoder-coverage shadow. Re-seeding beats half-applying it.
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

    if restore_rng:
        load_rng(d)

    counters = {"global_step": global_step, "step_count": step_count}
    return policy, optimizer, counters
