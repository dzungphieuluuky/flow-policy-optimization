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
    """Write bytes to a path atomically, durable across a crash.

    Args:
        path: Destination path.
        data: Bytes to write.
    """
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

    Args:
        path: Destination path.
        obj: Object to serialise with ``torch.save``.

    Raises:
        Whatever ``torch.save`` raised, after removing the temporary file.
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
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    global_step: int,
    ema_state: dict | None = None,
    save_rng: bool = True,
) -> None:
    """Save through LeRobot's ``HubMixin`` (+ optimizer.pt, counters.json).

    ``model.safetensors`` holds eager keys, so a fresh ``SJFPOFlowModel`` in
    :func:`load_policy_checkpoint` finds every key it expects.

    The frozen behavior snapshots are runtime-only and are refreshed before the
    next rollout, so their duplicate parameters are not persisted.

    Args:
        dir_path: Directory the checkpoint is written to; created if absent.
        policy: The :class:`~sjfpo.modeling_sjfpo.SJFPOPolicy` to save.
        optimizer: Legacy single optimizer, for checkpoints predating the
            split.
        actor_optimizer: Actor optimizer for the split format.
        critic_optimizer: Critic optimizer for the split format; required
            alongside ``actor_optimizer``.
        global_step: Environment-step count recorded in the counters file.
        ema_state: Optional EMA shadow state, saved once the warmup allows.
        save_rng: Whether to persist torch and CUDA RNG states.

    Raises:
        ValueError: If ``policy`` is not an SJFPO policy, if ``global_step`` is
            negative, or if the optimizer arguments are inconsistent.
    """
    from sjfpo.modeling_sjfpo import SJFPOPolicy  # lazy: import-light module

    if not isinstance(policy, SJFPOPolicy):
        raise ValueError(
            f"save_policy_checkpoint needs an SJFPOPolicy, got {type(policy).__name__}"
        )
    if int(global_step) < 0:
        raise ValueError(f"global_step={global_step} must be >= 0")
    if (actor_optimizer is None) != (critic_optimizer is None):
        raise ValueError("actor_optimizer and critic_optimizer must be provided together")
    if actor_optimizer is None and optimizer is None:
        raise ValueError("provide actor/critic optimizers or legacy optimizer")

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

    if actor_optimizer is not None:
        optimizer_state = {
            "format": 2,
            "actor": actor_optimizer.state_dict(),
            "critic": critic_optimizer.state_dict(),
        }
    else:
        optimizer_state = optimizer.state_dict()
    _atomic_torch_save(d / OPTIMIZER_FILE, optimizer_state)

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
    """Rebuild a single optimizer from the current config and load its state.

    Fails loudly on any structural mismatch (wrong number of param groups,
    wrong key set) rather than relying on ``load_state_dict``'s opaque
    error messages.

    Args:
        policy: Policy supplying the param groups and preset.
        d: Checkpoint directory holding ``optimizer.pt``.

    Returns:
        The optimizer with its restored state.

    Raises:
        ResumeError: If the file cannot be unpickled or the state does not
            match the current config.
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


def _load_optimizer_pair(policy, d: Path):
    """Load the split actor/critic optimizers, falling back to the legacy format.

    Args:
        policy: Policy supplying the param groups and presets.
        d: Checkpoint directory holding ``optimizer.pt``.

    Returns:
        Tuple of the actor optimizer and the critic optimizer, or the legacy
        optimizer and None.

    Raises:
        ResumeError: If the file cannot be unpickled, is a split-format
            payload with missing keys, or the state does not match the config.
    """
    try:
        payload = torch.load(
            d / OPTIMIZER_FILE, map_location="cpu", weights_only=True
        )
    except Exception as e:
        raise ResumeError(
            f"failed to unpickle {d / OPTIMIZER_FILE}: {e}. "
            "If the optimizer was subclassed with non-tensor state, re-save "
            "with weights_only=False at your own risk."
        ) from e

    if not isinstance(payload, dict) or payload.get("format") != 2:
        return _load_optimizer(policy, d), None

    try:
        actor_state = payload["actor"]
        critic_state = payload["critic"]
    except (KeyError, TypeError) as e:
        raise ResumeError(f"malformed split optimizer payload in {d}: {e}") from e

    actor = policy.config.get_optimizer_preset(lr=policy.config.policy_lr).build(
        policy.get_actor_optim_params()["params"]
    )
    critic = policy.config.get_optimizer_preset(lr=policy.config.critic_lr).build(
        policy.get_critic_optim_params()["params"]
    )
    try:
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
    except Exception as e:
        raise ResumeError(f"split optimizer load_state_dict failed: {e}") from e
    return actor, critic


def _load_rng(d: Path) -> None:
    """Restore torch and CUDA RNG state, best effort.

    A missing or corrupt RNG file is ignored rather than raised on, since
    losing bit-reproducibility should not block a resume.

    Args:
        d: Checkpoint directory that may hold ``rng.pt``.
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
    New split checkpoints return ``(policy, actor_optimizer, critic_optimizer,
    counters)``; legacy checkpoints return ``(policy, optimizer, counters)``.

    Args:
        dir_path: Checkpoint directory written by
            :func:`save_policy_checkpoint`.
        device: Device the loaded policy is moved to; left in place when
            omitted.
        restore_rng: Whether to restore the saved RNG state.
        action_low: Per-dimension lower env action bounds.
        action_high: Per-dimension upper env action bounds.

    Returns:
        The policy, its optimizers, and the restored counters.

    Raises:
        ResumeError: If the checkpoint is incomplete or any saved component is
            corrupt or structurally mismatched.
        ValueError: If only one of ``action_low`` and ``action_high`` is given.
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

    actor_optimizer, critic_optimizer = _load_optimizer_pair(policy, d)

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
    if critic_optimizer is None:
        return policy, actor_optimizer, counters
    return policy, actor_optimizer, critic_optimizer, counters
