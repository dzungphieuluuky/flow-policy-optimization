"""Policy checkpoint IO (HuggingFace style): save_pretrained/from_pretrained.

``save_policy_checkpoint`` writes ``config.json`` + ``model.safetensors``
(unwrapped eager keys, so artifacts are compile-mode independent) plus
``optimizer.pt`` and ``counters.json``; ``load_policy_checkpoint`` rebuilds
everything from the directory.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

OPTIMIZER_FILE = "optimizer.pt"
COUNTERS_FILE = "counters.json"


class ResumeError(RuntimeError):
    """Unresumable checkpoint (missing/corrupt files)."""


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
        opt_state = torch.load(
            d / OPTIMIZER_FILE, map_location="cpu", weights_only=True
        )
    except Exception as e:
        raise ResumeError(f"failed to unpickle {d / OPTIMIZER_FILE}: {e}") from e
    optimizer.load_state_dict(opt_state)
    try:
        counters = json.loads((d / COUNTERS_FILE).read_text(encoding="utf-8"))
        policy.step_count = int(counters["step_count"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise ResumeError(f"corrupt counters in {d}: {e}") from e
    return (
        policy,
        optimizer,
        {"global_step": int(counters["global_step"]), "step_count": policy.step_count},
    )
