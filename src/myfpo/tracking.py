"""Log scalars to wandb and tensorboard.

Two backends, one call site each. The trainer builds the backend list, then
calls :func:`log_metrics` and :func:`finish_backends`. An empty list means the
run logs nothing, which is also how the trainer knows to skip the metric pass.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "build_backends",
    "build_run_name",
    "finish_backends",
    "log_metrics",
]

_REPORT_TO = ("wandb", "tensorboard", "all", "none")


def build_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    """The run name: method, env, seed, the knobs that differ, and a timestamp.

    The timestamp tells two runs of the same config apart in the wandb list.
    """
    tags: list[str] = []
    get = cfg_dict.get
    if (v := get("action_horizon")) is not None:
        tags.append(f"h{v}")
    if (v := get("hidden_dim")) is not None and (d := get("dit_depth")) is not None:
        tags.append(f"d{v}x{d}")
    if (v := get("num_learning_epochs")) is not None:
        tags.append(f"ep{v}")
    if (v := get("policy_lr")) is not None:
        tags.append(f"lr{v:g}" if isinstance(v, float) else f"lr{v}")
    if (v := get("clip_param")) is not None:
        tags.append(f"eps{v:g}" if isinstance(v, float) else f"eps{v}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "_".join([f"{method}_{env_name}_s{seed}", *tags, stamp])


class WandbBackend:
    """Writes scalars to a wandb run."""

    def __init__(
        self,
        *,
        name: str,
        method: str,
        env_name: str,
        seed: int,
        total_steps: int,
        cfg_dict: dict,
    ):
        import wandb

        self._wandb = wandb
        wandb.init(
            project=method,
            group=f"{method}_{env_name}",
            name=name,
            tags=["mujoco"],
            config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
            job_type="smoke" if total_steps <= 10000 else "train",
            mode=os.environ.get("WANDB_MODE", "online"),
            reinit="finish_previous",
        )

    def log(self, data: dict[str, float], step: int) -> None:
        self._wandb.log(data, step=step)

    def finish(self, exit_code: int) -> None:
        self._wandb.finish(exit_code=exit_code)


class TensorboardBackend:
    """Writes scalars to a tensorboard event file."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, flush_secs=30)

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            if value is None:
                continue
            self.writer.add_scalar(key, float(value), step)

    def finish(self, _exit_code: int) -> None:
        self.writer.close()


def build_backends(
    cfg,
    *,
    out_dir: str | Path,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
) -> list:
    """The backends a run logs to. An empty list means no logging."""
    report_to = cfg.report_to
    if report_to not in _REPORT_TO:
        raise ValueError(
            f"report_to={report_to!r} must be one of {', '.join(_REPORT_TO)}"
        )
    if report_to == "none":
        return []

    cfg_dict = dataclasses.asdict(cfg)
    name = build_run_name(method, env_name, seed, cfg_dict)
    backends: list = []
    if report_to in ("wandb", "all"):
        backends.append(
            WandbBackend(
                name=name,
                method=method,
                env_name=env_name,
                seed=seed,
                total_steps=total_steps,
                cfg_dict=cfg_dict,
            )
        )
    if report_to in ("tensorboard", "all"):
        log_dir = Path(out_dir) / "tensorboard" / name
        backends.append(TensorboardBackend(str(log_dir)))
    return backends


def log_metrics(backends: list, data: dict[str, float], step: int) -> None:
    """Write one scalar dict to every backend."""
    for backend in backends:
        backend.log(data, step)


def finish_backends(backends: list, exit_code: int = 0) -> None:
    """Close every backend."""
    for backend in backends:
        backend.finish(exit_code)
