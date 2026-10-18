from __future__ import annotations

import dataclasses
from pathlib import Path

from .base import (
    ExperimentTracker,
    ExperimentTrackerBackend,
    compute_eval_summary_metrics,
)

__all__ = [
    "ExperimentTracker",
    "ExperimentTrackerBackend",
    "build_tracker",
    "compute_eval_summary_metrics",
]

_REPORT_TO = ("wandb", "tensorboard", "all", "none")


def build_tracker(
    cfg,
    *,
    out_dir: str | Path,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
) -> ExperimentTracker:
    """Build the tracker fan-out for one run."""
    report_to = getattr(cfg, "report_to", "none")
    if report_to not in _REPORT_TO:
        raise ValueError(
            f"report_to={report_to!r} must be one of {', '.join(_REPORT_TO)}"
        )
    if report_to == "none":
        return ExperimentTracker([])

    cfg_dict = dataclasses.asdict(cfg)
    backends: list[ExperimentTrackerBackend] = []
    if report_to in ("wandb", "all"):
        from .wandb import WandbBackend

        backends.append(
            WandbBackend(
                method=method,
                env_name=env_name,
                seed=seed,
                total_steps=total_steps,
                cfg_dict=cfg_dict,
            )
        )
    if report_to in ("tensorboard", "all"):
        from .base import build_run_name
        from .tensorboard import TensorboardBackend

        log_dir = Path(out_dir) / "tensorboard" / build_run_name(
            method, env_name, seed, cfg_dict
        )
        backends.append(TensorboardBackend(str(log_dir)))
    return ExperimentTracker(backends)
