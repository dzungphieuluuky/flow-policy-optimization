"""wandb logging helper — enabled by default.

Only file that imports wandb. wandb is a REQUIRED dependency (module-level
import: a missing package is a hard ImportError, never a silent no-op). The
ONLY off switch is ``SJFPO_DISABLE_WANDB=1``. When enabled, ``wandb.init`` runs
unconditionally — if it raises (missing key, no network) the exception
propagates and the run FAILS FAST.
"""

from __future__ import annotations

import os

import numpy as np

import wandb


def wandb_disabled() -> bool:
    """Return True only when the single off switch is set."""
    return os.environ.get("SJFPO_DISABLE_WANDB") == "1"


def resolve_wandb_job_type(total_steps: int) -> str:
    """Map the training budget to the wandb job_type label."""
    return "smoke" if total_steps <= 10000 else "train"


def open_wandb_run(
    *,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    cfg_dict: dict,
) -> None:
    """Open the per-algo run; init errors propagate (fail fast)."""
    wandb.init(
        project=method,
        group=f"{method}_{env_name}",
        name=None,
        tags=[method, "mujoco", "cpu-first"],
        config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
        job_type=resolve_wandb_job_type(total_steps),
        mode=os.environ.get("WANDB_MODE", "online"),
        reinit="finish_previous",
        force=True,
    )


def init_wandb_run_for_method(
    *,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    cfg_dict: dict,
) -> bool:
    """Init a run in the per-algo project.

    Uses the default wandb step (``step=global_step``) as the x-axis so all
    charts render as line graphs, matching wandb system auto-logging.

    Args:
        method: algo name — also the wandb project (``"sjfpo"/"ppo"/"sac"``).
        env_name: MuJoCo env id.
        seed: run seed (stored in config; the run name is wandb-generated).
        total_steps: training budget (<=10000 → job_type "smoke").
        cfg_dict: base config dict.

    Returns:
        True when enabled, False when disabled.

    Raises:
        Exception: any ``wandb.init`` error propagates — a
            failed init FAILS FAST, never degrades silently.
    """
    if wandb_disabled():
        return False
    open_wandb_run(
        method=method,
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=cfg_dict,
    )
    return True


def log_metrics(data: dict, step: int) -> None:
    """Log a dict at a global step; no-op when disabled."""
    if wandb_disabled():
        return
    wandb.log(data, step=step)


def log_eval_details(
    eval_results: list[dict], all_eval_returns: list[float], global_step: int
) -> None:
    """Log per-episode Table + reward improvement (shared runner/baselines shape).

    Single commit per step so scalar history stays dense (one row per
    global_step, like wandb system auto-logging).
    """
    if wandb_disabled():
        return
    data: dict = {}
    if eval_results:
        rows = [(r["episode"], r["return"], r["length"]) for r in eval_results]
        data["eval/episodes"] = wandb.Table(
            columns=["episode", "return", "length"],
            data=[list(r) for r in rows],
        )
    if len(all_eval_returns) >= 2:
        prev, curr = all_eval_returns[-2], all_eval_returns[-1]
        data["eval/reward_improvement"] = curr - prev
        data["eval/reward_improvement_pct"] = (curr - prev) / max(abs(prev), 1e-8)
    if data:
        log_metrics(data, step=global_step)


def log_eval_summary(
    mean_return: float,
    eval_results: list[dict],
    all_eval_returns: list[float],
    eval_count: int,
    global_step: int,
    *,
    r_rand: float | None = None,
    consecutive_below: int | None = None,
) -> None:
    """Log all eval metrics plus per-episode details.

    Single shared home for the runner and baselines eval-logging payload
    (previously duplicated per loop); one commit per step so eval/* renders
    as line graphs. No-op when disabled.
    """
    if wandb_disabled():
        return
    returns = [r["return"] for r in eval_results]
    eval_lengths = [r["length"] for r in eval_results]
    data: dict = {
        "eval/mean_return": mean_return,
        "eval/min_return": float(np.min(returns)) if returns else 0.0,
        "eval/max_return": float(np.max(returns)) if returns else 0.0,
        "eval/std_return": float(np.std(returns)) if returns else 0.0,
        "eval/mean_length": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
        "eval/max_length": float(np.max(eval_lengths)) if eval_lengths else 0.0,
        "eval/std_length": float(np.std(eval_lengths)) if eval_lengths else 0.0,
        "eval/episode_count": eval_count,
    }
    if r_rand is not None:
        data["eval/random_baseline"] = float(r_rand)
    if consecutive_below is not None:
        data["eval/consecutive_below"] = int(consecutive_below)
    if eval_results:
        rows = [(r["episode"], r["return"], r["length"]) for r in eval_results]
        data["eval/episodes"] = wandb.Table(
            columns=["episode", "return", "length"],
            data=[list(r) for r in rows],
        )
    if len(all_eval_returns) >= 2:
        prev, curr = all_eval_returns[-2], all_eval_returns[-1]
        data["eval/reward_improvement"] = curr - prev
        data["eval/reward_improvement_pct"] = (curr - prev) / max(abs(prev), 1e-8)
    log_metrics(data, step=global_step)


def finish_run(exit_code: int = 0) -> None:
    """Finish the run; no-op when disabled."""
    if wandb_disabled():
        return
    wandb.finish(exit_code=exit_code)
