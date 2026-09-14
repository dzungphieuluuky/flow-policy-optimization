"""wandb logging — sole wandb importer, enabled by default.

REQUIRED dependency (missing package = hard ImportError). Only off switch:
``SJFPO_DISABLE_WANDB=1``. Failed init fails fast, never degrades silently.
"""

from __future__ import annotations

import os

import numpy as np

import wandb


def wandb_disabled() -> bool:
    """Return True only when the single off switch is set."""
    return os.environ.get("SJFPO_DISABLE_WANDB") == "1"


def init_wandb_run_for_method(
    *,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    cfg_dict: dict,
) -> bool:
    """Init a run in the per-algo project (``method``); False when disabled."""
    if wandb_disabled():
        return False
    wandb.init(
        project=method,
        group=f"{method}_{env_name}",
        name=None,
        tags=[method, "mujoco", "cpu-first"],
        config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
        job_type="smoke" if total_steps <= 10000 else "train",
        mode=os.environ.get("WANDB_MODE", "online"),
        reinit="finish_previous",
        force=True,
    )
    return True


def log_metrics(data: dict, step: int) -> None:
    """Log a dict at a global step; no-op when disabled."""
    if wandb_disabled():
        return
    wandb.log(data, step=step)


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
    """Eval metrics + per-episode table in one commit (shared runner/baselines home)."""
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
