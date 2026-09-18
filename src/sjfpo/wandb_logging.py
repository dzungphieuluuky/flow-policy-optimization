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


def compact_value(v) -> str:
    """Short wandb-safe rendering: ints plain, floats via ``:g``."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    """``{method}_{env}_s{seed}_{tags}`` from major config keys.

    Fixed tag order, missing keys skipped, so sjfpo/ppo/sac dicts all work:
    e.g. ``sjfpo_Hopper-v5_s3407_top0.3_d128x8_ep16_lr0.0001_eps0.05_mc16``.
    """
    tags: list[str] = []
    get = cfg_dict.get
    if get("top_p_spectral_tokens") is not None:
        tags.append(f"top{compact_value(get('top_p_spectral_tokens'))}")
    if get("hidden_dim") is not None and get("dit_depth") is not None:
        tags.append(f"d{get('hidden_dim')}x{get('dit_depth')}")
    epochs = get("num_epochs", get("learning_epochs"))
    if epochs is not None:
        tags.append(f"ep{epochs}")
    lr = get("optimizer_lr", get("lr"))
    if lr is not None:
        tags.append(f"lr{compact_value(lr)}")
    eps = get("epsilon_clip", get("ratio_clip"))
    if eps is not None:
        tags.append(f"eps{compact_value(eps)}")
    if get("num_mc_samples") is not None:
        tags.append(f"mc{get('num_mc_samples')}")
    base = f"{method}_{env_name}_s{seed}"
    return base if not tags else f"{base}_{'_'.join(tags)}"


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
        name=build_run_name(method, env_name, seed, cfg_dict),
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


def log_eval_video(frames: list, global_step: int, *, fps: int = 30) -> None:
    """One eval episode as wandb.Video; no-op when disabled/empty/unrenderable."""
    if wandb_disabled() or not frames:
        return
    try:
        import numpy as _np

        video = _np.stack([_np.asarray(f, dtype=_np.uint8) for f in frames])
        if video.ndim != 4:
            return
        log_metrics({"eval/video": wandb.Video(video, fps=fps, format="mp4")}, step=global_step)
    except Exception:
        pass


def finish_run(exit_code: int = 0) -> None:
    """Finish the run; no-op when disabled."""
    if wandb_disabled():
        return
    wandb.finish(exit_code=exit_code)
