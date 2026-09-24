"""wandb logging — direct wandb usage (installed and enabled)."""

from __future__ import annotations

import os

import numpy as np
import wandb


# Direct wandb usage (library installed and enabled).
# Enabled only when the caller passes ``wandb_enabled=True``.


# ---------------------------------------------------------------------------
# Run-name helpers
# ---------------------------------------------------------------------------


def compact_value(v) -> str:
    """Short wandb-safe rendering: ints plain, floats via ``:g``."""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:g}"
    return str(v)


def build_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    tags: list[str] = []
    get = cfg_dict.get
    if (v := get("top_p_spectral_tokens")) is not None:
        tags.append(f"top{compact_value(v)}")
    if (v := get("hidden_dim")) is not None and (d := get("dit_depth")) is not None:
        tags.append(f"d{v}x{d}")
    if (v := get("num_epochs", get("learning_epochs"))) is not None:
        tags.append(f"ep{v}")
    if (v := get("policy_lr", get("lr"))) is not None:
        tags.append(f"lr{v:g}" if isinstance(v, float) else f"lr{v}")
    if (v := get("epsilon_clip", get("ratio_clip"))) is not None:
        tags.append(f"eps{compact_value(v)}")
    if get("use_per_bin_clip"):
        tags.append(
            f"bin{compact_value(get('epsilon_clip_lowfreq'))}"
            f"-{compact_value(get('epsilon_clip_highfreq'))}"
        )
    if (v := get("num_mc_samples")) is not None:
        tags.append(f"mc{v}")
    base = f"{method}_{env_name}_s{seed}"
    return base if not tags else f"{base}_{'_'.join(tags)}"


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


def init_wandb_run_for_method(
    *,
    method: str,
    env_name: str,
    seed: int,
    total_steps: int,
    cfg_dict: dict,
    wandb_enabled: bool = True,
) -> bool:
    if not wandb_enabled:
        return False
    wandb.init(
        project=method,
        group=f"{method}_{env_name}",
        name=build_run_name(method, env_name, seed, cfg_dict),
        tags=[method, "mujoco"],
        config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
        job_type="smoke" if total_steps <= 10000 else "train",
        mode=os.environ.get("WANDB_MODE", "online"),
        reinit="finish_previous",
    )
    return True


def finish_run(exit_code: int = 0) -> None:
    wandb.finish(exit_code=exit_code)


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------


def log_metrics(data: dict, step: int) -> None:
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
    best_return: float | None = None,
    steps_to_best: int | None = None,
) -> None:
    returns = [r["return"] for r in eval_results]
    lengths = [r["length"] for r in eval_results]
    successes = [r["success"] for r in eval_results if r.get("success") is not None]
    jerks = [r["jerk"] for r in eval_results if r.get("jerk") is not None]
    data = {
        "eval/mean_return": float(mean_return),
        "eval/min_return": float(np.min(returns)) if returns else 0.0,
        "eval/max_return": float(np.max(returns)) if returns else 0.0,
        "eval/std_return": float(np.std(returns)) if returns else 0.0,
    }
    if best_return is not None:
        data["eval/best_mean_return"] = float(best_return)
    if steps_to_best is not None:
        data["eval/steps_to_best"] = int(steps_to_best)
    if r_rand is not None:
        data["eval/random_baseline"] = float(r_rand)
    if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
        prev = all_eval_returns[-2]
        data["eval/reward_improvement"] = float(mean_return - prev)
        data["eval/reward_improvement_pct"] = float(
            (mean_return - prev) / max(abs(prev), 1e-8)
        )
    if consecutive_below is not None:
        data["eval/consecutive_below"] = int(consecutive_below)

    data.update(
        {
            "eval/mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval/max_length": float(np.max(lengths)) if lengths else 0.0,
            "eval/std_length": float(np.std(lengths)) if lengths else 0.0,
            "eval/episode_count": int(eval_count),
        }
    )
    if successes:
        data["eval/success_rate"] = float(np.mean(successes))
    if jerks:
        data["eval/mean_jerk"] = float(np.mean(jerks)) if jerks else 0.0
        data["eval/max_jerk"] = float(np.max(jerks)) if jerks else 0.0
    if eval_results:
        data["eval/episodes"] = wandb.Table(
            columns=["episode", "return", "length", "success", "jerk"],
            data=[[r["episode"], r["return"], r["length"], r.get("success"), r.get("jerk")]
                  for r in eval_results],
        )
    log_metrics(data, step=global_step)


def log_eval_video(frames: list, global_step: int, *, fps: int = 30) -> None:
    if not frames:
        return
    try:
        video = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    except (ValueError, TypeError):
        return
    if video.ndim != 4:
        return
    try:
        log_metrics({"eval/video": wandb.Video(video, fps=fps, format="mp4")}, step=global_step)
    except Exception:
        return
