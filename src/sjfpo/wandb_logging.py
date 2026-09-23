"""wandb logging — sole wandb importer, enabled by default.

Disable by setting ``SJFPO_DISABLE_WANDB`` to any of ``1``/``true``/``yes``/``on``
(case-insensitive). ``WANDB_MODE=disabled`` is also honored.

The module is import-safe when wandb is not installed: every function checks
availability before touching the ``wandb`` package, and ``wandb_disabled()``
returns ``True`` if the library is missing.
"""

from __future__ import annotations

import os

import numpy as np

try:  # import-safe when wandb is absent
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _wandb = None
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enable / disable
# ---------------------------------------------------------------------------

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "t", "y"})


def wandb_disabled() -> bool:
    """Return True when wandb should be treated as off.

    Checks (in order): library availability, ``SJFPO_DISABLE_WANDB`` truthiness,
    and wandb's own ``WANDB_MODE=disabled`` convention.
    """
    if not _WANDB_AVAILABLE:
        return True
    raw = os.environ.get("SJFPO_DISABLE_WANDB", "")
    if raw.strip().lower() in _TRUE_VALUES:
        return True
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return True
    return False


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
        if v != v:  # NaN
            return "nan"
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:g}"
    return str(v)


def build_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    """``{method}_{env}_s{seed}_{tags}`` from major config keys.

    Fixed tag order, missing keys skipped, so ``sjfpo``/``ppo``/``sac``
    runs share a naming convention when they share a config layout.
    """
    tags: list[str] = []
    get = cfg_dict.get

    if (v := get("top_p_spectral_tokens")) is not None:
        tags.append(f"top{compact_value(v)}")
    if (v := get("hidden_dim")) is not None and (d := get("dit_depth")) is not None:
        tags.append(f"d{v}x{d}")
    if (v := get("num_epochs", get("learning_epochs"))) is not None:
        tags.append(f"ep{v}")
    if (v := get("optimizer_lr", get("lr"))) is not None:
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
) -> bool:
    """Init a run in the per-method project (``method``).

    Returns ``False`` when disabled. Swallows exceptions from ``wandb.init``
    so a wandb outage does not abort training.
    """
    if wandb_disabled():
        return False
    try:
        _wandb.init(
            project=method,
            group=f"{method}_{env_name}",
            name=build_run_name(method, env_name, seed, cfg_dict),
            tags=[method, "mujoco"],
            config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
            job_type="smoke" if total_steps <= 10000 else "train",
            mode=os.environ.get("WANDB_MODE", "online"),
            reinit="finish_previous",
        )
    except Exception:
        # Fall through to no-op: training continues without metrics.
        return False
    return True


def finish_run(exit_code: int = 0) -> None:
    """Finish the run; no-op when disabled. Tolerant of older wandb signatures."""
    if wandb_disabled():
        return
    try:
        _wandb.finish(exit_code=exit_code)
    except TypeError:
        # Very old wandb predates exit_code=; call bare.
        _wandb.finish()


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

def log_metrics(data: dict, step: int) -> None:
    """Log a dict at a global step; no-op when disabled."""
    if wandb_disabled():
        return
    _wandb.log(data, step=step)


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
    """Eval metrics + per-episode table in one commit.

    ``all_eval_returns`` should already include the current eval's mean
    (i.e., ``all_eval_returns[-1] == mean_return``); the previous entry
    ``all_eval_returns[-2]`` is used for the improvement delta. If that
    invariant doesn't hold, the improvement fields are skipped.
    """
    if wandb_disabled():
        return

    returns = [r["return"] for r in eval_results]
    lengths = [r["length"] for r in eval_results]
    successes = [r["success"] for r in eval_results if r.get("success") is not None]
    jerks = [r["jerk"] for r in eval_results if r.get("jerk") is not None]

    data = {
        "eval/mean_return": float(mean_return),
        "eval/min_return": float(np.min(returns)) if returns else 0.0,
        "eval/max_return": float(np.max(returns)) if returns else 0.0,
        "eval/std_return": float(np.std(returns)) if returns else 0.0,
        "eval/mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "eval/max_length": float(np.max(lengths)) if lengths else 0.0,
        "eval/std_length": float(np.std(lengths)) if lengths else 0.0,
        "eval/episode_count": eval_count,
    }
    if successes:
        data["eval/success_rate"] = float(np.mean(successes))
    if jerks:
        data["eval/mean_jerk"] = float(np.mean(jerks)) if jerks else 0.0
        data["eval/max_jerk"] = float(np.max(jerks)) if jerks else 0.0
    if best_return is not None:
        data["eval/best_mean_return"] = float(best_return)
    if steps_to_best is not None:
        data["eval/steps_to_best"] = int(steps_to_best)
    if r_rand is not None:
        data["eval/random_baseline"] = float(r_rand)
    if consecutive_below is not None:
        data["eval/consecutive_below"] = int(consecutive_below)

    if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
        prev = all_eval_returns[-2]
        data["eval/reward_improvement"] = float(mean_return - prev)
        data["eval/reward_improvement_pct"] = float(
            (mean_return - prev) / max(abs(prev), 1e-8)
        )

    if eval_results:
        data["eval/episodes"] = _wandb.Table(
            columns=["episode", "return", "length", "success", "jerk"],
            data=[[r["episode"], r["return"], r["length"], r.get("success"), r.get("jerk")]
                  for r in eval_results],
        )

    log_metrics(data, step=global_step)


def log_eval_video(frames: list, global_step: int, *, fps: int = 30) -> None:
    """One eval episode as ``wandb.Video``; no-op when disabled/empty/unrenderable.

    Returns silently on shape mismatch — the caller shouldn't crash the run
    because a renderer produced inconsistent frames.
    """
    if wandb_disabled() or not frames:
        return
    try:
        video = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
    except (ValueError, TypeError):
        return
    if video.ndim != 4:
        return
    try:
        log_metrics(
            {"eval/video": _wandb.Video(video, fps=fps, format="mp4")},
            step=global_step,
        )
    except Exception:
        # Video encoding can fail on codec/backend issues; not fatal.
        return