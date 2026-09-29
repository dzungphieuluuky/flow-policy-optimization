from __future__ import annotations

from typing import Protocol

import numpy as np

def compact_value(v) -> str:
    """Render a metric compactly for a wandb run name.

    Bools become ints, floats use ``:g``, non-finite floats become
    ``nan``/``inf``/``-inf`` instead of an empty or padded string.
    """
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
    """wandb run name from method, env, seed, and key settings.

    Only settings that distinguish runs get appended, so the name stays
    readable in the UI. Form: ``method_env_sseed_tag_tag``.
    """
    tags: list[str] = []
    get = cfg_dict.get
    if (v := get("num_kept_tokens")) is not None:
        tags.append(f"k{compact_value(v)}")
    if (v := get("hidden_dim")) is not None and (d := get("dit_depth")) is not None:
        tags.append(f"d{v}x{d}")
    if (v := get("num_learning_epochs", get("num_epochs"))) is not None:
        tags.append(f"ep{v}")
    if (v := get("policy_lr", get("lr"))) is not None:
        tags.append(f"lr{v:g}" if isinstance(v, float) else f"lr{v}")
    if (v := get("clip_param", get("epsilon_clip"))) is not None:
        tags.append(f"eps{compact_value(v)}")
    if (v := get("num_mc_samples")) is not None:
        tags.append(f"mc{v}")
    base = f"{method}_{env_name}_s{seed}"
    return base if not tags else f"{base}_{'_'.join(tags)}"


def eval_summary_metrics(
    mean_return: float,
    eval_results: list[dict],
    all_eval_returns: list[float],
    eval_count: int,
    *,
    r_rand: float | None = None,
    consecutive_below: int | None = None,
    best_return: float | None = None,
    steps_to_best: int | None = None,
) -> dict:
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
    return data


class ExperimentTrackerBackend(Protocol):
    def log(self, data: dict[str, float], step: int) -> None: ...
    def log_video(self, frames: list, step: int, *, fps: int) -> None: ...
    def finish(self, exit_code: int) -> None: ...


class ExperimentTracker:
    """Fan every call out to each configured backend."""

    def __init__(self, backends: list[ExperimentTrackerBackend]):
        self.backends = backends

    @property
    def enabled(self) -> bool:
        return bool(self.backends)

    def log(self, data: dict[str, float], step: int) -> None:
        for b in self.backends:
            b.log(data, step)

    def log_video(self, frames: list, step: int, *, fps: int = 30) -> None:
        for b in self.backends:
            b.log_video(frames, step, fps=fps)

    def finish(self, exit_code: int = 0) -> None:
        for b in self.backends:
            b.finish(exit_code)
