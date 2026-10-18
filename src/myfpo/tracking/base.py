from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

_FEATURE_KEYS = ("input_features", "output_features")

def compact_value(v) -> str:
    """Render a metric as short text for a wandb run name."""
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


def config_hash(cfg_dict: dict) -> str:
    """Six hex characters over every config value. Two configs cannot then"""
    items = [
        (key, compact_value(value))
        for key, value in sorted(cfg_dict.items())
        if key not in _FEATURE_KEYS
    ]
    return hashlib.sha256(repr(items).encode()).hexdigest()[:6]


def build_run_name(method: str, env_name: str, seed: int, cfg_dict: dict) -> str:
    """Build the wandb run name. The name holds the method, the env, the seed,"""
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
        tags.append(f"eps{compact_value(v)}")
    if (v := get("num_mc_samples")) is not None:
        tags.append(f"mc{v}")
    parts = [f"{method}_{env_name}_s{seed}", *tags, config_hash(cfg_dict)]
    return "_".join(parts)


def compute_eval_summary_metrics(
    mean_return: float,
    eval_results: list[dict],
    all_eval_returns: list[float],
    eval_count: int,
    *,
    r_rand: float | None = None,
    best_return: float | None = None,
    steps_to_best: int | None = None,
) -> dict:
    """Return the eval scalars for one eval window."""
    returns = [r["return"] for r in eval_results]
    lengths = [r["length"] for r in eval_results]
    successes = [r["success"] for r in eval_results if r.get("success") is not None]
    jerks = [r["jerk"] for r in eval_results if r.get("jerk") is not None]

    def stat(values, fn):
        return float(fn(values)) if values else float("nan")

    data = {
        "eval/return_mean": float(mean_return),
        "eval/return_min": stat(returns, np.min),
        "eval/return_max": stat(returns, np.max),
        "eval/return_std": stat(returns, np.std),
    }
    if best_return is not None:
        data["eval/return_best"] = float(best_return)
    if steps_to_best is not None:
        data["eval/steps_to_best"] = int(steps_to_best)
    if r_rand is not None:
        data["eval/random_policy_return"] = float(r_rand)
    if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
        prev = all_eval_returns[-2]
        data["eval/return_improvement"] = float(mean_return - prev)
        data["eval/return_improvement_frac"] = float(
            (mean_return - prev) / max(abs(prev), 1e-8)
        )
    data.update(
        {
            "eval/mean_length": stat(lengths, np.mean),
            "eval/max_length": stat(lengths, np.max),
            "eval/std_length": stat(lengths, np.std),
            "eval/episode_count": int(eval_count),
        }
    )
    if successes:
        data["eval/success_rate"] = float(np.mean(successes))
    if jerks:
        data["eval/mean_jerk"] = float(np.mean(jerks))
        data["eval/max_jerk"] = float(np.max(jerks))
    return data


class ExperimentTrackerBackend(Protocol):
    def log(self, data: dict[str, float], step: int) -> None: ...
    def finish(self, exit_code: int) -> None: ...


class ExperimentTracker:
    """Send every call to each configured backend."""

    def __init__(self, backends: list[ExperimentTrackerBackend]):
        self.backends = backends

    @property
    def enabled(self) -> bool:
        return bool(self.backends)

    def log(self, data: dict[str, float], step: int) -> None:
        for b in self.backends:
            b.log(data, step)

    def finish(self, exit_code: int = 0) -> None:
        for b in self.backends:
            b.finish(exit_code)
