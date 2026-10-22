"""The pieces of the training loop that do not need the training loop."""

from __future__ import annotations

from dataclasses import dataclass

from myfpo.configuration import MyfpoConfig

__all__ = ["WindowSummary", "build_train_metrics", "summarise_window"]


@dataclass
class WindowSummary:
    """What one eval window found."""

    reported_results: list[dict]
    summaries: dict[str, dict]
    mean_return: float
    stop_return: float
    improved: bool


def summarise_window(
    results_by_sampling: dict[str, list[dict]],
    *,
    reported_name: str,
    stop_name: str,
    eval_seed: int,
    summarise,
    best_return: float | None,
) -> WindowSummary:
    """Summarise every start, then read the two means the runner needs."""
    summaries = {
        name: summarise(rows, seed=eval_seed)
        for name, rows in results_by_sampling.items()
    }
    return WindowSummary(
        reported_results=results_by_sampling[reported_name],
        summaries=summaries,
        mean_return=summaries[reported_name]["return_mean"],
        stop_return=summaries[stop_name]["return_mean"],
        improved=best_return is None or summaries[stop_name]["return_mean"] > best_return,
    )


GAE_DIAG_LOG_KEYS = {
    "linked_frac": "gae_linked_frac",
    "advantage_std": "gae_advantage_std",
    "return_rms": "returns_rms",
    "advantage_std_over_return_std": "advantage_std_over_return_std",
}

ROLLOUT_ABS_KEYS = (
    "raw_action_abs_mean",
    "raw_action_abs_max",
    "executed_action_abs_mean",
    "executed_action_abs_max",
)


def build_train_metrics(
    *,
    cfg: MyfpoConfig,
    info: dict,
    batch: dict,
    rollout: dict,
    grad_norms: dict[str, float],
    above_clamp_frac: float,
    both_clamped_frac: float,
    drift_means: list[float],
    rollout_jerk: float | None,
    rollout_seam_jerk: float | None,
    policy_lr: float,
    critic_lr: float,
) -> dict:
    """Return the ``train/*`` rows for one optimiser step."""
    data = {f"train/{k}": v for k, v in info.items()}
    data["train/grad_norm_before_clip"] = grad_norms["total"]
    data["train/grad_norm_actor_before_clip"] = grad_norms["actor"]
    data["train/grad_norm_critic_before_clip"] = grad_norms["critic"]
    data["train/cfm_above_clamp_frac"] = above_clamp_frac
    data["train/cfm_both_clamped_frac"] = both_clamped_frac
    # The learning rate of the step that is about to run. It is logged here,
    # on every update, so the curve has one point per update and not one per
    # evaluation.
    data["train/policy_lr"] = policy_lr
    data["train/critic_lr"] = critic_lr
    if cfg.log_drift_diagnostics and drift_means:
        data["train/boot_value_relative_drift"] = drift_means[0]
        data["train/value_target_drift"] = drift_means[1]
    if rollout_jerk is not None:
        data["train/rollout_jerk"] = rollout_jerk
    if rollout_seam_jerk is not None:
        data["train/rollout_seam_jerk"] = rollout_seam_jerk
    for key in ROLLOUT_ABS_KEYS:
        data[f"train/{key}"] = rollout[key]
    for key in ("collect_s", "actor_step_s", "critic_step_s"):
        data[f"train/{key}"] = rollout.get(key, 0.0)
    gae_diag = batch.get("gae_diagnostics")
    if gae_diag:
        data.update(
            {
                f"train/{log}": gae_diag[src]
                for src, log in GAE_DIAG_LOG_KEYS.items()
            }
        )
    data["train/total_loss"] = (
        info["policy_loss"] + cfg.value_loss_coef * info["value_loss"]
    )
    return data
