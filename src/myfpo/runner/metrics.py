"""The training-metric log chain, as one pure function."""

from __future__ import annotations

from myfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["build_train_metrics"]


_GAE_DIAG_LOG_KEYS = {
    "linked_frac": "gae_linked_frac",
    "advantage_std": "gae_advantage_std",
    "return_rms": "returns_rms",
    "advantage_std_over_return_std": "advantage_std_over_return_std",
}

_ROLLOUT_ABS_KEYS = (
    "action_abs_mean",
    "action_abs_mean_executed",
    "action_abs_max_executed",
)


def build_train_metrics(
    *,
    cfg: SJFPOConfig,
    info: dict,
    batch: dict,
    rollout: dict,
    grad_norms: dict[str, float],
    above_clamp_frac: float,
    both_clamped_frac: float,
    drift_means: list[float],
    rollout_jerk: float | None,
    rollout_seam_jerk: float | None,
) -> dict:
    """Return the ``train/*`` rows for one optimiser step."""
    data = {f"train/{k}": v for k, v in info.items()}
    data["train/grad_norm_before_clip"] = grad_norms["total"]
    data["train/grad_norm_actor_before_clip"] = grad_norms["actor"]
    data["train/grad_norm_critic_before_clip"] = grad_norms["critic"]
    data["train/cfm_above_clamp_frac"] = above_clamp_frac
    data["train/cfm_both_clamped_frac"] = both_clamped_frac
    if cfg.log_drift_diagnostics and drift_means:
        data["train/boot_value_relative_drift"] = drift_means[0]
        data["train/value_target_drift"] = drift_means[1]
    if rollout_jerk is not None:
        data["train/rollout_jerk"] = rollout_jerk
    if rollout_seam_jerk is not None:
        data["train/rollout_seam_jerk"] = rollout_seam_jerk
    for key in _ROLLOUT_ABS_KEYS:
        data[f"train/{key}"] = rollout[key]
    gae_diag = batch.get("gae_diagnostics")
    if gae_diag:
        data.update(
            {
                f"train/{log}": gae_diag[src]
                for src, log in _GAE_DIAG_LOG_KEYS.items()
            }
        )
    data["train/total_loss"] = (
        info["policy_loss"] + cfg.value_loss_coef * info["value_loss"]
    )
    return data