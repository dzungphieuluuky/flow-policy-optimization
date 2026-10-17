"""The training-metric log chain, as one pure function.

``SJFPOPolicy.forward`` returns an ``info`` dict. The trainer used to turn that
into the ``train/*`` log rows inline, in a thirty-line block inside the
optimiser loop. That block was hard to test, because reaching it meant running
an optimiser step, and hard to read, because it sat between two abort checks.

The function here takes the same inputs and returns the same dict. It reads no
instance state and it logs nothing. So a caller can build the rows without a
policy, and a test can build them without a step.

Every key keeps the name it had. Renaming a log key breaks every run already
recorded, so the names are the contract, not the contents.
"""

from __future__ import annotations

from myfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["build_train_metrics"]


# GAE diagnostic name to logged name. The prefix is deliberately not uniform.
# The key return_rms is logged as returns_rms.
_GAE_DIAG_LOG_KEYS = {
    "linked_frac": "gae_linked_frac",
    "advantage_std": "gae_advantage_std",
    "return_rms": "returns_rms",
    "advantage_std_over_return_std": "advantage_std_over_return_std",
}

# Always present in compute_rollout_diagnostics' output.
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
    grad_norm_mean_pre: float,
    above_clamp_frac: float,
    both_clamped_frac: float,
    drift_means: list[float],
    rollout_jerk: float | None,
    rollout_seam_jerk: float | None,
) -> dict:
    """Return the ``train/*`` rows for one optimiser step.

    The argument ``info`` is the forward pass's own dict. Its keys are already
    the bare names, so each one takes the ``train/`` prefix here.

    The argument ``rollout`` is the batch, not the rollout function's output. The
    keys read from it are the ones ``compute_rollout_diagnostics`` fills.

    The function adds ``train/total_loss`` itself, because it is the only place
    that knows both loss terms and the coefficient.
    """
    data = {f"train/{k}": v for k, v in info.items()}
    data["train/grad_norm_before_clip"] = grad_norm_mean_pre
    data["train/cfm_above_clamp_frac"] = above_clamp_frac
    data["train/cfm_both_clamped_frac"] = both_clamped_frac
    if cfg.log_drift_diagnostics and drift_means:
        data["train/boot_value_relative_drift"] = drift_means[0]
        data["train/value_target_drift"] = drift_means[1]
    if rollout_jerk is not None:
        data["train/rollout_jerk"] = rollout_jerk
    if rollout_seam_jerk is not None:
        # Non-zero here with a zero rollout_jerk is the
        # stale-input / missing-RTC signature.
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