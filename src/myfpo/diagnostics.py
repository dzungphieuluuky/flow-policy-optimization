"""Boundary and value diagnostics, kept out of the hot path.

Everything here reads artifacts the training loop already produces; nothing here
feeds the loss. The expensive drift path is only reachable when
``cfg.log_drift_diagnostics`` is set; ``training.py`` imports it lazily inside
that branch.
"""

from __future__ import annotations

import torch

from myfpo.actions import to_critic_action, to_env_action
from myfpo.flow import subsampled_min
from myfpo.nn import rms
from myfpo.env import resolve_action_bounds

__all__ = ["compute_rollout_diagnostics", "executed_chunk", "measure_value_drift"]


def executed_chunk(
    batch: dict[str, torch.Tensor], action_low, action_high,
    action_processing_mode: str = "tanh_squash",
) -> torch.Tensor:
    """The env-facing action of a stored chunk.

    The batch holds the flow's output, which is what the policy was
    trained on. The env received it mapped into env bounds.
    """
    low, high = resolve_action_bounds(action_low, action_high)
    action = batch["action"]
    device = action.device
    low_f = low.to(device=device, dtype=torch.float32)
    high_f = high.to(device=device, dtype=torch.float32)
    return to_env_action(action.float(), low_f, high_f, mode=action_processing_mode)


@torch.no_grad()
def compute_rollout_diagnostics(
    batch: dict[str, torch.Tensor], *, action_low, action_high,
    env_action: torch.Tensor | None = None,
    action_processing_mode: str = "tanh_squash",
) -> dict[str, float]:
    """Read the post-rollout scalars with one device-to-host transfer.

    Two action spaces are reported: the raw chunk the flow produced, which
    is what the policy is trained on, and the chunk the env received.
    """
    action = batch["action"].float()
    is_pad = batch["action_is_pad"]
    executed = ~is_pad
    if env_action is None:
        env_action = executed_chunk(batch, action_low, action_high, action_processing_mode)

    exec_action = env_action[executed]
    scalars = [
        executed.sum(),
        action.abs().max(),
        action.abs().mean(),
        exec_action.abs().mean(),
        exec_action.abs().max(),
    ]
    keys = [
        "executed_action_steps",
        "raw_action_abs_max",
        "raw_action_abs_mean",
        "executed_action_abs_mean",
        "executed_action_abs_max",
    ]
    return dict(zip(keys, torch.stack(scalars).tolist()))


@torch.no_grad()
def measure_value_drift(policy, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RMS of the bootstrap delta relative to the start values and to the raw returns."""
    cfg = policy.config
    if "boot_values_start" not in batch or "boot_actions" not in batch:
        raise ValueError(
            "the batch carries no bootstrap values; the rollout only builds "
            "them when log_drift_diagnostics is on"
        )
    boot_obs = batch["boot_observations"].to(device=device, dtype=torch.float32)
    boot_actions_raw = batch["boot_actions"].to(device=device, dtype=torch.float32)
    start = batch["boot_values_start"].to(device).reshape(-1)
    boot_exec_frac = torch.full(
        (boot_obs.shape[0],),
        cfg.num_action_steps / float(cfg.action_horizon),
        device=boot_obs.device,
        dtype=torch.float32,
    )
    # Match the masking in bootstrap_values: squash, then zero out steps
    # beyond num_action_steps, so live and target critics read the same chunk.
    boot_mask = (
        torch.arange(cfg.action_horizon, device=boot_obs.device)
        .unsqueeze(0) < cfg.num_action_steps
    ).unsqueeze(-1).float()
    boot_actions = to_critic_action(
        boot_actions_raw, policy.action_low, policy.action_high,
        mode=getattr(cfg, "action_processing_mode", "tanh_squash"), mask=boot_mask,
    )
    critic_was_training = policy.critic.training
    policy.critic.eval()
    try:
        delta = (
            subsampled_min(policy.critic(boot_obs, boot_actions, boot_exec_frac).float())
            - start
        )
    finally:
        policy.critic.train(critic_was_training)
    drift_rms = rms(delta)
    return (
        drift_rms / (rms(start) + 1e-8),
        drift_rms / (rms(batch["raw_returns"].to(device)) + 1e-8),
    )
