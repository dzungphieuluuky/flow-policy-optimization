"""The rollout procedure, as a module function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from myfpo.collector import (
    RolloutBuffer,
    collect_chunks_vec,
    resolve_action_bounds,
    scale_action_chunk,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.gae import compute_chunk_advantages
from myfpo.metrics import (
    compute_gae_diagnostics,
    sample_eps,
    sample_mc_timesteps,
    subsampled_min,
)
from myfpo.utils import (
    autocast_context,
    resolve_storage_dtype,
    to_device_async,
    to_numpy_1d,
)

__all__ = ["RolloutBuffers", "run_rollout"]


@dataclass
class RolloutBuffers:
    """The buffers a rollout reuses between calls."""

    host: RolloutBuffer | None = None
    device: RolloutBuffer | None = None
    h2d_event: torch.cuda.Event | None = None

    def wait_for_copy(self) -> None:
        """Block until the recorded copy has landed, then release the event."""
        if self.h2d_event is not None:
            self.h2d_event.synchronize()
            self.h2d_event = None


def run_rollout(
    policy,
    buffers: RolloutBuffers,
    env,
    total_steps: int,
    *,
    action_low: float = -1.0,
    action_high: float = 1.0,
    noisy: bool = True,
) -> dict[str, torch.Tensor]:
    """Roll out open-loop chunks, build a training batch."""
    if total_steps <= 0:
        raise ValueError(f"total_steps={total_steps} must be positive")
    cfg = policy.config
    device = next(policy.parameters()).device
    storage_dtype = resolve_storage_dtype(device)

    def propose_batch(states: torch.Tensor) -> torch.Tensor:
        chunk = policy.predict_action_chunk(
            {OBS_STATE: to_device_async(states, device)}, start_from_zero=not noisy
        )
        return chunk.cpu()

    if buffers.h2d_event is not None:
        buffers.h2d_event.synchronize()
        buffers.h2d_event = None

    num_envs = int(getattr(env, "num_envs", 1))
    action_dim = int(policy.config.action_feature.shape[0])
    obs_dim = int(policy.config.robot_state_feature.shape[0])
    max_chunks = max(1, total_steps)
    buffer_key = (
        max_chunks,
        total_steps,
        cfg.n_obs_steps,
        obs_dim,
        cfg.action_horizon,
        action_dim,
    )

    def make_buffer(**extra) -> RolloutBuffer:
        """Host and device buffers share every shape argument."""
        return RolloutBuffer(
            max_chunks=max_chunks,
            max_steps=total_steps,
            obs_steps=cfg.n_obs_steps,
            obs_dim=obs_dim,
            action_horizon=cfg.action_horizon,
            action_dim=action_dim,
            **extra,
        )

    if (
        buffers.host is None
        or buffers.host.shape_key() != buffer_key
    ):
        buffers.host = make_buffer(pin_memory=device.type == "cuda")
    buffer = buffers.host
    buffer.reset()

    collect_kwargs = dict(
        total_steps=total_steps,
        action_low=action_low,
        action_high=action_high,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        buffer=buffer,
    )
    if num_envs == 1:
        import gymnasium as gym

        envs = gym.vector.SyncVectorEnv([lambda: env])
        try:
            collect_chunks_vec(envs, propose_batch, **collect_kwargs)
        finally:
            envs.close()
    else:
        collect_chunks_vec(env, propose_batch, **collect_kwargs)

    if device.type == "cuda":
        if (
            buffers.device is None
            or buffers.device.shape_key() != buffer.shape_key()
        ):
            buffers.device = make_buffer(device=device)
        buffer.copy_to(buffers.device, non_blocking=True)
        buffers.h2d_event = torch.cuda.Event()
        buffers.h2d_event.record()
        active = buffers.device.active()
    else:
        active = buffer.active()
    n_chunks = buffer.n_chunks
    horizon = cfg.action_horizon

    lengths = active["chunk_lengths"]
    chunk_starts = active["chunk_starts"]
    all_rewards = active["step_rewards"]
    actions = active["chunk_actions"]
    states = active["chunk_state"]

    with torch.inference_mode():
        next_states = active["chunk_next_state"]
        boot_actions = actions.clone()
        needs_proposal = ~active["chunk_terminated"]
        if bool(needs_proposal.any()):
            proposed = policy.predict_action_chunk(
                {OBS_STATE: to_device_async(next_states[needs_proposal], device)},
                start_from_zero=False,
            )
            low_t, range_t = resolve_action_bounds(action_low, action_high)
            boot_actions[needs_proposal] = scale_action_chunk(
                proposed, low_t.to(proposed.device), range_t.to(proposed.device)
            )[1]

        all_chunk_normalised_states = policy.preprocessor(
            {OBS_STATE: torch.cat([states, next_states], dim=0)}
        )[OBS_STATE]
        normalised_states = all_chunk_normalised_states[:n_chunks]
        normalised_next_states = all_chunk_normalised_states[n_chunks:]
        exec_length_frac = policy.exec_length_frac(lengths)
        fitted_steps = (lengths.float().mean() if lengths.numel() > 0 else torch.tensor(float(cfg.action_horizon), device=lengths.device, dtype=torch.float32))
        boot_exec_frac = policy.exec_length_frac(torch.full_like(lengths, int(round(fitted_steps.item()))))
        critic_cond = (
            policy.encoder(all_chunk_normalised_states)
            if cfg.critic_share_encoder
            else None
        )
        chunk_cond = critic_cond[:n_chunks] if critic_cond is not None else None
        next_cond = critic_cond[n_chunks:] if critic_cond is not None else None
        with autocast_context(device):
            q_current_chunk = subsampled_min(
                policy.critic(
                    policy.critic_inputs(normalised_states, chunk_cond),
                    actions,
                    exec_length_frac,
                ).float()
            )
        with autocast_context(device):
            q_bootstrap = subsampled_min(
                policy.critic(
                    policy.critic_inputs(normalised_next_states, next_cond),
                    boot_actions,
                    boot_exec_frac,
                ).float()
            )

    raw_advantages, raw_returns = compute_chunk_advantages(
        all_rewards,
        q_current_chunk,
        q_bootstrap,
        chunk_starts,
        lengths,
        chunk_terminated=active["chunk_terminated"],
        chunk_size=cfg.action_horizon,
        gamma=cfg.gamma,
        chunk_truncated=active["chunk_truncated"],
        chunk_env_id=active["chunk_env_id"],
        gae_lambda=cfg.gae_lambda,
    )
    normalised_advantages = raw_advantages
    if cfg.normalize_advantage and raw_advantages.numel() > 0:
        normalised_advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(correction=0).clamp_min(1e-8)
        )
    normalised_advantages = normalised_advantages.clamp(
        -cfg.advantage_clamp_neg, cfg.advantage_clamp_pos
    )
    gae_diagnostics = compute_gae_diagnostics(
        active["chunk_env_id"][:n_chunks],
        active["chunk_terminated"][:n_chunks],
        active["chunk_truncated"][:n_chunks],
        raw_advantages,
        raw_returns,
    )
    raw_advantages = raw_advantages.to(storage_dtype)
    normalised_advantages = normalised_advantages.to(storage_dtype)

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)

    t = sample_mc_timesteps(cfg, n_chunks, device)
    eps = sample_eps(cfg, n_chunks, action_dim, device)


    behavior_cfm_loss, behavior_cfm_loss_token = policy.compute_behavior_cfm_losses(
        normalised_states, actions, t, eps, action_is_pad
    )

    return {
        OBS_STATE: states,
        ACTION: actions,
        "action_is_pad": action_is_pad,
        "raw_advantages": raw_advantages,
        "normalised_advantages": normalised_advantages,
        "raw_returns": raw_returns,
        "normalizer_observations": active["step_frames"].detach(),
        "boot_observations": all_chunk_normalised_states[n_chunks:].to(storage_dtype),
        "boot_actions": boot_actions.to(storage_dtype),
        "boot_values_start": q_bootstrap.unsqueeze(1),
        "mc_t": t,
        "mc_eps": eps,
        "gae_diagnostics": gae_diagnostics,
        "behavior_cfm_loss": behavior_cfm_loss,
        "behavior_cfm_loss_token": behavior_cfm_loss_token,
    }
