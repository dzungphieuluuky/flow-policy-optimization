"""The rollout procedure, as a module function.

``SJFPOPolicy.generate_rollout`` used to hold 262 lines: build the plan, step the
vec env, copy the batch, run the advantage bootstrap. It read twelve attributes
off the policy, six of which were the policy's own methods.

The body moved here. The policy keeps a one-line wrapper, so no call site
changed and ``policy.generate_rollout(env, n)`` still reads the same.

Three of the twelve attributes were not the policy at all. They were a host
buffer, a device buffer, and the CUDA event that says the copy landed. Those now
live in :class:`RolloutBuffers`, which the policy owns and passes in. The
procedure needs a place to keep a pinned host buffer alive between calls, and a
named holder says that better than three underscore fields on the policy.

The parameter ``policy`` is a :class:`~myfpo.modeling_sjfpo.SJFPOPolicy`. The
module does not import that class, because the class imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from myfpo.collector import (
    RolloutBuffer,
    collect_chunks_vec,
    resolve_action_bounds,
    scale_and_clip_chunk,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.gae import compute_chunk_advantages
from myfpo.metrics import (
    compute_gae_diagnostics,
    sample_eps,
    sample_mc_timesteps,
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
    """The buffers a rollout reuses between calls.

    The host buffer is pinned so the copy to the device can be non-blocking. It
    is kept rather than reallocated, because a fresh pinned allocation per
    rollout would cost more than the copy saves.

    The device buffer holds the same shape as the host one. It exists because
    the critic reads the batch on the device and the trainer reads it on the
    host, and one buffer cannot be both.

    The event records the copy. The next call waits on it, so a slow copy from
    the previous rollout cannot be read as this rollout's data.
    """

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
    """Roll out open-loop chunks, build a training batch.

    See :mod:`myfpo.gae`. A chunk can run one step and then terminate. So a
    count can reach one.

    The batch carries one raw post-step frame per executed step. The frames
    stay separate from the overlapping obs windows. So the normalizer does
    not count a frame twice.

    The batch also carries the per-chunk and per-MC CFM draws. The policy
    loss reuses those draws for the old value and for the new value.
    """
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
    # A single env becomes a 1-env vec env. One collector, so the chunk
    # contract cannot drift between the n==1 and n>1 paths.
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

    # Two obs windows per chunk, the start and the end. The preprocessor
    # sees 2 * n_chunks, not 2 * n_steps.
    # The normalizer updates from normalizer_observations.
    # The bootstrap a' is drawn fresh at the end state of each chunk.
    # That end state is the target of QC. A terminated row keeps the chunk.
    with torch.inference_mode():
        next_states = active["chunk_next_state"]
        # Seeded from the chunk, so no row is garbage.
        # The backup scales by ~terminated. 0 * NaN is NaN.
        boot_actions = actions.clone()
        # Only a live chunk needs a proposal. A terminated chunk keeps the
        # cloned chunk actions. Nothing reads those values. The backup masks
        # them with (1 - terminated).
        needs_proposal = ~active["chunk_terminated"]
        if bool(needs_proposal.any()):
            # The call predict_action_chunk returns the env bounds. The
            # tensor chunk_actions is scaled to [-1, 1]. The critic reads
            # the scaled form. So the proposal takes the same affine
            # transform.
            proposed = policy.predict_action_chunk(
                {OBS_STATE: to_device_async(next_states[needs_proposal], device)},
                start_from_zero=False,
            )
            low_t, range_t = resolve_action_bounds(action_low, action_high)
            boot_actions[needs_proposal] = scale_and_clip_chunk(
                proposed, low_t.to(proposed.device), range_t.to(proposed.device)
            )[1]

        all_chunk_normalised_states = policy.preprocessor(
            {OBS_STATE: torch.cat([states, next_states], dim=0)}
        )[OBS_STATE]
        normalised_states = all_chunk_normalised_states[:n_chunks]
        normalised_next_states = all_chunk_normalised_states[n_chunks:]
        # The bootstrap chunk only ever executes n_action_steps, so its
        # fraction is n_action_steps/H, NOT H/H. Passing H/H
        # reintroduces the mismatch this feature removes.
        exec_length_frac = policy.exec_length_frac(lengths)
        boot_exec_frac = policy.exec_length_frac(
            torch.full_like(lengths, cfg.n_action_steps)
        )
        # amin across members. That is for the backup only, never in the
        # loss.
        # The value loss averages the same members.
        # The shared encoder runs here so the critic can read it. One pass
        # covers the chunk and its bootstrap. The weights are the weights of
        # the actor.
        critic_cond = (
            policy.encoder(all_chunk_normalised_states)
            if cfg.critic_share_encoder
            else None
        )
        chunk_cond = critic_cond[:n_chunks] if critic_cond is not None else None
        next_cond = critic_cond[n_chunks:] if critic_cond is not None else None
        with autocast_context(device):
            q_current_chunk = policy.critic(
                policy.critic_inputs(normalised_states, chunk_cond),
                actions,
                exec_length_frac,
            ).float().amin(dim=0)
        # The code is already inside the outer inference_mode. Autocast is
        # the only context this block adds.
        with autocast_context(device):
            q_bootstrap = policy.critic(
                policy.critic_inputs(normalised_next_states, next_cond),
                boot_actions,
                boot_exec_frac,
            ).float().amin(dim=0)

    # One chunk is one macro-transition. Sum the rewards, then discount by
    # gamma**h.
    # This is the rsl-rl space. Q predicts raw returns, and the target is the
    # raw return, so Q is the true expected return.
    raw_advantages, raw_returns = compute_chunk_advantages(
        all_rewards,
        q_current_chunk,
        q_bootstrap,
        chunk_starts,
        lengths,
        chunk_terminated=active["chunk_terminated"],
        chunk_size=cfg.action_horizon,
        gamma=cfg.gamma,
        # The code reads both values only when gae_lambda > 0. QC ignores
        # both.
        # The value chunk_truncated is the ENV limit, not the step budget.
        # The two limits are separate. So the code never reads a live
        # episode as finished.
        chunk_truncated=active["chunk_truncated"],
        chunk_env_id=active["chunk_env_id"],
        gae_lambda=cfg.gae_lambda,
    )
    normalised_advantages = raw_advantages
    # The rationale for correction=0 is in compute_gae_diagnostics.
    if cfg.normalize_advantage and raw_advantages.numel() > 0:
        normalised_advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(correction=0).clamp_min(1e-8)
        )
    # FPO++ clamps inside the minibatch loop, right before the surrogate.
    # This code clamps here instead. The clamp is a fixed elementwise bound,
    # so one clamp per rollout gives the same numbers in one fewer pass.
    # The order is after normalisation, as theirs is.
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
    # raw_returns stays fp32. It is the critic target, and it is the only
    # stored tensor that is not O(1) (returns_rms runs to 35). There, the 8
    # mantissa bits of bf16 lose real precision.

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)

    t = sample_mc_timesteps(cfg, n_chunks, device)
    eps = sample_eps(cfg, n_chunks, action_dim, device)

    chunk_clip_counts = active["chunk_clip_counts"][:n_chunks]

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
        # Critic drift from the bootstrap value that the targets used.
        # Target critic. That is the price of the slow teacher.
        # Without one, the target's own drift.
        "boot_observations": all_chunk_normalised_states[n_chunks:].to(storage_dtype),
        "boot_actions": boot_actions.to(storage_dtype),
        "boot_values_start": q_bootstrap.unsqueeze(1),
        "mc_t": t,
        "mc_eps": eps,
        "chunk_clip_counts": chunk_clip_counts,
        "gae_diagnostics": gae_diagnostics,
        "behavior_cfm_loss": behavior_cfm_loss,
        "behavior_cfm_loss_token": behavior_cfm_loss_token,
    }
