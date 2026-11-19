"""flow_policy_optimization.rollout: buffer, SameStep collector and rollout driver."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from flow_policy_optimization.config import (
    FlowPolicyOptimizationConfig,
    resolve_action_bounds,
)
from flow_policy_optimization.objective import bound_for_storage, compute_chunk_advantages, to_env_action
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy

class RolloutBuffer:
    """Fixed-capacity fp32 CPU storage for chunk and executed-step rows."""

    def __init__(self, *, max_chunks: int, max_steps: int, obs_steps: int, obs_dim: int, action_horizon: int, action_dim: int):
        if max_chunks < 1 or max_steps < 1:
            raise ValueError("max_chunks and max_steps must be positive")
        self.max_chunks, self.max_steps = max_chunks, max_steps
        self.obs_steps, self.obs_dim = obs_steps, obs_dim
        self.action_horizon, self.action_dim = action_horizon, action_dim

        self.normalizer_observations = torch.zeros((max_steps, obs_dim), dtype=torch.float32)
        self.step_rewards = torch.zeros((max_steps,), dtype=torch.float32)
        self.chunk_state = torch.zeros((max_chunks, obs_steps, obs_dim), dtype=torch.float32)
        self.chunk_next_state = torch.zeros((max_chunks, obs_steps, obs_dim), dtype=torch.float32)
        self.chunk_actions = torch.zeros((max_chunks, action_horizon, action_dim), dtype=torch.float32)
        self.chunk_starts = torch.zeros((max_chunks,), dtype=torch.int64)
        self.chunk_lengths = torch.zeros((max_chunks,), dtype=torch.int64)
        self.chunk_terminated = torch.zeros((max_chunks,), dtype=torch.bool)
        self.chunk_truncated = torch.zeros((max_chunks,), dtype=torch.bool)
        self.chunk_env_id = torch.zeros((max_chunks,), dtype=torch.int16)
        self.chunk_prev = torch.full((max_chunks,), -1, dtype=torch.int64)
        self._last_chunk: dict[int, int] = {}
        self.n_chunks = 0
        self.n_steps = 0
        self.carry: torch.Tensor | None = None

    def reset(self) -> None:
        self.n_chunks = self.n_steps = 0
        self._last_chunk.clear()

    def active_prefix(self) -> dict[str, torch.Tensor]:
        nc, ns = self.n_chunks, self.n_steps
        return {
            "normalizer_observations": self.normalizer_observations[:ns],
            "step_rewards": self.step_rewards[:ns],
            "chunk_state": self.chunk_state[:nc],
            "chunk_next_state": self.chunk_next_state[:nc],
            "chunk_actions": self.chunk_actions[:nc],
            "chunk_starts": self.chunk_starts[:nc],
            "chunk_lengths": self.chunk_lengths[:nc],
            "chunk_terminated": self.chunk_terminated[:nc],
            "chunk_truncated": self.chunk_truncated[:nc],
            "chunk_env_id": self.chunk_env_id[:nc],
            "chunk_prev": self.chunk_prev[:nc],
        }


    def write_chunks(
        self, *, start_window: torch.Tensor, next_window: torch.Tensor, actions: torch.Tensor,
        lengths: torch.Tensor, frames: torch.Tensor, rewards: torch.Tensor,
        terminated: torch.Tensor, truncated: torch.Tensor, env_ids, n_steps: int,
    ) -> None:
        k = int(lengths.numel())
        if k == 0:
            return
        if self.n_chunks + k > self.max_chunks or n_steps < 1 or self.n_steps + n_steps > self.max_steps:
            raise RuntimeError(f"chunk capacity exceeded: {k} chunks and {n_steps} steps do not fit")
        lo, hi = self.n_chunks, self.n_chunks + k
        slo, shi = self.n_steps, self.n_steps + n_steps
        mask = (torch.arange(self.action_horizon)[None, :] < lengths[:, None]).reshape(-1)

        self.chunk_state[lo:hi] = start_window
        self.chunk_next_state[lo:hi] = next_window
        self.chunk_actions[lo:hi] = actions
        self.chunk_starts[lo:hi] = slo + torch.cumsum(lengths, 0) - lengths
        self.chunk_lengths[lo:hi] = lengths
        self.chunk_terminated[lo:hi] = terminated
        self.chunk_truncated[lo:hi] = truncated
        self.chunk_env_id[lo:hi] = torch.as_tensor(env_ids, dtype=torch.int16)

        prev_list = []
        for j, raw_e in enumerate(torch.as_tensor(env_ids).detach().cpu().tolist()):
            e = int(raw_e)
            prev_list.append(self._last_chunk.get(e, -1))
            self._last_chunk[e] = lo + j
        self.chunk_prev[lo:hi] = torch.tensor(prev_list, dtype=torch.int64)
        self.normalizer_observations[slo:shi] = frames.reshape(-1, self.obs_dim)[mask]
        self.step_rewards[slo:shi] = rewards.reshape(-1)[mask]
        self.n_chunks, self.n_steps = hi, shi
def is_torch_env(envs) -> bool:
    return callable(getattr(envs, "step_torch", None))
def take_final_observations(info, done_np, reference: torch.Tensor, strict: bool) -> torch.Tensor:
    """Replace done rows of ``reference`` with true terminal observations from ``info``."""
    rows = np.flatnonzero(done_np)
    if rows.size == 0:
        return reference
    final = info.get("final_obs") if isinstance(info, dict) else None
    if final is None and isinstance(info, dict):
        final = info.get("final_observation")
    if final is None:
        if strict:
            raise ValueError(f"SAME_STEP env reset but supplied no final observation for done rows {rows.tolist()}")
        return reference

    out = reference.clone()
    n_envs = reference.shape[0]
    idx = torch.as_tensor(rows, dtype=torch.long)

    def _select(vals: torch.Tensor) -> torch.Tensor:
        if vals.shape[0] == n_envs:
            return vals[idx]
        if vals.shape[0] == rows.size:
            return vals
        raise ValueError(f"final observation has {vals.shape[0]} rows, expected {n_envs} or {rows.size}")

    if torch.is_tensor(final):
        out[idx] = _select(final.to(device=reference.device, dtype=reference.dtype))
    else:
        arr = np.asarray(final)
        if arr.dtype == object:
            for k, i in enumerate(rows):
                val = arr[i] if arr.shape[0] == n_envs else arr[k]
                if val is None:
                    if strict:
                        raise ValueError(f"final observation for done row {i} is None")
                    continue
                out[i] = torch.as_tensor(np.asarray(val, dtype=np.float32), device=reference.device)
        else:
            out[idx] = _select(torch.as_tensor(arr, device=reference.device, dtype=reference.dtype))
    return out
def collect_chunks_vec(
    envs, policy, chunks_per_env: int, action_low, action_high, *,
    num_obs_steps: int, num_action_steps: int, buffer: RolloutBuffer,
    carry: torch.Tensor | None = None, action_noise_std: float = 0.0, seed: int | None = None,
    max_steps_per_env: int | None = None,
) -> RolloutBuffer:
    """Collect executed chunks from a lockstep SameStep vector env."""
    low, high = resolve_action_bounds(action_low, action_high)
    if not is_torch_env(envs):
        mode = getattr(envs, "autoreset_mode", None)
        mode_name = str(getattr(mode, "name", None) or mode or "").upper().replace("_", "")
        if mode_name != "SAMESTEP":
            raise ValueError(f"collector requires SameStep autoreset; got {mode!r}")

    n = int(envs.num_envs)
    act_dim = int(np.asarray(envs.single_action_space.shape).prod())
    targets = np.full(n, int(chunks_per_env), dtype=np.int64)
    cap = np.iinfo(np.int64).max if max_steps_per_env is None else max(1, int(max_steps_per_env))
    closed = np.zeros(n, dtype=np.int64)

    if seed is None:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    if carry is None:
        obs0, _ = envs.reset(seed=[seed + i for i in range(n)])
        obs0_t = obs0.detach().cpu().float() if torch.is_tensor(obs0) else torch.from_numpy(np.asarray(obs0, dtype=np.float32))
        windows = obs0_t.clone() if obs0_t.dim() == 3 else obs0_t[:, None, :].expand(n, num_obs_steps, -1).clone()
    else:
        windows = carry.float().clone()

    obs_dim = windows.shape[-1]
    horizon = buffer.action_horizon
    active = np.zeros(n, dtype=bool)
    step_limits = np.zeros(n, dtype=np.int64)
    executed_total = np.zeros(n, dtype=np.int64)
    remaining = int(targets.sum())
    chunk_pos = torch.zeros(n, dtype=torch.long)
    raw_plan = torch.zeros(n, horizon, act_dim)
    exec_plan = torch.zeros(n, horizon, act_dim)
    start_window = torch.zeros(n, num_obs_steps, obs_dim)
    reward_win = torch.zeros(n, horizon)
    obs_win = torch.zeros(n, horizon, obs_dim)
    next_actions = torch.zeros(n, act_dim)
    torch_env = is_torch_env(envs)

    while remaining > 0:
        need = np.flatnonzero((~active) & (closed < targets) & (executed_total < cap))
        if need.size:
            wf = torch.as_tensor(need, dtype=torch.long)
            states = windows[wf]
            raw = policy(states).detach().cpu().float()
            raw_plan[wf] = raw
            exec_plan[wf] = to_env_action(raw, low, high)
            start_window[wf] = states
            chunk_pos[wf] = 0
            step_limits[need] = np.minimum(num_action_steps, cap - executed_total[need])
            active[need] = True
        # Every env stops when its step cap is exhausted before the chunk target.
        if not active.any() and need.size == 0:
            break

        idle = np.flatnonzero(~active)
        live = torch.from_numpy(np.flatnonzero(active))
        next_actions = torch.zeros(n, act_dim)
        if live.numel():
            next_actions[live] = exec_plan[live, chunk_pos[live].clamp(max=horizon - 1)]
        if idle.size:
            # Idle rows still step in lockstep. Run them on-policy and keep
            # their windows fresh: their transitions are not stored, but a
            # stale window would poison the carry into the next rollout.
            idle_t = torch.as_tensor(idle, dtype=torch.long)
            idle_raw = policy(windows[idle_t]).detach().cpu().float()
            next_actions[idle_t] = to_env_action(idle_raw, low, high)[:, 0]

        if torch_env:
            obs_r, rew_r, term_r, trunc_r, info = envs.step_torch(next_actions)
            raw_obs, rew_t = obs_r.detach().cpu().float(), rew_r.detach().cpu().float()
            term_np, trunc_np = term_r.cpu().numpy().astype(bool), trunc_r.cpu().numpy().astype(bool)
        else:
            obs_r, rew_r, term_r, trunc_r, info = envs.step(next_actions.numpy())
            raw_obs = torch.from_numpy(np.asarray(obs_r, dtype=np.float32))
            rew_t = torch.from_numpy(np.asarray(rew_r, dtype=np.float32))
            term_np, trunc_np = np.asarray(term_r, dtype=bool), np.asarray(trunc_r, dtype=bool)

        done_np = term_np | trunc_np
        reset_frames = raw_obs if done_np.any() else None
        obs_t = take_final_observations(info, done_np, raw_obs, strict=True) if done_np.any() else raw_obs

        # Every row slides by one frame, every step.
        windows = torch.cat((windows[:, 1:], obs_t[:, None]), dim=1)

        if live.numel():
            p = chunk_pos[live]
            reward_win[live, p] = rew_t[live]
            obs_win[live, p] = obs_t[live]
            chunk_pos[live] = p + 1

        if idle.size:
            # Done rows that are not mid-chunk get their window reset.
            d_idle = done_np[idle]
            if d_idle.any():
                d_rows = torch.as_tensor(idle[d_idle])
                windows[d_rows] = reset_frames[d_rows].unsqueeze(1).expand(-1, num_obs_steps, -1)

        closing = np.flatnonzero(active & (done_np | (chunk_pos.numpy() >= step_limits)))
        if closing.size:
            rows = torch.as_tensor(closing)
            lens_np = chunk_pos.numpy()[closing]
            n_new = int(lens_np.sum())
            lengths = torch.as_tensor(lens_np, dtype=torch.long)
            next_win = windows[rows].clone()
            stored = raw_plan[rows]
            if action_noise_std > 0:
                stored = bound_for_storage(stored + action_noise_std * torch.randn_like(stored))
            buffer.write_chunks(
                start_window=start_window[rows], next_window=next_win, actions=stored,
                lengths=lengths, frames=obs_win[rows], rewards=reward_win[rows],
                terminated=torch.as_tensor(term_np)[rows], truncated=torch.as_tensor(trunc_np)[rows],
                env_ids=closing, n_steps=n_new,
            )
            executed_total[closing] += lens_np
            closed[closing] += 1
            remaining -= closing.size
            active[closing] = False
            c_done = closing[done_np[closing]]
            if c_done.size:
                cd_rows = torch.as_tensor(c_done)
                windows[cd_rows] = reset_frames[cd_rows].unsqueeze(1).expand(-1, num_obs_steps, -1)

    buffer.carry = windows
    return buffer
@dataclasses.dataclass
class RolloutState:
    buffer: RolloutBuffer
def _rollout_budget(cfg: FlowPolicyOptimizationConfig, n_envs: int, remaining_steps: int) -> int:
    """Chunk budget for one rollout.

    A full rollout is ``chunks_per_update // n_envs`` chunks per env. A final
    partial rollout covers the remaining steps in one chunk wave instead of
    running a full rollout past the step budget; chunk granularity cannot
    match ``total_steps`` exactly, so the last rollout overshoots by under a
    chunk wave.
    """
    full_chunks_per_env = max(1, int(cfg.chunks_per_update) // n_envs)
    num_action_steps = int(cfg.num_action_steps)
    wave = n_envs * num_action_steps
    if remaining_steps >= full_chunks_per_env * wave:
        return full_chunks_per_env
    return max(1, math.ceil(remaining_steps / wave))
def make_rollout_state(obs_dim: int, action_dim: int, cfg: FlowPolicyOptimizationConfig, chunks_hint: int | None = None) -> RolloutState:
    chunks = max(1, int(cfg.chunks_per_update) if chunks_hint is None else int(chunks_hint))
    return RolloutState(
        buffer=RolloutBuffer(
            max_chunks=chunks, max_steps=chunks * int(cfg.num_action_steps),
            obs_steps=cfg.num_obs_steps, obs_dim=obs_dim,
            action_horizon=cfg.action_horizon, action_dim=action_dim,
        )
    )


@torch.no_grad()
def run_rollout(policy: FlowPolicyOptimizationPolicy, env, chunks_per_env: int, *, action_low, action_high, state: RolloutState, max_steps_per_env: int | None = None) -> dict:
    """Collect one rollout and compute V-anchored GAE + behavior CFM losses."""
    if chunks_per_env <= 0:
        raise ValueError(f"chunks_per_env={chunks_per_env} must be positive")
    cfg = policy.config
    device = next(policy.parameters()).device
    buffer = state.buffer
    buffer.reset()

    collect_chunks_vec(
        env, policy.predict_action_chunk, chunks_per_env, action_low, action_high,
        num_obs_steps=cfg.num_obs_steps, num_action_steps=cfg.num_action_steps,
        buffer=buffer, carry=buffer.carry,
        max_steps_per_env=max_steps_per_env,
    )
    active = buffer.active_prefix()
    n_chunks = buffer.n_chunks
    policy.obs_normalizer.update(active["normalizer_observations"])

    states = active["chunk_state"].to(device)
    next_states = active["chunk_next_state"].to(device)
    actions = active["chunk_actions"].to(device)
    lengths = active["chunk_lengths"].to(device)

    all_norm = policy.obs_normalizer(torch.cat([states, next_states], dim=0))
    norm_states, norm_next = all_norm[:n_chunks], all_norm[n_chunks:]

    v_cur = policy.critic.forward_v(norm_states)
    v_nxt = policy.critic_target.forward_v(norm_next)

    adv, returns = compute_chunk_advantages(
        active["step_rewards"], v_cur, v_nxt,
        active["chunk_starts"].reshape(-1), lengths.reshape(-1),
        chunk_terminated=active["chunk_terminated"].reshape(-1),
        chunk_size=cfg.action_horizon, gamma=cfg.gamma,
        chunk_truncated=active["chunk_truncated"].reshape(-1),
        chunk_prev=active["chunk_prev"], gae_lambda=cfg.gae_lambda,
    )
    if cfg.normalize_advantage and adv.numel() > 0:
        adv = (adv - adv.mean()) / adv.std(correction=0).clamp_min(1e-8)

    action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    lo, hi = cfg.min_sampled_timestep, cfg.max_sampled_timestep
    t = torch.rand(n_chunks, cfg.num_mc_samples, cfg.action_horizon, 1, device=device) * (hi - lo) + lo
    eps = torch.randn(n_chunks, cfg.num_mc_samples, cfg.action_horizon, buffer.action_dim, device=device)
    beh_loss, beh_loss_tok = policy.compute_behavior_cfm_losses(norm_states, actions, t, eps, action_is_pad)

    return {
        "observation.state": states,
        "action": actions,
        "action_is_pad": action_is_pad,
        "chunk_truncated": active["chunk_truncated"],
        "normalized_advantages": adv,
        "raw_returns": returns,
        "boot_observations": norm_next,
        "mc_t": t,
        "mc_eps": eps,
        "behavior_cfm_loss": beh_loss,
        "behavior_cfm_loss_token": beh_loss_tok,
        "step_rewards": active["step_rewards"],
    }


# ---------------------------------------------------------------------------
# Optimizer & Update
