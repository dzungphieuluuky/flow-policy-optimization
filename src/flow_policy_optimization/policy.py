"""flow_policy_optimization.policy: the flow actor + encoder + critic module."""

from __future__ import annotations

import copy
from collections import deque

import torch
from torch import nn

from flow_policy_optimization.config import FlowPolicyOptimizationConfig, RunningMeanStd, resolve_action_bounds
from flow_policy_optimization.networks import Critic, FlowMLP, StateEncoder
from flow_policy_optimization.objective import (
    expand_mc_samples,
    sample_action,
    to_env_action,
    token_cfm_loss,
)

class FlowPolicyOptimizationPolicy(nn.Module):
    """Flow actor + state encoder + single cross-attention critic."""

    def __init__(self, config: FlowPolicyOptimizationConfig, obs_dim: int, action_dim: int, action_low, action_high):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = FlowMLP(action_dim, config.action_horizon, config.hidden_dim, config.time_embed_dim)
        self.encoder = StateEncoder(obs_dim, config.hidden_dim, config.num_obs_steps)
        critic_width = config.critic_hidden_dim
        self.critic = Critic(
            obs_dim, critic_width, action_dim, config.num_obs_steps, config.action_horizon,
            num_layers=config.critic_num_layers, state_hidden=config.hidden_dim,
        )
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.obs_normalizer = RunningMeanStd(obs_dim)
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=config.num_obs_steps)
        self.action_buffer: deque[torch.Tensor] = deque()
        low, high = resolve_action_bounds(action_low, action_high)
        self.register_buffer("action_low", low.float(), persistent=False)
        self.register_buffer("action_high", high.float(), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.critic_target.eval()
        return self

    def reset(self) -> None:
        self.frame_buffer.clear()
        self.action_buffer.clear()

    def _to_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        return actions / self.config.actor_scale

    def _from_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.config.actor_scale

    def compute_predicted_velocity(self, x_t, t, condition):
        v = self.actor(x_t, t, condition)
        bound = self.config.velocity_clip_value
        return v if bound is None else v.clamp(-bound, bound)

    @torch.inference_mode()
    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(device=self.obs_normalizer.mean.device, dtype=torch.float32)
        return self.encoder(self.obs_normalizer(states))

    @torch.inference_mode()
    def sample_chunk(self, condition: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        shape = (condition.shape[0], cfg.action_horizon, self.action_dim)
        eps = torch.randn(*shape, device=condition.device)
        chunk = sample_action(
            self.compute_predicted_velocity, shape, condition,
            num_steps=cfg.num_steps_noise_start, eps=eps, device=condition.device,
            t_min=cfg.min_sampled_timestep, t_max=cfg.max_sampled_timestep,
        ).float()
        chunk = self._from_flow_space(chunk)
        if self.training and cfg.action_perturb_std > 0:
            chunk = chunk + cfg.action_perturb_std * torch.randn_like(chunk)
        return chunk.clamp(-1.0, 1.0)

    def predict_action_chunk(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.dim() == 2 and states.shape[0] == self.config.num_obs_steps:
            states = states.unsqueeze(0)
        elif states.dim() != 3:
            raise ValueError(f"observation.state must be [B, T, obs] or [T, obs]; got {tuple(states.shape)}")
        return self.sample_chunk(self.encode_states(states))

    def select_action(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.action_low.device)
        if obs.dim() == 1:
            if self.frame_buffer:
                self.frame_buffer.append(obs.detach())
            else:
                for _ in range(self.config.num_obs_steps):
                    self.frame_buffer.append(obs.detach().clone())
            window = torch.stack(tuple(self.frame_buffer), dim=0)
        elif obs.dim() == 2:
            window = obs
        else:
            raise ValueError(f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}")
        if not self.action_buffer:
            chunk = self.predict_action_chunk(window)
            n_exec = max(1, min(self.config.num_action_steps, chunk.shape[1]))
            env_chunk = to_env_action(chunk, self.action_low, self.action_high)
            self.action_buffer.extend(env_chunk[0, i] for i in range(n_exec))
        return self.action_buffer.popleft()

    @torch.no_grad()
    def compute_behavior_cfm_losses(
        self, normalized_states: torch.Tensor, actions: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, action_is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self._to_flow_space(actions)
        x_t, flat_t, flow_shape, num_mc = expand_mc_samples(actions, eps, t)
        cond = self.encoder(normalized_states)
        cond_flat = cond.unsqueeze(1).expand(cond.shape[0], num_mc, cond.shape[1]).reshape(-1, cond.shape[1])
        v_old = self.compute_predicted_velocity(x_t, flat_t, cond_flat).float()
        return token_cfm_loss(v_old, actions, eps, action_is_pad, self.config, n=eps.shape[0], num_mc=num_mc, flow_shape=flow_shape)


# ---------------------------------------------------------------------------
# Chunk GAE, RolloutBuffer, and SameStep Collector
