"""SJFPO policy (LeRobot-native): flow head + spectral encoder + critic + obs stats.

Batch contract for ``forward`` (fp32, on policy device): observation.state
[B, T, obs], action [B, H, action_dim] real scaled, action_is_pad [B, H]
bool (absent = all executed), advantages/returns [B] (frozen GAE),
values [B] optional (v_old at collection for clipped value loss).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import einx
import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import bootstrap_action_lengths, collect_chunks
from sjfpo.dct_utils import dct, idct, mask_unexecuted_tail
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_mc_simulation,
    sample,
    sample_noise,
    sample_timesteps,
)
from sjfpo.gae import compute_advantages
from sjfpo.utils import autocast_context
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFOPolicyModel,
    SpectralEncoder,
)


class SJFPOAborted(RuntimeError):
    """Non-finite training loss (message names the offending term)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SJFPOPolicy(PreTrainedPolicy):
    config_class = SJFPOConfig  # must match the string in @register_subclass
    name = "sjfpo"

    def __init__(
        self, config: SJFPOConfig, dataset_stats: dict[str, Any] | None = None
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        obs_dim = int(config.robot_state_feature.shape[0])
        act_dim = int(config.action_feature.shape[0])
        self.actor = SJFOPolicyModel(
            act_dim, obs_dim,
            hidden_size=config.hidden_dim,
            spectral_pool=config.spectral_pool,
        )
        self.spectral = SpectralEncoder(
            obs_dim,
            top_p_spectral_tokens=config.top_p_spectral_tokens,
            n_obs_steps=config.n_obs_steps,
        )
        self.critic = Critic(
            obs_dim,
            act_dim,
            hidden_size=config.hidden_dim,
            n_obs_steps=config.n_obs_steps,
        )
        self.obs_normalizer = RunningMeanStd(obs_dim)
        # Frozen behavior snapshot (PPO π_old): unit ratios at init, refreshed
        # per collection round; rides state_dict but excluded from the optimizer.
        self.old_actor = copy.deepcopy(self.actor)
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        del dataset_stats  # stats stay online (RunningMeanStd), never frozen

    @staticmethod
    def compute_target_velocity(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """v* = a − ε broadcast over MC draws: [B, *flow] x [B, m, *flow] -> [B, m, *flow]."""
        return a.unsqueeze(1) - eps

    def compute_predicted_velocity(
        self, x_t, tau, condition, state, old=False
    ) -> torch.Tensor:
        """Velocity from the live head, or the frozen snapshot when old."""
        net = self.old_actor if old else self.actor
        return net(x_t, tau, condition, state)

    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    def snapshot_behavior_policy(self) -> None:
        """Freeze current weights as the behavior policy for l_old.

        Call once per collection round, before any update on the new batch.
        """
        self.old_actor = copy.deepcopy(self.actor)

    def get_optim_params(self) -> dict:
        """Live weights only (spectral is parameter-free, snapshot never trains)."""
        return {
            "params": list(self.actor.parameters())
            + list(self.critic.parameters())
        }

    def _window_from_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce [B, T, obs] / [B, obs] / [T, obs] to a [B, T, obs] window."""
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.dim() == 3:
            return states
        if states.dim() == 2:
            if states.shape[0] == self.config.n_obs_steps:
                return einx.id("t d -> 1 t d", states)
            return einx.id("d -> 1 t d", states, t=self.config.n_obs_steps)
        raise ValueError(
            f"observation.state must be [B, T, obs], [T, obs] or [B, obs]; got {tuple(states.shape)}"
        )

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Full action chunk [B, H, action_dim]; ``noise=True`` samples, else zeros."""
        noise = kwargs.get("noise", False)
        cfg = self.config
        states = self._window_from_batch(batch[OBS_STATE])
        states = states.to(next(self.actor.parameters()).device)
        if self.obs_normalizer is not None:
            states = self.obs_normalizer(states)
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.horizon))
        act_dim = int(self.config.action_feature.shape[0])
        with torch.inference_mode(), autocast_context(states.device):
            condition = self.spectral(states)
            shape = (states.shape[0], n_coeffs, act_dim)
            eps = None if noise else torch.zeros(*shape, device=states.device)
            steps = cfg.euler_train if noise else cfg.euler_eval
            x0 = sample(
                self.actor,
                shape,
                condition,
                steps=steps,
                eps=eps,
                device=states.device,
                state=states,
            )
            x0 = x0.float()
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, cfg.horizon - n_coeffs))
        raw = idct(x0, dim=1)
        return raw.clamp(-1.0, 1.0)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """One action per call: replan a chunk, drain its executed prefix open-loop."""
        obs = torch.as_tensor(batch[OBS_STATE], dtype=torch.float32)
        if obs.dim() == 1:
            if not self._frame_buffer:
                for _ in range(self.config.n_obs_steps):
                    self._frame_buffer.append(obs.clone())
            else:
                self._frame_buffer.append(obs)
            window = torch.stack(tuple(self._frame_buffer), dim=0)  # [T, obs]
        elif obs.dim() == 2:
            window = obs
        else:
            raise ValueError(
                f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}"
            )
        if not self._action_buffer:
            chunk = self.predict_action_chunk({OBS_STATE: window}, **kwargs)
            n_execute = max(1, min(self.config.n_action_steps, chunk.shape[1]))
            for i in range(n_execute - 1, -1, -1):
                self._action_buffer.append(chunk[0, i])
        return self._action_buffer.popleft()

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop chunks; return a forward-ready batch.

        Full-horizon noisy rollouts, V(s,a) under frozen stats, chunk GAE.
        Obs stats stay frozen here; ``forward`` updates them in train mode.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device

        def _propose(state: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: state.to(device)}, noise=noisy
            )[0]
            return chunk.cpu()  # env stepping stays on CPU

        chunks = collect_chunks(
            env,
            _propose,
            total_steps,
            action_low,
            action_high,
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.horizon,
        )
        lengths = torch.tensor(
            [int(c["n_action_steps"]) for c in chunks], dtype=torch.long
        )  # [N] CPU; compiled critic takes the device copy below
        next_lengths = torch.tensor(
            bootstrap_action_lengths(chunks, cfg.horizon), dtype=torch.long
        )  # [N] CPU
        with torch.inference_mode():
            states = torch.stack([c[OBS_STATE] for c in chunks]).to(device)
            next_states = torch.stack([c["next.observation.state"] for c in chunks]).to(device)
            states_n = self.obs_normalizer(states)
            next_states_n = self.obs_normalizer(next_states)
            a_exec = mask_unexecuted_tail(
                torch.stack([c["action"] for c in chunks]).to(device), lengths, dim=1
            )
            a_next_exec = mask_unexecuted_tail(
                torch.stack([c["next.action"] for c in chunks]).to(device),
                next_lengths,
                dim=1,
            )
            with autocast_context(device):
                values = self.critic(states_n, a_exec, lengths.to(device)).float()
                v_next_all = self.critic(next_states_n, a_next_exec, next_lengths.to(device)).float()
            terminated = torch.tensor([bool(c[DONE]) for c in chunks], device=device)
            next_values = torch.where(terminated, 0.0, v_next_all)
        all_rewards = [r for c in chunks for r in c[REWARD].tolist()]
        advantages, returns = compute_advantages(
            all_rewards,
            values.cpu().numpy(),
            next_values.cpu().numpy(),
            lengths.numpy(),
            np.array([bool(c[DONE]) for c in chunks]),
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            normalize=False,
            truncated=np.array([bool(c[TRUNCATED]) for c in chunks]),
        )
        is_pad = torch.arange(cfg.horizon).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(1)
        return {
            OBS_STATE: torch.stack([c[OBS_STATE] for c in chunks]).to(device),
            ACTION: torch.stack([c["action"] for c in chunks]).to(device),
            "action_is_pad": is_pad.to(device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "values": values.detach().to(device),
        }

    def forward(
        self, batch: dict[str, torch.Tensor], reduction: str = "mean"
    ) -> tuple[torch.Tensor, dict | None]:
        """Training loss from a generate_rollout batch (advantages/returns precomputed)."""
        cfg = self.config
        states = torch.as_tensor(batch[OBS_STATE], dtype=torch.float32)
        actions = torch.as_tensor(batch[ACTION], dtype=torch.float32)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)  # [B]
        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.horizon:
            raise ValueError(
                f"action horizon {actions.shape[-2]} != horizon {cfg.horizon}"
            )
        if self.training:
            self.obs_normalizer.update(states)  # train batches only; rollout/eval frozen
        states = self.obs_normalizer(states)
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            lengths = torch.full(
                (actions.shape[0],), actions.shape[1],
                dtype=torch.long, device=actions.device,
            )  # [B] all executed
            a_exec = actions
        else:
            is_pad = torch.as_tensor(is_pad, dtype=torch.bool, device=actions.device)
            lengths = (~is_pad).sum(dim=1)  # [B] executed steps
            a_exec = actions * (~is_pad).to(actions.dtype).unsqueeze(-1)
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.horizon))
        action_coeffs = dct(a_exec, dim=1)[:, :n_coeffs, :]
        condition = self.spectral(states)
        n = states.shape[0]
        tau = sample_timesteps(cfg.num_mc_samples, cfg.tau_range, device=states.device)
        eps = sample_noise(
            (n, cfg.num_mc_samples, n_coeffs, actions.shape[-1]),
            device=states.device,
        )

        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            action_coeffs, eps, tau
        )
        x_t_flat = (
            (1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep
        ).reshape(-1, *flow_shape)  # (1−t)·ε+t·a
        cond_flat = einx.id(
            "n m k d -> (n m) k d", einx.id("n k d -> n m k d", condition, m=num_mc)
        )
        state_flat = einx.id(
            "n m t d -> (n m) t d", einx.id("n t d -> n m t d", states, m=num_mc)
        )
        tau_flat = einx.id("m -> n m", tau, n=n).reshape(n * num_mc)

        # Frozen snapshot (l_old) vs live weights; DCT/CFM/ratio math stays fp32.
        with torch.inference_mode(), autocast_context(states.device):
            v_old = self.compute_predicted_velocity(
                x_t_flat, tau_flat, cond_flat, state_flat, old=True
            ).float()
        with autocast_context(states.device):
            v_new = self.compute_predicted_velocity(
                x_t_flat, tau_flat, cond_flat, state_flat
            ).float()

        v_star = self.compute_target_velocity(action_coeffs, eps)
        l_old = compute_cfm_loss(
            v_old.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="sample"
        )
        l_new = compute_cfm_loss(
            v_new.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="sample"
        )

        rho = compute_policy_ratio(l_old, l_new, advantage=advantages.unsqueeze(1))
        psi = compute_aspo_surrogate(
            rho, advantages.unsqueeze(1), cfg.epsilon_low, cfg.epsilon_high
        )
        policy_per_chunk = -psi.mean(dim=1)
        with autocast_context(states.device):
            v_pred = self.critic(states, a_exec, lengths).float()
        returns_d = returns.to(v_pred.device)
        se = (v_pred - returns_d).pow(2)
        if "values" in batch:  # PPO-style clipped value loss against v_old
            old_values = torch.as_tensor(batch["values"], dtype=torch.float32).to(
                v_pred.device
            )
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.clip_range_vf, cfg.clip_range_vf
            )
            value_per_chunk = torch.max(se, (v_clipped - returns_d).pow(2))
        else:
            value_per_chunk = se
        if reduction == "mean":
            policy_loss = policy_per_chunk.mean()
            value_loss = value_per_chunk.mean()
            loss = policy_loss + cfg.vf_coef * value_loss
        elif reduction == "none":
            loss = policy_per_chunk + cfg.vf_coef * value_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")
        if not bool(torch.isfinite(loss).all()):
            raise SJFPOAborted(
                f"Non-finite loss: policy={float(policy_per_chunk.mean().detach())}, "
                f"value={float(value_per_chunk.mean().detach())}"
            )
        with torch.inference_mode():
            mean_log_ratio = float((l_old - l_new).mean())
        return loss, {
            "policy_loss": float(policy_per_chunk.mean().detach()),
            "value_loss": float(value_per_chunk.mean().detach()),
            "mean_log_ratio": mean_log_ratio,
        }
