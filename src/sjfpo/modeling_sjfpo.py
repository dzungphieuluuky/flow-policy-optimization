"""SJFPO policy (LeRobot-native).

Single ``PreTrainedPolicy`` owning every SJFPO nn.Module as attributes:
policy flow head (with its policy state encoder), spectral encoder, critic
(with its critic state encoder), and the running obs normalizer. Mirrors the
math of the ``collector -> GAE -> update`` path, reorganized around
LeRobot batch dicts keyed by ``observation.state`` / ``action``.

Batch contract for ``forward`` (all fp32, on the policy device):
    observation.state: [B, T, obs] raw obs windows.
    action: [B, H, Da] real-space scaled actions (DCT + truncation inside).
    action_is_pad: optional [B, H] bool, True = unexecuted tail rows
        (LeRobot-native executed-step mask; absent = all executed).
    advantages: [B] raw chunk advantages from chunk GAE.
    returns: [B] frozen GAE value targets.
    values: optional [B] v_old at collection (PPO-style clipped value loss).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import einx
import torch

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.dct_utils import dct, idct
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    pair_tensor,
    sample,
)
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFOPolicyModel,
    SpectralEncoder,
)


class SJFPOAborted(RuntimeError):
    """Raised when NaN/Inf is detected in training losses.

    Attributes:
        message: Human-readable description of which loss term became non-finite.
    """

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
        config.validate_features()  # not called automatically by the base class
        obs_dim = int(config.robot_state_feature.shape[0])
        act_dim = int(config.action_feature.shape[0])
        self.actor = SJFOPolicyModel(
            act_dim, obs_dim, hidden_size=config.hidden_dim
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
        # Frozen behavior-policy snapshot (PPO π_θold): same weights at init
        # (unit ratios), refreshed once per collection round via
        # snapshot_behavior_policy(). Registered like a target net so it rides
        # state_dict, but excluded from get_optim_params().
        self.old_actor = copy.deepcopy(self.actor)
        # Single-env inference buffers (mirrors SjfpoEvalWrapper semantics).
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        # Update counter (checkpoint compat; incremented by the driver loop).
        self.step_count = 0
        # Accepted but unused: obs stats stay online (RunningMeanStd) rather
        # than frozen dataset stats.
        del dataset_stats

    @staticmethod
    def compute_target_velocity(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Target velocity v* = ε − a broadcast over MC draws.

        Args:
            a: Anchors, shape [B, *flow_shape].
            eps: Noise draws, shape [B, m, *flow_shape].

        Returns:
            v* of shape [B, m, *flow_shape] (independent of t by construction).
        """
        return eps - a.unsqueeze(1)  # [B, m, *flow_shape]

    def compute_predicted_velocity(
        self, x_t, tau, condition, state, old=False
    ) -> torch.Tensor:
        """Velocity prediction from the live head (or frozen snapshot if old).

        Args:
            x_t: Interpolated points, shape [B * m, *flow_shape].
            tau: Times, shape [B * m].
            condition/state: MC-repeated conditioning.
            old: Read the frozen behavior snapshot (l_old) instead of live weights.
        """
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
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        # Single AdamW group over live weights only: spectral is
        # parameter-free, the normalizer holds buffers, and the behavior
        # snapshot never trains.
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

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: bool = False
    ) -> torch.Tensor:
        """Return the full action chunk (B, horizon, action_dim) for the current observation."""
        cfg = self.config
        states = self._window_from_batch(batch[OBS_STATE])  # [B, T, obs]
        if self.obs_normalizer is not None:
            states = self.obs_normalizer(states)
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.horizon))
        act_dim = int(self.config.action_feature.shape[0])
        with torch.inference_mode():
            condition = self.spectral(states)  # [B, K, obs] raw DCT bins
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
            )  # [B, Ka, Da]
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, cfg.horizon - n_coeffs))
        raw = idct(x0, dim=1)
        return raw.clamp(-1.0, 1.0)  # [B, H, Da] scaled actions

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a single action for the current timestep (called every step at inference).

        Single-env contract: batch[observation.state] is [obs] (live step) or
        [T, obs] (prefilled window). Executes the ``n_action_steps`` prefix of
        each predicted chunk open-loop, then replans.
        """
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
            chunk = self.predict_action_chunk({OBS_STATE: window})  # [1, H, Da]
            n_execute = max(1, min(self.config.n_action_steps, chunk.shape[1]))
            for i in range(n_execute - 1, -1, -1):
                self._action_buffer.append(chunk[0, i])
        return self._action_buffer.popleft()  # [Da]

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict | None]:
        """Compute the training loss.

        Returns `(loss, output_dict)` with logging-friendly Python natives.
        Advantages/returns come precomputed (chunk GAE over the rollout that
        produced this batch); the policy never bootstraps inside ``forward``.
        """
        cfg = self.config
        states = torch.as_tensor(batch[OBS_STATE], dtype=torch.float32)  # [B, T, obs]
        actions = torch.as_tensor(batch[ACTION], dtype=torch.float32)  # [B, H, Da]
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)  # [B]
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
            # Running stats update here (train batches only); rollout and eval
            # normalize frozen through the shared module.
            self.obs_normalizer.update(states)
        states = self.obs_normalizer(states)
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            lengths = [actions.shape[1]] * actions.shape[0]
            a_exec = actions
        else:
            is_pad = torch.as_tensor(is_pad, dtype=torch.bool, device=actions.device)
            lengths = (~is_pad).sum(dim=1)  # [B] executed steps
            a_exec = actions * (~is_pad).to(actions.dtype).unsqueeze(-1)
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.horizon))
        action_coeffs = dct(a_exec, dim=1)[:, :n_coeffs, :]  # [N, Ka, Da]
        condition = self.spectral(states)  # [N, K, obs] raw DCT bins
        n = states.shape[0]
        tau = (
            torch.rand(cfg.num_mc_samples, device=states.device)
            * (cfg.tau_range[1] - cfg.tau_range[0])
            + cfg.tau_range[0]
        )  # [m]
        eps = torch.randn(
            n, cfg.num_mc_samples, n_coeffs, actions.shape[-1], device=states.device
        )  # [N, m, Ka, Da]

        # --- Shared MC interpolation: flats serving both likelihoods (same draw,
        # or rho is pure MC noise).
        tau_bn, eps_flat, a_rep, flow_shape, num_mc = pair_tensor(
            action_coeffs, eps, tau
        )
        x_t_flat = (
            tau_bn[:, None] * eps_flat + (1.0 - tau_bn[:, None]) * a_rep
        ).reshape(-1, *flow_shape)  # [N*m, *flow_shape] t·ε+(1−t)·a
        cond_flat = einx.id(
            "n m k d -> (n m) k d", einx.id("n k d -> n m k d", condition, m=num_mc)
        )
        state_flat = einx.id(
            "n m t d -> (n m) t d", einx.id("n t d -> n m t d", states, m=num_mc)
        )
        tau_flat = einx.id("m -> n m", tau, n=n).reshape(n * num_mc)

        # --- Velocity predictions: frozen snapshot (l_old) vs live weights.
        with torch.inference_mode():
            v_old = self.compute_predicted_velocity(
                x_t_flat, tau_flat, cond_flat, state_flat, old=True
            )
        v_new = self.compute_predicted_velocity(
            x_t_flat, tau_flat, cond_flat, state_flat
        )

        # --- Per-sample errors: target velocities + clamped flow-space means.
        v_star = self.compute_target_velocity(action_coeffs, eps)  # [N, m, *flow_shape]
        l_old = compute_cfm_loss(
            v_old.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="sample"
        )  # [N, m]
        l_new = compute_cfm_loss(
            v_new.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="sample"
        )  # [N, m]

        rho = compute_policy_ratio(l_old, l_new, adv=advantages.unsqueeze(1))
        psi = compute_aspo_surrogate(rho, advantages.unsqueeze(1), cfg.clip_range)
        policy_loss = -psi.mean()
        v_pred = self.critic(states, a_exec, lengths)  # [N] values
        if "values" in batch:  # v_old at collection: PPO-style clipped value loss
            old_values = torch.as_tensor(batch["values"], dtype=torch.float32).to(
                v_pred.device
            )
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.clip_range_vf, cfg.clip_range_vf
            )
            value_loss = torch.max(
                (v_pred - returns.to(v_pred.device)).pow(2),
                (v_clipped - returns.to(v_pred.device)).pow(2),
            ).mean()
        else:
            value_loss = torch.nn.functional.mse_loss(v_pred, returns.to(v_pred.device))
        loss = policy_loss + cfg.vf_coef * value_loss
        if not torch.isfinite(loss):
            raise SJFPOAborted(
                f"Non-finite loss: policy={float(policy_loss.detach())}, "
                f"value={float(value_loss.detach())}"
            )
        with torch.inference_mode():
            mean_log_ratio = float((l_old - l_new).mean())
        return loss, {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "mean_log_ratio": mean_log_ratio,
        }
