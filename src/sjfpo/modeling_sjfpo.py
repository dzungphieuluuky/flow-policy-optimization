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
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import bootstrap_action_lengths, collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import dct, idct, mask_actions
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_mc_simulation,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo.gae import compute_advantages
from sjfpo.utils import autocast_context, to_numpy_1d, to_tensor_1d
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFPOFlowModel,
    SpectralEncoder,
)
from sjfpo.processor_sjfpo import make_sjfpo_pre_post_processors


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
        action_dim = int(config.action_feature.shape[0])
        self.actor = SJFPOFlowModel(
            action_dim,
            obs_dim,
            hidden_dim=config.hidden_dim,
            spectral_condition_mode=config.spectral_condition_mode,
        )
        self.spectral = SpectralEncoder(
            obs_dim,
            top_p_spectral_tokens=config.top_p_spectral_tokens,
            n_obs_steps=config.n_obs_steps,
        )
        self.critic = Critic(
            obs_dim,
            action_dim,
            hidden_dim=config.hidden_dim,
            n_obs_steps=config.n_obs_steps,
        )
        self.obs_normalizer = RunningMeanStd(obs_dim)
        # Real action bounds (env.action_space); predict/select return REAL
        # actions, training batches stay scaled [-1, 1]. Non-persistent so
        # old checkpoints load unchanged; build_policy sets them per run.
        self.register_buffer("action_low", torch.full((action_dim,), -1.0), persistent=False)
        self.register_buffer("action_high", torch.full((action_dim,), 1.0), persistent=False)
        # Frozen behavior snapshot (PPO π_old): unit ratios at init, refreshed
        # per collection round; rides state_dict but excluded from the optimizer.
        # Inference-only (requires_grad off): nothing ever backprops through
        # it, and DDP treats grad-less requires_grad params as an error.
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)
        # Slow EMA anchor for the KL loss; inference-only (never compiled,
        # never optimized), rides state_dict so checkpoints resume it.
        self.ema_actor = copy.deepcopy(self.actor)
        self.ema_actor.requires_grad_(False)
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        # LeRobot-fashion pre/post pair sharing the live normalizer + bounds.
        self.preprocessor, self.postprocessor = make_sjfpo_pre_post_processors(self)
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

    def set_action_bounds(self, low: np.ndarray | torch.Tensor, high: np.ndarray | torch.Tensor) -> None:
        """Real env bounds for unscaling; defaults (±1) are identity."""
        # copy_ keeps buffer registration (device moves still follow .to()).
        self.action_low.copy_(to_tensor_1d(low, dtype=torch.float32, device=self.action_low.device))
        self.action_high.copy_(to_tensor_1d(high, dtype=torch.float32, device=self.action_high.device))

    def snapshot_behavior_policy(self) -> None:
        """Freeze current weights as the behavior policy for l_old.

        Call once per collection round, before any update on the new batch.
        """
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)  # deepcopy inherits live grad flags

    def get_optim_params(self) -> dict:
        """Actor (decay / no-decay) + critic groups (spectral/EMA never train).

        Zero-init layers (adaLN regressors incl. adaLN_out, final out) skip
        weight decay so decay never fights the identity-at-init design.
        """
        decay, no_decay = [], []
        for name, p in self.actor.named_parameters():
            # Strip the torch.compile wrapper prefix so compiled/eager group
            # identically (else save/load optimizer states stop matching).
            short = name.split("_orig_mod.")[-1]
            if "adaLN" in short or short.startswith("out."):
                no_decay.append(p)
            else:
                decay.append(p)
        return {
            "params": [
                {"params": decay},
                {"params": no_decay, "weight_decay": 0.0},
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.config.optimizer_lr_critic,
                },
            ]
        }

    @torch.no_grad()
    def update_ema(self) -> None:
        """Fold live weights into the EMA anchor (call after optimizer.step)."""
        live = getattr(self.actor, "_orig_mod", self.actor)
        decay = self.config.ema_decay
        for p_ema, p in zip(self.ema_actor.parameters(), live.parameters()):
            p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce [B, T, obs] / [B, obs] / [T, obs] to a [B, T, obs] window."""
        states = to_tensor(states, dtype=torch.float32)
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
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Full action chunk [B, H, action_dim]; ``noise=True`` samples, else zeros."""
        noise = kwargs.get("noise", False)
        cfg = self.config
        states = self.standardize_obs_tensor(batch[OBS_STATE])
        states = states.to(next(self.actor.parameters()).device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]
        top_k_spectral_action_tokens = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
        action_dim = int(self.config.action_feature.shape[0])
        with torch.inference_mode(), autocast_context(states.device):
            condition = self.spectral(states)
            shape = (states.shape[0], top_k_spectral_action_tokens, action_dim)
            eps = None if noise else torch.zeros(*shape, device=states.device)
            num_inference_steps = cfg.num_euler_steps if noise else cfg.num_inference_steps
            x0 = sample_action(
                self.compute_predicted_velocity,
                shape,
                condition,
                num_inference_steps=num_inference_steps,
                eps=eps,
                device=states.device,
                state=states,
            )
            x0 = x0.float()
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, cfg.action_horizon - top_k_spectral_action_tokens))
        scaled = idct(x0, dim=1).clamp(-1.0, 1.0)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """One action per call: replan a chunk, drain its executed prefix open-loop."""
        obs = to_tensor(batch[OBS_STATE], dtype=torch.float32)
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

        def _propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: states.to(device)}, noise=noisy
            )
            return chunk.cpu()

        if int(getattr(env, "num_envs", 1)) > 1:
            chunks = collect_chunks_vec(
                env,
                _propose_batch,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
            )
        else:
            chunks = collect_chunks(
                env,
                _propose,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
            )
        executed_n_action_steps = to_tensor(
            [int(c["executed_n_action_steps"]) for c in chunks], dtype=torch.long
        )  # [N] CPU; compiled critic takes the device copy below
        next_executed_n_action_steps = to_tensor(
            bootstrap_action_lengths(chunks, cfg.action_horizon), dtype=torch.long
        )  # [N] CPU
        with torch.inference_mode():
            states = torch.stack([c[OBS_STATE] for c in chunks]).to(device)
            next_states = torch.stack([c["next.observation.state"] for c in chunks]).to(
                device
            )
            states_n = self.preprocessor({OBS_STATE: states})[OBS_STATE]
            next_states_n = self.preprocessor({OBS_STATE: next_states})[OBS_STATE]
            a_exec = mask_actions(
                torch.stack([c["action"] for c in chunks]).to(device), executed_n_action_steps, dim=1
            )
            a_next_exec = mask_actions(
                torch.stack([c["next.action"] for c in chunks]).to(device),
                next_executed_n_action_steps,
                dim=1,
            )
            with autocast_context(device):
                values = self.critic(states_n, self.postprocessor({ACTION: a_exec})[ACTION], executed_n_action_steps.to(device)).float()
                v_next_all = self.critic(
                    next_states_n, self.postprocessor({ACTION: a_next_exec})[ACTION], next_executed_n_action_steps.to(device)
                ).float()
            terminated = to_tensor([bool(c[DONE]) for c in chunks], dtype=torch.bool, device=device)
            next_values = torch.where(terminated, 0.0, v_next_all)
        all_rewards = np.concatenate([to_numpy_1d(c[REWARD]) for c in chunks])
        advantages, returns = compute_advantages(
            all_rewards,
            to_numpy_1d(values),
            to_numpy_1d(next_values),
            to_numpy_1d(executed_n_action_steps).astype(np.int64),
            np.array([bool(c[DONE]) for c in chunks]),
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            normalize=True,
            truncated=np.array([bool(c[TRUNCATED]) for c in chunks]),
        )
        is_pad = torch.arange(cfg.action_horizon).unsqueeze(0) >= to_tensor(
            executed_n_action_steps, dtype=torch.long
        ).unsqueeze(1)
        return {
            OBS_STATE: torch.stack([c[OBS_STATE] for c in chunks]).to(device),
            ACTION: torch.stack([c["action"] for c in chunks]).to(device),
            "action_is_pad": is_pad.to(device),
            "advantages": to_tensor(advantages, dtype=torch.float32, device=device),
            "returns": to_tensor(returns, dtype=torch.float32, device=device),
            "values": values.detach().to(device),
        }

    def forward(
        self, batch: dict[str, torch.Tensor], reduction: str = "mean"
    ) -> tuple[torch.Tensor, dict | None]:
        """Compute the combined training loss (ASPO policy + value + KL) from a rollout batch.

        Inputs (batch dict from ``generate_rollout``):
            observation.state: [B, T, obs_dim]
            action:              [B, H, action_dim]
            advantages:          [B]  (frozen GAE; never optimized)
            returns:             [B]  (GAE targets for value loss)
            values:              [B]  (optional; frozen v_old for clipped value loss)
            action_is_pad:       [B, H] bool (optional; if missing, full horizon executed)

        Outputs:
            (loss, info_dict) where ``loss`` is scalar if ``reduction="mean"``
            or per-chunk [B] if ``reduction="none"``. ``info_dict`` holds detached
            scalar summaries: ``policy_loss``, ``value_loss`` (kl_loss removed),
            ``mean_log_ratio``, plus diagnostics (auto-logged to wandb as
            ``train/...``): ``flow_rho_mean/std``, ``flow_v_star_rms``, and
            ``cond_*`` RMS norms (``temb``, ``pooled``, ``c``, per-block mean
            ``gate_msa/mlp`` and ``1+scale``) — gates near 0 mean the spectrum
            is unused (expected early under adaLN-Zero).

        Gradient behavior:
            - ``actor`` and ``critic`` receive gradients.
            - ``old_actor`` (behavior snapshot) is frozen (``inference_mode``).
            - ``ema_actor`` is frozen (``requires_grad_(False)``) — KL compares
              ``v_new`` (live) to ``v_ema`` (EMA anchor) without EMA updates.
        """
        cfg = self.config
        states = to_tensor(batch[OBS_STATE], dtype=torch.float32)
        actions = to_tensor(batch[ACTION], dtype=torch.float32)
        advantages = to_tensor(batch["advantages"], dtype=torch.float32)
        returns = to_tensor(batch["returns"], dtype=torch.float32)  # [B]
        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action action_horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )
        if self.training:
            self.preprocessor.update_stats(
                states
            )  # train batches only; rollout/eval frozen
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]
        is_pad = batch.get("action_is_pad")
        if is_pad is None:
            executed_n_action_steps = torch.full(
                (actions.shape[0],),
                actions.shape[1],
                dtype=torch.long,
                device=actions.device,
            )  # [B] all executed
            a_exec = actions
        else:
            is_pad = to_tensor(is_pad, dtype=torch.bool, device=actions.device)
            executed_n_action_steps = (~is_pad).sum(dim=1)  # [B] executed steps
            a_exec = actions * (~is_pad).to(actions.dtype).unsqueeze(-1)
        top_k_spectral_action_tokens = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
        spectral_action_tokens = dct(a_exec, dim=1)[:, :top_k_spectral_action_tokens, :]
        condition = self.spectral(states)
        n = states.shape[0]
        tau = sample_timesteps(
            cfg.num_mc_samples,
            (cfg.min_timestep, cfg.max_timestep),
            device=states.device,
        )
        eps = sample_noise(
            (n, cfg.num_mc_samples, top_k_spectral_action_tokens, actions.shape[-1]),
            device=states.device,
        )

        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
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
        cond_stats = dict(self.actor.last_cond_stats) 
        # KL/EMA anchor removed from pipeline per instruction.
        kl_loss = torch.tensor(0.0)
        v_star = self.compute_target_velocity(spectral_action_tokens, eps)
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
            v_pred = self.critic(states, self.postprocessor({ACTION: a_exec})[ACTION], executed_n_action_steps).float()
        returns_d = returns.to(v_pred.device)
        # Per-batch standardization: the critic learns normalized-return space so
        # the value loss scale stays stable across reward scales. old_values use
        # the same batch stats so the PPO-style clip stays in the same space.
        returns_mean = returns_d.mean()
        returns_std = returns_d.std(unbiased=False).clamp_min(1e-8)
        returns_norm = (returns_d - returns_mean) / returns_std
        se = (v_pred - returns_norm).pow(2)
        if "values" in batch:  # PPO-style clipped value loss against v_old
            old_values = to_tensor(
                batch["values"], dtype=torch.float32, device=v_pred.device
            )
            old_values_norm = (old_values - returns_mean) / returns_std
            v_clipped = old_values_norm + (v_pred - old_values_norm).clamp(
                -cfg.clip_range_vf, cfg.clip_range_vf
            )
            value_per_chunk = torch.max(se, (v_clipped - returns_norm).pow(2))
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
            rho_mean = float(rho.detach().mean())
            rho_std = float(rho.detach().std(unbiased=False))
            v_star_rms = float(v_star.detach().pow(2).mean().sqrt())
            # Phase 0 per-bin trust diagnostics (read-only): same elementwise
            # ratio math as the scalar path, unreduced over bins. The scalar
            # loss above is untouched; these stats only answer "is rho flat
            # in k?" ahead of any per-bin clipping experiment.
            l_old_bin = compute_cfm_loss(
                v_old.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="bin"
            )
            l_new_bin = compute_cfm_loss(
                v_new.reshape(n, num_mc, *flow_shape), v_star, reduce_mode="bin"
            )
            rho_bin = compute_policy_ratio(
                l_old_bin, l_new_bin, advantage=advantages.view(n, 1, 1)
            )
            rho_bin_mean = rho_bin.mean(dim=(0, 1))  # [K] over batch, mc
            n_bins = rho_bin_mean.numel()
            split = max(1, n_bins // 2)
            rho_bin_range = float(rho_bin_mean.max() - rho_bin_mean.min())
            rho_bin_low = float(rho_bin_mean[:split].mean())
            rho_bin_high = float(
                rho_bin_mean[split:].mean()
                if n_bins > 1
                else rho_bin_mean.mean()
            )
            rho_bin_outside = float(
                (
                    (rho_bin < 1.0 - cfg.epsilon_low)
                    | (rho_bin > 1.0 + cfg.epsilon_high)
                )
                .float()
                .mean()
            )
        return loss, {
            "policy_loss": float(policy_per_chunk.mean().detach()),
            "value_loss": float(value_per_chunk.mean().detach()),
            "mean_log_ratio": mean_log_ratio,
            "flow_rho_mean": rho_mean,
            "flow_rho_std": rho_std,
            "flow_rho_bin_range": rho_bin_range,
            "flow_rho_bin_lowfreq": rho_bin_low,
            "flow_rho_bin_highfreq": rho_bin_high,
            "flow_rho_bin_outside_frac": rho_bin_outside,
            "flow_v_star_rms": v_star_rms,
            **{f"cond_{k}": float(v.detach()) for k, v in cond_stats.items()},
        }
