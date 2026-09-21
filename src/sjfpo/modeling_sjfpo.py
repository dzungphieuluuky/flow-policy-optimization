"""SJFPO policy (LeRobot-native): flow head + state conditioner + critic + obs stats.

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
from sjfpo.collector import collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import dct, idct
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_mc_simulation,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo.gae import compute_advantages_torch
from sjfpo.utils import autocast_context, filter_low_freq, resolve_storage_dtype, to_device_async, to_tensor_1d
STORAGE_DTYPE = resolve_storage_dtype()  # cached once
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFPOFlowModel,
    StateConditioner,
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
            hidden_dim=config.hidden_dim,
        )
        # Fused state conditioning (Flamingo-style: state queries DCT bins),
        # computed externally; the flow head only sees the [B, D] vector.
        self.conditioner = StateConditioner(
            obs_dim,
            hidden_dim=config.hidden_dim,
        )
        self.critic = Critic(
            obs_dim,
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
        self.old_conditioner = copy.deepcopy(self.conditioner)
        self.old_conditioner.requires_grad_(False)
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

    def compute_predicted_velocity(self, x_t, tau, states, old=False) -> torch.Tensor:
        """Velocity from the live head, or the frozen snapshot when old."""
        conditioner = self.old_conditioner if old else self.conditioner
        net = self.old_actor if old else self.actor
        return net(x_t, tau, conditioner(states))

    def reset(self):
        """Reset per-episode state. Called by lerobot-eval at the start of each episode."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    def set_action_bounds(self, low: np.ndarray | torch.Tensor, high: np.ndarray | torch.Tensor) -> None:
        """Real env bounds for unscaling; defaults (±1) are identity."""
        # copy_ keeps buffer registration (device moves still follow .to()).
        self.action_low.copy_(to_tensor_1d(low, dtype=STORAGE_DTYPE, device=self.action_low.device))
        self.action_high.copy_(to_tensor_1d(high, dtype=STORAGE_DTYPE, device=self.action_high.device))

    def snapshot_behavior_policy(self) -> None:
        """Freeze current weights as the behavior policy for l_old.

        Call once per collection round, before any update on the new batch.
        """
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)  # deepcopy inherits live grad flags
        self.old_conditioner = copy.deepcopy(self.conditioner)
        self.old_conditioner.requires_grad_(False)

    def get_optim_params(self) -> dict:
        """Actor (decay / no-decay) + conditioner + critic groups.

        Single LR everywhere (PPO standard); snapshots never train. Zero-init
        layers (adaLN regressors incl. adaLN_out, final out) skip weight decay
        so decay never fights the identity-at-init design.
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
                {"params": list(self.conditioner.parameters())},
                {"params": list(self.critic.parameters())},
            ]
        }

    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce [B, T, obs] / [B, obs] / [T, obs] to a [B, T, obs] window."""
        states = to_tensor(states, dtype=STORAGE_DTYPE)
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
        # ------- standardize + normalize obs window -------
        states = self.standardize_obs_tensor(batch[OBS_STATE])
        states = to_device_async(states, next(self.actor.parameters()).device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]
        token_dim = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
        action_dim = int(self.config.action_feature.shape[0])
        with torch.inference_mode(), autocast_context(states.device):
            # ------- fuse state into one condition vector -------
            condition = self.conditioner(states)  # [B, D] fused outside the flow head
            shape = (states.shape[0], token_dim, action_dim)
            eps = None if noise else torch.zeros(*shape, device=states.device)
            num_inference_steps = cfg.num_euler_steps if noise else cfg.num_inference_steps
            # ------- denoise action chunk with Euler sampler -------
            x0 = sample_action(
                lambda x, t, c: self.actor(x, t, c),
                shape,
                condition,
                num_inference_steps=num_inference_steps,
                eps=eps,
                device=states.device,
            )
            x0 = x0.float()
        # ------- pad + decode chunk back to action space -------
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, cfg.action_horizon - token_dim))
        scaled = idct(x0, dim=1).clamp(-1.0, 1.0)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """One action per call: replan a chunk, drain its executed prefix open-loop."""
        obs = to_tensor(batch[OBS_STATE], dtype=STORAGE_DTYPE)
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

        Full-horizon noisy rollouts, V(s) under frozen stats, chunk GAE.
        Obs stats stay frozen here; ``forward`` updates them in train mode.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device

        def propose(state: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(state, device)}, noise=noisy
            )[0]
            return chunk.cpu()  # env stepping stays on CPU

        def propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(states, device)}, noise=noisy
            )
            return chunk.cpu()

        from sjfpo.mjx_env import mjx_collect_chunks, resolve_env_name, want_mjx

        env_name = resolve_env_name(env)

        if want_mjx():
            # ------- GPU-batched MJX rollout (Blackwell box; raises if unavailable) -------
            chunks = mjx_collect_chunks(
                self,
                env_name,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
                num_envs=cfg.num_envs,
                seed=int(torch.randint(0, 2**31, (1,)).item()),
            )
        elif int(getattr(env, "num_envs", 1)) > 1:
            # ------- collect open-loop chunks from env -------
            chunks = collect_chunks_vec(
                env,
                propose_batch,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
            )
        else:
            # ------- collect open-loop chunks from env -------
            chunks = collect_chunks(
                env,
                propose,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
            )
        # Torch-only path (no numpy roundtrip): lengths/flags built on device.
        lengths = torch.as_tensor(
            [int(c["executed_n_action_steps"]) for c in chunks],
            dtype=torch.long,
            device=device,
        )  # [N] on device
        # Stacked once: reused for critic scoring below and the return dict
        # (stacking must stay outside inference_mode so forward() autograd
        # can consume the batch).
        states = to_device_async(
            torch.stack([c[OBS_STATE] for c in chunks]), device
        )
        next_states = to_device_async(
            torch.stack([c["next.observation.state"] for c in chunks]), device
        )
        with torch.inference_mode():
            states_n = self.preprocessor({OBS_STATE: states})[OBS_STATE]
            next_states_n = self.preprocessor({OBS_STATE: next_states})[OBS_STATE]
            # ------- score states with frozen V(s) critic -------
            with autocast_context(device):
                values = self.critic(states_n).float()
                v_next_all = self.critic(next_states_n).float()
            terminated = torch.as_tensor(
                [bool(c[DONE]) for c in chunks], dtype=torch.bool, device=device
            )
            truncated = torch.as_tensor(
                [bool(c[TRUNCATED]) for c in chunks], dtype=torch.bool, device=device
            )
            next_values = torch.where(terminated, 0.0, v_next_all)
        all_rewards = torch.cat(
            [c[REWARD].reshape(-1).to(device=device, dtype=torch.float64) for c in chunks]
        )
        # ------- chunk GAE into advantages + returns (torch, fp64) -------
        advantages, returns = compute_advantages_torch(
            all_rewards,
            values.detach(),
            next_values.detach(),
            lengths,
            terminated,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            normalize=True,
            truncated=truncated,
        )
        advantages = advantages.to(STORAGE_DTYPE)
        returns = returns.to(STORAGE_DTYPE)
        is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        n_chunks = len(chunks)
        action_dim = int(self.config.action_feature.shape[0])
        # ------- common random numbers: one shared (tau, eps) draw per round -------
        # Stored in the batch and reused by every forward() on it, so l_old and
        # l_new always compare the same MC samples (OFF rollout storage). The
        # behavior snapshot is frozen per round (train loop), never per step.
        top_k = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
        tau = sample_timesteps(
            cfg.num_mc_samples, (cfg.min_timestep, cfg.max_timestep), device=device
        )
        eps = sample_noise((n_chunks, cfg.num_mc_samples, top_k, action_dim), device=device)
        actions = to_device_async(torch.stack([c["action"] for c in chunks]), device)
        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": to_device_async(is_pad, device),
            "advantages": to_device_async(
                to_tensor(advantages, dtype=STORAGE_DTYPE), device
            ),
            "returns": to_device_async(to_tensor(returns, dtype=STORAGE_DTYPE), device),
            "values": values.detach().to(device),
            "tau": tau.unsqueeze(0).expand(n_chunks, -1).contiguous(),
            "eps": eps,
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
            action_is_pad:       [B, H] bool (accepted but ignored: the CFM loss
                always sees the full proposed chunk, paper-original)
            tau:                 [B, m] (stored rollout draw; reused for CRN)
            eps:                 [B, m, K, act] (stored rollout draw)

        Outputs:
            (loss, info_dict) where ``loss`` is scalar if ``reduction="mean"``
            or per-chunk [B] if ``reduction="none"``. ``info_dict`` holds detached
            scalar summaries: ``policy_loss``, ``value_loss``,
            ``explained_var``, ``mean_log_ratio``, plus diagnostics (auto-logged to wandb as
            ``train/...``): ``flow_rho_mean/std``, ``flow_v_star_rms``, and
            ``cond_*`` RMS norms (``temb``, ``pooled``, ``c``, per-block mean
            ``gate_msa/mlp`` and ``1+scale``) — gates near 0 mean the spectrum
            is unused (expected early under adaLN-Zero).

        Gradient behavior:
            - ``actor`` and ``critic`` receive gradients.
            - ``old_actor`` (behavior snapshot) is frozen (``inference_mode``).
        """
        cfg = self.config
        states = to_tensor(batch[OBS_STATE], dtype=STORAGE_DTYPE)
        actions = to_tensor(batch[ACTION], dtype=STORAGE_DTYPE)
        advantages = to_tensor(batch["advantages"], dtype=STORAGE_DTYPE)
        returns = to_tensor(batch["returns"], dtype=STORAGE_DTYPE)  # [B]
        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action action_horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )
        # Observation stats frozen for this rollout/update cycle (updated once
        # outside PPO epochs, not per epoch/minibatch, to avoid drift during
        # optimization: old_policy sees the same normalized inputs as at
        # collection time).
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]
        # Gradient checkpointing (Blackwell / memory optimization)
        import torch.utils.checkpoint as cp
        def _actor_ckpt(s):
            return self.actor(s)
        actor_wrapper = lambda s: cp.checkpoint(_actor_ckpt, s, use_reentrant=False)
        token_dim = max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon)) if cfg.use_dct_transform else cfg.action_horizon
        # FPO / spectral flow-matching action representation: full chunk [B, H, D]
        # (not masked to executed steps); DCT transforms the temporal dimension.
        # Paper-original: the loss sees the FULL proposed chunk. Unexecuted
        # tails (early termination / budget cutoff) only shape GAE via
        # executed_n_action_steps; they are never zero-masked here (masking
        # before the DCT leaks zeros into every frequency bin).
        if cfg.use_dct_transform:
            spectral_action_tokens = filter_low_freq(dct(actions, dim=1), token_dim, dim=1)
        else:
            spectral_action_tokens = actions
        n = states.shape[0]
        # ------- MC draws: reuse the rollout's stored (tau, eps) when present ---
        # (common random numbers: l_old/l_new compare identical samples across
        # all epochs, OFF rollout storage). Batches built by hand (tests) fall
        # back to a fresh draw.
        if "tau" in batch and "eps" in batch:
            tau = to_tensor(batch["tau"], dtype=STORAGE_DTYPE, device=states.device)
            tau = tau[0] if tau.dim() == 2 else tau
            eps = to_tensor(batch["eps"], dtype=STORAGE_DTYPE, device=states.device)
            if tuple(eps.shape) != (
                n,
                cfg.num_mc_samples,
                token_dim,
                actions.shape[-1],
            ):
                raise ValueError(
                    f"stored eps shape {tuple(eps.shape)} != "
                    f"({(n, cfg.num_mc_samples, token_dim, actions.shape[-1])})"
                )
        else:
            tau = sample_timesteps(
                cfg.num_mc_samples,
                (cfg.min_timestep, cfg.max_timestep),
                device=states.device,
            )
            eps = sample_noise(
                (n, cfg.num_mc_samples, token_dim if cfg.use_dct_transform else actions.shape[-1], actions.shape[-1]),
                device=states.device,
            )

        # ------- interpolate noisy anchors along the flow -------
        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
        )
        x_t_flat = (
            (1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep
        ).reshape(-1, *flow_shape)  # (1−t)·ε+t·a
        # ------- fuse states once, replicate over MC draws -------
        cond_flat = einx.id(
            "n m d -> (n m) d", einx.id("n d -> n m d", self.conditioner(states), m=num_mc)
        )

        # ------- live velocities vs frozen snapshot velocities -------
        # (tau_bn already holds the flattened MC times; no second expansion.)
        with torch.inference_mode(), autocast_context(states.device):
            old_cond_flat = einx.id(
                "n m d -> (n m) d",
                einx.id("n d -> n m d", self.old_conditioner(states), m=num_mc),
            )
            v_old = self.old_actor(x_t_flat, tau_bn, old_cond_flat).float()
        with autocast_context(states.device):
            v_new = self.actor(x_t_flat, tau_bn, cond_flat).float()
        cond_stats = dict(self.actor.last_cond_stats)
        # ------- CFM losses under target velocities -------
        v_star = self.compute_target_velocity(spectral_action_tokens, eps)
        v_old_r = v_old.reshape(n, num_mc, *flow_shape)
        v_new_r = v_new.reshape(n, num_mc, *flow_shape)
        # ------- ASPO trust-region policy loss (paper Eq.12-13, OFF clamps) ----
        adv_mc = advantages.unsqueeze(1)
        rho_bin = None
        eps_bin = None
        if cfg.use_per_bin_clip:
            # Phase 1 per-bin trust region: mean-of-psi over DCT bins with a
            # two-group epsilon (tight on low-frequency intent bins, loose on
            # high-frequency detail). Same elementwise ratio/ASPO math, no
            # fpo_core change; OFF-faithful scalar path when disabled.
            l_old_bin = compute_cfm_loss(v_old_r, v_star, reduce_mode="bin")
            l_new_bin = compute_cfm_loss(v_new_r, v_star, reduce_mode="bin")
            rho_bin = compute_policy_ratio(l_old_bin, l_new_bin)
            n_bins = rho_bin.shape[-1]
            split = max(1, n_bins // 2)
            eps_bin = torch.full(
                (n_bins,), cfg.epsilon_clip_highfreq, device=rho_bin.device
            )
            eps_bin[:split] = cfg.epsilon_clip_lowfreq
            psi = compute_aspo_surrogate(rho_bin, adv_mc.unsqueeze(-1), eps_bin)
            policy_per_chunk = -psi.mean(dim=(1, 2))
        l_old = compute_cfm_loss(v_old_r, v_star, reduce_mode="sample")
        l_new = compute_cfm_loss(v_new_r, v_star, reduce_mode="sample")
        rho = compute_policy_ratio(l_old, l_new)
        # Log ratio overflow diagnostics (suggestion #2).
        with torch.inference_mode():
            log_rho = (l_old.detach() - l_new).clamp(-10.0, 10.0)  # approximate
            log_rho_max = float(log_rho.abs().max())
            log_rho_clip_frac = float((log_rho.abs() > 10.0).float().mean())
        if rho_bin is None:  # scalar path owns the loss; per-bin logged only
            psi = compute_aspo_surrogate(
                rho, adv_mc, cfg.epsilon_clip
            )
            policy_per_chunk = -psi.mean(dim=1)
        # ------- value loss on raw returns -------
        with autocast_context(states.device):
            v_pred_raw = self.critic(states).float()
        # Fix 3/12: enforce [B] contract; guard broadcasting / shape bugs.
        v_pred = v_pred_raw.squeeze(-1)
        returns_d = returns.to(v_pred.device).reshape(-1)
        if v_pred.shape != returns_d.shape:
            raise ValueError(
                f"value/return shape mismatch: v_pred={tuple(v_pred.shape)}, "
                f"returns_d={tuple(returns_d.shape)}; critic output must be [B] (got {tuple(v_pred_raw.shape)})"
            )
        se = (v_pred - returns_d).pow(2)
        value_error_max = float(se.max().detach())
        value_pred_max = float(v_pred.abs().max().detach())
        return_max = float(returns_d.abs().max().detach())
        if cfg.use_clipped_value_loss and "values" in batch:
            old_values = to_tensor(
                batch["values"], dtype=STORAGE_DTYPE, device=v_pred.device
            )
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.epsilon_clip, cfg.epsilon_clip
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
            rho_mean = float(rho.detach().mean())
            rho_std = float(rho.detach().std(unbiased=False))
            v_star_rms = float(v_star.detach().pow(2).mean().sqrt())
            # PPO standard critic health: 1 - Var(G - V) / Var(G).
            returns_var = returns_d.detach().var(unbiased=False).clamp_min(1e-8)
            explained_var = float(
                1.0 - (returns_d.detach() - v_pred.detach()).var(unbiased=False) / returns_var
            )
            # Phase 0 per-bin trust diagnostics (read-only on the scalar path;
            # live trust region when use_per_bin_clip): same elementwise
            # ratio math, unreduced over bins. Reuses the loss path's rho_bin
            # when present so logged stats match the applied trust region.
            if rho_bin is None:
                l_old_bin = compute_cfm_loss(v_old_r, v_star, reduce_mode="bin")
                l_new_bin = compute_cfm_loss(v_new_r, v_star, reduce_mode="bin")
                rho_bin = compute_policy_ratio(l_old_bin, l_new_bin)
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
                    (rho_bin < 1.0 - cfg.epsilon_clip)
                    | (rho_bin > 1.0 + cfg.epsilon_clip)
                )
                .float()
                .mean()
            )
            # Per-group clip fractions against the eps actually applied
            # (per-bin eps when enabled, scalar eps for both groups otherwise).
            eps_low = cfg.epsilon_clip_lowfreq if eps_bin is not None else cfg.epsilon_clip
            eps_high = cfg.epsilon_clip_highfreq if eps_bin is not None else cfg.epsilon_clip
            rho_bin_low_outside = float(
                (
                    (rho_bin[..., :split] < 1.0 - eps_low)
                    | (rho_bin[..., :split] > 1.0 + eps_low)
                )
                .float()
                .mean()
            )
            rho_bin_high_outside = (
                float(
                    (
                        (rho_bin[..., split:] < 1.0 - eps_high)
                        | (rho_bin[..., split:] > 1.0 + eps_high)
                    )
                    .float()
                    .mean()
                )
                if n_bins > 1
                else rho_bin_low_outside
            )
        return loss, {
            "policy_loss": float(policy_per_chunk.mean().detach()),
            "value_loss": float(value_per_chunk.mean().detach()),
            "explained_var": explained_var,
            "mean_log_ratio": mean_log_ratio,
            "log_rho_max": log_rho_max,
            "log_rho_clip_frac": log_rho_clip_frac,
            "value_error_max": value_error_max,
            "value_pred_max": value_pred_max,
            "return_max": return_max,
            "flow_rho_mean": rho_mean,
            "flow_rho_std": rho_std,
            "flow_rho_bin_range": rho_bin_range,
            "flow_rho_bin_lowfreq": rho_bin_low,
            "flow_rho_bin_highfreq": rho_bin_high,
            "flow_rho_bin_outside_frac": rho_bin_outside,
            "flow_rho_bin_low_outside_frac": rho_bin_low_outside,
            "flow_rho_bin_high_outside_frac": rho_bin_high_outside,
            "flow_v_star_rms": v_star_rms,
            **{f"cond_{k}": float(v.detach()) for k, v in cond_stats.items()},
        }
