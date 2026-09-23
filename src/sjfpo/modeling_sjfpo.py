"""SJFPO policy: flow head + state conditioner + critic.

Batch contract: ``[B, T, obs]`` states, ``[B, H, action_dim]`` action chunks,
frozen GAE advantages/returns from the rollout.

Performance notes
-----------------
* Diagnostics returned by ``forward`` are batched into a single device→host
  transfer (one ``.tolist()``) instead of ~20 ``float()`` syncs per call.
* ``rho_bin`` is computed once per forward, not once per branch.
* Gradient checkpointing actually wraps the actor call, gated on
  ``config.use_gradient_checkpointing``.
* ``compute_predicted_velocity`` is only invoked when the caller requests
  frozen-snapshot velocities via ``return_stats=True`` (they're still needed
  for the CFM comparison; the note above documents the intent).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import einx
import numpy as np
import torch
import torch.utils.checkpoint as cp

from lerobot.policies import PreTrainedPolicy
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import (
    decode_coeffs_to_velocity,
    extract_dct_coeffs,
    sample_freq_weighted_noise,
)
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
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFPOFlowModel,
    StateConditioner,
)
from sjfpo.processor_sjfpo import make_sjfpo_pre_post_processors
from sjfpo.utils import (
    autocast_context,
    resolve_storage_dtype,
    to_device_async,
    to_tensor_1d,
)

STORAGE_DTYPE = resolve_storage_dtype()


class SJFPOAborted(RuntimeError):
    """Non-finite training loss (message names the offending term)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Small helpers (avoid repeating shape arithmetic)
# ---------------------------------------------------------------------------


def _token_dim(cfg: SJFPOConfig) -> int:
    """Number of spectral (or raw) tokens per action chunk."""
    if cfg.use_dct_transform:
        return max(1, cfg.action_horizon - 1) + 1
    return cfg.action_horizon


def _sample_spectral_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw MC noise matching the token layout (DCT or raw)."""
    if cfg.use_dct_transform:
        M_modes = max(1, cfg.action_horizon - 1)
        eps = sample_freq_weighted_noise(
            M=M_modes,
            d=action_dim,
            T=1.0,
            B=n * cfg.num_mc_samples,
            sigma=1.0,
        )
        return eps.reshape(n, cfg.num_mc_samples, M_modes + 1, action_dim).to(device)
    return sample_noise(
        (n, cfg.num_mc_samples, cfg.action_horizon, action_dim), device=device
    )


def _actor_call(actor, x_t, tau, cond, *, use_ckpt: bool):
    """Actor forward, optionally under gradient checkpointing."""
    if use_ckpt:
        return cp.checkpoint(actor, x_t, tau, cond, use_reentrant=False)
    return actor(x_t, tau, cond)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class SJFPOPolicy(PreTrainedPolicy):
    config_class = SJFPOConfig
    name = "sjfpo"

    def __init__(
        self, config: SJFPOConfig, dataset_stats: dict[str, Any] | None = None
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        obs_dim = int(config.robot_state_feature.shape[0])
        action_dim = int(config.action_feature.shape[0])

        self.actor = SJFPOFlowModel(action_dim, hidden_dim=config.hidden_dim)
        self.conditioner = StateConditioner(obs_dim, hidden_dim=config.hidden_dim)
        self.critic = Critic(
            obs_dim, hidden_dim=config.hidden_dim, n_obs_steps=config.n_obs_steps
        )
        self.obs_normalizer = RunningMeanStd(obs_dim)

        self.register_buffer(
            "action_low", torch.full((action_dim,), -1.0), persistent=False
        )
        self.register_buffer(
            "action_high", torch.full((action_dim,), 1.0), persistent=False
        )

        # Frozen behavior snapshot (PPO π_old); refreshed per collection round.
        self.old_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.old_conditioner = (
            copy.deepcopy(self.conditioner).requires_grad_(False).eval()
        )

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        self.preprocessor, self.postprocessor = make_sjfpo_pre_post_processors(self)
        del dataset_stats  # kept for base-class signature compatibility

    # -- module state --------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        self.old_actor.eval()
        self.old_conditioner.eval()
        return self

    def reset(self):
        self._frame_buffer.clear()
        self._action_buffer.clear()

    def set_action_bounds(
        self,
        low: np.ndarray | torch.Tensor,
        high: np.ndarray | torch.Tensor,
    ) -> None:
        self.action_low.copy_(
            to_tensor_1d(low, dtype=STORAGE_DTYPE, device=self.action_low.device)
        )
        self.action_high.copy_(
            to_tensor_1d(high, dtype=STORAGE_DTYPE, device=self.action_high.device)
        )

    def snapshot_behavior_policy(self) -> None:
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.requires_grad_(False)
        self.old_conditioner = copy.deepcopy(self.conditioner)
        self.old_conditioner.requires_grad_(False)

    def get_optim_params(self) -> dict:
        decay, no_decay = [], []
        for name, p in self.actor.named_parameters():
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

    # -- velocity helpers ----------------------------------------------

    @staticmethod
    def compute_target_velocity(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """v* = a − ε broadcast over MC draws: [B, *flow] x [B, m, *flow] → [B, m, *flow]."""
        return a.unsqueeze(1) - eps

    def compute_predicted_velocity(self, x_t, tau, states, old: bool = False):
        conditioner = self.old_conditioner if old else self.conditioner
        net = self.old_actor if old else self.actor
        return net(x_t, tau, conditioner(states))

    # -- observation coercion ------------------------------------------

    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        states = to_tensor(states, dtype=STORAGE_DTYPE)
        if states.dim() == 3:
            return states
        if states.dim() == 2:
            if states.shape[0] == self.config.n_obs_steps:
                return einx.id("t d -> 1 t d", states)
            return einx.id("d -> 1 t d", states, t=self.config.n_obs_steps)
        raise ValueError(
            f"observation.state must be [B, T, obs], [T, obs] or [B, obs]; "
            f"got {tuple(states.shape)}"
        )

    # -- inference -----------------------------------------------------

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Full action chunk ``[B, H, action_dim]``; ``noise=True`` samples, else zeros."""
        noise = kwargs.get("noise", False)
        cfg = self.config
        device = next(self.actor.parameters()).device

        states = self.standardize_obs_tensor(batch[OBS_STATE])
        states = to_device_async(states, device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        token_dim = _token_dim(cfg)
        action_dim = int(self.config.action_feature.shape[0])

        with torch.inference_mode(), autocast_context(device):
            condition = self.conditioner(states)
            shape = (states.shape[0], token_dim, action_dim)
            eps = None if noise else torch.zeros(*shape, device=device)
            num_steps = cfg.num_euler_steps if noise else cfg.num_inference_steps
            x0 = sample_action(
                lambda x, t, c: self.actor(x, t, c),
                shape,
                condition,
                num_inference_steps=num_steps,
                eps=eps,
                device=device,
            ).float()

        if cfg.use_dct_transform:
            tau_grid = torch.linspace(
                0.0, 1.0, cfg.action_horizon, device=x0.device, dtype=torch.float32
            )
            scaled = decode_coeffs_to_velocity(
                x0, tau=tau_grid, T=1.0, K_original=cfg.action_horizon
            )[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)
        else:
            scaled = x0[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)

        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        obs = to_tensor(batch[OBS_STATE], dtype=STORAGE_DTYPE)
        if obs.dim() == 1:
            if not self._frame_buffer:
                for _ in range(self.config.n_obs_steps):
                    self._frame_buffer.append(obs.detach().clone())
            else:
                self._frame_buffer.append(obs.detach())
            window = torch.stack(tuple(self._frame_buffer), dim=0)
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

    # -- rollout -------------------------------------------------------

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop chunks; return a forward-ready batch."""
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device

        def propose(state: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(state, device)}, noise=noisy
            )[0]
            return chunk.cpu()

        def propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(states, device)}, noise=noisy
            )
            return chunk.cpu()

        from sjfpo.mjx_env import mjx_collect_chunks, resolve_env_name, want_mjx

        env_name = resolve_env_name(env)
        if want_mjx():
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
            chunks = collect_chunks(
                env,
                propose,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.action_horizon,
            )

        n_chunks = len(chunks)
        action_dim = int(self.config.action_feature.shape[0])
        token_dim = _token_dim(cfg)

        # Stack once, reuse everywhere.
        lengths = torch.as_tensor(
            [int(c["executed_n_action_steps"]) for c in chunks],
            dtype=torch.long,
            device=device,
        )
        states = to_device_async(torch.stack([c[OBS_STATE] for c in chunks]), device)
        next_states = to_device_async(
            torch.stack([c["next.observation.state"] for c in chunks]), device
        )
        actions = to_device_async(torch.stack([c["action"] for c in chunks]), device)
        terminated = torch.as_tensor(
            [bool(c[DONE]) for c in chunks], dtype=torch.bool, device=device
        )
        truncated = torch.as_tensor(
            [bool(c[TRUNCATED]) for c in chunks], dtype=torch.bool, device=device
        )
        all_rewards = torch.cat(
            [
                c[REWARD].reshape(-1).to(device=device, dtype=torch.float32)
                for c in chunks
            ]
        )

        # Sanity: chunk shape.
        if actions.dim() != 3 or actions.shape[1] != cfg.action_horizon:
            raise ValueError(
                f"chunk actions shape {tuple(actions.shape)} != expected "
                f"[B, H={cfg.action_horizon}, D={action_dim}]"
            )

        with torch.inference_mode():
            states_n = self.preprocessor({OBS_STATE: states})[OBS_STATE]
            next_states_n = self.preprocessor({OBS_STATE: next_states})[OBS_STATE]
            with autocast_context(device):
                values = self.critic(states_n).float()
                v_next_all = self.critic(next_states_n).float()
            next_values = torch.where(terminated, 0.0, v_next_all)

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
            clamp=cfg.advantage_clamp,  # was silently defaulted before
        )
        advantages = advantages.to(STORAGE_DTYPE)
        returns = returns.to(STORAGE_DTYPE)

        is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        tau = sample_timesteps(
            cfg.num_mc_samples, (cfg.min_timestep, cfg.max_timestep), device=device
        )
        eps = _sample_spectral_eps(cfg, n_chunks, action_dim, device)

        # Single shape check against the layout that will actually be used.
        expected_eps_shape = (n_chunks, cfg.num_mc_samples, token_dim, action_dim)
        if tuple(eps.shape) != expected_eps_shape:
            raise ValueError(
                f"CRN eps shape {tuple(eps.shape)} != expected {expected_eps_shape}"
            )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": is_pad,
            "advantages": advantages,
            "returns": returns,
            "values": values.detach(),
            "tau": tau.unsqueeze(0).expand(n_chunks, -1).contiguous(),
            "eps": eps,
        }

    # -- training forward ---------------------------------------------

    def forward(
        self, batch: dict[str, torch.Tensor], reduction: str = "mean"
    ) -> tuple[torch.Tensor, dict | None]:
        """Combined ASPO policy + value loss from a ``generate_rollout`` batch."""
        cfg = self.config
        device = next(self.parameters()).device

        states = to_device_async(
            to_tensor(batch[OBS_STATE], dtype=STORAGE_DTYPE), device
        )
        actions = to_device_async(to_tensor(batch[ACTION], dtype=STORAGE_DTYPE), device)
        advantages = to_tensor(batch["advantages"], dtype=STORAGE_DTYPE).to(device)
        returns = to_tensor(batch["returns"], dtype=STORAGE_DTYPE).to(device)

        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )

        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        M_modes = max(1, cfg.action_horizon - 1)
        token_dim = M_modes + 1 if cfg.use_dct_transform else cfg.action_horizon
        n = states.shape[0]
        action_dim = actions.shape[-1]

        spectral_action_tokens = (
            extract_dct_coeffs(actions, M=M_modes) if cfg.use_dct_transform else actions
        )

        # Reuse stored CRN draws if present, else sample fresh ones.
        if "tau" in batch and "eps" in batch:
            tau = to_tensor(batch["tau"], dtype=STORAGE_DTYPE, device=device)
            tau = tau[0] if tau.dim() == 2 else tau
            eps = to_tensor(batch["eps"], dtype=STORAGE_DTYPE, device=device)
            expected = (n, cfg.num_mc_samples, token_dim, action_dim)
            if tuple(eps.shape) != expected:
                raise ValueError(f"stored eps shape {tuple(eps.shape)} != {expected}")
        else:
            tau = sample_timesteps(
                cfg.num_mc_samples, (cfg.min_timestep, cfg.max_timestep), device=device
            )
            eps = _sample_spectral_eps(cfg, n, action_dim, device)

        # Noisy anchors.
        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
        )
        x_t_flat = (
            (1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep
        ).reshape(-1, *flow_shape)

        # Condition fusion: [n, D] -> [n*m, D].
        cond_flat = einx.id(
            "n m d -> (n m) d",
            einx.id("n d -> n m d", self.conditioner(states), m=num_mc),
        )
        with torch.inference_mode(), autocast_context(device):
            old_cond_flat = einx.id(
                "n m d -> (n m) d",
                einx.id("n d -> n m d", self.old_conditioner(states), m=num_mc),
            )
            v_old = self.old_actor(x_t_flat, tau_bn, old_cond_flat).float()

        with autocast_context(device):
            # Stats return path — no more self.actor.last_cond_stats mutation.
            v_new, cond_stats = (
                _actor_call(
                    self.actor,
                    x_t_flat,
                    tau_bn,
                    cond_flat,
                    use_ckpt=cfg.use_gradient_checkpointing,
                )
                if not cfg.use_gradient_checkpointing
                else (
                    # checkpoint() returns only the output; split the stats
                    # request into a separate non-checkpointed call for logging.
                    _actor_call(self.actor, x_t_flat, tau_bn, cond_flat, use_ckpt=True),
                    None,
                )
            )
            v_new = v_new.float()

        # When checkpointing, re-run just the modulation MLPs to grab stats —
        # cheap relative to attention, and only on steps that log.
        if cond_stats is None:
            with torch.inference_mode():
                _, cond_stats = self.actor(
                    x_t_flat[:1], tau_bn[:1], cond_flat[:1], return_stats=True
                )

        # -- CFM losses under target velocities (fp32 for stability) --
        with torch.autocast(device_type=device.type, enabled=False):
            v_star_f = self.compute_target_velocity(spectral_action_tokens, eps).float()
            v_old_r_f = v_old.reshape(n, num_mc, *flow_shape)
            v_new_r_f = v_new.reshape(n, num_mc, *flow_shape)

            l_old_bin = compute_cfm_loss(v_old_r_f, v_star_f, reduce_mode="bin")
            l_new_bin = compute_cfm_loss(v_new_r_f, v_star_f, reduce_mode="bin")
            rho_bin = compute_policy_ratio(l_old_bin, l_new_bin)  # computed once

            l_old = compute_cfm_loss(v_old_r_f, v_star_f, reduce_mode="sample")
            l_new = compute_cfm_loss(v_new_r_f, v_star_f, reduce_mode="sample")
            rho = compute_policy_ratio(l_old, l_new)

            adv_mc = advantages.float().unsqueeze(1)

            if cfg.use_per_bin_clip:
                n_bins = rho_bin.shape[-1]
                split = max(1, n_bins // 2)
                eps_bin = torch.full(
                    (n_bins,), cfg.epsilon_clip_highfreq, device=device
                )
                eps_bin[:split] = cfg.epsilon_clip_lowfreq
                psi = compute_aspo_surrogate(rho_bin, adv_mc.unsqueeze(-1), eps_bin)
                policy_per_chunk = -psi.mean(dim=(1, 2))
            else:
                psi = compute_aspo_surrogate(rho, adv_mc, cfg.epsilon_clip)
                policy_per_chunk = -psi.mean(dim=1)

        # -- Value loss (Huber) --
        with autocast_context(device):
            v_pred_raw = self.critic(states).float()
        v_pred = v_pred_raw.squeeze(-1)
        returns_d = returns.reshape(-1)
        if v_pred.shape != returns_d.shape:
            raise ValueError(
                f"value/return shape mismatch: v_pred={tuple(v_pred.shape)}, "
                f"returns_d={tuple(returns_d.shape)}"
            )
        se = torch.nn.functional.smooth_l1_loss(
            v_pred, returns_d, reduction="none", beta=1.0
        )
        if cfg.use_clipped_value_loss and "values" in batch:
            old_values = to_tensor(batch["values"], dtype=STORAGE_DTYPE, device=device)
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.epsilon_clip, cfg.epsilon_clip
            )
            clipped_se = torch.nn.functional.smooth_l1_loss(
                v_clipped, returns_d, reduction="none", beta=1.0
            )
            value_per_chunk = torch.maximum(se, clipped_se)
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
                f"Non-finite loss: "
                f"policy_loss={float(policy_per_chunk.mean().detach())}, "
                f"value_loss={float(value_per_chunk.mean().detach())}"
            )

        info = self._compute_info(
            l_old=l_old,
            l_new=l_new,
            rho=rho,
            rho_bin=rho_bin,
            v_star_f=v_star_f,
            v_pred=v_pred,
            returns_d=returns_d,
            policy_per_chunk=policy_per_chunk,
            value_per_chunk=value_per_chunk,
            advantages=advantages,
            cond_stats=cond_stats,
            cfg=cfg,
        )
        return loss, info

    # -- diagnostics (single device->host transfer) --------------------

    @torch.inference_mode()
    def _compute_info(
        self,
        *,
        l_old,
        l_new,
        rho,
        rho_bin,
        v_star_f,
        v_pred,
        returns_d,
        policy_per_chunk,
        value_per_chunk,
        advantages,
        cond_stats,
        cfg,
    ) -> dict[str, float]:
        """All diagnostic reductions in one go, batched into a single sync.

        Every field is a Python float, but the device↔host transfer happens
        exactly once (via a single ``.tolist()`` at the end).
        """
        # -- log-ratio on raw delta (never clamped-then-checked) --
        delta = l_old - l_new
        log_rho_abs = delta.abs()
        log_rho_max = log_rho_abs.max()
        log_rho_clip_frac = (log_rho_abs > cfg.log_ratio_cap).float().mean()

        # -- rho stats --
        rho_mean = rho.mean()
        rho_std = rho.std(correction=0)

        # -- per-bin rho --
        rho_bin_mean = rho_bin.mean(dim=(0, 1))
        n_bins = rho_bin_mean.numel()
        split = max(1, n_bins // 2)
        rho_bin_range = rho_bin_mean.max() - rho_bin_mean.min()
        rho_bin_low = rho_bin_mean[:split].mean()
        rho_bin_high = (
            rho_bin_mean[split:].mean() if n_bins > 1 else rho_bin_mean.mean()
        )
        rho_bin_outside = (
            ((rho_bin < 1.0 - cfg.epsilon_clip) | (rho_bin > 1.0 + cfg.epsilon_clip))
            .float()
            .mean()
        )
        eps_low = cfg.epsilon_clip_lowfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        eps_high = (
            cfg.epsilon_clip_highfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        )
        low_slice = rho_bin[..., :split]
        rho_bin_low_outside = (
            ((low_slice < 1.0 - eps_low) | (low_slice > 1.0 + eps_low)).float().mean()
        )
        if n_bins > 1:
            high_slice = rho_bin[..., split:]
            rho_bin_high_outside = (
                ((high_slice < 1.0 - eps_high) | (high_slice > 1.0 + eps_high))
                .float()
                .mean()
            )
        else:
            rho_bin_high_outside = rho_bin_low_outside

        # -- value-health stats --
        returns_var = returns_d.var(correction=0).clamp_min(1e-8)
        explained_var = 1.0 - (returns_d - v_pred).var(correction=0) / returns_var
        value_error_max = torch.nn.functional.smooth_l1_loss(
            v_pred, returns_d, reduction="none", beta=1.0
        ).max()
        value_pred_max = v_pred.abs().max()
        return_max = returns_d.abs().max()
        v_star_rms = v_star_f.pow(2).mean().sqrt()

        # -- Concrete diagnostics (Step 3) --
        l_old_std = l_old.std(correction=0)
        l_new_std = l_new.std(correction=0)
        corr_denom = l_old_std * l_new_std + 1e-8
        l_old_new_corr = (
            (l_old - l_old.mean()) * (l_new - l_new.mean())
        ).mean() / corr_denom
        # -- pack into one tensor and transfer once --
        cond_keys = list(cond_stats.keys()) if cond_stats else []
        scalars = [
            policy_per_chunk.mean(),
            value_per_chunk.mean(),
            explained_var,
            delta.mean(),
            log_rho_max,
            log_rho_clip_frac,
            value_error_max,
            value_pred_max,
            return_max,
            rho_mean,
            rho_std,
            rho_bin_range,
            rho_bin_low,
            rho_bin_high,
            rho_bin_outside,
            rho_bin_low_outside,
            rho_bin_high_outside,
            v_star_rms,
            l_old.mean(),
            l_new.mean(),
            l_old_std,
            l_new_std,
            l_old_new_corr,
            advantages.abs().mean(),
            returns_d.std(correction=0),
            v_pred.std(correction=0),
        ]
        scalars.extend(
            cond_stats[k].detach()
            if torch.is_tensor(cond_stats[k])
            else torch.tensor(float(cond_stats[k]), device=delta.device)
            for k in cond_keys
        )
        packed = torch.stack([s.detach().reshape(()) for s in scalars]).float().cpu()
        vals = packed.tolist()

        keys = [
            "policy_loss",
            "value_loss",
            "explained_var",
            "mean_log_ratio",
            "log_rho_max",
            "log_rho_clip_frac",
            "value_error_max",
            "value_pred_max",
            "return_max",
            "flow_rho_mean",
            "flow_rho_std",
            "flow_rho_bin_range",
            "flow_rho_bin_lowfreq",
            "flow_rho_bin_highfreq",
            "flow_rho_bin_outside_frac",
            "flow_rho_bin_low_outside_frac",
            "flow_rho_bin_high_outside_frac",
            "flow_v_star_rms",
            "train/l_old_mean",
            "train/l_new_mean",
            "train/l_old_std",
            "train/l_new_std",
            "train/l_old_new_corr",
            "train/adv_mean_abs",
            "train/value_target_std",
            "train/value_pred_std",
        ]
        info = dict(zip(keys, vals))
        for i, k in enumerate(cond_keys):
            info[f"cond_{k}"] = vals[len(keys) + i]
        return info
