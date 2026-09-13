"""SJFPO trainer: FPO++ update assembly with NaN guards.

Plan task 8. Implements the single-step (or multi-epoch) gradient update for
the SJFPO agent, combining two loss terms:

    total = ASPO_surrogate_mean
          + vf_coef · MSE(V(s,a), TD_target)   (V(s,a) bootstraps V(s',a'))

The actor predicts action velocities ONLY (DiT, DCT space); there is no
next-state prediction. AdamW(lr, wd); grad-norm clip; fp32 only.

Tensor dimension conventions (used throughout this module):
    B: batch size (number of chunks)
    N: batch size alias (number of chunks)
    T: state window length (config.n_obs_steps)
    H: predicted action-chunk length (config.chunk_size)
    K: spectral tokens (default 5)
    D: latent dimension hidden_size (default 128)
    Da: action dimension
    Ds: state dimension
    num_mc_samples: Monte Carlo samples for flow likelihood (config.num_mc_samples)

MuJoCo-only: FPO++ path (ASPO + V(s,a) critic).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import einx
import torch
import torch.nn as nn

from sjfpo import config
from sjfpo.collector import Transition
from sjfpo.fpo_core import aspo_surrogate, policy_ratio, total_update_loss
from sjfpo.networks import (
    StateEncoder,
    SJFOPolicyModel,
    SpectralEncoder,
    Critic,
)
from sjfpo.dct_utils import dct, idct
from sjfpo.utils import is_bf16_available, resolve_device, safe_float_value


class SJFPOAborted(RuntimeError):
    """Raised when NaN/Inf is detected in training losses.

    Attributes:
        message: Human-readable description of which loss term became non-finite.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SJFPOTrainer:
    """FPO++ trainer assembling all loss terms and managing gradient routing.

    Attributes:
        policy_head: DiT policy velocity network (SJFOPolicyModel, actions only).
        critic: State-action value function (Critic, cross-attention fusion).
        spectral: Parameter-free spectral encoder over raw obs (SpectralEncoder).
        critic_state_encoder: Critic frame encoder (StateEncoder), trainable.
        obs_normalizer: Running obs stats (RunningMeanStd) or None (no norm).
        cfg: Hyperparameters (SJFPOConfig).
        optimizer: AdamW optimizer over trainable parameters.
        step_count: Internal update counter (checkpoint compat).

    Args:
        policy_head: DiT policy velocity network.
        critic: State-action value function.
        spectral: Parameter-free spectral encoder over the raw obs window.
        critic_state_encoder: Critic frame encoder.
        cfg: Hyperparameters (optional, defaults to config.SJFPO_CFG).
        device: Compute device (default: CUDA if available, else CPU).
        obs_normalizer: Running obs stats module (optional, updated on train batches).
    """

    def __init__(
        self,
        policy_head: SJFOPolicyModel,
        critic: Critic,
        spectral: SpectralEncoder,
        critic_state_encoder: StateEncoder,
        cfg: config.SJFPOConfig | None = None,
        device: torch.device | str | None = None,
        obs_normalizer=None,
    ):
        self.device = torch.device(device) if device is not None else resolve_device()
        self.policy_head = policy_head.to(self.device)
        self.critic = critic.to(self.device)
        self.spectral = spectral.to(self.device)
        self.critic_state_encoder = critic_state_encoder.to(self.device)
        self.obs_normalizer = (
            None if obs_normalizer is None else obs_normalizer.to(self.device)
        )
        self.cfg = cfg or config.SJFPO_CFG

        # Collect all trainable parameters (policy model + critic + critic
        # trunk). The spectral encoder is parameter-free (raw DCT bins), so
        # the policy head's cond_proj is the only learnable conditioning path.
        # The critic MUST be optimized against its TD bootstrap target.
        params = (
            list(policy_head.parameters())
            + list(critic.parameters())
            + list(critic_state_encoder.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.wd,
        )

        self.step_count = 0

    @staticmethod
    def stack_chunk_batch(chunks: Sequence[Transition], meta: Mapping[str, Any]):
        """Stack transition tensors; derive env-specific dims from data."""
        states = torch.stack([c["observation.state"] for c in chunks])  # [N, T, obs]
        actions = torch.stack([c["action"] for c in chunks])  # [N, H, act]
        next_states = torch.stack(
            [c["next.observation.state"] for c in chunks]
        )  # [N, T, obs]
        next_actions = torch.stack([c["next.action"] for c in chunks])  # [N, H, act]
        advantages = meta["advantages"]  # [N]
        act_dim = int(actions.shape[-1])  # scalar env action dim
        chunk_size = int(actions.shape[-2])  # scalar predicted horizon H
        return (
            states,
            actions,
            next_states,
            next_actions,
            advantages,
            act_dim,
            chunk_size,
        )

    def compute_policy_terms(self, actions, cond, states, advantages, N, cfg):
        """Old/new MC log-likelihoods plus ASPO surrogate policy loss."""
        n_coeffs = int(actions.shape[-2])  # truncated Ka rows, not full H
        # Single shared (eps, tau) draw: l_old and l_new must compare old vs
        # new weights on the SAME samples, or rho is pure MC noise (eval decay).
        # tau: fixed uniform quadrature nodes (no 1-D sampling noise, no clumping).
        # eps: antithetic pairs {e, -e} halving Gaussian variance at equal cost.
        num_mc = cfg.num_mc_samples
        act_dim = int(actions.shape[-1])
        tau_range = cfg.tau_range
        tau = torch.linspace(
            tau_range[0], tau_range[1], num_mc, device=self.device
        )  # [num_mc_samples]
        half = num_mc // 2
        eps_half = torch.randn(N, half, n_coeffs, act_dim, device=self.device)
        eps = torch.cat([eps_half, -eps_half], dim=1)  # [N, 2*half, Ka, act]
        if eps.shape[1] < num_mc:  # odd num_mc_samples: one extra unpaired draw
            eps = torch.cat(
                [eps, torch.randn(N, 1, n_coeffs, act_dim, device=self.device)], dim=1
            )
        with torch.inference_mode():
            l_old = self.compute_log_likelihood(
                actions,
                cond,
                states,
                eps=eps,
                tau=tau,
                act_dim=int(actions.shape[-1]),
                n_coeffs=n_coeffs,
            )  # [N, num_mc_samples]
        l_new = self.compute_log_likelihood(
            actions,
            cond,
            states,
            eps=eps,
            tau=tau,
            act_dim=int(actions.shape[-1]),
            n_coeffs=n_coeffs,
        )
        rho = policy_ratio(l_old, l_new, adv=advantages.unsqueeze(1))
        psi = aspo_surrogate(rho, advantages.unsqueeze(1), cfg.clip_range)
        # Maximize the ASPO surrogate → minimize its negation (PPO sign convention).
        return -psi.mean(), l_old, l_new, tau

    def compute_critic_targets(
        self, chunks: Sequence[Transition], v_next: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Chunk-level TD targets: R_k + gamma^{L_k} * V(s',a') (terminated: R_k).

        R_k intradiscounts only executed rewards; bootstrap uses the stored
        behavior next-action, so V(s,a) bootstraps V(next state, next action).

        Args:
            chunks: Transition dicts in batch order.
            v_next: [N] current-critic values at (next_state, next_action).
            gamma: Discount factor.

        Returns:
            TD targets [N] fp32 (caller detaches v_next before passing).
        """
        targets = []
        for c, vn in zip(chunks, v_next):
            rew = c["next.reward"].to(torch.float32)  # [L_k] executed rewards
            discounts = gamma ** torch.arange(rew.numel(), dtype=torch.float32)
            r_chunk = float((discounts * rew).sum())  # scalar intradiscounted R_k
            if bool(c["next.done"]):
                targets.append(r_chunk)  # terminated: no bootstrap
            else:
                length = int(c["n_action_steps"])  # executed steps L_k
                targets.append(r_chunk + (gamma**length) * float(vn))
        return torch.tensor(targets, dtype=torch.float32, device=self.device)  # [N]

    def combine_and_guard_losses(
        self, policy_loss, value_loss, cfg, diagnostics_sink
    ):
        """Weight total loss; raise SJFPOAborted with sink dump if non-finite."""
        total_loss = total_update_loss(
            policy_loss,
            value_loss,
            vf_coef=cfg.vf_coef,
        )
        if not torch.isfinite(total_loss):
            diag: dict[str, float] = {
                "policy_loss": safe_float_value(policy_loss),
                "value_loss": safe_float_value(value_loss),
                "clip_fraction": float("nan"),
                "mean_log_ratio": float("nan"),
                "clamp_engagement": float("nan"),
            }
            if diagnostics_sink is not None:
                Path(diagnostics_sink).write_text(json.dumps(diag, indent=2))
            raise SJFPOAborted(
                f"Non-finite loss: policy={diag['policy_loss']}, "
                f"value={diag['value_loss']}"
            )
        return total_loss

    def apply_optimizer_step(self, total_loss):
        """Backward + global grad-clip + AdamW step."""
        self.optimizer.zero_grad()
        total_loss.backward()
        all_params = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(all_params, self.cfg.grad_clip)
        self.optimizer.step()
        self.step_count += 1

    def build_update_diagnostics(
        self, l_old, l_new, policy_loss, value_loss, cfg
    ):
        """Assemble the finite diagnostics dict returned by assemble_update."""
        with torch.inference_mode():
            log_ratios = (l_old - l_new).mean(dim=1)
            clip_frac = float((log_ratios.abs() > cfg.log_ratio_cap).float().mean())
            mean_lr = float(log_ratios.mean())
            clamp_eng = float(
                (log_ratios.abs() > cfg.log_ratio_cap - 1e-3).float().mean()
            )
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "clip_fraction": clip_frac,
            "mean_log_ratio": mean_lr,
            "clamp_engagement": clamp_eng,
        }

    def assemble_update(
        self,
        chunks: Sequence[Transition],
        meta: Mapping[str, Any],
        *,
        diagnostics_sink: str | Path | None = None,
    ) -> dict[str, float]:
        """Assemble MuJoCo update (ASPO + V(s,a) critic).

        Args:
            chunks: Sequence of Transition dicts (LeRobot keys) with fixed
                shapes ([T, obs] states, [H, act] actions, [H, act] next actions).
            meta: Dictionary with advantages for the update path.
            diagnostics_sink: Optional path to write diagnostic JSON on NaN/Inf.

        Returns:
            Dictionary of loss components and diagnostics.

        Raises:
            SJFPOAborted: If any loss component is non-finite.
            ValueError: If stacked action/state shapes disagree with config.
        """
        N = len(chunks)
        cfg = self.cfg
        states, actions, next_states, next_actions, advantages, act_dim, chunk_size = (
            self.stack_chunk_batch(chunks, meta)
        )
        dev = self.device
        states = states.to(dev)  # [N, T, obs]
        actions = actions.to(dev)  # [N, H, act]
        next_states = next_states.to(dev)  # [N, T, obs]
        next_actions = next_actions.to(dev)  # [N, H, act]
        advantages = advantages.to(dev)  # [N]
        if self.obs_normalizer is not None:
            # Running stats update here (train batches only); rollout and eval
            # normalize frozen through the shared module.
            self.obs_normalizer.update(torch.cat([states, next_states]))
            states = self.obs_normalizer(states)
            next_states = self.obs_normalizer(next_states)
        if chunk_size != cfg.chunk_size:
            raise ValueError(
                f"stacked action horizon {chunk_size} != "
                f"config.chunk_size {cfg.chunk_size}"
            )
        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"stacked state length {states.shape[-2]} != "
                f"config.n_obs_steps {cfg.n_obs_steps}"
            )
        # Fit only executed steps: zero-pad unexecuted tail rows (terminated /
        # truncated / budget-cut) in real space, back to DCT for the policy.
        # Full-length chunks get an all-ones mask (exact no-op).
        lengths = torch.tensor(
            [int(c["n_action_steps"]) for c in chunks], device=dev
        )  # [N]
        exec_mask = (
            torch.arange(chunk_size, device=dev).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(actions.dtype).unsqueeze(-1)  # [N, H, 1]
        a_exec = idct(actions, dim=1) * exec_mask  # [N, H, Da]
            # Policy path: spectral tokens straight from the raw obs window.
        # Autocast to bf16 only where the GPU supports it (else fp32 no-op).
        use_bf16 = dev.type == "cuda" and is_bf16_available()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            cond = self.spectral(states)  # [N, K, obs] raw DCT bins
            # Denoise in truncated coefficient space: keep the top-p low-frequency
            # action bins (same top_p as the spectral tokens); the critic keeps
            # the full-H chunk.
            n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.chunk_size))
            action_coeffs = dct(a_exec, dim=1)[:, :n_coeffs, :]  # [N, Ka, act]
            policy_loss, l_old, l_new, tau = self.compute_policy_terms(
                action_coeffs, cond, states, advantages, N, cfg
            )
            # Critic path: critic trunk encodes states for V(s,a) (decoupled).
            z_critic = self.critic_state_encoder(states)  # [N, T, Z]
            z_critic_next = self.critic_state_encoder(
                next_states
            )  # [N, T, Z] for TD bootstrap
            # Standard critic recursion: V(s,a) bootstraps V(s',a') under inference_mode.
            # The critic judges clean real-space actions: IDCT the stored DCT
            # coefficients (same pad+IDCT the rollout path applies to model output).
            v_pred = self.critic(z_critic, a_exec)  # [N] executed-steps actions
            with torch.inference_mode():
                v_next = self.critic(
                    z_critic_next, idct(next_actions, dim=1)
                )  # [N] fixed bootstrap
            td_targets = self.compute_critic_targets(chunks, v_next, cfg.gamma)
            if "values" in meta:  # v_old at collection: PPO-style clipped value loss
                old_values = meta["values"].to(dev)  # [N]
                v_clipped = old_values + (v_pred - old_values).clamp(
                    -cfg.clip_range, cfg.clip_range
                )
                value_loss = torch.max(
                    (v_pred - td_targets).pow(2), (v_clipped - td_targets).pow(2)
                ).mean()
            else:  # callers without old values (tests): plain TD regression
                value_loss = nn.functional.mse_loss(v_pred, td_targets)
            total_loss = self.combine_and_guard_losses(
                policy_loss, value_loss, cfg, diagnostics_sink
            )
        self.apply_optimizer_step(total_loss)
        return self.build_update_diagnostics(
            l_old, l_new, policy_loss, value_loss, cfg
        )

    def interpolate_mc_batch(self, actions, eps, tau, N, cfg):
        """Interpolate x_t = tau*eps + (1-tau)*a and target velocity v*."""
        tau_exp = einx.id("m -> 1 m 1 1", tau)  # [1, num_mc_samples, 1, 1]
        a_exp = einx.id("n h a -> n 1 h a", actions)  # [N, 1, H, act]
        x_t = tau_exp * eps + (1.0 - tau_exp) * a_exp
        x_t_flat = einx.id("n m h a -> (n m) h a", x_t)  # [N*num_mc_samples, H, act]
        v_star = eps - a_exp
        v_star_flat = einx.id(
            "n m h a -> (n m) h a", v_star
        )  # [N*num_mc_samples, H, act]
        return x_t_flat, v_star_flat

    def compute_log_likelihood(
        self,
        actions: torch.Tensor,
        cond: torch.Tensor,
        states: torch.Tensor,
        *,
        act_dim: int,
        n_coeffs: int,
        eps: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-sample flow log-likelihood for each chunk.

        Returns [N, num_mc_samples] tensor of l_i values.
        """
        N = actions.shape[0]
        cfg = self.cfg

        if eps is None:
            eps = torch.randn(
                N, cfg.num_mc_samples, n_coeffs, act_dim, device=self.device
            )
        if tau is None:
            tau_range = cfg.tau_range
            tau = (
                torch.rand(cfg.num_mc_samples, device=self.device)
                * (tau_range[1] - tau_range[0])
                + tau_range[0]
            )

            # Expand conditioning for MC samples
        cond_mc = einx.id(
            "n k d -> n m k d", cond, m=cfg.num_mc_samples
        )  # [N, num_mc_samples, K, obs]
        cond_flat = einx.id(
            "n m k d -> (n m) k d", cond_mc
        )  # [N*num_mc_samples, K, obs]
        state_mc = einx.id(
            "n t d -> n m t d", states, m=cfg.num_mc_samples
        )  # [N, num_mc_samples, T, obs]
        state_flat = einx.id(
            "n m t d -> (n m) t d", state_mc
        )  # [N*num_mc_samples, T, obs]
        x_t_flat, v_star_flat = self.interpolate_mc_batch(actions, eps, tau, N, cfg)
        # Forward through policy model
        tau_flat = einx.id("m -> n m", tau, n=N).reshape(
            N * cfg.num_mc_samples
        )  # [N*num_mc_samples] via einx broadcast
        v_hat = self.policy_head(x_t_flat, tau_flat, cond_flat, state_flat)

        # Squared error, mean over flow dims, per sample
        se = (v_hat - v_star_flat).pow(2).mean(dim=(-2, -1))  # [N*num_mc_samples]
        l = einx.id("(n m) -> n m", se, n=N, m=cfg.num_mc_samples).clamp(
            -cfg.cfm_clamp, cfg.cfm_clamp
        )  # [N, num_mc_samples]
        return l
