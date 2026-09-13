"""SJFPO trainer: FPO++ update assembly with NaN guards.

Plan task 8. Implements the single-step (or multi-epoch) gradient update for
the SJFPO agent, combining two loss terms:

    total = ASPO_surrogate_mean
          + vf_coef · MSE(V(s,a), GAE_returns)   (frozen collection-time
            returns; callers without returns fall back to one-step TD)

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

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import einx
import torch
import torch.nn as nn

from sjfpo import config
from sjfpo.collector import Transition, bootstrap_action_lengths
from sjfpo.fpo_core import compute_aspo_surrogate, compute_policy_ratio
from sjfpo.networks import (
    SJFOPolicyModel,
    SpectralEncoder,
    Critic,
)
from sjfpo.dct_utils import dct, mask_unexecuted_tail
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
        critic: State-action value function (Critic, owns its state encoder).
        spectral: Parameter-free spectral encoder over raw obs (SpectralEncoder).
        obs_normalizer: Running obs stats (RunningMeanStd) or None (no norm).
        cfg: Hyperparameters (SJFPOConfig).
        optimizer: AdamW optimizer over trainable parameters.
        step_count: Internal update counter (checkpoint compat).

    Args:
        policy_head: DiT policy velocity network.
        critic: State-action value function (owns its state encoder).
        spectral: Parameter-free spectral encoder over the raw obs window.
        cfg: Hyperparameters (optional, defaults to config.SJFPO_CFG).
        device: Compute device (default: CUDA if available, else CPU).
        obs_normalizer: Running obs stats module (optional, updated on train batches).
    """

    def __init__(
        self,
        policy_head: SJFOPolicyModel,
        critic: Critic,
        spectral: SpectralEncoder,
        cfg: config.SJFPOConfig | None = None,
        device: torch.device | str | None = None,
        obs_normalizer=None,
    ):
        self.device = torch.device(device) if device is not None else resolve_device()
        self.policy_head = policy_head.to(self.device)
        self.critic = critic.to(self.device)
        self.spectral = spectral.to(self.device)
        self.obs_normalizer = (
            None if obs_normalizer is None else obs_normalizer.to(self.device)
        )
        # Frozen behavior-policy snapshot (PPO π_θold): refreshed once per
        # collection round via snapshot_policy(); l_old always reads this.
        # Never in the optimizer — plain attribute, not a trained submodule.
        self.old_policy_head: SJFOPolicyModel | None = None
        self.cfg = cfg or config.SJFPO_CFG

        # Collect all trainable parameters (policy model + critic, which owns
        # its state encoder). The spectral encoder is parameter-free (raw DCT
        # bins), so the policy head's cond_proj is the only learnable
        # conditioning path. The critic MUST be optimized against its value target.
        params = list(policy_head.parameters()) + list(critic.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.wd,
        )

        self.step_count = 0

    def snapshot_policy(self) -> None:
        """Freeze current weights as the behavior policy for l_old.

        Call once per collection round, before any update on the new batch
        (the driver does this in execute_trainer_update). Later epochs on the
        same batch keep comparing against collection-time weights.
        """
        self.old_policy_head = copy.deepcopy(self.policy_head)

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
        """Old/new MC log-likelihoods plus ASPO surrogate policy loss.

        l_old reads the frozen snapshot (behavior weights); l_new reads the
        live policy. Without a snapshot (unit tests) both read live weights.
        """
        old_policy = (
            self.old_policy_head
            if self.old_policy_head is not None
            else self.policy_head
        )
        n_coeffs = int(actions.shape[-2])  # truncated Ka rows, not full H
        # Single shared plain-MC draw: l_old and l_new must compare old vs new
        # weights on the SAME samples, or rho is pure MC noise (eval decay).
        # Uniform tau + standard-normal eps (no fixed grids: they desync
        # train-time t from the Euler query points used at inference).
        num_mc = cfg.num_mc_samples
        act_dim = int(actions.shape[-1])
        tau_range = cfg.tau_range
        tau = (
            torch.rand(num_mc, device=self.device)
            * (tau_range[1] - tau_range[0])
            + tau_range[0]
        )  # [num_mc_samples]
        eps = torch.randn(
            N, num_mc, n_coeffs, act_dim, device=self.device
        )  # [N, num_mc_samples, Ka, act]
        with torch.inference_mode():
            l_old = self.compute_log_likelihood(
                actions,
                cond,
                states,
                eps=eps,
                tau=tau,
                act_dim=int(actions.shape[-1]),
                n_coeffs=n_coeffs,
                net=old_policy,
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
        rho = compute_policy_ratio(l_old, l_new, adv=advantages.unsqueeze(1))
        psi = compute_aspo_surrogate(rho, advantages.unsqueeze(1), cfg.clip_range)
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
        total_loss = policy_loss + cfg.vf_coef * value_loss
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
            meta: Dictionary with advantages for the update path, plus values
                (v_old for PPO-style clipping) and returns (frozen GAE value
                targets; without them the critic falls back to one-step TD).
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
        lengths = [int(c["n_action_steps"]) for c in chunks]
        a_exec = mask_unexecuted_tail(actions, lengths, dim=1)  # [N, H, Da]
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
            # Critic path: V(s,a) from raw states (owned critic state encoder).
            v_pred = self.critic(
                states, a_exec, lengths
            )  # [N] masked-mean over executed rows only
            if "returns" in meta:  # frozen GAE returns (PPO semi-gradient target)
                td_targets = meta["returns"].to(dev)  # [N]
            else:  # callers without returns (tests): one-step TD target
                # Bootstrap on executed-steps next-actions (same mask rule as values).
                next_lengths = bootstrap_action_lengths(chunks, chunk_size)
                a_next_exec = mask_unexecuted_tail(
                    next_actions, next_lengths, dim=1
                )  # [N, H, Da]
                with torch.inference_mode():
                    v_next = self.critic(
                        next_states, a_next_exec, next_lengths
                    )  # [N] fixed bootstrap
                td_targets = self.compute_critic_targets(chunks, v_next, cfg.gamma)
            if "values" in meta:  # v_old at collection: PPO-style clipped value loss
                old_values = meta["values"].to(dev)  # [N]
                v_clipped = old_values + (v_pred - old_values).clamp(
                    -cfg.clip_range_vf, cfg.clip_range_vf
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
        net=None,
    ) -> torch.Tensor:
        """Compute per-sample flow log-likelihood for each chunk.

        Returns [N, num_mc_samples] tensor of l_i values.
        net defaults to the live policy head (l_new); pass the frozen
        snapshot for l_old.
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
        model = self.policy_head if net is None else net
        v_hat = model(x_t_flat, tau_flat, cond_flat, state_flat)

        # Squared error, mean over flow dims, per sample
        se = (v_hat - v_star_flat).pow(2).mean(dim=(-2, -1))  # [N*num_mc_samples]
        l = einx.id("(n m) -> n m", se, n=N, m=cfg.num_mc_samples).clamp(
            -cfg.cfm_clamp, cfg.cfm_clamp
        )  # [N, num_mc_samples]
        return l
