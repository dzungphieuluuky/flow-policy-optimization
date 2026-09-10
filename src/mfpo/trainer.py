"""MFPO trainer: FPO++ update assembly with aux/SigReg terms and NaN guards.

Plan task 8. Implements the single-step (or multi-epoch) gradient update for
the MFPO agent, combining four loss terms:

    total = ASPO_surrogate_mean
          + c_value · MSE(value_pred, scaled_targets)
          + lambda_aux · aux_CFM_loss
          + lambda_sigreg · sigreg_loss (Epps-Pulley CF normality test)

AdamW(lr, wd); grad-norm clip; fp32 only.

Tensor dimension conventions (used throughout this module):
    B: batch size (number of chunks)
    N: context length (default 16)
    H: chunk length / horizon (config.chunk_len)
    K: spectral tokens (default 5)
    D: latent dimension Z_dim (default 128)
    Dk: spectral projector output dim (default 64)
    Da: action dimension
    Ds: state dimension
    N_mc: Monte Carlo samples for flow likelihood (config.N_mc)

MuJoCo-only: FPO++ path (ASPO + value + aux CFM + SigReg).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import einx
import torch
import torch.nn as nn

from mfpo import config
from mfpo.collector import ChunkRecord
from mfpo.fpo_core import aspo_surrogate, cfm_loss, policy_ratio, total_update_loss
from mfpo.networks import (
    AuxStateFlowHead,
    ContextTrunk,
    EMAProjection,
    JointFlowTransformer,
    PerBranchFlowLoss,
    PolicyFlowHead,
    SpectralTokens,
    ValueCritic,
    sigreg_losses,
)
from mfpo.utils import safe_float_value


class MFPOAborted(RuntimeError):
    """Raised when NaN/Inf is detected in training losses.

    Attributes:
        message: Human-readable description of which loss term became non-finite.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MFPOTrainer:
    """FPO++ trainer assembling all loss terms and managing gradient routing.

    Attributes:
        trunk: Shared frame encoder (ContextTrunk).
        policy_head: Policy velocity network (PolicyFlowHead).
        aux_head: Auxiliary future-state velocity network (AuxStateFlowHead).
        critic: Value function (ValueCritic).
        ema_proj: Online/target observation projection (EMAProjection).
        spectral: Frozen-stat spectral feature extractor (SpectralTokens).
        cfg: Hyperparameters (MFPOConfig).
        joint_flow: Optional joint flow transformer (JointFlowTransformer).
        optimizer: AdamW optimizer over trainable parameters.
        _step_count: Internal counter for EMA update scheduling.

    Args:
        trunk: Shared frame encoder.
        policy_head: Policy velocity network.
        aux_head: Auxiliary future-state velocity network.
        critic: Value function.
        ema_proj: Online/target observation projection.
        spectral: Frozen-stat spectral feature extractor.
        cfg: Hyperparameters (optional, defaults to config.MFPO_CFG).
        joint_flow: Optional joint flow transformer.
    """

    def __init__(
        self,
        trunk: ContextTrunk,
        policy_head: PolicyFlowHead,
        aux_head: AuxStateFlowHead,
        critic: ValueCritic,
        ema_proj: EMAProjection,
        spectral: SpectralTokens,
        cfg: config.MFPOConfig | None = None,
        joint_flow: JointFlowTransformer | None = None,
    ):
        self.trunk = trunk
        self.policy_head = policy_head
        self.aux_head = aux_head
        self.critic = critic
        self.ema_proj = ema_proj
        self.spectral = spectral
        self.cfg = cfg or config.MFPO_CFG
        self.joint_flow = joint_flow

        # Collect all trainable parameters (trunk + policy head)
        params = list(trunk.parameters()) + list(policy_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.wd,
        )

        self._step_count = 0

    @staticmethod
    def stack_chunk_batch(chunks: Sequence[ChunkRecord], meta: Mapping[str, Any]):
        """Stack chunk tensors; derive env-specific dims from data."""
        contexts = torch.stack([c.context for c in chunks])  # [N, T, obs] -- list stack stays torch (heterogeneous ChunkRecord)
        action_dcts = torch.stack([c.action_dct for c in chunks])  # [N, H, act]
        returns = meta["returns"]  # [N]
        advantages = meta["advantages"]  # [N]
        values = meta["values"]  # [N]
        act_dim = int(action_dcts.shape[-1])  # scalar env action dim
        chunk_len = int(action_dcts.shape[-2])  # scalar executed horizon
        return contexts, action_dcts, returns, advantages, values, act_dim, chunk_len

    def compute_policy_terms(self, action_dcts, z, cond, advantages, N, cfg):
        """Old/new MC log-likelihoods plus ASPO surrogate policy loss."""
        with torch.no_grad():
            l_old = self.compute_log_likelihood(
                action_dcts,
                z,
                cond,
                act_dim=int(action_dcts.shape[-1]),
                chunk_len=int(action_dcts.shape[-2]),
            )  # [N, N_mc]
        eps = torch.randn(N, cfg.N_mc, int(action_dcts.shape[-2]), int(action_dcts.shape[-1]))
        tau_range = cfg.tau_range
        tau = torch.rand(cfg.N_mc) * (tau_range[1] - tau_range[0]) + tau_range[0]
        l_new = self.compute_log_likelihood(
            action_dcts,
            z,
            cond,
            eps=eps,
            tau=tau,
            act_dim=int(action_dcts.shape[-1]),
            chunk_len=int(action_dcts.shape[-2]),
        )
        rho = policy_ratio(l_old, l_new, adv=advantages.unsqueeze(1))
        psi = aspo_surrogate(rho, advantages.unsqueeze(1), cfg.epsilon_clip)
        return psi.mean(), l_old, l_new, tau

    def compute_aux_loss(self, z, cond, tau, chunk_len, cfg):
        """Auxiliary state-space CFM loss over Z_dim latents."""
        aux_target = torch.randn_like(z[:, :chunk_len, :])  # [N, chunk_len, Z_dim]
        aux_eps = torch.randn(z.shape[0], cfg.N_mc, chunk_len, cfg.Z_dim)
        return cfm_loss(
            self.aux_head,
            aux_target,
            aux_eps,
            tau,
            cond[:, :1, :],
        )

    def combine_and_guard_losses(
        self, policy_loss, value_loss, aux_loss, sigreg_loss, cfg, diagnostics_sink
    ):
        """Weight total loss; raise MFPOAborted with sink dump if non-finite."""
        total_loss = total_update_loss(
            policy_loss,
            value_loss,
            aux_loss,
            sigreg_loss,
            c_value=cfg.c_value,
            lambda_aux=cfg.lambda_aux,
            lambda_sigreg=cfg.lambda_sigreg,
        )
        if not torch.isfinite(total_loss):
            diag: dict[str, float] = {
                "policy_loss": safe_float_value(policy_loss),
                "value_loss": safe_float_value(value_loss),
                "aux_loss": safe_float_value(aux_loss),
                "sigreg_loss": safe_float_value(sigreg_loss),
                "clip_fraction": float("nan"),
                "mean_log_ratio": float("nan"),
                "clamp_engagement": float("nan"),
            }
            if diagnostics_sink is not None:
                Path(diagnostics_sink).write_text(json.dumps(diag, indent=2))
            raise MFPOAborted(
                f"Non-finite loss: policy={diag['policy_loss']}, "
                f"value={diag['value_loss']}, aux={diag['aux_loss']}, "
                f"sigreg={diag['sigreg_loss']}"
            )
        return total_loss

    def apply_optimizer_step(self, total_loss):
        """Backward + grad-clip + AdamW step + EMA update."""
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trunk.parameters(), self.cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.policy_head.parameters(),
            self.cfg.grad_clip,
        )
        self.optimizer.step()
        self.ema_proj.ema_update(step=self._step_count)
        self._step_count += 1

    def build_update_diagnostics(
        self, l_old, l_new, policy_loss, value_loss, aux_loss, sigreg_loss, cfg
    ):
        """Assemble the finite diagnostics dict returned by assemble_update."""
        with torch.no_grad():
            log_ratios = (l_old - l_new).mean(dim=1)
            clip_frac = float((log_ratios.abs() > cfg.log_ratio_cap).float().mean())
            mean_lr = float(log_ratios.mean())
            clamp_eng = float(
                (log_ratios.abs() > cfg.log_ratio_cap - 1e-3).float().mean()
            )
        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "aux_loss": float(aux_loss),
            "sigreg_loss": float(sigreg_loss),
            "clip_fraction": clip_frac,
            "mean_log_ratio": mean_lr,
            "clamp_engagement": clamp_eng,
        }

    def assemble_update(
        self,
        chunks: Sequence[ChunkRecord],
        meta: Mapping[str, Any],
        *,
        diagnostics_sink: str | Path | None = None,
    ) -> dict[str, float]:
        """Assemble MuJoCo update (ASPO + value head).

        Args:
            chunks: Sequence of ChunkRecord objects containing context, actions, rewards.
            meta: Dictionary with returns, advantages, values for the update path.
            diagnostics_sink: Optional path to write diagnostic JSON on NaN/Inf.

        Returns:
            Dictionary of loss components and diagnostics.

        Raises:
            MFPOAborted: If any loss component is non-finite.
        """
        N = len(chunks)
        cfg = self.cfg
        contexts, action_dcts, returns, advantages, values, act_dim, chunk_len = self.stack_chunk_batch(chunks, meta)
        z = self.trunk(contexts)  # [N, T, Z]
        cond = self.spectral(z)  # [N, K, 2Z]
        policy_loss, l_old, l_new, tau = self.compute_policy_terms(action_dcts, z, cond, advantages, N, cfg)
        v_pred = self.critic(z)  # [N]
        value_loss = nn.functional.mse_loss(v_pred, returns)
        aux_loss = self.compute_aux_loss(z, cond, tau, chunk_len, cfg)
        sigreg_loss = sigreg_losses(z)
        total_loss = self.combine_and_guard_losses(policy_loss, value_loss, aux_loss, sigreg_loss, cfg, diagnostics_sink)
        self.apply_optimizer_step(total_loss)
        return self.build_update_diagnostics(l_old, l_new, policy_loss, value_loss, aux_loss, sigreg_loss, cfg)

    def interpolate_mc_batch(self, action_dcts, eps, tau, N, cfg):
        """Interpolate x_t = tau*eps + (1-tau)*a and target velocity v*."""
        tau_exp = einx.id("m -> 1 m 1 1", tau)  # [1, N_mc, 1, 1]
        a_exp = einx.id("n h a -> n 1 h a", action_dcts)  # [N, 1, H, act]
        x_t = tau_exp * eps + (1.0 - tau_exp) * a_exp
        x_t_flat = einx.id("n m h a -> (n m) h a", x_t)  # [N*N_mc, H, act]
        v_star = eps - a_exp
        v_star_flat = einx.id("n m h a -> (n m) h a", v_star)  # [N*N_mc, H, act]
        return x_t_flat, v_star_flat

    def compute_log_likelihood(
        self,
        action_dcts: torch.Tensor,
        z: torch.Tensor,
        cond: torch.Tensor,
        *,
        act_dim: int,
        chunk_len: int,
        eps: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-sample flow log-likelihood for each chunk.

        Returns [N, N_mc] tensor of l_i values.
        """
        N = action_dcts.shape[0]
        cfg = self.cfg

        if eps is None:
            eps = torch.randn(N, cfg.N_mc, chunk_len, act_dim)
        if tau is None:
            tau_range = cfg.tau_range
            tau = torch.rand(cfg.N_mc) * (tau_range[1] - tau_range[0]) + tau_range[0]

        # Expand conditioning for MC samples
        cond_mc = einx.id("n k d -> n m k d", cond, m=cfg.N_mc)  # [N, N_mc, K, 2Z]
        cond_flat = einx.id("n m k d -> (n m) k d", cond_mc)  # [N*N_mc, K, 2Z]
        x_t_flat, v_star_flat = self.interpolate_mc_batch(action_dcts, eps, tau, N, cfg)
        # Forward through policy head
        tau_flat = einx.id("m -> n m", tau, n=N).reshape(N * cfg.N_mc)  # [N*N_mc] via einx broadcast
        v_hat = self.policy_head(x_t_flat, tau_flat, cond_flat)

        # Squared error, mean over flow dims, per sample
        se = (v_hat - v_star_flat).pow(2).mean(dim=(-2, -1))  # [N*N_mc]
        l = einx.id("(n m) -> n m", se, n=N, m=cfg.N_mc).clamp(-cfg.cfm_clamp, cfg.cfm_clamp)  # [N, N_mc]
        return l
