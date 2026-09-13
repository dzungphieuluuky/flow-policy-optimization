"""Tests for SJFPO trainer update step (plan task 8 — RED phase).

Covers:
  (a) assemble_update returns finite diagnostics with correct keys
  (b) gradient routing with zeroed loss terms
  (c) 3 optimizer steps decrease or maintain finite losses
  (d) NaN injection raises SJFPOAborted with diagnostics sink
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.config import SJFPO_CFG, SJFPOConfig
from sjfpo.networks import (
    StateEncoder,
    SJFOPolicyModel,
    SpectralEncoder,
    Critic,
)
from sjfpo.collector import Transition
from sjfpo.dct_utils import idct
from sjfpo.trainer import SJFPOTrainer, SJFPOAborted

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 17
ACT_DIM = 4


def make_trainer(**overrides) -> SJFPOTrainer:
    """Construct a trainer with small dims for fast tests."""
    cfg = SJFPOConfig(**overrides) if overrides else SJFPOConfig()
    # SpectralEncoder outputs [B, K, obs] raw DCT bins straight from the obs window.
    policy_head = SJFOPolicyModel(ACT_DIM, OBS_DIM, hidden_size=cfg.hidden_size)
    critic = Critic(hidden_size=cfg.hidden_size, act_dim=ACT_DIM)
    spectral = SpectralEncoder(OBS_DIM, n_obs_steps=cfg.n_obs_steps)
    critic_state_encoder = StateEncoder(
        OBS_DIM, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
    )
    return SJFPOTrainer(
        policy_head=policy_head,
        critic=critic,
        spectral=spectral,
        critic_state_encoder=critic_state_encoder,
        cfg=cfg,
    )


def make_chunks(n: int = 128) -> tuple[list[Transition], dict]:
    """Create synthetic transitions with pre-computed advantages."""
    cfg = SJFPOConfig()
    t_len, h_len, n_steps = cfg.n_obs_steps, cfg.chunk_size, 8
    chunks: list[Transition] = []
    for _ in range(n):
        chunks.append(
            {
                "observation.state": torch.randn(t_len, OBS_DIM),
                "action": torch.randn(h_len, ACT_DIM) * 0.5,
                "next.action": torch.randn(h_len, ACT_DIM) * 0.5,
                "next.observation.state": torch.randn(t_len, OBS_DIM),
                "next.reward": torch.randn(n_steps).abs() * 0.1,
                "next.done": torch.tensor(False),
                "next.truncated": torch.tensor(False),
                "n_action_steps": torch.tensor(n_steps),
                "clip_count": torch.tensor(0),
            }
        )
    advantages = torch.randn(n) * 0.1
    meta = {"advantages": advantages}
    return chunks, meta


def expected_diag_keys() -> set[str]:
    return {
        "policy_loss",
        "value_loss",
        "clip_fraction",
        "mean_log_ratio",
        "clamp_engagement",
    }

    # ---------------------------------------------------------------------------
    # (a) assemble_update returns finite diagnostics with correct keys
    # ---------------------------------------------------------------------------


class TestAssembleUpdate:
    def test_returns_correct_keys_and_finite(self):
        trainer = make_trainer()
        chunks, meta = make_chunks(128)
        diag = trainer.assemble_update(chunks, meta)

        assert set(diag.keys()) == expected_diag_keys()
        for k, v in diag.items():
            assert isinstance(v, float), f"diagnostics['{k}'] is not float: {type(v)}"
            assert math.isfinite(v), f"diagnostics['{k}'] is not finite: {v}"

    def test_deterministic_same_input(self):
        """Two identical trainers on same input produce same diagnostics."""
        chunks, meta = make_chunks(64)
        # Seed before each trainer init + before each assemble_update call
        # (assemble_update uses torch.randn for eps/tau internally)
        torch.manual_seed(0)
        t1 = make_trainer()
        torch.manual_seed(0)
        d1 = t1.assemble_update(chunks, meta)

        torch.manual_seed(0)
        t2 = make_trainer()
        torch.manual_seed(0)
        d2 = t2.assemble_update(chunks, meta)

        for k in d1:
            assert d1[k] == d2[k], f"diagnostics['{k}'] not deterministic"

            # ---------------------------------------------------------------------------
            # (b) gradient routing with zeroed loss terms
            # ---------------------------------------------------------------------------


class TestGradientRouting:
    def test_aspo_only_reaches_policy_head_and_spectral(self):
        """Policy head + spectral projector receive grads through the surrogate path."""
        trainer = make_trainer()
        chunks, meta = make_chunks(32)

        # DiT adaLN-Zero cold start: gates/output leave zero over the first
        # updates, so deep grads are exactly 0.0 in fp32 until the head
        # awakens. Warm up (cap 10) until time_mlp sees grads, then probe.
        for _ in range(10):
            trainer.assemble_update(chunks, meta)
            if trainer.policy_head.time_mlp.gate.weight.grad.abs().sum() > 0:
                break
        trainer.optimizer.zero_grad()
        diag = trainer.assemble_update(chunks, meta)
        assert math.isfinite(diag["policy_loss"])

        # Policy head must have gradients
        for name, p in trainer.policy_head.named_parameters():
            assert p.grad is not None, f"policy_head.{name} has no grad"
            assert p.grad.abs().sum() > 0, f"policy_head.{name} grad is zero"

            # Spectral encoder is parameter-free (raw DCT bins): nothing to check.
        assert list(trainer.spectral.parameters()) == []

            # Critic also receives grads through total_loss.
        for name, p in trainer.critic.named_parameters():
            assert p.grad is not None, f"critic.{name} has no grad"

    def test_critic_only_reaches_critic(self):
        """Critic TD gradients reach the critic but NOT the policy head."""
        trainer = make_trainer()
        chunks, meta = make_chunks(8)
        dev = trainer.device
        states, actions, _, _, _, _, _ = trainer.stack_chunk_batch(chunks, meta)
        states, actions = states.to(dev), actions.to(dev)
        trainer.optimizer.zero_grad()
        with torch.no_grad():
            z = trainer.critic_state_encoder(states)
        v_pred = trainer.critic(z, actions)
        target = torch.zeros_like(v_pred) + 5.0
        loss = torch.nn.functional.mse_loss(v_pred, target)
        loss.backward()
        for name, p in trainer.critic.named_parameters():
            assert p.grad is not None, f"critic.{name} has no grad"
            assert p.grad.abs().sum() > 0, f"critic.{name} grad is zero"
        for name, p in trainer.policy_head.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() == 0, (
                    f"policy_head.{name} has nonzero grad from critic loss"
                )

                # ---------------------------------------------------------------------------
                # (c) 3 optimizer steps — all losses stay finite
                # ---------------------------------------------------------------------------


class TestOptimizerSteps:
    def test_losses_finite_after_steps(self):
        trainer = make_trainer()
        chunks, meta = make_chunks(128)
        for _ in range(3):
            diag = trainer.assemble_update(chunks, meta)
            for k in ("policy_loss", "value_loss"):
                assert math.isfinite(diag[k]), f"{k} not finite: {diag[k]}"

                # ---------------------------------------------------------------------------
                # (d) NaN injection raises SJFPOAborted with diagnostics written
                # ---------------------------------------------------------------------------


class TestNaNInjection:
    def test_nan_loss_raises_sjfpo_aborted(self):
        trainer = make_trainer()
        chunks, meta = make_chunks(32)

        # Inject NaN into chunk rewards to poison the critic TD target.
        for c in chunks:
            c["next.reward"] = torch.full_like(c["next.reward"], float("nan"))

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = Path(tmpdir) / "diagnostics.json"

            with pytest.raises(SJFPOAborted):
                trainer.assemble_update(chunks, meta, diagnostics_sink=sink)

                # Diagnostics should have been written before the abort
            assert sink.exists(), "diagnostics sink not written before SJFPOAborted"
            data = json.loads(sink.read_text())
            assert "policy_loss" in data

            # ---------------------------------------------------------------------------
            # (f) critic parameters are optimized (S-critic regression)
            # ---------------------------------------------------------------------------


class TestCriticOptimized:
    def test_critic_params_in_optimizer(self):
        """Every critic parameter must be stepped by the optimizer."""
        trainer = make_trainer()
        opt_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        for name, p in trainer.critic.named_parameters():
            assert id(p) in opt_ids, f"critic.{name} not in optimizer"

    def test_critic_moves_after_update(self):
        """Nonzero TD error must move critic weights after one update."""
        trainer = make_trainer()
        before = {n: p.clone() for n, p in trainer.critic.named_parameters()}
        chunks, meta = make_chunks(32)
        for c in chunks:
            c["next.reward"] = c["next.reward"] + 5.0  # force nonzero TD error
        trainer.assemble_update(chunks, meta)
        moved = any(
            not torch.equal(p, before[n]) for n, p in trainer.critic.named_parameters()
        )
        assert moved, "critic parameters did not move after update"

        # ---------------------------------------------------------------------------
        # (g) spectral projector parameters are optimized (S-projector regression)
        # ---------------------------------------------------------------------------


class TestSpectralParameterFree:
    def test_spectral_has_no_params(self):
        """Raw-DCT encoder carries no parameters and needs no optimizer slot."""
        trainer = make_trainer()
        assert list(trainer.spectral.parameters()) == []
        opt_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        head_ids = {id(p) for p in trainer.policy_head.parameters()}
        assert head_ids <= opt_ids, "policy head (incl. cond_proj) not optimized"

    def test_cond_proj_moves_after_update(self):
        """Cond-path loss must move the policy head's cond_proj after updates."""
        trainer = make_trainer()
        before = trainer.policy_head.cond_proj.weight.clone()
        chunks, meta = make_chunks(32)
        # Same adaLN-Zero cold start as above: cond_proj only moves
        # once conditioning grads reach it (cap 10 updates, else real bug).
        moved = False
        for _ in range(10):
            trainer.assemble_update(chunks, meta)
            moved = not torch.equal(trainer.policy_head.cond_proj.weight, before)
            if moved:
                break
        assert moved, "cond_proj did not move after update"

        # ---------------------------------------------------------------------------
        # (h) critic trunk gradients (SigReg removed; trunk trains via TD only)
        # ---------------------------------------------------------------------------


class TestCriticTrunkGrads:
    def test_trunk_receives_grads(self):
        """Policy head (incl. cond_proj) and critic trunk must get grads."""
        trainer = make_trainer()
        chunks, meta = make_chunks(16)
        trainer.assemble_update(chunks, meta)
        assert trainer.policy_head.cond_proj.weight.grad is not None, (
            "cond_proj missing grads"
        )
        critic_grads = [
            p.grad for _, p in trainer.critic_state_encoder.named_parameters()
        ]
        assert all(g is not None for g in critic_grads), (
            "critic trunk params missing grads"
        )


class TestSharedMCSamples:
    def test_identical_weights_give_unit_ratios(self):
        """l_old/l_new share one draw: identical weights must give rho == 1."""
        from sjfpo.fpo_core import policy_ratio

        torch.manual_seed(0)
        trainer = make_trainer()
        cfg = trainer.cfg
        n = 4
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.chunk_size))
        actions = torch.randn(n, n_coeffs, ACT_DIM)
        cond = torch.randn(n, 5, OBS_DIM)
        states = torch.randn(n, cfg.n_obs_steps, OBS_DIM)
        advantages = torch.randn(n) * 0.1
        _, l_old, l_new, _ = trainer.compute_policy_terms(
            actions, cond, states, advantages, n, cfg
        )
        assert torch.equal(l_old, l_new)
        rho = policy_ratio(l_old, l_new, adv=advantages.unsqueeze(1))
        assert torch.equal(rho, torch.ones_like(rho))

        # ---------------------------------------------------------------------------
        # (i) critic bootstraps V(s',a') with terminated masking
        # ---------------------------------------------------------------------------


class TestCriticBootstrap:
    def td_chunks(self):
        cfg = SJFPOConfig()
        chunks = []
        for _ in range(4):
            chunks.append(
                {
                    "observation.state": torch.randn(cfg.n_obs_steps, OBS_DIM),
                    "action": torch.randn(cfg.chunk_size, ACT_DIM) * 0.5,
                    "next.action": torch.randn(cfg.chunk_size, ACT_DIM) * 0.5,
                    "next.observation.state": torch.randn(cfg.n_obs_steps, OBS_DIM),
                    "next.reward": torch.ones(8),
                    "next.done": torch.tensor(False),
                    "next.truncated": torch.tensor(False),
                    "n_action_steps": torch.tensor(8),
                    "clip_count": torch.tensor(0),
                }
            )
        return chunks

    def test_terminated_masks_bootstrap(self):
        trainer = make_trainer()
        chunks = self.td_chunks()
        v_next = torch.tensor([10.0, 10.0, 10.0, 10.0])
        live = trainer.compute_critic_targets(chunks, v_next, trainer.cfg.gamma)
        for c in chunks:
            c["next.done"] = torch.tensor(True)
        dead = trainer.compute_critic_targets(chunks, v_next, trainer.cfg.gamma)
        # terminated target == intradiscounted rewards only, independent of v_next
        gamma = trainer.cfg.gamma
        n = int(chunks[0]["n_action_steps"])
        r = float(sum(gamma**i for i in range(n)))
        assert torch.allclose(dead.cpu(), torch.full((4,), r)), dead
        assert bool((live > dead).all()), "live bootstrap must exceed R-only target"

    def test_value_loss_matches_manual_td(self):
        trainer = make_trainer()
        chunks, meta = make_chunks(8)
        dev = trainer.device
        states, actions, next_states, next_actions, adv, _, _ = (
            trainer.stack_chunk_batch(chunks, meta)
        )
        states, actions, next_states, next_actions = (
            states.to(dev),
            actions.to(dev),
            next_states.to(dev),
            next_actions.to(dev),
        )
        with torch.no_grad():
            z = trainer.critic_state_encoder(states)
            # Mirror the trainer: the critic judges IDCT real-space actions.
            v_pred = trainer.critic(z, idct(actions, dim=1))
            v_next = trainer.critic(
                trainer.critic_state_encoder(next_states),
                idct(next_actions, dim=1),
            )
        expected = trainer.compute_critic_targets(chunks, v_next, trainer.cfg.gamma)
        manual = torch.nn.functional.mse_loss(v_pred, expected)
        diag = trainer.assemble_update(chunks, meta)
        assert diag["value_loss"] == pytest.approx(manual.item())
