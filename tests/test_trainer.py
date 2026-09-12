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
from sjfpo.trainer import SJFPOTrainer, SJFPOAborted

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 17
ACT_DIM = 4


def make_trainer(**overrides) -> SJFPOTrainer:
    """Construct a trainer with small dims for fast tests."""
    cfg = SJFPOConfig(**overrides) if overrides else SJFPOConfig()
    policy_state_encoder = StateEncoder(
        OBS_DIM, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
    )
    # SpectralEncoder outputs [B, K, spectral_dim=64] conditioning tokens.
    policy_head = SJFOPolicyModel(ACT_DIM, hidden_size=cfg.hidden_size)
    critic = Critic(hidden_size=cfg.hidden_size, act_dim=ACT_DIM)
    spectral = SpectralEncoder(hidden_size=cfg.hidden_size, n_obs_steps=cfg.n_obs_steps)
    critic_state_encoder = StateEncoder(
        OBS_DIM, n_obs_steps=cfg.n_obs_steps, hidden_size=cfg.hidden_size
    )
    return SJFPOTrainer(
        policy_state_encoder=policy_state_encoder,
        policy_head=policy_head,
        critic=critic,
        spectral=spectral,
        critic_state_encoder=critic_state_encoder,
        cfg=cfg,
    )


def make_chunks(n: int = 128) -> tuple[list[Transition], dict]:
    """Create synthetic transitions with pre-computed advantages."""
    cfg = SJFPOConfig()
    t_len, h_len, n_steps = cfg.n_obs_steps, cfg.chunk_size, cfg.n_action_steps
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
        "sigreg_loss",
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
    def test_aspo_only_reaches_policy_head_and_trunk(self):
        """Policy + trunk receive grads through the surrogate path."""
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

            # Policy trunk must have gradients (through policy path)
        for name, p in trainer.policy_state_encoder.named_parameters():
            assert p.grad is not None, f"policy_state_encoder.{name} has no grad"

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
            for k in ("policy_loss", "value_loss", "sigreg_loss"):
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


class TestSpectralProjectorOptimized:
    def test_spectral_params_in_optimizer(self):
        """Every spectral parameter (projector) must be stepped by optimizer."""
        trainer = make_trainer()
        opt_ids = {
            id(p) for group in trainer.optimizer.param_groups for p in group["params"]
        }
        params = list(trainer.spectral.parameters())
        assert params, "spectral module has no parameters to optimize"
        for name, p in trainer.spectral.named_parameters():
            assert id(p) in opt_ids, f"spectral.{name} not in optimizer"

    def test_spectral_projector_moves_after_update(self):
        """Cond-path loss must move the spectral projector after updates."""
        trainer = make_trainer()
        before = {n: p.clone() for n, p in trainer.spectral.named_parameters()}
        chunks, meta = make_chunks(32)
        # Same adaLN-Zero cold start as above: the projector only moves
        # once conditioning grads reach it (cap 10 updates, else real bug).
        moved = False
        for _ in range(10):
            trainer.assemble_update(chunks, meta)
            moved = any(
                not torch.equal(p, before[n])
                for n, p in trainer.spectral.named_parameters()
            )
            if moved:
                break
        assert moved, "spectral projector did not move after update"

        # ---------------------------------------------------------------------------
        # (h) SigReg on the critic latent only (no next-state terms)
        # ---------------------------------------------------------------------------


class TestCriticLatentSigReg:
    def test_sigreg_follows_state_and_action_not_next_state(self):
        """Seeded: state/action moves move sigreg; next_states must NOT."""

        # Fresh identically-initialized trainers per call: assemble_update is
        # stateful (weights move), so cross-call equality needs equal inits.
        def sigreg(data):
            torch.manual_seed(0)
            trainer = make_trainer()
            torch.manual_seed(7)
            return trainer.assemble_update(data, make_chunks(16)[1])["sigreg_loss"]

        torch.manual_seed(0)
        chunks, _ = make_chunks(16)
        base = sigreg(chunks)
        act2 = [dict(c, **{"action": torch.randn(16, ACT_DIM)}) for c in chunks]
        assert sigreg(act2) != base, "sigreg must see the critic latent (action path)"
        obs2 = [
            dict(c, **{"observation.state": torch.randn(16, OBS_DIM)}) for c in chunks
        ]
        assert sigreg(obs2) != base, "sigreg must see the critic latent (state path)"
        next2 = [
            dict(c, **{"next.observation.state": torch.randn(16, OBS_DIM)})
            for c in chunks
        ]
        assert sigreg(next2) == base, "sigreg must NOT see next-states anymore"

    def test_trunk_receives_grads(self):
        """Each trunk must get grads through its own path (policy/critic)."""
        trainer = make_trainer()
        chunks, meta = make_chunks(16)
        trainer.assemble_update(chunks, meta)
        policy_grads = [
            p.grad for _, p in trainer.policy_state_encoder.named_parameters()
        ]
        assert all(g is not None for g in policy_grads), (
            "policy trunk params missing grads"
        )
        critic_grads = [
            p.grad for _, p in trainer.critic_state_encoder.named_parameters()
        ]
        assert all(g is not None for g in critic_grads), (
            "critic trunk params missing grads"
        )

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
                    "next.reward": torch.ones(cfg.n_action_steps),
                    "next.done": torch.tensor(False),
                    "next.truncated": torch.tensor(False),
                    "n_action_steps": torch.tensor(cfg.n_action_steps),
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
            v_pred = trainer.critic(z, actions)
            v_next = trainer.critic(
                trainer.critic_state_encoder(next_states), next_actions
            )
        expected = trainer.compute_critic_targets(chunks, v_next, trainer.cfg.gamma)
        manual = torch.nn.functional.mse_loss(v_pred, expected)
        diag = trainer.assemble_update(chunks, meta)
        assert diag["value_loss"] == pytest.approx(manual.item())
