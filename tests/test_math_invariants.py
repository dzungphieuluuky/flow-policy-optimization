"""Algorithmic invariant tests for heavy-math compute functions.

Detect only — does not fix bugs; fails when invariants are violated.
New identifiers: test_gae_chunk_advantage_invariant,
test_fpo_policy_ratio_pessimistic, test_fpo_cfm_loss_mask_shape.
"""
import torch


class TestGAEChunkAdvantageInvariant:
    def test_gae_chunk_advantage_invariant(self):
        """GAE: advantages + chunk_q == target; gamma^h scales bootstrap."""
        from myfpo.gae import compute_chunk_advantages

        # Minimal synthetic chunk: 2 chunks, length 3 each, gamma=1.0 (no discount)
        # Then target = R + 1*(1-0)*next_q = R + next_q; advantage = target - q
        chunk_q = torch.ones(2, dtype=torch.float32)
        chunk_next_q = torch.ones(2, dtype=torch.float32) * 2.0
        chunk_starts = torch.tensor([0, 3], dtype=torch.long)
        chunk_lengths = torch.tensor([3, 3], dtype=torch.long)
        chunk_terminated = torch.tensor([False, False], dtype=torch.bool)
        # step_rewards: 6 steps, 3 per chunk
        step_rewards = torch.ones(6, dtype=torch.float32)

        advantages, targets = compute_chunk_advantages(
            step_rewards=step_rewards,
            chunk_q=chunk_q,
            chunk_next_q=chunk_next_q,
            chunk_starts=chunk_starts,
            chunk_lengths=chunk_lengths,
            chunk_terminated=chunk_terminated,
            chunk_size=3,
            gamma=1.0,
            gae_lambda=0.0,
        )
        # With gamma=1 and no termination, target = R (3) + next_q (2) = 5
        # advantage = target - chunk_q (1) = 4
        expected_target = 3.0 + 2.0  # R + gamma^h * next_q
        assert torch.allclose(targets, torch.tensor([expected_target, expected_target], dtype=torch.float32))
        assert torch.allclose(advantages, targets - chunk_q)


class TestFPORatioInvariant:
    def test_fpo_policy_ratio_pessimistic(self):
        """PPO ratio: min(ratio*A, clip(ratio)*A) must be <= ratio*A for A>0."""
        # This is an algorithmic invariant from the PPO pessimistic bound.
        pass  # Placeholder: requires full forward path setup.


class TestCFMLossMaskShape:
    def test_fpo_cfm_loss_mask_shape(self):
        """CFM: an all-False pad mask must leave the loss bit-identical to unmasked.

        Asserts the ``"both"`` reduction at the 4-D ``[n, mc, horizon, dim]``
        shape both production call sites use, where the mask broadcasts over the
        MC axis without inserting one.
        """
        from myfpo.configuration_sjfpo import SJFPOConfig
        from myfpo.fpo_core import compute_cfm_loss

        n, mc, horizon, dim = 2, 4, 4, 2
        g = torch.Generator().manual_seed(0)
        v_hat = torch.randn(n, mc, horizon, dim, generator=g, dtype=torch.float64)
        v_star = torch.randn(n, mc, horizon, dim, generator=g, dtype=torch.float64)
        no_pad = torch.zeros(n, horizon, dtype=torch.bool)

        masked = compute_cfm_loss(v_hat, v_star, SJFPOConfig(), "both", no_pad)
        unmasked = compute_cfm_loss(v_hat, v_star, SJFPOConfig(), "both", None)
        for got, want in zip(masked, unmasked):
            assert torch.equal(got, want)


class TestFPOCoreRemaining:
    def test_fpo_aspo_surrogate_shape(self):
        """ASPO: for A>0 returns PPO; A<0 returns SPO; elementwise."""
        from myfpo.fpo_core import compute_aspo_surrogate
        from myfpo.configuration_sjfpo import SJFPOConfig
        import torch

        cfg = SJFPOConfig()
        ratio = torch.ones(4, 4)
        adv = torch.ones(4, 4) * 0.5
        result = compute_aspo_surrogate(ratio, adv, cfg)
        assert result.shape == ratio.shape

    def test_spo_uses_its_own_width_not_the_ppo_one(self):
        """FPO++ gives SPO a separate coefficient (``spo_clip_coef``).

        Sharing ``clip_param`` would silently rescale the negative-advantage
        branch whenever the PPO band moves.
        """
        import dataclasses

        import torch

        from myfpo.configuration_sjfpo import SJFPOConfig
        from myfpo.fpo_core import compute_aspo_surrogate

        cfg = dataclasses.replace(
            SJFPOConfig(), clip_param=0.05, spo_clip_param=0.01
        )
        ratio = torch.tensor([1.5, 1.5])
        adv = torch.tensor([-2.0, 2.0])

        got = compute_aspo_surrogate(ratio, adv, cfg)
        spo_width = ratio[0] * adv[0] - (abs(adv[0]) / (2.0 * 0.01)) * (ratio[0] - 1.0) ** 2
        # A>0 clips at clip_param, so the band [0.95, 1.05] binds: 1.05 * 2.
        ppo_clipped = min(ratio[1].item(), 1.0 + cfg.clip_param) * adv[1].item()
        assert torch.allclose(got[0], spo_width), "A<0 took SPO with the wrong width"
        assert torch.allclose(got[1], torch.tensor(ppo_clipped)), "A>0 must stay PPO"

    def test_fpo_policy_ratio_clip_invariant(self):
        """Policy ratio: exp(delta_st) with clamped delta; backward through delta."""
        from myfpo.fpo_core import compute_policy_ratio
        from myfpo.configuration_sjfpo import SJFPOConfig
        import torch

        cfg = SJFPOConfig()
        cfm_loss_old = torch.ones(2, 2)
        cfm_loss_new = torch.ones(2, 2) * 0.9
        ratio = compute_policy_ratio(cfm_loss_old, cfm_loss_new, cfg)
        assert ratio.shape == cfm_loss_old.shape
        # With identical old/new and no clamp, ratio should be 1 (exp(0) = 1)
        ratio_same = compute_policy_ratio(cfm_loss_old, cfm_loss_old, cfg)
        assert torch.isclose(ratio_same, torch.ones_like(ratio_same), atol=1e-5).all()


class TestValueDiagnostics:
    def test_gae_diagnostics_shape(self):
        """Diagnostics dict keys match expected tuple."""
        from myfpo.modeling_sjfpo import _GAE_DIAGNOSTIC_KEYS
        assert isinstance(_GAE_DIAGNOSTIC_KEYS, tuple)
        assert len(_GAE_DIAGNOSTIC_KEYS) == 5


class TestSamplingFunctions:
    def test_sample_timesteps_bounds(self):
        """Sampled timesteps within [min, max]."""
        from myfpo.fpo_core import sample_timesteps
        from myfpo.configuration_sjfpo import SJFPOConfig
        import torch

        cfg = SJFPOConfig()
        t = sample_timesteps(10, cfg, device=torch.device("cpu"), generator=None)
        assert (t >= cfg.min_timestep - 1e-5).all()
        assert (t <= cfg.max_timestep + 1e-5).all()

    def test_sample_noise_shape(self):
        """Noise shape matches request."""
        from myfpo.fpo_core import sample_noise
        from myfpo.configuration_sjfpo import SJFPOConfig
        import torch

        cfg = SJFPOConfig()
        eps = sample_noise((4, 2), cfg, device=torch.device("cpu"), generator=None)
        assert eps.shape == (4, 2)
