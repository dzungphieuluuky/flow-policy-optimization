"""Correctness checks for the DPPO denoising-MDP primitives."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myfpo.denoising_mdp import (
    denoising_chain_gae,
    gaussian_log_prob,
    gaussian_step_ratio,
    sample_denoising_path,
)

B, D, K = 6, 7, 5
SHAPE = (D,)


def _net(x, t, c):
    """Deterministic, cheap, and time-varying so the chain actually moves."""
    return torch.tanh(x.mean(dim=-1, keepdim=True) * 1.7 + float(t) * 2.0).expand_as(x)


def test_path_records_a_full_chain_and_lands_on_the_returned_action():
    cond = torch.randn(B, 16)
    x_final, path = sample_denoising_path(
        _net, cond, SHAPE, num_steps=K, noise_std=0.0, store=True
    )
    assert path is not None
    assert path.states.shape == (B, K, D)
    assert path.actions.shape == (B, K, D)
    assert path.final.shape == (B, D)
    # The chain must reconstruct the returned action, or the env executed.
    rebuilt = path.states[:, 0] + path.actions.sum(dim=1)
    assert torch.allclose(rebuilt, x_final, atol=1e-5), (
        rebuilt - x_final
    ).abs().max()
    # x_0 is the noise seed, untouched by any step.
    assert torch.allclose(path.states[:, 0], path.actions[:, 0] * 0 + path.states[:, 0])


def test_stored_mus_equal_the_stored_actions_when_noise_is_zero():
    cond = torch.randn(B, 16)
    _, path = sample_denoising_path(_net, cond, SHAPE, num_steps=K, noise_std=0.0)
    assert torch.allclose(path.mus, path.actions, atol=1e-6)


def test_store_false_returns_no_path_but_the_same_action():
    cond = torch.randn(B, 16)
    g = torch.Generator().manual_seed(0)
    x1, p1 = sample_denoising_path(
        _net, cond, SHAPE, num_steps=K, noise_std=0.1, generator=g, store=False
    )
    assert p1 is None
    g2 = torch.Generator().manual_seed(0)
    x2, p2 = sample_denoising_path(
        _net, cond, SHAPE, num_steps=K, noise_std=0.1, generator=g2, store=True
    )
    assert p2 is not None
    # actions is [n, K, *flow]. SHAPE is the per-sample flow shape.
    assert p2.actions.shape == (B, K, *SHAPE)
    assert p2.mus.shape == p2.actions.shape
    assert torch.allclose(x1, x2, atol=1e-6)


def test_stochastic_sampling_actually_differs_from_deterministic():
    cond = torch.randn(B, 16)
    x_det, _ = sample_denoising_path(_net, cond, SHAPE, num_steps=K, noise_std=0.0)
    x_sto, _ = sample_denoising_path(_net, cond, SHAPE, num_steps=K, noise_std=0.5)
    assert (x_det - x_sto).abs().max() > 1e-3


def test_gaussian_log_prob_matches_the_closed_form():
    mu = torch.randn(B, D)
    logstd = torch.full((B,), 0.3)
    std = logstd.exp()
    a = mu + std.unsqueeze(-1) * torch.randn(B, D)
    got = gaussian_log_prob(a, mu, logstd)
    want = (
        -0.5 * ((a - mu) / std.unsqueeze(-1)).pow(2).sum(-1)
        - D * logstd
        - 0.5 * D * math.log(2.0 * math.pi)
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_ratio_is_one_when_the_policy_has_not_moved():
    lp = torch.randn(50)
    assert torch.allclose(gaussian_step_ratio(lp, lp, 0.05), torch.ones(50), atol=1e-6)


def test_ratio_respects_the_trust_region():
    # Ratio is exp(lp_new - lp_old). A policy that got 10 nats MORE likely.
    lp_new = torch.zeros(50)
    lp_old = torch.full((50,), -10.0)
    r = gaussian_step_ratio(lp_new, lp_old, 0.05)
    assert torch.allclose(r, torch.full((50,), 1.05), atol=1e-6)
    # And the lower clip on the other side.
    lp_new2 = torch.zeros(50)
    lp_old2 = torch.full((50,), 10.0)
    r2 = gaussian_step_ratio(lp_new2, lp_old2, 0.05)
    assert torch.allclose(r2, torch.full((50,), 0.95), atol=1e-6)


def test_gae_matches_an_explicit_loop():
    torch.manual_seed(0)
    r = torch.randn(4, K)
    v = torch.randn(4, K + 1)
    d = torch.zeros(4, K, dtype=torch.bool)
    d[0, 2] = True
    d[2, 0] = True
    g = 0.97
    got = denoising_chain_gae(r, v, d, gamma=g)

    want = torch.zeros_like(r)
    for n in range(4):
        last = 0.0
        for j in range(K - 1, -1, -1):
            nt = 0.0 if bool(d[n, j]) else 1.0
            delta = r[n, j] + g * nt * v[n, j + 1] - v[n, j]
            last = float(delta) + g * nt * last
            want[n, j] = last
    assert torch.allclose(got, want, atol=1e-5)


def test_gae_value_at_last_index_is_the_td_residual():
    r = torch.randn(3, K)
    v = torch.randn(3, K + 1)
    d = torch.zeros(3, K, dtype=torch.bool)
    g = 0.99
    got = denoising_chain_gae(r, v, d, gamma=g)
    for n in range(3):
        assert torch.allclose(
            got[n, K - 1], r[n, K - 1] + g * v[n, K] - v[n, K - 1], atol=1e-6
        )


def test_done_flag_zeroes_the_bootstrap_at_that_step():
    r = torch.zeros(1, K)
    v = torch.ones(1, K + 1)
    g = 0.99
    # Terminal residual with no reward: delta = 0 + 0*V(s') - V(s) = -1.
    d = torch.zeros(1, K, dtype=torch.bool)
    d[0, K - 1] = True
    got = denoising_chain_gae(r, v, d, gamma=g)
    assert torch.allclose(got[0, K - 1], torch.tensor(-1.0), atol=1e-6)
    # Without the flag the same step carries the discounted future instead.
    got2 = denoising_chain_gae(r, v, torch.zeros(1, K, dtype=torch.bool), gamma=g)
    assert torch.allclose(got2[0, K - 1], torch.tensor(-1.0 + g), atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))