"""FPO++ core math tests (plan task 5) — written BEFORE src/mfpo/fpo_core.py.

Covers the nine numeric gates from plan checkbox 5:
(a) toy 1D Gaussian reference-equivalence gate (Metis blocker #1),
(b) fitted-CFM < shuffled-target CFM on >=95% of fixtures,
(c) per-sample ratio shape [chunks, N_mc=16] + degeneracy guard (blocker #2),
(d) ASPO hand-computed surrogates (tolerance 1e-3),
(e) STE log-ratio clamp: value capped at exp(cap), gradient still flows,
(f) negative-advantage clamp: effective Delta-l capped at +cfm_clamp,
(g) finite-difference gradient check of the policy loss (<5% rel. err),
(h) Euler sampler: descending t-grid linspace(1,0,steps+1), x <- x - dt*v_hat,
    zero-sampling (eps=0) deterministic,
(i) space-consistency: DCT-coefficient-space sampling -> IDCT -> clip inside
    bounds; DCT(IDCT(x)) == x roundtrip < 1e-5.

Binding conventions exercised throughout: x_t = t*eps + (1-t)*a (t=1 noise),
v* = eps - a, tau ~ U[0.005, 0.995], N_mc = 16, fp32 only.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mfpo import config  # noqa: E402
from mfpo.dct_utils import dct, idct  # noqa: E402
from mfpo.networks import PolicyFlowHead, timestep_embedding  # noqa: E402

from mfpo import fpo_core  # noqa: E402  (RED until implemented)

_TAU_LO, _TAU_HI = config.MFPO_CFG.tau_range


class VelocityMLP(nn.Module):
    """Toy velocity net MLP[in+8 -> 128 -> 128 -> out] ELU.

    Signature matches the project flow heads: ``net(x_t, t, cond)`` returning
    a tensor with the same shape as ``x_t`` (trailing dims flattened into the
    input features). ``cond`` is accepted and ignored (unconditional toy).
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 8, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_t: torch.Tensor, t, cond=None) -> torch.Tensor:
        temb = timestep_embedding(t, dim=8)
        if temb.shape[0] != x_t.shape[0]:
            temb = temb.expand(x_t.shape[0], -1)
        z = self.net(torch.cat([x_t.reshape(x_t.shape[0], -1), temb], dim=-1))
        return z.reshape(x_t.shape)


def _draw_tau(shape, generator) -> torch.Tensor:
    """tau ~ U[0.005, 0.995] (binding tau_range from config)."""
    return _TAU_LO + (_TAU_HI - _TAU_LO) * torch.rand(shape, generator=generator)


# ---------------------------------------------------------------------------
# (a) Reference-equivalence gate: integrated velocity field reproduces the
#     data distribution moments (<5% relative error on mean AND variance).
# ---------------------------------------------------------------------------


def test_toy_gaussian_reference_gate(capsys):
    torch.manual_seed(0)  # pins network init (plan requirement)
    net = VelocityMLP(in_dim=1, out_dim=1)
    gen = torch.Generator().manual_seed(0)  # draws the dataset ONCE
    data = 2.0 + 0.5 * torch.randn(4000, 1, generator=gen)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):  # full-batch CFM steps
        # eps [n, N_mc=1, F=1]; tau scalar per step; v* = eps - a inside cfm_loss
        eps = torch.randn(4000, 1, 1)
        tau = float(torch.empty(1).uniform_(_TAU_LO, _TAU_HI))
        loss = fpo_core.cfm_loss(net, data, eps, tau, cond=None)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        samples = fpo_core.euler_sample(
            net,
            (4000, 1),
            cond=None,
            steps=64,
            generator=torch.Generator().manual_seed(7),
        )
    mean_hat = samples.mean().item()
    var_hat = samples.var(unbiased=True).item()
    mean_err = abs(mean_hat - 2.0) / 2.0
    var_err = abs(var_hat - 0.25) / 0.25
    print(f"moments OK mean_err={mean_err:.4f} var_err={var_err:.4f}")
    assert mean_err < 0.05, f"mean {mean_hat:.4f} too far from 2.0"
    assert var_err < 0.05, f"var {var_hat:.4f} too far from 0.25"


# ---------------------------------------------------------------------------
# (b) Fitted check: per-point CFM loss under the correctly-trained net is
#     lower than under a net trained with row-shuffled noise targets.
# ---------------------------------------------------------------------------


def test_fitted_loss_below_shuffled_targets():
    torch.manual_seed(11)
    n = 1000
    gen = torch.Generator().manual_seed(11)
    data = 2.0 + 0.5 * torch.randn(n, 1, generator=gen)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))

    net_fit = VelocityMLP(1, 1)
    net_shuf = VelocityMLP(1, 1)
    net_shuf.load_state_dict(copy.deepcopy(net_fit.state_dict()))  # same init

    def train(net: VelocityMLP, shuffle: bool) -> None:
        # Manual CFM loop (same interpolation/target conventions as cfm_loss)
        # so the SHUFFLED variant can corrupt the TARGETS (eps - a) themselves;
        # permuting eps rows alone would leave the (a, eps) joint unchanged
        # (iid noise is exchangeable) and the comparison would be vacuous.
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(5)
        for _ in range(3000):
            eps = torch.randn(n, 1, generator=g)
            tau = float(torch.empty(1).uniform_(_TAU_LO, _TAU_HI, generator=g))
            x_t = tau * eps + (1.0 - tau) * data  # t*eps + (1-t)*a
            v_hat = net(x_t, torch.full((n,), tau), None)
            target = eps - data  # v* = eps - a
            if shuffle:
                target = target[perm]  # break target<->anchor pairing
            loss = ((v_hat - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    train(net_fit, shuffle=False)
    train(net_shuf, shuffle=True)

    n_fix = 40  # fixtures are points CONTAINED in the training dataset
    fixtures = data[:n_fix]
    g_eval = torch.Generator().manual_seed(13)
    eps_ev = torch.randn(n_fix, 256, 1, generator=g_eval)
    tau_ev = _draw_tau((n_fix, 256), g_eval)
    with torch.no_grad():
        l_fit = fpo_core.flow_loglik(net_fit, fixtures, eps_ev, tau_ev, None)
        l_shuf = fpo_core.flow_loglik(net_shuf, fixtures, eps_ev, tau_ev, None)
    wins = int((l_fit.mean(dim=1) < l_shuf.mean(dim=1)).sum().item())
    print(f"fitted<shuffled on {wins}/{n_fix} fixtures")
    assert wins >= math.ceil(0.95 * n_fix)  # integer count: no float-boundary flake


# ---------------------------------------------------------------------------
# (c) Per-sample ratios: shape [chunks, N_mc=16]; ONE perturbed update keeps
#     log-ratios non-degenerate (Var > 0) and centered (|mean| < 0.5).
# ---------------------------------------------------------------------------


def test_ratio_shape_and_degeneracy_guard():
    torch.manual_seed(21)
    net_old = VelocityMLP(in_dim=8 * 3)
    net_new = copy.deepcopy(net_old)
    a = torch.randn(4, 8, 3)
    g = torch.Generator().manual_seed(22)
    eps_tr = torch.randn(4, config.MFPO_CFG.N_mc, 8, 3, generator=g)
    tau_tr = _draw_tau((4, config.MFPO_CFG.N_mc), g)

    opt = torch.optim.Adam(net_new.parameters(), lr=1e-3)
    loss = fpo_core.cfm_loss(net_new, a, eps_tr, tau_tr, cond=None)
    opt.zero_grad()
    loss.backward()
    opt.step()  # the ONE perturbed update

    eps_ev = torch.randn(4, config.MFPO_CFG.N_mc, 8, 3, generator=g)
    tau_ev = _draw_tau((4, config.MFPO_CFG.N_mc), g)
    with torch.no_grad():
        l_old = fpo_core.flow_loglik(net_old, a, eps_ev, tau_ev, None)
        l_new = fpo_core.flow_loglik(net_new, a, eps_ev, tau_ev, None)
        rho = fpo_core.flow_ratio(l_old, l_new)

    assert rho.shape == (4, config.MFPO_CFG.N_mc)
    log_rho = torch.log(rho)
    assert log_rho.var(unbiased=True).item() > 0.0, "ratios degenerated to a constant"
    assert abs(log_rho.mean().item()) < 0.5, "one update moved ratios too far"


# ---------------------------------------------------------------------------
# (d) ASPO piecewise trust region against hand-computed values (tol 1e-3).
# ---------------------------------------------------------------------------


def test_aspo_surrogate_hand_computed():
    eps_clip = config.MFPO_CFG.epsilon_clip  # 0.05
    # Positive advantage: PPO-clip branch.
    psi_ppo = fpo_core.aspo_surrogate(torch.tensor(1.04), torch.tensor(1.0), eps_clip)
    assert psi_ppo.item() == pytest.approx(1.04, abs=1e-3)
    # Negative advantage: SPO branch  rho*A - (|A|/(2*eps))*(rho-1)^2.
    psi_spo = fpo_core.aspo_surrogate(torch.tensor(1.04), torch.tensor(-1.0), eps_clip)
    expected = -1.04 - (1.0 / 0.1) * 0.04**2  # = -1.056
    assert psi_spo.item() == pytest.approx(expected, abs=1e-3)
    # Mixed batch routes each element through its own branch.
    rho = torch.tensor([1.04, 1.04])
    adv = torch.tensor([1.0, -1.0])
    mixed = fpo_core.aspo_surrogate(rho, adv, eps_clip)
    assert mixed[0].item() == pytest.approx(1.04, abs=1e-3)
    assert mixed[1].item() == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# (e) STE log-ratio clamp: Delta_l = 50 -> value exp(10), gradient flows.
# ---------------------------------------------------------------------------


def test_ste_clamp_value_and_gradient():
    l_old = torch.tensor([60.0])
    l_new = torch.tensor([10.0], requires_grad=True)  # Delta_l = 50 > cap 10
    rho = fpo_core.flow_ratio(l_old, l_new)
    assert rho.item() == pytest.approx(math.exp(config.MFPO_CFG.log_ratio_cap), rel=1e-6)
    rho.backward()
    assert l_new.grad is not None
    assert l_new.grad.abs().item() > 0.0, "straight-through clamp blocked gradients"
    assert l_new.grad.item() == pytest.approx(
        -math.exp(config.MFPO_CFG.log_ratio_cap), rel=1e-5
    )


# ---------------------------------------------------------------------------
# (f) Negative-advantage clamp: l_new - l_old effectively capped at +20.
# ---------------------------------------------------------------------------


def test_neg_advantage_clamp_caps_delta():
    l_old = torch.zeros(1)
    l_new = torch.full((1,), 100.0)  # raw Delta_l = -100
    adv = torch.full((1,), -1.0)
    rho = fpo_core.policy_ratio(l_old, l_new, adv)
    # Effective Delta_l becomes -cfm_clamp = -20, so rho = exp(-20) (NOT exp(-100)).
    assert rho.item() == pytest.approx(math.exp(-config.MFPO_CFG.cfm_clamp), rel=1e-6)


# ---------------------------------------------------------------------------
# (g) Finite-difference gradient check of the policy loss w.r.t. one
#     velocity-net weight (rel. error < 5%).
# ---------------------------------------------------------------------------


def test_policy_loss_fd_gradient():
    torch.manual_seed(31)
    net = VelocityMLP(in_dim=8 * 1)  # small problem keeps FD numerics clean
    a = torch.randn(1, 8, 1)
    g = torch.Generator().manual_seed(32)
    eps = torch.randn(1, config.MFPO_CFG.N_mc, 8, 1, generator=g)
    tau = _draw_tau((1, config.MFPO_CFG.N_mc), g)
    adv = torch.tensor([1.0])

    with torch.no_grad():
        l_old = fpo_core.flow_loglik(net, a, eps, tau, None)

    def policy_loss_value() -> torch.Tensor:
        l_new = fpo_core.flow_loglik(net, a, eps, tau, None)
        return fpo_core.policy_loss(l_old.detach(), l_new, adv)

    policy_loss_value().backward()
    param = net.net[-1].weight  # last Linear, direct path to the loss
    analytic = param.grad[0, 0].item()
    assert analytic != 0.0, "gradient did not reach the velocity net"

    h = 1e-3
    with torch.no_grad():
        w = param[0, 0]
        orig = w.item()
        w.add_(h)
        l_plus = policy_loss_value().item()
        w.sub_(2.0 * h)
        l_minus = policy_loss_value().item()
        w.add_(h)
        assert w.item() == pytest.approx(orig, abs=1e-8)
    fd = (l_plus - l_minus) / (2.0 * h)
    rel_err = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
    print(f"fd grad check: analytic={analytic:.6e} fd={fd:.6e} rel_err={rel_err:.2%}")
    assert rel_err < 0.05


# ---------------------------------------------------------------------------
# (h) Euler sampler: descending linspace(1, 0, steps+1) grid, update
#     x <- x - dt * v_hat, zero-sampling deterministic given weights.
# ---------------------------------------------------------------------------


def test_euler_sampler_grid_update_and_zero_noise():
    torch.manual_seed(41)
    net = VelocityMLP(1, 1)
    seen_ts: list[float] = []

    class Probe(nn.Module):
        def forward(self, x_t, t, cond=None):
            seen_ts.append(float(t))
            return net(x_t, t, cond)

    out = fpo_core.euler_sample(
        Probe(), (5, 1), None, steps=64, generator=torch.Generator().manual_seed(42)
    )

    grid = torch.linspace(1.0, 0.0, 65)  # 65 nodes -> 64 descending intervals
    assert len(seen_ts) == 64
    assert torch.allclose(torch.tensor(seen_ts), grid[:-1], atol=1e-7)
    assert all(seen_ts[i] > seen_ts[i + 1] for i in range(len(seen_ts) - 1)), (
        "t-grid must be DESCENDING from 1 (noise) to 0 (data)"
    )

    # Replicate the exact recursion x <- x - dt * v_hat(t_left) manually.
    eps = torch.randn(5, 1, generator=torch.Generator().manual_seed(42))
    x = eps.clone()
    with torch.no_grad():
        for i in range(64):
            x = x - (grid[i] - grid[i + 1]) * net(x, grid[i], None)
    assert torch.allclose(out, x, atol=1e-6), "sampler deviates from x <- x - dt*v_hat"

    # Zero-sampling path (eps = 0) is deterministic given the weights.
    z1 = fpo_core.euler_sample(net, (5, 1), None, steps=16, eps=torch.zeros(5, 1))
    z2 = fpo_core.euler_sample(net, (5, 1), None, steps=16, eps=torch.zeros(5, 1))
    assert torch.equal(z1, z2)


# ---------------------------------------------------------------------------
# (i) SPACE-CONSISTENCY gate: sample action chunks in DCT coefficient space,
#     map back via IDCT, clip to bounds; DCT(IDCT(x)) == x roundtrip < 1e-5.
# ---------------------------------------------------------------------------


def test_space_consistency_dct_pipeline():
    torch.manual_seed(51)
    head = PolicyFlowHead(action_dim=3, cond_dims=(config.MFPO_CFG.Z_dim,))
    cond = torch.zeros(4, config.MFPO_CFG.context_len, config.MFPO_CFG.Z_dim)
    with torch.no_grad():
        coeffs = fpo_core.euler_sample(
            head, (4, 8, 3), cond, steps=16, generator=torch.Generator().manual_seed(52)
        )
        actions = idct(coeffs, dim=1)  # coefficient space -> action space
        low, high = -0.5, 0.5
        clipped = actions.clamp(min=low, max=high)
        assert (clipped <= high).all() and (clipped >= low).all(), (
            "executed actions escaped the action bounds"
        )
        assert int((clipped != actions).sum()) > 0, (
            "clip never engaged; bounds too loose"
        )

    rt_coeffs = (dct(idct(coeffs, dim=1), dim=1) - coeffs).abs().max().item()
    rt_actions = (idct(dct(actions, dim=1), dim=1) - actions).abs().max().item()
    assert rt_coeffs < 1e-5
    assert rt_actions < 1e-5
