"""FPO++ core math tests: gates (a) toy-Gaussian moments, (b) fitted < shuffled CFM,
(c) ratio shapes, (d) ASPO hand-computed, (e) STE clamp + grad flow,
(f) negative-advantage clamp, (g) finite-difference grad check,
(h) Euler grid/update/zero-noise, (i) DCT space-consistency roundtrip.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402

_DEFAULTS = SJFPOConfig()
from sjfpo.dct_utils import dct, idct  # noqa: E402
from sjfpo.networks import SJFPOFlowModel, SinusoidalEncoding  # noqa: E402
from sjfpo.modeling_sjfpo import SJFPOPolicy  # noqa: E402 (velocity helpers under test)

from sjfpo import fpo_core  # noqa: E402  (RED until implemented)

TAU_LO, TAU_HI = _DEFAULTS.min_timestep, _DEFAULTS.max_timestep


def _cfg(**overrides):
    return replace(_DEFAULTS, **overrides)


def compute_cfm_loss(v_hat, v_star, reduce_mode="scalar", clamp=None):
    cfg = _DEFAULTS if clamp is None else _cfg(cfm_clamp=clamp)
    return fpo_core.compute_cfm_loss(v_hat, v_star, cfg, reduce_mode)


def compute_aspo_surrogate(rho, advantage, epsilon_clip=None):
    epsilon = _DEFAULTS.epsilon_clip if epsilon_clip is None else epsilon_clip
    return fpo_core.compute_aspo_surrogate(rho, advantage, _cfg(use_aspo=True), epsilon)


def compute_policy_ratio(l_old, l_new, cfm_clamp=None, cap=None):
    cfg = _DEFAULTS
    if cfm_clamp is not None:
        cfg = _cfg(cfm_clamp=cfm_clamp)
    if cap is not None:
        cfg = _cfg(log_ratio_cap=cap, cfm_clamp=cfg.cfm_clamp)
    return fpo_core.compute_policy_ratio(l_old, l_new, cfg)


def compute_policy_loss(l_old, l_new, advantage, epsilon_clip=None):
    return fpo_core.compute_policy_loss(l_old, l_new, advantage, _DEFAULTS)


def sample_timesteps(num_samples, tau_range=None, device=None, mode="uniform", **kwargs):
    cfg = _cfg(timestep_sampling_mode=mode)
    if tau_range is not None:
        cfg = _cfg(
            timestep_sampling_mode=mode,
            min_timestep=tau_range[0],
            max_timestep=tau_range[1],
        )
    return fpo_core.sample_timesteps(
        num_samples, cfg, torch.device("cpu") if device is None else device, None
    )


def sample_noise(shape, device=None, mode="normal", **kwargs):
    cfg = _cfg(noise_sampling_mode=mode)
    return fpo_core.sample_noise(
        shape, cfg, torch.device("cpu") if device is None else device, None
    )


def sample_action(
    predict_velocity,
    shape,
    condition,
    num_inference_steps,
    eps=None,
    generator=None,
    device=None,
):
    if device is None:
        device = condition.device if isinstance(condition, torch.Tensor) else "cpu"
    return fpo_core.sample_action(
        predict_velocity, shape, condition, num_inference_steps, eps, generator, device
    )


class VelocityMLP(nn.Module):
    """Toy velocity net MLP[in+8 -> 128 -> 128 -> out] SiLU.

    Signature matches the project flow heads: ``net(x_t, t, condition)``
    returning a tensor with the same shape as ``x_t`` (trailing dims flattened
    into the input features). ``condition`` is REQUIRED (positional, may be
    None) and ignored (unconditional toy).
    """

    def __init__(self, in_dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + 8, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_t: torch.Tensor, t, condition) -> torch.Tensor:
        temb = SinusoidalEncoding(8, scale=_DEFAULTS.timestep_scale).encode_times(t)
        if temb.shape[0] != x_t.shape[0]:
            temb = temb.expand(x_t.shape[0], -1)
        z = self.net(torch.cat([x_t.reshape(x_t.shape[0], -1), temb], dim=-1))
        return z.reshape(x_t.shape)


def draw_tau(shape, generator) -> torch.Tensor:
    """tau ~ U[0, 1] (binding min/max_timestep from config, paper-original)."""
    return TAU_LO + (TAU_HI - TAU_LO) * torch.rand(shape, generator=generator)

    # (a) Reference-equivalence gate: integrated velocity field reproduces the
    #     data distribution moments (<5% relative error on mean AND variance).


def test_toy_gaussian_reference_gate(capsys):
    torch.manual_seed(0)  # pins network init (plan requirement)
    net = VelocityMLP(in_dim=1, out_dim=1)
    gen = torch.Generator().manual_seed(0)  # draws the dataset ONCE
    data = 2.0 + 0.5 * torch.randn(4000, 1, generator=gen)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):  # full-batch CFM steps
        # eps [n, num_mc_samples=1, F=1]; tau scalar per step
        eps = torch.randn(4000, 1, 1)
        tau = float(torch.empty(1).uniform_(TAU_LO, TAU_HI))
        x_t = (1.0 - tau) * eps + tau * data.unsqueeze(1)  # [4000, 1, 1] (1−t)·ε+t·a
        v_hat = net(x_t.reshape(4000, 1), torch.full((4000,), tau), None).reshape(
            4000, 1, 1
        )
        v_star = SJFPOPolicy.compute_target_velocity(data, eps)  # [4000, 1, 1]
        loss = compute_cfm_loss(v_hat, v_star)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        # 256 steps: coarse Euler (64) contracts variance by ~6% through any
        # imperfect field, which would confound this field-correctness gate.
        samples = sample_action(
            lambda x, t, c: net(x, t, c),
            (4000, 1),
            condition=None,
            num_inference_steps=256,
            generator=torch.Generator().manual_seed(7),
        )
    mean_hat = samples.mean().item()
    var_hat = samples.var(unbiased=True).item()
    mean_err = abs(mean_hat - 2.0) / 2.0
    var_err = abs(var_hat - 0.25) / 0.25
    print(f"moments OK mean_err={mean_err:.4f} var_err={var_err:.4f}")
    assert mean_err < 0.05, f"mean {mean_hat:.4f} too far from 2.0"
    assert var_err < 0.05, f"var {var_hat:.4f} too far from 0.25"

    # (b) Fitted check: per-point CFM loss under the correctly-trained net is
    #     lower than under a net trained with row-shuffled noise targets.


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
        # Manual CFM loop (same interpolation/target conventions as compute_cfm_loss)
        # so the SHUFFLED variant can corrupt the TARGETS (a - eps) themselves;
        # permuting eps rows alone would leave the (a, eps) joint unchanged
        # (iid noise is exchangeable) and the comparison would be vacuous.
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(5)
        for _ in range(3000):
            eps = torch.randn(n, 1, generator=g)
            tau = float(torch.empty(1).uniform_(TAU_LO, TAU_HI, generator=g))
            x_t = (1.0 - tau) * eps + tau * data  # (1-t)*eps + t*a
            v_hat = net(x_t, torch.full((n,), tau), None)
            target = data - eps  # v* = a - eps
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
    tau_ev = draw_tau((n_fix, 256), g_eval)
    with torch.no_grad():
        t = tau_ev.unsqueeze(-1)  # [40, 256, 1]
        x_t = (1.0 - t) * eps_ev + t * fixtures.unsqueeze(1)  # [40, 256, 1]
        v_star = SJFPOPolicy.compute_target_velocity(fixtures, eps_ev)
        l_fit = compute_cfm_loss(
            net_fit(x_t.reshape(40 * 256, 1), tau_ev.reshape(40 * 256), None).reshape(
                40, 256, 1
            ),
            v_star,
            reduce_mode="sample",
        )
        l_shuf = compute_cfm_loss(
            net_shuf(x_t.reshape(40 * 256, 1), tau_ev.reshape(40 * 256), None).reshape(
                40, 256, 1
            ),
            v_star,
            reduce_mode="sample",
        )
    wins = int((l_fit.mean(dim=1) < l_shuf.mean(dim=1)).sum().item())
    print(f"fitted<shuffled on {wins}/{n_fix} fixtures")
    assert wins >= math.ceil(0.95 * n_fix)  # integer count: no float-boundary flake

    # (c) Per-sample ratios: shape [chunks, num_mc_samples=16]; ONE perturbed update keeps
    #     log-ratios non-degenerate (Var > 0) and centered (|mean| < 0.5).


def test_ratio_shape_and_degeneracy_guard():
    torch.manual_seed(21)
    net_old = VelocityMLP(in_dim=8 * 3)
    net_new = copy.deepcopy(net_old)
    a = torch.randn(4, 8, 3)
    g = torch.Generator().manual_seed(22)
    eps_tr = torch.randn(4, _DEFAULTS.num_mc_samples, 8, 3, generator=g)
    tau_tr = draw_tau((4, _DEFAULTS.num_mc_samples), g)

    opt = torch.optim.Adam(net_new.parameters(), lr=1e-3)
    m = _DEFAULTS.num_mc_samples
    t = tau_tr.unsqueeze(-1).unsqueeze(-1)  # [4, m, 1, 1]
    x_t = (1.0 - t) * eps_tr + t * a.unsqueeze(1)  # [4, m, 8, 3]
    v_star_tr = SJFPOPolicy.compute_target_velocity(a, eps_tr)
    loss = compute_cfm_loss(
        net_new(x_t.reshape(4 * m, 8, 3), tau_tr.reshape(4 * m), None).reshape(
            4, m, 8, 3
        ),
        v_star_tr,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()  # the ONE perturbed update

    eps_ev = torch.randn(4, _DEFAULTS.num_mc_samples, 8, 3, generator=g)
    tau_ev = draw_tau((4, _DEFAULTS.num_mc_samples), g)
    with torch.no_grad():
        t = tau_ev.unsqueeze(-1).unsqueeze(-1)  # [4, m, 1, 1]
        x_t = (1.0 - t) * eps_ev + t * a.unsqueeze(1)  # [4, m, 8, 3]
        v_star = SJFPOPolicy.compute_target_velocity(a, eps_ev)
        l_old = compute_cfm_loss(
            net_old(x_t.reshape(4 * m, 8, 3), tau_ev.reshape(4 * m), None).reshape(
                4, m, 8, 3
            ),
            v_star,
            reduce_mode="sample",
        )
        l_new = compute_cfm_loss(
            net_new(x_t.reshape(4 * m, 8, 3), tau_ev.reshape(4 * m), None).reshape(
                4, m, 8, 3
            ),
            v_star,
            reduce_mode="sample",
        )
        rho = compute_policy_ratio(l_old, l_new)

    assert rho.shape == (4, _DEFAULTS.num_mc_samples)
    log_rho = torch.log(rho)
    assert log_rho.var(unbiased=True).item() > 0.0, "ratios degenerated to a constant"
    assert abs(log_rho.mean().item()) < 0.5, "one update moved ratios too far"

    # (d) ASPO piecewise trust region against hand-computed values (tol 1e-3).


def test_aspo_surrogate_hand_computed():
    eps = _DEFAULTS.epsilon_clip  # 0.05, single paper-original parameter
    # Positive advantage inside bounds: PPO-clip branch is the identity.
    psi_ppo = compute_aspo_surrogate(
        torch.tensor(1.02), torch.tensor(1.0), eps
    )
    assert psi_ppo.item() == pytest.approx(1.02, abs=1e-3)
    # Above the upper bound: clipped to 1 + eps.
    psi_hi = compute_aspo_surrogate(torch.tensor(1.3), torch.tensor(1.0), eps)
    assert psi_hi.item() == pytest.approx(1.0 + eps, abs=1e-3)
    # Below the lower bound: min picks the unclipped (smaller) value.
    psi_lo = compute_aspo_surrogate(torch.tensor(0.7), torch.tensor(1.0), eps)
    assert psi_lo.item() == pytest.approx(0.7, abs=1e-3)
    # Negative advantage: SPO branch  rho*A - (|A|/(2*eps))*(rho-1)^2.
    psi_spo = compute_aspo_surrogate(
        torch.tensor(1.02), torch.tensor(-1.0), eps
    )
    expected = -1.02 - (1.0 / (2 * eps)) * 0.02**2
    assert psi_spo.item() == pytest.approx(expected, abs=1e-3)
    # Mixed batch routes each element through its own branch.
    rho = torch.tensor([1.3, 1.02])
    advantage = torch.tensor([1.0, -1.0])
    mixed = compute_aspo_surrogate(rho, advantage, eps)
    assert mixed[0].item() == pytest.approx(1.0 + eps, abs=1e-3)
    assert mixed[1].item() == pytest.approx(expected, abs=1e-3)
    # OFF boundary: zero advantage routes through SPO but both branches are 0.
    zero = compute_aspo_surrogate(torch.tensor(1.5), torch.tensor(0.0), eps)
    assert zero.item() == pytest.approx(0.0, abs=1e-9)


def test_fpo_helpers_use_explicit_policy_config():
    cfg = _cfg(use_aspo=True, cfm_clamp=1.0, log_ratio_cap=0.5, epsilon_clip=0.1)
    rho = fpo_core.compute_policy_ratio(
        torch.tensor([2.0]), torch.tensor([0.0]), cfg
    )
    assert rho.item() == pytest.approx(math.exp(0.5), rel=1e-6)
    aspo = fpo_core.compute_aspo_surrogate(
        torch.tensor(1.02), torch.tensor(-1.0), cfg, cfg.epsilon_clip
    )
    assert aspo.item() < -1.02


def test_policy_ratio_ste_cap_value_and_gradient():
    # OFF cfm_diff_clamp_max=10: Delta=50 -> exp(10) forward, gradient flows.
    l_old = torch.tensor([60.0])
    l_new = torch.tensor([10.0], requires_grad=True)
    rho = compute_policy_ratio(l_old, l_new)
    assert rho.item() == pytest.approx(math.exp(10.0), rel=1e-5)
    rho.backward()
    assert l_new.grad is not None
    assert l_new.grad.item() == pytest.approx(-math.exp(10.0), rel=1e-5)


def test_large_loss_clamped_for_negative_advantage():
    # l_new=100 -> max-clamped to cfm_clamp=20, then log-ratio STE-capped at 10.
    l_old = torch.zeros(1)
    l_new = torch.full((1,), 100.0)
    rho = compute_policy_ratio(l_old, l_new)
    assert rho.item() == pytest.approx(math.exp(-10.0), rel=1e-6)


def test_symmetric_cfm_clamp_applies_to_both():
    # OFF cfm_loss_clamp=20 binds old and new alike.
    l_old = torch.full((2,), 100.0)
    l_new = torch.zeros(2)
    rho = compute_policy_ratio(l_old, l_new)
    assert rho.tolist() == pytest.approx(
        [math.exp(10.0)] * 2, rel=1e-5
    )  # (20-0)->STE 10

    # (g) Finite-difference gradient check of the policy loss w.r.t. one
    #     velocity-net weight (rel. error < 5%).


def test_policy_loss_fd_gradient():
    torch.manual_seed(31)
    net = VelocityMLP(in_dim=8 * 1)  # small problem keeps FD numerics clean
    a = torch.randn(1, 8, 1)
    g = torch.Generator().manual_seed(32)
    eps = torch.randn(1, _DEFAULTS.num_mc_samples, 8, 1, generator=g)
    tau = draw_tau((1, _DEFAULTS.num_mc_samples), g)
    advantage = torch.tensor([1.0])
    m = _DEFAULTS.num_mc_samples
    t = tau.unsqueeze(-1).unsqueeze(-1)  # [1, m, 1, 1]
    x_t = (1.0 - t) * eps + t * a.unsqueeze(1)  # [1, m, 8, 1]
    x_t_flat = x_t.reshape(m, 8, 1)
    tau_flat = tau.reshape(m)
    v_star = SJFPOPolicy.compute_target_velocity(a, eps)  # [1, m, 8, 1]

    def v_hat_of(net_) -> torch.Tensor:
        return net_(x_t_flat, tau_flat, None).reshape(1, m, 8, 1)

    with torch.no_grad():
        l_old = compute_cfm_loss(v_hat_of(net), v_star, reduce_mode="sample")

    def policy_loss_value() -> torch.Tensor:
        l_new = compute_cfm_loss(v_hat_of(net), v_star, reduce_mode="sample")
        return compute_policy_loss(l_old.detach(), l_new, advantage)

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

    # (h) Euler sampler: ascending linspace(0, 1, steps+1) grid, update
    #     x <- x + dt * v_hat, zero-sampling deterministic given weights.


def test_euler_sampler_grid_update_and_zero_noise():
    torch.manual_seed(41)
    net = VelocityMLP(1, 1)
    seen_ts: list[float] = []

    class Probe(nn.Module):
        def forward(self, x_t, t, condition=None):
            seen_ts.append(float(t))
            return net(x_t, t, condition)

    out = sample_action(
        Probe(),
        (5, 1),
        None,
        num_inference_steps=64,
        generator=torch.Generator().manual_seed(42),
    )

    grid = torch.linspace(0.0, 1.0, 65)  # 65 nodes -> 64 ascending intervals
    assert len(seen_ts) == 64
    assert torch.allclose(torch.tensor(seen_ts), grid[:-1], atol=1e-7)
    assert all(seen_ts[i] < seen_ts[i + 1] for i in range(len(seen_ts) - 1)), (
        "t-grid must be ASCENDING from 0 (noise) to 1 (data)"
    )

    # Replicate the exact recursion x <- x + dt * v_hat(t_left) manually.
    eps = torch.randn(5, 1, generator=torch.Generator().manual_seed(42))
    x = eps.clone()
    with torch.no_grad():
        for i in range(64):
            x = x + (grid[i + 1] - grid[i]) * net(x, grid[i], None)
    assert torch.allclose(out, x, atol=1e-6), "sampler deviates from x <- x + dt*v_hat"

    # Zero-sampling path (eps = 0) is deterministic given the weights.
    z1 = sample_action(
        net, (5, 1), None, num_inference_steps=16, eps=torch.zeros(5, 1)
    )
    z2 = sample_action(
        net, (5, 1), None, num_inference_steps=16, eps=torch.zeros(5, 1)
    )
    assert torch.equal(z1, z2)

    # (i) SPACE-CONSISTENCY gate: sample action chunks in DCT coefficient space,
    #     map back via IDCT, clip to bounds; DCT(IDCT(x)) == x roundtrip < 1e-5.


def test_space_consistency_dct_pipeline():
    torch.manual_seed(51)
    policy = SJFPOFlowModel(
        action_dim=3,
        action_horizon=_DEFAULTS.action_horizon,
        hidden_dim=_DEFAULTS.hidden_dim,
        dit_depth=_DEFAULTS.dit_depth,
        dit_heads=_DEFAULTS.dit_heads,
        dit_mlp_ratio=_DEFAULTS.dit_mlp_ratio,
        dit_time_dim=_DEFAULTS.dit_time_dim,
        timestep_scale=_DEFAULTS.timestep_scale,
    )
    condition = torch.zeros(4, _DEFAULTS.hidden_dim)  # pre-fused state vector
    with torch.no_grad():
        coeffs = sample_action(
            policy,
            (4, 8, 3),
            condition,
            num_inference_steps=16,
            generator=torch.Generator().manual_seed(52),
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

    # (j) compute_cfm_loss reduce_mode: "scalar" vs "sample" shapes +
    #     validation (unified flow_loglik lives behind reduce_mode="sample").


def test_compute_cfm_loss_reduce_mode_contract():
    torch.manual_seed(61)
    v_hat = torch.randn(2, 4, 3)
    v_star = torch.randn(2, 4, 3)
    scalar = compute_cfm_loss(v_hat, v_star)
    assert scalar.dim() == 0
    per_sample = compute_cfm_loss(v_hat, v_star, reduce_mode="sample")
    assert per_sample.shape == (2, 4)
    # OFF sqrt convention: sample = sum/sqrt(D); mean over draws recovers the
    # scalar mean up to sqrt(D_flat).
    assert per_sample.mean().item() == pytest.approx(
        scalar.item() * math.sqrt(3), rel=1e-5
    )
    with pytest.raises(ValueError, match="reduce_mode"):
        compute_cfm_loss(v_hat, v_star, reduce_mode="mean")
    with pytest.raises(ValueError, match="shape"):
        compute_cfm_loss(v_hat, v_star[..., :2])


def test_compute_cfm_loss_bin_mode_parity():
    """Phase 0: per-bin sqrt losses average back to the sample path up to sqrt(K)."""
    torch.manual_seed(62)
    # Small errors so the clamp never binds on either path.
    v_hat = torch.randn(2, 4, 5, 3) * 0.1
    v_star = torch.randn(2, 4, 5, 3) * 0.1
    per_bin = compute_cfm_loss(v_hat, v_star, reduce_mode="bin")
    assert per_bin.shape == (2, 4, 5)
    per_sample = compute_cfm_loss(v_hat, v_star, reduce_mode="sample")
    # bin_k = sum_D/sqrt(D); sample = sum/sqrt(K*D) = bin.mean * sqrt(K).
    torch.testing.assert_close(per_bin.mean(dim=-1) * math.sqrt(5), per_sample)
    # Clamp binds on both paths (OFF cfm_loss_clamp).
    big = compute_cfm_loss(
        torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2) * 10.0, reduce_mode="bin"
    )
    assert bool((big == _DEFAULTS.cfm_clamp).all())


def test_sample_timesteps_modes_and_contract():
    assert sample_timesteps(16).shape == (16,)
    for mode in ("uniform", "normal"):
        t = sample_timesteps(4096, (TAU_LO, TAU_HI), mode=mode)
        assert t.shape == (4096,) and t.dtype == torch.float32
        assert bool(((t >= TAU_LO) & (t <= TAU_HI)).all()), mode
    u = sample_timesteps(4096, mode="uniform")
    assert u.mean().item() == pytest.approx(0.5 * (TAU_LO + TAU_HI), abs=0.03)
    g = sample_timesteps(4096, mode="normal")
    assert g.std().item() < u.std().item(), "normal should concentrate mid-window"
    b = sample_timesteps(
        4096, (TAU_LO, TAU_HI), mode="beta", alpha=1.5, beta=1.0
    )
    assert b.shape == (4096,) and b.dtype == torch.float32
    assert bool(((b >= TAU_LO) & (b <= TAU_HI)).all()), "beta scaled to [lo, hi]"
    with pytest.raises(ValueError, match="mode"):
        sample_timesteps(4, mode="logit-normal")


def test_sample_noise_modes_and_contract():
    assert sample_noise((2, 4, 3)).shape == (2, 4, 3)
    torch.manual_seed(7)
    z = sample_noise((2, 16, 8, 3), mode="normal")
    assert z.mean().item() == pytest.approx(0.0, abs=0.05)
    assert z.var().item() == pytest.approx(1.0, rel=0.1)
    w = sample_noise((2, 16, 8, 3), mode="uniform")
    assert bool(((w >= -math.sqrt(3)) & (w <= math.sqrt(3))).all())
    assert w.var().item() == pytest.approx(1.0, rel=0.1)
    with pytest.raises(ValueError, match="mode"):
        sample_noise((4,), mode="cauchy")
