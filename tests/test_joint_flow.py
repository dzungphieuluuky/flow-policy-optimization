"""Task 7 TDD — JointFlowTransformer per spec."""

import pytest
import torch

from mfpo.config import MFPOConfig

B, H, D, K = 4, 8, 128, 5
Da = 7


def _make_joint():
    from mfpo.networks import JointFlowTransformer

    return JointFlowTransformer(z_dim=D)


def test_forward_shape_split():
    mod = _make_joint()
    z_t = torch.randn(B, H, Da + D)
    t = torch.rand(B)
    Z = torch.randn(B, H, D)
    v = mod(z_t, t, Z)
    assert v.shape == (B, H, Da + D)
    va, vs = v.split([Da, D], dim=-1)
    assert va.shape == (B, H, Da)
    assert vs.shape == (B, H, D)


def test_loss_decreases():
    from mfpo.networks import JointFlowTransformer, PerBranchFlowLoss

    torch.manual_seed(0)
    mod = JointFlowTransformer(z_dim=D)
    loss_mod = PerBranchFlowLoss()
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    z0 = torch.zeros(B, H, Da + D)
    Xt = torch.ones(B, H, Da + D)
    t = torch.full((B,), 0.5)
    z_t = 0.5 * Xt
    u_star = Xt - z0
    Z = torch.zeros(B, H, D)
    with torch.no_grad():
        v = mod(z_t, t, Z)
        l0 = loss_mod(v, u_star).item()
    for _ in range(20):
        opt.zero_grad()
        v = mod(z_t, t, Z)
        loss = loss_mod(v, u_star)
        loss.backward()
        opt.step()
    with torch.no_grad():
        v = mod(z_t, t, Z)
        l1 = loss_mod(v, u_star).item()
    assert l1 < l0 * 0.95, f"not decreased >5%: {l0} -> {l1}"


def test_t_conditioning_varies():
    mod = _make_joint()
    z_t = torch.randn(B, H, Da + D)
    Z = torch.randn(B, H, D)
    v0 = mod(z_t, torch.zeros(B), Z)
    v1 = mod(z_t, torch.ones(B), Z)
    assert not torch.allclose(v0, v1)


def test_context_conditioning_varies():
    mod = _make_joint()
    z_t = torch.randn(B, H, Da + D)
    t = torch.rand(B)
    Z0 = torch.randn(B, H, D)
    Z1 = torch.randn(B, H, D)
    v0 = mod(z_t, t, Z0)
    v1 = mod(z_t, t, Z1)
    assert not torch.allclose(v0, v1)


def test_determinism():
    mod = _make_joint()
    z_t = torch.randn(B, H, Da + D)
    t = torch.rand(B)
    Z = torch.randn(B, H, D)
    v1 = mod(z_t, t, Z)
    v2 = mod(z_t, t, Z)
    assert torch.equal(v1, v2)


def test_t_convention_pin():
    z0 = torch.randn(B, H, Da + D)
    Xt = torch.randn(B, H, Da + D)
    for t_val in [0.0, 1.0]:
        t = torch.full((B,), t_val)
        z_t = (1 - t.view(-1, 1, 1)) * z0 + t.view(-1, 1, 1) * Xt
        if t_val == 0:
            assert torch.allclose(z_t, z0)
        else:
            assert torch.allclose(z_t, Xt)
