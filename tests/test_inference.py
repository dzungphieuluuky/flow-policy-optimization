import torch
import pytest
from mfpo.inference import euler_integrate_joint, decode_joint


def test_identity_recovery():
    B, H, Dad, steps = 2, 8, 135, 16
    Da = 7
    torch.manual_seed(0)
    X_star = torch.randn(B, H, Dad)
    z0 = torch.randn(B, H, Dad)
    Z = torch.randn(B, 16, 128)

    def v_id(z, t, Z, T=None):
        return X_star - z0

    X_hat = euler_integrate_joint(v_id, z0, Z, steps=steps)
    assert torch.allclose(X_hat, X_star, atol=1e-4), (
        f"max {(X_hat - X_star).abs().max()}"
    )


def test_steps_zero():
    with pytest.raises(ValueError):
        euler_integrate_joint(
            lambda z, t, Z: z, torch.randn(2, 2, 4), torch.randn(2, 2, 4), steps=0
        )


def test_deterministic():
    B, H = 2, 4
    z0 = torch.randn(B, H, 10)
    Z = torch.randn(B, 4, 8)

    def v(z, t, Z):
        return torch.zeros_like(z)

    a = euler_integrate_joint(v, z0, Z, steps=8)
    b = euler_integrate_joint(v, z0, Z, steps=8)
    assert torch.equal(a, b)


def test_decode_gripper():
    B, H, Da, D = 2, 8, 7, 128
    X_hat = torch.randn(B, H, Da + D)
    A_hat, S_hat = decode_joint(X_hat)
    assert A_hat.shape == (B, H, Da)
    assert S_hat.shape == (B, H, D)
    grip = A_hat[..., -2:]
    uniq = set(torch.unique(grip).tolist())
    assert uniq.issubset({0.0, 1.0}), f"grip not discrete {uniq}"


def test_zero_sampling():
    B, H = 2, 4
    z0 = torch.zeros(B, H, 6)
    Z = torch.randn(B, 4, 6)

    def v(z, t, Z):
        return torch.ones_like(z)

    X1 = euler_integrate_joint(v, z0, Z, steps=8)
    X2 = euler_integrate_joint(v, z0, Z, steps=8)
    assert torch.equal(X1, X2)
