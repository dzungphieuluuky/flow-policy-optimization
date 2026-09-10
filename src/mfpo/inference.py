"""MFPO inference — joint Euler ODE 0→1 + IDCT decode.

NOTEBOOK-ONLY surface (kept per deslop plan default-KEEP): no prod module
imports this file. Live consumers are tests/test_inference.py and the
notebook builder (scripts/build_notebook.py). Do NOT merge into fpo_core:
Euler direction mismatch (ascending 0→1 here vs descending there).

CELLORDER: 7

Provides the deterministic inference path: Euler integration of the joint
flow transformer from noise to data (ascending t=0→1)
followed by denormalize + IDCT and discrete gripper clipping.

Tensor conventions (MuJoCo flat Box):
    B: batch size
    N: context length (16)
    H: horizon / chunk length (8)
    Da: action dimension (7: xyz + axis-angle/cnt + gripper 2)
    Ds: state/latent dimension (128, equals D)
    D: Z_dim (128)
    K: spectral tokens (5)
    Dk: spectral projector dim (64)

Sequence: Z [B,N,D] + T_FFT [B,5,64] ──condition──► v_theta(z_t,t) ──► z_{k+1}
          z0 ~ N(0,I) [B,H,Da+Ds] ──Euler steps 0→1──► X_hat [B,H,Da+Ds]
          X_hat ──split+denormalize+IDCT──► A_hat [B,H,Da], S_hat [B,H,Ds] (+ gripper {0,1})
"""

from __future__ import annotations

import einx
import torch

from mfpo import config
from mfpo.dct_utils import denormalize_joint_targets, get_frozen_target_stats, idct


def query_velocity(v_theta, z, t, Z, T_FFT):
    """Call v_theta with 4-arg/3-arg tolerance (TypeError fallback)."""
    try:
        if T_FFT is not None:
            return v_theta(z, t, Z, T_FFT)  # [B, H, Da+Ds]
        return v_theta(z, t, Z)  # [B, H, Da+Ds]
    except TypeError:
        return v_theta(z, t, Z)  # [B, H, Da+Ds] — signature fallback


def run_euler_sweep(v_theta, z0, Z, T_FFT, steps):
    """Advance z0 through `steps` uniform Euler updates from t=0 to t=1."""
    B = z0.shape[0]  # B
    z = z0.clone()  # [B, H, Da+Ds] — clone so caller retains z0
    dt = 1.0 / steps  # scalar float
    for k in range(steps):  # K iterations (16 or 8)
        t = torch.full((B,), k / steps, device=z0.device, dtype=z0.dtype)  # [B]
        v = query_velocity(v_theta, z, t, Z, T_FFT)  # [B, H, Da+Ds]
        z = z + dt * v  # [B, H, Da+Ds]
    return z  # [B, H, Da+Ds]


def euler_integrate_joint(v_theta, z0: torch.Tensor, Z: torch.Tensor, T_FFT: torch.Tensor | None = None, steps: int | None = None, **kwargs) -> torch.Tensor:
    """Integrate joint flow ODE dz/dt = v_theta(z,t,cond) from t=0 to t=1.

    Uses forward Euler with uniform spacing. Ascending convention
    (0=noise → 1=data)::

        t_k = k / steps, dt = 1/steps
        z_{k+1} = z_k + dt * v_theta(z_k, t_k, Z, [T_FFT])

    Args:
        v_theta: Velocity network `v(z,t,Z[,T_FFT]) -> v` with signature
            ``(z [B,H,Da+Ds], t [B], Z [B,N,D], T_FFT [B,5,64]|None) -> [B,H,Da+Ds]``.
            Both 3-arg and 4-arg variants are tolerated (TypeError fallback).
        z0: Initial noise sample. Shape ``[B, H, Da+Ds]``.
        Z: Context latent from fusion. Shape ``[B, N, D]``.
        T_FFT: Optional spectral tokens. Shape ``[B, 5, 64]`` or ``None``.
        steps: Number of Euler steps. Defaults to 16.
        **kwargs: Forwarded to ``v_theta`` if accepted (unused in tests, kept for compat).

    Returns:
        Integrated trajectory ``X_hat`` of shape ``[B, H, Da+Ds]``.

    Raises:
        ValueError: If ``steps <= 0``.

    Example:
        >>> X_hat = euler_integrate_joint(joint_flow, z0, Z, T_FFT, steps=16)  # [B,8,135]
    """
    if steps is None:
        steps = 16  # scalar int
    if steps <= 0:
        raise ValueError(f"steps must be >0, got {steps}")
    return run_euler_sweep(v_theta, z0, Z, T_FFT, steps)  # [B, H, Da+Ds]


def resolve_joint_stats(X_hat, stats):
    """Fetch frozen per-branch stats or fall back to unit stats for tests."""
    if stats is not None:
        return stats  # caller-supplied {std_a [Da], std_s [Ds]}
    try:
        return get_frozen_target_stats()  # {std_a [Da], std_s [Ds]}
    except RuntimeError:
        Da = X_hat.shape[-1] - 128  # Da inferred from last dim
        if Da <= 0:
            Da = 7  # default action dim
        return {
            "std_a": torch.ones(Da, device=X_hat.device),  # [Da]
            "std_s": torch.ones(128, device=X_hat.device),  # [Ds=128]
        }


def discretize_gripper(A_hat):
    """Discretize last two action dims to {0,1}; pass through if <2 dims."""
    if A_hat.shape[-1] < 2:  # gripper invariant needs ≥2 dims
        return A_hat  # [B, H, Da] unchanged
    grip = A_hat[..., -2:]  # [B, H, 2]
    grip = (grip > 0.5).float()  # [B, H, 2] discrete {0,1}
    return einx.id("b h da1, b h da2 -> b h (da1 + da2)", A_hat[..., :-2], grip)  # [B, H, Da]


def decode_joint(
    X_hat: torch.Tensor,
    stats: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denormalize + IDCT decode joint prediction to action and state.

    Inverts the joint DCT target pipeline: ``X_hat [B,H,Da+Ds]``
    → per-branch denormalize → IDCT (dim=1) → ``A_hat [B,H,Da]``, ``S_hat [B,H,Ds]``.
    Actions are soft-clipped to ``[-1.5, 1.5]`` then gripper discretized.

    Args:
        X_hat: Predicted joint trajectory in normalized DCT space.
            Shape ``[B, H, Da+Ds]`` where ``Ds`` is typically 128 (D).
        stats: Frozen per-branch stats ``{std_a [Da], std_s [Ds]}``. When ``None``,
            fetches from ``get_frozen_target_stats()``; if unavailable (unit tests),
            falls back to unit stats (all ones).

    Returns:
        Tuple ``(A_hat, S_hat)`` where
        ``A_hat`` has shape ``[B, H, Da]`` with ``A_hat[...,-2:] ∈ {0,1}`` discrete,
        ``S_hat`` has shape ``[B, H, D]`` auxiliary future state.

    Example:
        >>> A_hat, S_hat = decode_joint(X_hat)  # X_hat [B,8,135] → A [B,8,7] S [B,8,128]
    """
    stats = resolve_joint_stats(X_hat, stats)  # {std_a [Da], std_s [Ds]}
    A_dct, S_dct = denormalize_joint_targets(X_hat, stats)  # [B,H,Da] , [B,H,Ds]
    A_hat = idct(A_dct, dim=1)  # [B, H, Da] — orthonormal IDCT
    S_hat = idct(S_dct, dim=1)  # [B, H, Ds]
    A_hat = A_hat.clamp(-1.5, 1.5)  # [B, H, Da] — soft clip before discretization
    A_hat = discretize_gripper(A_hat)  # [B, H, Da] gripper ∈ {0,1}
    return A_hat, S_hat  # [B,H,Da] , [B,H,Ds]
