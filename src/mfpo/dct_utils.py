"""Orthonormal DCT-II / DCT-III utilities (pure torch, fp32).

Plan task 3. The FPO++ flow pipeline operates in ORTHONORMAL DCT-II
coefficient space of the scaled action chunk (plan Scope §Flow space,
BINDING): ``a_dct = DCT(a_chunk)``, noise/velocity/Euler all live in
coefficient space, and execution maps back via ``IDCT``. Orthonormality
(MᵀM = I) is required so the transform preserves Euclidean measure — a
Gaussian in coefficient space is a Gaussian in action space and log-densities
are invariant under an orthogonal change of variables (log-det = 0).

Implementation constraint (plan Must-NOT-do): NO scipy at runtime — the
transform is a plain matmul against a precomputed orthonormal DCT-II cosine
basis matrix held as a buffer-less plain tensor (module-level cache), fp32
throughout.

Definitions (orthonormal DCT-II), N = basis size:
    M[k, n] = sqrt(2/N) · c_k · cos(π·(2n+1)·k / (2N)),
    c_0 = 1/sqrt(2), c_k = 1 for k > 0.
Rows k are frequencies, columns n are time samples, so for a tensor with the
transform axis last: ``dct(x) = x @ Mᵀ`` and the inverse (DCT-III, the
transpose of an orthogonal matrix is its inverse) is ``idct(y) = y @ M``.

Tensor conventions:
    B: batch size
    H: action-chunk horizon (default 8)
    Da: action dimension (env-dependent)
    D: state dimension (default 128)
    N: DCT basis size (typically H=8 or Da)

All tensors are fp32.

Retained public API (dead in the src train path, kept per fn-consolidation D1):
``discounted_chunk_reward`` (tests/test_dct.py) and ``capture_training_stats``
(scripts/build_notebook.py expert-stats path); both listed in ``__all__``.
"""

from __future__ import annotations

import math

import einx
import torch

__all__ = [
    "dct",
    "idct",
    "discounted_chunk_reward",
    "capture_training_stats",
    "normalize_joint_targets",
    "denormalize_joint_targets",
]

# Module-level cache of precomputed orthonormal bases, keyed by size N.
# Plain tensors (buffer-less per plan); read-only after construction.
BASIS_CACHE: dict[int, torch.Tensor] = {}


def dct_basis_matrix(n: int) -> torch.Tensor:
    """Return the [n, n] orthonormal DCT-II basis M with MᵀM = I (fp32).

    Args:
        n: Basis size (sequence length).

    Returns:
        Orthonormal DCT-II basis matrix of shape [n, n].
    """
    m = BASIS_CACHE.get(n)
    if m is None:
        k = einx.id(
            "n -> n 1", torch.arange(n, dtype=torch.float32)
        )  # [n, 1] frequency rows
        t = einx.id(
            "n -> 1 n", torch.arange(n, dtype=torch.float32)
        )  # [1, n] time columns
        m = math.sqrt(2.0 / n) * torch.cos(
            math.pi * (2.0 * t + 1.0) * k / (2.0 * n)
        )  # [n, n]
        m[0].mul_(1.0 / math.sqrt(2.0))  # c_0 normalization → orthonormal rows
        BASIS_CACHE[n] = m
    return m


def apply_matrix_along_dim(x: torch.Tensor, dim: int, mat: torch.Tensor) -> torch.Tensor:
    """Right-multiply the slice of ``x`` along ``dim`` by ``mat``.

    Args:
        x: Input tensor of shape [..., N, ...] where N is the dimension to transform.
        dim: Dimension index to apply transform along.
        mat: Transform matrix of shape [N, N].

    Returns:
        Transformed tensor with same shape as x.
    """
    dim = dim % x.dim()
    if dim != x.dim() - 1:
        x = x.movedim(dim, -1)
    out = x @ mat  # [..., N] @ [N, N] -> [..., N]
    if dim != out.dim() - 1:
        out = out.movedim(-1, dim)
    return out


def dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Orthonormal DCT-II of ``x`` along ``dim`` via matmul (fp32).

    For chunk tensors [B, H=8, A] use ``dim=1``; default transforms the last
    axis.

    Args:
        x: Input tensor of shape [..., N, ...].
        dim: Dimension to transform. Default -1 (last dimension).

    Returns:
        DCT-II coefficients of same shape as x.
    """
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]).t())


def idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse transform (DCT-III) along ``dim``: matmul with M itself.

    M is orthogonal so M⁻¹ = Mᵀ; applying DCT-III to coefficients returns the
    original signal exactly up to float round-off.

    Args:
        x: DCT coefficients of shape [..., N, ...].
        dim: Dimension to transform. Default -1 (last dimension).

    Returns:
        Reconstructed signal of same shape as x.
    """
    return apply_matrix_along_dim(x, dim, dct_basis_matrix(x.shape[dim % x.dim()]))


# retained: test-only public API — exercised directly by tests/test_dct.py;
# not called by the src train path. Keep per fn-consolidation D1.
def discounted_chunk_reward(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Σ_t γ^t · r_t along the last axis (chunk length 8).

    Supports batched input [..., 8]; returns a tensor with the batch dims
    preserved (scalar for a bare [8] input).

    Args:
        rewards: Reward tensor of shape [..., 8].
        gamma: Discount factor.

    Returns:
        Discounted sum of shape [...] (last dimension reduced).
    """
    n = rewards.shape[-1]
    weights = gamma ** torch.arange(
        n, dtype=rewards.dtype, device=rewards.device
    )  # [8]
    return (rewards * weights).sum(dim=-1)  # [...]


# ---------------------------------------------------------------------------
# Joint target normalization (per-branch frozen std)
# ---------------------------------------------------------------------------

# Module-level frozen target stats, set via set_frozen_target_stats().
_frozen_target_stats: dict[str, torch.Tensor] | None = None


# retained: public notebook API — used by scripts/build_notebook.py (expert-stats path);
# not called by the src train path. Keep per fn-consolidation D1.
def capture_training_stats(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Compute per-branch std vectors from a list of expert demo batches.

    Each batch must have keys ``"actions"`` [B, H, Da] and ``"z_future"`` [B, H, D].
    Returns ``{"std_a": [Da], "std_s": [D]}`` with zeros as mu (flow targets are
    zero-mean in DCT space). Std is clamped to a minimum of 1e-6 to avoid division
    by zero.

    Args:
        batches: List of dicts with keys "actions" [B, H, Da] and "z_future" [B, H, D].

    Returns:
        Dict with "std_a" [Da] and "std_s" [D].
    """
    all_actions = torch.cat(
        [b["actions"] for b in batches], dim=0
    )  # [N, H, Da] -- list cat stays torch
    all_z_future = torch.cat([b["z_future"] for b in batches], dim=0)  # [N, H, D]
    std_a = (
        einx.id("... da -> ( ... ) da", all_actions).std(dim=0).clamp(min=1e-6)
    )  # [Da]
    std_s = (
        einx.id("... ds -> ( ... ) ds", all_z_future).std(dim=0).clamp(min=1e-6)
    )  # [D]
    return {"std_a": std_a, "std_s": std_s}


def set_frozen_target_stats(stats: dict[str, torch.Tensor]) -> None:
    """Freeze target normalization stats (called once from expert dataloader).

    Args:
        stats: Dict with "std_a" [Da] and "std_s" [D].
    """
    global _frozen_target_stats
    _frozen_target_stats = stats


def get_frozen_target_stats() -> dict[str, torch.Tensor]:
    """Return current frozen target stats; raises RuntimeError if unset.

    Returns:
        Dict with "std_a" [Da] and "std_s" [D].

    Raises:
        RuntimeError: If frozen target stats not initialized.
    """
    if _frozen_target_stats is None:
        raise RuntimeError(
            "Frozen target stats not initialized. "
            "Call set_frozen_target_stats() from the expert dataloader first."
        )
    return _frozen_target_stats


def normalize_joint_targets(
    A_dct: torch.Tensor, S_dct: torch.Tensor, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Normalize DCT targets by per-branch frozen std → [B, H, Da + D].

    ``A_dct`` has shape [B, H, Da], ``S_dct`` has shape [B, H, D].
    Each branch is divided by its own frozen std vector (broadcast over B, H).
    Returns concatenated ``[A_dct / std_a, S_dct / std_s]``.

    Args:
        A_dct: Action DCT coefficients [B, H, Da].
        S_dct: State DCT coefficients [B, H, D].
        stats: Dict with "std_a" [Da] and "std_s" [D].

    Returns:
        Concatenated normalized targets [B, H, Da+D].

    Raises:
        RuntimeError: If stats missing required keys.
    """
    if "std_a" not in stats or "std_s" not in stats:
        raise RuntimeError(
            "frozen target stats missing 'std_a' or 'std_s'. "
            "Call set_frozen_target_stats() first."
        )
    std_a = stats["std_a"]  # [Da]
    std_s = stats["std_s"]  # [D]
    A_norm = A_dct / std_a  # [B, H, Da] / [Da] -> [B, H, Da]
    S_norm = S_dct / std_s  # [B, H, D] / [D] -> [B, H, D]
    return einx.id("b h da, b h ds -> b h (da + ds)", A_norm, S_norm)  # [B, H, Da+D]


def denormalize_joint_targets(
    X_norm: torch.Tensor, stats: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert per-branch normalization → (A_dct [B,H,Da], S_dct [B,H,D]).

    Args:
        X_norm: Normalized joint targets [B, H, Da+D].
        stats: Dict with "std_a" [Da] and "std_s" [D].

    Returns:
        Tuple (A_dct [B, H, Da], S_dct [B, H, D]).

    Raises:
        RuntimeError: If stats missing required keys.
    """
    if "std_a" not in stats or "std_s" not in stats:
        raise RuntimeError(
            "frozen target stats missing 'std_a' or 'std_s'. "
            "Call set_frozen_target_stats() first."
        )
    std_a = stats["std_a"]  # [Da]
    std_s = stats["std_s"]  # [D]
    Da = std_a.shape[0]
    A_dct = X_norm[..., :Da] * std_a  # [B, H, Da] * [Da] -> [B, H, Da]
    S_dct = X_norm[..., Da:] * std_s  # [B, H, D] * [D] -> [B, H, D]
    return A_dct, S_dct
