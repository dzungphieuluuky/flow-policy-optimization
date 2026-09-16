"""Bin-order sensitivity of spectral pooling (networks.SpectralAttentionPool)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.networks import SpectralAttentionPool  # noqa: E402


def test_attention_pooling_sees_bin_order():
    torch.manual_seed(0)
    pool = SpectralAttentionPool(obs_dim=4, hidden_dim=8, time_dim=8, n_heads=2)
    cond = torch.randn(2, 5, 4)
    temb = torch.randn(2, 8)
    pooled, _ = pool(cond, temb)
    pooled_perm, _ = pool(cond[:, torch.tensor([4, 3, 2, 1, 0]), :], temb)
    assert torch.isfinite(pooled).all()
    # Frequency codes make the pooling permutation-sensitive; bare tokens with
    # a single query would return the identical vector for any bin order.
    assert float((pooled - pooled_perm).abs().max()) > 1e-5
