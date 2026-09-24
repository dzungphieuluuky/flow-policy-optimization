"""Torch minibatch iteration helper."""

from __future__ import annotations

import torch

__all__ = ["mini_batch_indices"]


def mini_batch_indices(
    n: int, num_minibatches: int, device: torch.device | str, num_epochs: int = 1
):
    """Yield shuffled minibatch indices for each epoch."""
    if n < 1:
        raise ValueError(f"n={n} must be >= 1")
    n_mb = max(1, min(int(num_minibatches), n))
    device = torch.device(device)
    for _ in range(max(1, int(num_epochs))):
        perm = torch.randperm(n, device=device)
        for i in range(n_mb):
            start = (i * n) // n_mb
            stop = ((i + 1) * n) // n_mb
            yield perm[start:stop]
