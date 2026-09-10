"""Synthetic learning curve generator for testing analysis module.

rng seed 7; methods strong/weak/tie × seeds 0-4 × 20 eval points on the 50k-1M grid.
base curve: 100 * (1 - exp(-step / 200000))
strong adds: 200 * sigmoid((step - 300000) / 50000)
weak/tie share the formula with distinct rng streams.
Gaussian noise σ=5.
"""

from __future__ import annotations

import numpy as np


def generate_curves() -> dict[str, list[dict]]:
    """Generate synthetic learning curves for testing.

    Returns
    -------
    dict[str, list[dict]]
        Dictionary mapping method name to list of seed results.
        Each seed result is a dict with keys: 'seed', 'eval_points', 'returns'.
    """
    rng = np.random.default_rng(7)

    # 20 eval points from 50000 to 1000000
    eval_points = np.linspace(50000, 1000000, 20, dtype=int)

    # Base curve: 100 * (1 - exp(-step / 200000))
    base = 100 * (1 - np.exp(-eval_points / 200000))

    # Strong bonus: 200 / (1 + exp(-(step - 300000) / 50000))
    strong_bonus = 200 / (1 + np.exp(-(eval_points - 300000) / 50000))

    methods = ["strong", "weak", "tie"]
    seeds = range(5)

    result = {}

    for method in methods:
        method_data = []
        for seed in seeds:
            # Use distinct rng streams for each method/seed combination
            seed_rng = np.random.default_rng(7 + hash((method, seed)) % 10000)

            if method == "strong":
                # strong = base + strong_bonus + noise
                returns = (
                    base + strong_bonus + seed_rng.normal(0, 5, size=len(eval_points))
                )
            elif method == "weak":
                # weak = base - 50 + noise (slightly worse)
                returns = base - 50 + seed_rng.normal(0, 5, size=len(eval_points))
            else:  # tie
                # tie = base + noise (same as base)
                returns = base + seed_rng.normal(0, 5, size=len(eval_points))

            method_data.append(
                {
                    "seed": seed,
                    "eval_points": eval_points.tolist(),
                    "returns": returns.tolist(),
                }
            )
        result[method] = method_data

    return result
