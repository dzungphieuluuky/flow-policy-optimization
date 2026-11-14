"""Pin the training step: the loss, every metric, every gradient, at ``atol=0``.

What this covers
    One forward and one backward pass of the lean policy on a fixed batch, and
    every gradient of every named parameter. One arm only: the lean pole has no
    clamp diagnostics to force a second configuration.

What this does not cover
    Rollout collection, or the eval loop.

Usage
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py --write
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py

See ``README.md`` for which baseline is tracked and how to regenerate it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import build, compare, fixed_batch, flatten, pin_config

BASELINE = Path(__file__).with_name("train_old.pt")


def gradients(policy) -> dict:
    """Every named gradient, so a change in the backward pass cannot hide.

    A missing gradient is a value of ``None``. A zero gradient is ``0.0``.
    Those are different claims, so the pin records them differently.
    """
    out = {}
    for name, param in policy.named_parameters():
        if param.grad is None:
            out[name] = None
        else:
            out[name] = param.grad.detach().to(torch.float64).flatten().tolist()
    return out


def run_one(cfg) -> dict:
    """One forward and one backward, flattened to plain numbers."""
    from myfpo.myfpo import compute_loss

    policy = build(cfg)
    batch = fixed_batch(policy)
    loss, metrics = compute_loss(policy, batch, torch.device("cpu"))
    loss.backward()

    out = {"loss": float(loss.detach())}
    out.update(flatten(metrics, "metrics."))
    out.update(flatten(gradients(policy), "grad."))
    return out


CONFIGS = {"default": pin_config()}


def collect() -> dict:
    """Every configuration, under one key space."""
    out = {}
    for label, cfg in CONFIGS.items():
        for key, value in run_one(cfg).items():
            out[f"{label}.{key}"] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="Rewrite the baseline from the current code. Use only on purpose.",
    )
    args = parser.parse_args()

    new = collect()
    if args.write:
        torch.save(new, BASELINE)
        print(f"wrote {BASELINE} ({len(new)} keys)")
        return 0

    old = torch.load(BASELINE, weights_only=False) if BASELINE.exists() else None
    return 1 if compare("pin_train", new, old) else 0


if __name__ == "__main__":
    raise SystemExit(main())
