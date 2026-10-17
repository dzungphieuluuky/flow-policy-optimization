"""Pin the training step: every metric, every gradient, at ``atol=0``.

What this covers
    One forward pass of ``SJFPOPolicy`` on a fixed batch, the full ``info``
    dict it returns, and the gradient of every named parameter. It also runs
    the same pass a second time under a fresh policy of the other architecture,
    so a change in one actor cannot pass unnoticed because the other moved too.

What this does not cover
    Rollout collection, the eval loop, or anything in ``denoising_mdp.py``.

Usage
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py --write
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py

The baseline is a binary file and is not tracked. See ``README.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import build, compare, fixed_batch, flatten, pin_config  # noqa: E402

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


def run_one(arch: str) -> dict:
    """One forward pass and one backward pass, flattened to plain numbers."""
    cfg = pin_config()
    policy = build(cfg)
    if arch != cfg.actor_arch:
        import dataclasses

        cfg = dataclasses.replace(cfg, actor_arch=arch)
        policy = build(cfg)

    batch = fixed_batch(policy)
    loss, info = policy(batch, compute_info=True)
    loss.backward()

    out = {"loss": float(loss.detach())}
    out.update(flatten(info, "info."))
    out.update(flatten(gradients(policy), "grad."))
    return out


def collect() -> dict:
    """Both architectures, under one key space.

    Prefixing by architecture keeps the two apart. A key that exists for one
    architecture and not the other is a difference the pin must report.
    """
    out = {}
    for arch in ("dit", "resmlp"):
        for key, value in run_one(arch).items():
            out[f"{arch}.{key}"] = value
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