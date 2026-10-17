"""Pin the training step: every metric, every gradient, at ``atol=0``.

What this covers
    One forward pass of ``SJFPOPolicy`` on a fixed batch, the full ``info``
    dict it returns, and the gradient of every named parameter. It runs three
    configurations: the ``dit`` actor, the ``resmlp`` actor, and ``dit`` at a
    clamp low enough to saturate. A change in one actor cannot pass unnoticed
    because the other moved too, and the clamp arm exists because the default
    bound hides every clamp diagnostic at 0.0.

What this does not cover
    Rollout collection, the eval loop, or anything in ``denoising_mdp.py``.

Usage
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py --write
    PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py

The baseline is a binary file and is not tracked. See ``README.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
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


def run_one(cfg) -> dict:
    """One forward pass and one backward pass, flattened to plain numbers."""
    policy = build(cfg)
    batch = fixed_batch(policy)
    loss, info = policy(batch, compute_info=True)
    loss.backward()

    out = {"loss": float(loss.detach())}
    out.update(flatten(info, "info."))
    out.update(flatten(gradients(policy), "grad."))
    return out


# The configurations the pin covers. Each label is a key prefix.
#
# The clamp arm is not optional. At the default ``cfm_clamp=20.0`` both clamp
# diagnostics read 0.0, because no loss in this fixture reaches 20.0. A pin
# that only ran the default could not detect any change to the saturation
# figures. On 2026-10-17 that blindness hid a real defect: the diagnostic read
# the per-sample loss while the ratio clamped the per-token one, and the two
# agreed only because both were zero. The 0.5 arm puts the share strictly
# inside (0, 1), so the two axes are distinguishable.
CONFIGS = {
    "dit": pin_config(),
    "resmlp": dataclasses.replace(pin_config(), actor_arch="resmlp"),
    "dit.clamp0.5": dataclasses.replace(pin_config(), cfm_clamp=0.5),
}


def collect() -> dict:
    """Every configuration, under one key space.

    Prefixing by label keeps the arms apart. A key that exists for one arm and
    not another is a difference the pin must report.
    """
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