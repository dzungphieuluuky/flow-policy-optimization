"""Shared setup for the behaviour pins.

A pin does one job. It runs a fixed computation and records every number the
computation produces. A later run replays the same computation and compares.

The rule is ``atol=0``. A refactor that moves a float by one part in a hundred
is a behaviour change. These scripts exist to make such a change impossible to
miss, so they must fail on any difference at all. Do not add a tolerance.

Do not import these helpers from ``src``. A pin that shares code with the code
it checks cannot catch a defect in the shared part.
"""

from __future__ import annotations

from pathlib import Path

import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from myfpo.configuration import MyfpoConfig

# The shape of the pin problem. Small on purpose: a pin must run in seconds so
# it can be the gate on every commit.
OBS, ACT, HORIZON, MC, N_OBS = 6, 3, 4, 4, 2
HIDDEN, TIME_EMBED, HEADS, DEPTH = 32, 16, 2, 1

PIN_DIR = Path(__file__).resolve().parent

__all__ = [
    "ACT", "HORIZON", "MC", "N_OBS", "OBS", "PIN_DIR",
    "build", "compare", "fixed_batch", "flatten", "pin_config",
]


def pin_config(**kw) -> MyfpoConfig:
    """The one config every pin uses.

    ``compile`` stays off. A compiled module is not reproducible across
    machines, and a pin that fails on the second GPU is not a gate.
    """
    base = dict(
        action_horizon=HORIZON,
        n_action_steps=HORIZON,
        num_mc_samples=MC,
        hidden_dim=HIDDEN,
        time_embed_dim=TIME_EMBED,
        dit_depth=DEPTH,
        dit_heads=HEADS,
        n_obs_steps=N_OBS,
        compile=False,
        report_to="none",
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (OBS,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (ACT,))},
    )
    base.update(kw)
    return MyfpoConfig(**base)


def build(cfg: MyfpoConfig, seed: int = 0):
    """Build the policy and the batch together, under one seed.

    The seed is set before the policy is built, because parameter construction
    order fixes the random stream. Moving one ``nn.Module`` to a different
    position in ``__init__`` changes every later draw. The pin catches that.
    """
    from myfpo.training import build_policy

    torch.manual_seed(seed)
    return build_policy(
        OBS, ACT, cfg, device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )


def fixed_batch(policy, n: int = 3, seed: int = 1) -> dict:
    """A batch whose ``(t, eps)`` pairs the policy already agreed on.

    The behaviour CFM losses come from the same policy, so a fresh forward pass
    must report a ratio of exactly one. The pin therefore measures the training
    step only, not the learning.
    """
    from myfpo.fpo_core import sample_timesteps

    cfg = policy.config
    torch.manual_seed(seed)
    states = torch.randn(n, cfg.n_obs_steps, OBS)
    actions = torch.randn(n, HORIZON, ACT)
    t = sample_timesteps((n, MC, HORIZON), cfg, "cpu", None).unsqueeze(-1)
    eps = torch.randn(n, MC, HORIZON, ACT)
    pad = torch.zeros(n, HORIZON, dtype=torch.bool)

    pre = policy.preprocessor({"observation.state": states})["observation.state"]
    old, old_token = policy.compute_behavior_cfm_losses(pre, actions, t, eps, pad)
    return {
        "observation.state": states,
        "action": actions,
        "action_is_pad": pad,
        "normalised_advantages": torch.randn(n, generator=torch.Generator().manual_seed(7)),
        "raw_returns": torch.randn(n, generator=torch.Generator().manual_seed(8)),
        "mc_t": t,
        "mc_eps": eps,
        "behavior_cfm_loss": old,
        "behavior_cfm_loss_token": old_token,
    }


def flatten(obj, prefix: str = "") -> dict:
    """Turn a nested result into ``dotted.key -> number``.

    Every leaf becomes a plain Python float or list. A tensor would pin its
    dtype and device as well as its value, which makes the pin fail for a
    reason that has nothing to do with behaviour.
    """
    out: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, torch.Tensor):
        out[prefix.rstrip(".")] = obj.detach().to(torch.float64).cpu().flatten().tolist()
    elif isinstance(obj, (list, tuple)):
        # Keep the list as one leaf. ``compare`` already walks it element by
        # element and names the index, so splitting it into per-index keys
        # would multiply the key count by the tensor length and report the same
        # differences more slowly.
        out[prefix.rstrip(".") or "value"] = [
            float(v) if isinstance(v, (bool, int, float)) else v for v in obj
        ]
    elif isinstance(obj, (bool, int, float)):
        out[prefix.rstrip(".")] = float(obj)
    elif obj is None:
        out[prefix.rstrip(".")] = None
    else:
        out[prefix.rstrip(".")] = repr(obj)
    return out


def _walk_equal(a, b, path: str, bad: list):
    """Compare two leaves with ``atol=0``.

    Two floating point numbers are equal only when they are bit-identical.
    ``==`` on floats already means that, so the check stays a plain compare.
    """
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk_equal(x, y, f"{path}[{i}]", bad)
        return
    if isinstance(a, float) and isinstance(b, float):
        if a != b:  # atol=0 is the point of this module
            bad.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        bad.append(f"{path}: {a!r} != {b!r}")


def compare(name: str, new: dict, old: dict | None) -> int:
    """Report every difference. Return the count, so the shell can gate on it.

    A missing key and an extra key are both failures. A key that appears or
    disappears means the computation changed shape, which is a behaviour change
    even when no value moved.
    """
    if old is None:
        print(f"{name}: no baseline. Run with --write first.")
        return 1

    new_keys, old_keys = set(new), set(old)
    bad = []
    for key in sorted(old_keys - new_keys):
        bad.append(f"missing key: {key}")
    for key in sorted(new_keys - old_keys):
        bad.append(f"new key: {key}")
    for key in sorted(new_keys & old_keys):
        _walk_equal(new[key], old[key], key, bad)

    if bad:
        print(f"{name}: {len(bad)} difference(s) over {len(old_keys)} pinned keys")
        for line in bad[:40]:
            print(f"  {line}")
        if len(bad) > 40:
            print(f"  ... {len(bad) - 40} more")
        return len(bad)
    print(f"{name}: {len(old_keys)} keys, 0 mismatches")
    return 0