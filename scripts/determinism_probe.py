"""Measure the eval noise floor, so a reported gain can be read against it.

The evaluator uses fixed seeds, so two runs of one checkpoint over the same
seeds must agree. They do not always agree, because the rollout depends on CPU
thread scheduling. The gap is the noise floor. A comparison smaller than this
floor means nothing.

The script runs the same checkpoint twice and reports the gap in the mean, in
the median, and in the interval width. Run it before trusting any small delta.

Usage:
    .venv/bin/python scripts/determinism_probe.py \\
        --env Hopper-v5 --checkpoint outputs/<run>/checkpoints/best --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from myfpo.checkpointing import load_policy_checkpoint
from myfpo.evaluator import Evaluator
from myfpo.training import EvalPolicy
from myfpo.layers import resolve_device, set_seed


def run_once(policy, evaluator, eval_seed: int) -> dict[str, float]:
    """One full eval pass, summarized."""
    results = evaluator.evaluate(EvalPolicy(policy), out_dir=None)
    summary = evaluator.summarize(results, seed=eval_seed)
    summary["episode_returns"] = [float(r["return"]) for r in results]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the JSON result here")
    args = parser.parse_args()

    import gymnasium as gym

    set_seed(args.seed)
    device = resolve_device()
    env = gym.make(args.env)
    low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    if args.checkpoint is None:
        from myfpo.configuration import MyfpoConfig
        from myfpo.training import make_policy

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        cfg = MyfpoConfig(compile=False)
        policy = make_policy(
            obs_dim, act_dim, cfg, device=device, action_low=low, action_high=high
        )
        source = "fresh policy"
    else:
        policy, _opt, _counters = load_policy_checkpoint(
            args.checkpoint, device=device, action_low=low, action_high=high
        )
        source = str(args.checkpoint)

    evaluator = Evaluator(env, eval_seed=args.eval_seed, max_episodes=args.episodes)

    means, medians, widths, n_ep = [], [], [], []
    for _ in range(max(1, args.repeats)):
        s = run_once(policy, evaluator, args.eval_seed)
        means.append(s["return_mean"])
        medians.append(s["return_median"])
        widths.append(s["return_ci_high"] - s["return_ci_low"])
        n_ep.append(s["return_n"])

    def spread(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.max() - arr.min())

    out = {
        "env": args.env,
        "source": source,
        "episodes": args.episodes,
        "repeats": args.repeats,
        "return_mean_per_repeat": means,
        "return_median_per_repeat": medians,
        "ci_width_per_repeat": widths,
        "mean_noise_floor": spread(means),
        "median_noise_floor": spread(medians),
        "ci_width_median": float(np.median(widths)) if widths else float("nan"),
        "episodes_finite": n_ep[0] if n_ep else 0,
        "note": (
            "mean_noise_floor is the same-checkpoint eval gap. A gain below it "
            "is inside the noise. ci_width_median is the within-window "
            "episode spread, which the stopper now reads."
        ),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())