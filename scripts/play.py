#!/usr/bin/env python3
"""Evaluate a loaded myfpo policy and print the per-episode metrics.

The policy is stepped through the environment and the return, the length and
the jerk of every episode are printed. No video is written: the frames cost
more than the numbers are worth for a policy you are iterating on.
"""

from __future__ import annotations

import argparse
import sys


import torch

# Match train.py. Set the CUDA precision and benchmark options before any
# module load.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.backends.cudnn.benchmark = True
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )

import numpy as np

from myfpo.checkpointing import load_policy_checkpoint
from myfpo.evaluator import Evaluator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved myfpo policy and print the per-episode metrics."
    )
    parser.add_argument(
        "--env",
        default="Hopper-v5",
        help="Gym env name (must be in KNOWN_ENVS).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory (contains config.json + model.safetensors + optimizer.pt + counters.json).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic rollout.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for policy inference.",
    )

    args = parser.parse_args(argv)

    print(f"Loading checkpoint from: {args.checkpoint}")
    policy, _, counters = load_policy_checkpoint(args.checkpoint, device=args.device)
    policy.eval()
    print(f"Checkpoint loaded (global_step={counters.get('global_step', '?')})")

    import gymnasium as gym

    # The evaluator steps the env itself. No render_mode is needed.
    env = gym.make(args.env)
    try:
        evaluator = Evaluator(
            env=env,
            eval_seed=args.seed,
            max_episodes=args.episodes,
            max_steps=args.max_steps,
        )

        def policy_callable(obs: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                # Bare [obs]. select_action buffers the frame into a window
                # itself. A rank-2 input is read as a [T, obs] window, then
                # rejected unless T == n_obs_steps.
                batch = {"observation.state": torch.as_tensor(obs, dtype=torch.float32)}
                return policy.select_action(batch, start_from_zero=True).cpu().numpy()

        print(f"Running {args.episodes} episode(s)")
        for r in evaluator.evaluate(policy_callable):
            print(
                f"  Episode {r['episode']}: return={r['return']:.2f}, "
                f"length={r['length']}, success={r['success']}, jerk={r['jerk']}"
            )

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
