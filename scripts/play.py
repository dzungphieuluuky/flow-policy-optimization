#!/usr/bin/env python3
"""Evaluate a loaded lean-flow_policy_optimization policy and print the per-episode metrics.

The policy is stepped through the environment and the return, the length and
the jerk of every episode are printed. No video is written: the frames cost
more than the numbers are worth for a policy you are iterating on.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

# Match train.py. Set the CUDA precision and benchmark options before any
# module load.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = True
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )

from flow_policy_optimization.config import FlowPolicyOptimizationConfig
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy
from flow_policy_optimization.train import Evaluator, is_maniskill_id, load_checkpoint, register_gym_envs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved lean-flow_policy_optimization policy and print the per-episode metrics."
    )
    parser.add_argument(
        "--env",
        default="Hopper-v5",
        help="Gym env name.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory (contains checkpoint.pt).",
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

    import gymnasium as gym

    if is_maniskill_id(args.env):
        register_gym_envs()  # the single-env CPU path, for this CLI only

    env = gym.make(args.env)
    try:
        obs_dim = int(math.prod(env.observation_space.shape))
        action_dim = int(math.prod(env.action_space.shape))
        cfg = FlowPolicyOptimizationConfig()
        policy = FlowPolicyOptimizationPolicy(
            cfg,
            obs_dim,
            action_dim,
            np.asarray(env.action_space.low, dtype=np.float32),
            np.asarray(env.action_space.high, dtype=np.float32),
        ).to(args.device)

        print(f"Loading checkpoint from: {args.checkpoint}")
        step = load_checkpoint(args.checkpoint, policy)
        policy.eval()
        print(f"Checkpoint loaded (global_step={step})")

        evaluator = Evaluator(
            env=env,
            eval_seed=args.seed,
            max_episodes=args.episodes,
            max_steps=args.max_steps,
        )

        def policy_callable(obs: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                action = policy.select_action(obs)
            return action.detach().cpu().numpy()

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
