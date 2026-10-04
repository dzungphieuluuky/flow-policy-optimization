#!/usr/bin/env python3
"""Play / evaluate a loaded SJFPO policy with render mode video output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Match train.py: CUDA precision / benchmark setup before any module load
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from spectral_joint_policy.checkpointing import load_policy_checkpoint  # noqa: E402
from spectral_joint_policy.evaluator import Evaluator, render_episode_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SJFPO policy with video frames."
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
        "--render_mode",
        default="rgb_array",
        choices=["rgb_array", "human", "ansi"],
        help="Gym render mode.",
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

    import gymnasium as gym  # noqa: E402

    env = gym.make(args.env, render_mode=args.render_mode)
    try:
        evaluator = Evaluator(
            env=env,
            eval_seed=args.seed,
            max_episodes=args.episodes,
            max_steps=args.max_steps,
        )

        def policy_callable(obs: np.ndarray) -> np.ndarray:
            import torch

            with torch.inference_mode():
                batch = {
                    "observation.state": torch.as_tensor(
                        obs, dtype=torch.float32
                    ).unsqueeze(0)
                }
                action = policy.select_action(batch, zero_sampling=True)
                return action.squeeze(0).cpu().numpy()

        print(
            f"Running {args.episodes} episode(s) with render_mode={args.render_mode!r}"
        )
        results = evaluator.evaluate(policy_callable)
        for r in results:
            print(
                f"  Episode {r['episode']}: return={r['return']:.2f}, "
                f"length={r['length']}, success={r['success']}, jerk={r['jerk']}"
            )

        if args.render_mode == "rgb_array":
            frames, ret = render_episode_frames(
                env_id=args.env,
                policy=policy_callable,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            # Note: frames returned as numpy arrays; caller can save with imageio/ffmpeg
            print(f"  First episode frames: {len(frames)} frames, return={ret:.2f}")
            print(
                f"  (Save frames manually to video if needed; frame count: {len(frames)})"
            )
        else:
            print(
                f"  Render mode '{args.render_mode}': video output handled by env window/display."
            )

    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
