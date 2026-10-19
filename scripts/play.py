#!/usr/bin/env python3
"""Play / evaluate a loaded myfpo policy, writing the rollout to an mp4.

Prints the per-episode metrics and the video path. Notebook callers pass
``--video_out`` and embed the result with ``IPython.display.Video``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from myfpo.evaluator import Evaluator, render_episode_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved myfpo policy and write the rollout to an mp4."
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
        "--video_out",
        help="Write the rollout to this path as an mp4. Default: <checkpoint>/play.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate.",
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

    # There is no render_mode here. The Evaluator only steps the env. The video
    # pass builds its own rgb_array env inside render_episode_frames.
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

        frames, ret = render_episode_frames(
            env_id=args.env,
            policy=policy_callable,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    out = Path(args.video_out) if args.video_out else Path(args.checkpoint) / "play.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimwrite(out, frames, fps=args.fps, codec="libx264", quality=7)
    print(f"  Video: {len(frames)} frames, return={ret:.2f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
