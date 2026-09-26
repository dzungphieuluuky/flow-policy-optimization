"""Train the SJFPO MLP actor on a state-based MuJoCo environment.

A thin front end over :func:`sjfpo.train_sjfpo.train_sjfpo_run` that swaps the
policy class for :class:`~sjfpo.networks_mujoco.SJFPOPolicyMLP`. The collector,
GAE, CFM loss, ASPO trust region, optimizers, and checkpointing are all the
existing code paths — only the actor network differs from a default run.

The registry in ``configs/envs.yaml`` is the gate for which envs are accepted:
an id must be present and must not carry a non-MuJoCo ``requires`` entry, so
the Adroit ids and the classic-control ids are rejected up front rather than
failing later on a shape mismatch.

    python src/train_mujoco.py --env Hopper-v5 --total_steps 200000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

# Make src/ importable before any sjfpo.* import.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import gymnasium as gym  # noqa: E402

from sjfpo.configuration_sjfpo import SJFPOConfig  # noqa: E402
from sjfpo.networks_mujoco import SJFPOPolicyMLP  # noqa: E402
from sjfpo.train_sjfpo import train_sjfpo_run  # noqa: E402
from sjfpo.utils import detect_runtime_env, default_runs_base  # noqa: E402

ENV_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "envs.yaml"


def load_mujoco_envs() -> dict[str, dict]:
    """Load the env registry, keeping only the state-based MuJoCo entries.

    Returns:
        Mapping of env id to its registry entry.

    Raises:
        RuntimeError: If PyYAML is missing or ``configs/envs.yaml`` is absent.
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("pyyaml required to read configs/envs.yaml") from e
    if not ENV_REGISTRY_PATH.is_file():
        raise RuntimeError(f"missing env registry: {ENV_REGISTRY_PATH}")
    registry = yaml.safe_load(ENV_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    # ``requires: gymnasium`` / ``gymnasium-robotics`` marks the non-MuJoCo
    # entries; everything else in this file is a MuJoCo env id.
    return {
        k: v
        for k, v in registry.items()
        if isinstance(v, dict) and "requires" not in v
    }


def build_config(args: argparse.Namespace) -> SJFPOConfig:
    """Build the training config from the parsed arguments.

    The MLP defaults are deliberately smaller than the DiT ones: the channel
    MLP, not attention, dominates the parameter count, so the saving comes from
    halving ``dit_mlp_ratio`` and halving the depth rather than from dropping
    attention.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The configuration to train with.
    """
    return dataclasses.replace(
        SJFPOConfig(),
        dit_depth=args.depth,
        dit_mlp_ratio=args.mlp_ratio,
        hidden_dim=args.hidden_dim,
        n_action_steps=args.n_action_steps,
        action_horizon=args.action_horizon,
        use_dct_transform=not args.no_dct,
        num_envs=args.num_envs,
        chunks_per_update=args.chunks_per_update,
    )


def resolve_dims(env_id: str, env: gym.Env) -> tuple[int, int]:
    """Cross-check the registry dimensions against the live env space.

    Args:
        env_id: Environment id being trained on.
        env: The constructed environment.

    Returns:
        Tuple of ``(obs_dim, action_dim)``.

    Raises:
        ValueError: If the registry disagrees with the live space, which
            usually means a gymnasium version change.
    """
    entry = load_mujoco_envs()[env_id]
    reg = (int(entry["obs_dim"]), int(entry["action_dim"]))
    live = (
        int(math.prod(env.observation_space.shape)),
        int(math.prod(env.action_space.shape)),
    )
    if reg != live:
        raise ValueError(
            f"registry dims {reg} != live {env_id} dims {live}; "
            "update configs/envs.yaml"
        )
    return reg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the SJFPO MLP actor on a state-based MuJoCo env."
    )
    parser.add_argument("--env", default="Hopper-v5", help="Gymnasium MuJoCo env id")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--eval_steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out_dir", default=None, help="Defaults to the runtime run dir")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6, help="MLP-mixer blocks")
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--chunks_per_update", type=int, default=32)
    parser.add_argument("--no_dct", action="store_true", help="Use raw action chunks")
    parser.add_argument(
        "--wandb", action="store_true", help="Log to wandb (default: off)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one MLP-actor training job and print its summary as JSON.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 if the run diverged or raised.
    """
    args = parse_args(argv)

    mujoco_envs = load_mujoco_envs()
    if args.env not in mujoco_envs:
        raise ValueError(
            f"{args.env!r} is not a state-based MuJoCo env. Known ids: "
            f"{sorted(mujoco_envs)}"
        )

    cfg = build_config(args)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_runs_base() / f"sjfpo_mlp_{args.env}_seed{args.seed}"
    )

    env = gym.make(args.env)
    try:
        obs_dim, action_dim = resolve_dims(args.env, env)
        summary = train_sjfpo_run(
            env=env,
            out_dir=out_dir,
            cfg=cfg,
            seed=args.seed,
            total_steps=args.total_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            wandb_enabled=args.wandb,
            policy_cls=SJFPOPolicyMLP,
        )
    finally:
        env.close()

    summary["method"] = "sjfpo_mlp"
    summary["runtime"] = detect_runtime_env()
    summary["actor"] = {
        "type": "mlp_mixer",
        "hidden_dim": cfg.hidden_dim,
        "depth": cfg.dit_depth,
        "mlp_ratio": cfg.dit_mlp_ratio,
    }
    print(json.dumps(summary, indent=2))
    return 1 if summary.get("diverged") else 0


if __name__ == "__main__":
    raise SystemExit(main())
