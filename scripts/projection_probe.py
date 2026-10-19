"""Does the action projection move the trust region?

The velocity field is unbounded. The sampled action is clamped to the env
bounds. The CFM target is the clamped action. So the trust region measures how
well the flow matches a *projected* action, and the projection has no term.

This probe reads the three series the trainer already logs and reports their
correlation:

* ``train/action_clip_frac`` — how much of the proposal the bounds truncated.
* ``train/l_old_new_corr`` — how well the old and new CFM losses move together.
* ``train/clipfrac`` — the share of ratio elements outside the PPO band.

If ``action_clip_frac`` stays near zero, the projection is a non-issue and the
objective needs no change. If it rises and the other two follow, the projection
is driving the trust region and the target must change.

Usage:
    .venv/bin/python scripts/projection_probe.py --env Hopper-v5 --total_steps 20000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myfpo.configuration_sjfpo import SJFPOConfig

CLIP_SERIES = "train/action_clip_frac"
CORR_SERIES = "train/l_old_new_corr"
CLIPFRAC_SERIES = "train/clipfrac"
L_NEW_SERIES = "train/l_new_mean"


def tensorboard_dir(cfg, method: str, env_name: str, seed: int) -> Path:
    """Where the tensorboard backend writes for this run.

    ``build_tracker`` builds a path relative to the process working directory,
    not relative to ``out_dir``. So the event file lands under the repository
    ``tensorboard/`` directory, beside the run directory and not inside it.
    """
    from myfpo.tracking.base import build_run_name

    name = build_run_name(
        method, env_name, seed, dataclasses.asdict(cfg)
    )
    return Path("tensorboard") / name


def read_scalars(tb_dir: Path) -> dict[str, list[float]]:
    """Read every scalar the tensorboard backend wrote for one run."""
    from tensorboard.backend.event_processing import event_accumulator

    out: dict[str, list[float]] = {}
    for d in sorted(tb_dir.rglob("events.out.tfevents.*")):
        ea = event_accumulator.EventAccumulator(str(d.parent))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            out.setdefault(tag, []).extend(float(p.value) for p in ea.Scalars(tag))
    return out


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation, for two equal-length finite series.

    The rank form is used because these series are heavy-tailed and clipped at
    zero, so a Pearson value would report a slope where there is no trend.

    The value ``nan`` stands for a value it cannot compute. That covers a short
    series and a constant one. A constant series would otherwise rank as
    0, 1, 2, ... by position, and a second constant series would rank the same
    way, so the correlation would read as a perfect trend where there is none.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return float("nan")
    if float(a.max() - a.min()) == 0.0 or float(b.max() - b.min()) == 0.0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _align(series: dict[str, list[float]], a: str, b: str) -> tuple[list, list]:
    xs, ys = series.get(a, []), series.get(b, [])
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def projection_measurement(
    env_name: str,
    scales: list[float],
    seed: int,
    steps: int,
    checkpoint: str | None = None,
) -> list[dict]:
    """How often the real projection fires, against the actor output magnitude.

    The projection that matters is ``x0.clamp(-1.0, 1.0)`` in
    ``SJFPOPolicy.predict_action_chunk``. That clamp is the last step before the
    action leaves the policy, and the CFM target is built from the clamped
    chunk. So it enters the objective.

    The tracker does not see it. ``chunk_clip_counts`` comes from
    ``scale_action_chunk``, which runs in the collector. The value it receives
    has already been clamped to [-1, 1] by the policy and then mapped onto the
    env bounds by the postprocessor, whose ``addcmul`` is exact. So that count is
    structurally zero, and ``train/action_clip_frac`` never leaves zero.

    This function replicates the sampling of ``predict_action_chunk`` — the
    encoder call, ``sample_noise``, ``sample_action`` — and reports the
    distribution of ``x0`` before that clamp. The replication is deliberate:
    ``predict_action_chunk`` does not return the pre-clamp value.

    The argument ``scales`` perturbs the actor output layer additively. The layer
    is zero at init, so scaling it multiplicatively does nothing and a fresh
    policy returns ``eps`` unchanged. That row is the untrained floor.

    The argument ``checkpoint`` measures a trained policy. That is the number a
    run would log.
    """
    import gymnasium as gym

    from myfpo.checkpointing import load_policy_checkpoint
    from myfpo.fpo_core import clip_velocity, sample_action, sample_noise
    from myfpo.train_sjfpo import build_policy

    rows: list[dict] = []
    for scale in scales:
        env = gym.make(env_name)
        try:
            low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
            high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = int(np.prod(env.action_space.shape))
            torch.manual_seed(seed)
            if checkpoint:
                policy, _opt, _counters = load_policy_checkpoint(
                    checkpoint, device="cpu", action_low=low, action_high=high
                )
                cfg = policy.config
            else:
                cfg = dataclasses.replace(SJFPOConfig(), compile=False)
                policy = build_policy(
                    obs_dim,
                    act_dim,
                    cfg,
                    device="cpu",
                    action_low=low,
                    action_high=high,
                )
            with torch.no_grad():
                if scale:
                    gen = torch.Generator().manual_seed(seed)
                    shape_w = policy.actor.out.weight.shape
                    shape_b = policy.actor.out.bias.shape
                    policy.actor.out.weight.add_(
                        scale * torch.randn(shape_w, generator=gen)
                    )
                    policy.actor.out.bias.add_(
                        scale * torch.randn(shape_b, generator=gen)
                    )
                policy.update_observation_stats(torch.randn(steps, obs_dim))
                states = torch.randn(steps, cfg.n_obs_steps, obs_dim)
                normalised = policy.preprocessor({"observation.state": states})[
                    "observation.state"
                ]
                shape = (steps, cfg.action_horizon, act_dim)
                condition = policy.encoder(normalised)
                eps = sample_noise(shape, cfg, "cpu", None)
                x0 = sample_action(
                    lambda x, t, c, policy=policy, cfg=cfg: clip_velocity(
                        policy.actor(x, t, c), cfg.transported_clip_value
                    ),
                    shape,
                    condition,
                    num_steps=cfg.num_steps_noise_start,
                    eps=eps,
                    device="cpu",
                ).float()
            mag = x0.abs()
            rows.append(
                {
                    "actor_out_perturbation": scale,
                    "pre_clamp_frac_over_1": float((mag > 1.0).float().mean()),
                    "pre_clamp_frac_over_2": float((mag > 2.0).float().mean()),
                    "pre_clamp_abs_max": float(mag.max()),
                    "pre_clamp_abs_p99": float(torch.quantile(mag.flatten(), 0.99)),
                    "pre_clamp_abs_mean": float(mag.mean()),
                }
            )
        finally:
            env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--out-dir", default="outputs/projection_probe")
    parser.add_argument("--scales", default="0,0.02,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import gymnasium as gym

    from myfpo.train_sjfpo import train_sjfpo_run

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    curve = projection_measurement(
        args.env, scales, seed=args.seed, steps=32, checkpoint=args.checkpoint
    )
    print(json.dumps({"projection_curve": curve}, indent=2), flush=True)

    if args.skip_train:
        return 0

    env = gym.make(args.env)
    # Four workers. The default of 64 forkserver processes exhausts a CPU box.
    cfg = SJFPOConfig(
        report_to="tensorboard",
        compile=False,
        num_envs=4,
        chunks_per_update=256,
        batch_size=64,
        eval_max_episodes=10,
    )
    try:
        result = train_sjfpo_run(
            env=env,
            out_dir=Path(args.out_dir) / f"seed{args.seed}",
            cfg=cfg,
            seed=args.seed,
            total_steps=args.total_steps,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
        )
    finally:
        env.close()

    run_dir = Path(args.out_dir) / f"seed{args.seed}"
    tb_dir = tensorboard_dir(cfg, "sjfpo", args.env, args.seed)
    series = read_scalars(tb_dir)
    if CLIP_SERIES not in series:
        print(
            json.dumps(
                {
                    "error": f"{CLIP_SERIES} not found",
                    "available": sorted(k for k in series if k.startswith("train/")),
                },
                indent=2,
            )
        )
        return 1

    clip = series[CLIP_SERIES]
    corr_x, corr_y = _align(series, CLIP_SERIES, CORR_SERIES)
    cf_x, cf_y = _align(series, CLIP_SERIES, CLIPFRAC_SERIES)
    ln_x, ln_y = _align(series, CLIP_SERIES, L_NEW_SERIES)

    peak = float(np.nanmax(clip)) if clip else float("nan")
    frac_nonzero = float(np.mean(np.asarray(clip) > 0.0)) if clip else float("nan")
    # Case A closes the item. Case B asks for a code change. The threshold is
    # 1 percent of coordinates, which is where the projection stops being noise.
    threshold = 0.01
    case = "B" if peak > threshold else "A"

    out = {
        "env": args.env,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "steps_logged": len(clip),
        "action_clip_frac_mean": float(np.nanmean(clip)),
        "action_clip_frac_peak": peak,
        "action_clip_frac_steps_above_zero": frac_nonzero,
        "spearman_clip_vs_l_old_new_corr": spearman(corr_x, corr_y),
        "spearman_clip_vs_clipfrac": spearman(cf_x, cf_y),
        "spearman_clip_vs_l_new_mean": spearman(ln_x, ln_y),
        "threshold": threshold,
        "case": case,
        "verdict": (
            "case A: the projection is near zero. Close the item. Log the "
            "number and change no objective."
            if case == "A"
            else "case B: the projection bites. The CFM target must move to the "
            "unclipped proposal, or the trust region must model the projection."
        ),
        "run": {
            "best_return": result.get("best_return"),
        },
        "tensorboard_dir": str(tb_dir),
        "run_dir": str(run_dir),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())