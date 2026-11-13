"""myfpo evaluator. The module runs deterministic rollouts and writes a per-eval
CSV, and computes the jerk diagnostics that expose chunk-boundary artifacts.

Two evaluators, one result schema. :class:`Evaluator` steps a single numpy env
(MuJoCo, and the ManiSkill CPU path). :class:`VectorEvaluator` steps a batched
torch env and never leaves the device, which is what the ManiSkill GPU path
uses.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myfpo.configuration import validate_num_action_steps
from myfpo.actions import to_env_action
from myfpo.rollout import resolve_action_bounds


CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_LOW_Q = 0.025
BOOTSTRAP_HIGH_Q = 0.975


@contextmanager
def _pinned_eval_mode(policy: Any):
    """Pin the wrapped module to eval mode for a whole eval window.

    The adapter used to toggle the mode on every action. The flag is cheap in
    eager mode, but a per-call toggle is the pattern the perf guidance warns
    about, and it defeats a compiled eval path. Restore the mode in ``finally``
    so a failed episode still leaves the module as it was found.
    """
    module = getattr(policy, "policy", None)
    if not isinstance(module, torch.nn.Module):
        yield
        return
    was = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was)


def bootstrap_interval(
    returns: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    low_q: float = BOOTSTRAP_LOW_Q,
    high_q: float = BOOTSTRAP_HIGH_Q,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A percentile interval for the mean of ``returns``."""
    values = np.asarray([float(r) for r in returns], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size < 2:
        return mean, float("nan"), float("nan")
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError(f"need 0 <= low_q < high_q <= 1, got ({low_q}, {high_q})")
    if resamples < 1:
        raise ValueError(f"resamples={resamples} must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    low = float(np.percentile(means, 100.0 * low_q))
    high = float(np.percentile(means, 100.0 * high_q))
    return mean, low, high


def median_of(returns: list[float]) -> float:
    """The median of ``returns``. It returns ``nan`` for an empty input."""
    values = np.asarray([float(r) for r in returns], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _check_eval_counts(max_episodes: int, max_steps: int | None) -> None:
    """Validate the two eval budgets shared by both evaluators."""
    if max_episodes < 0:
        raise ValueError(f"max_episodes={max_episodes} must be >= 0")
    if max_steps is not None and max_steps < 0:
        raise ValueError(f"max_steps={max_steps} must be >= 0 or None")


def _mean_squared_second_diff(rows: np.ndarray) -> float:
    """Mean squared second difference. ``rows`` needs at least three rows."""
    second_diff = rows[2:] - 2.0 * rows[1:-1] + rows[:-2]
    return float(np.square(second_diff).mean())


def _finish_eval(out_dir, append_csv, results):
    """Write ``eval_history.csv`` when asked, and return the results."""
    if out_dir is not None:
        write_eval_csv(results, out_dir, append=append_csv)
    return results


def write_eval_csv(
    results: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write ``eval_history.csv``, columns ``[episode, return, length, success, jerk]``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    file_exists = csv_path.is_file()
    mode = "a" if append and file_exists else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not (append and file_exists):
            writer.writeheader()
        writer.writerows(results)
    return csv_path


def compute_action_jerk(actions: np.ndarray) -> float | None:
    """Mean squared second-difference of actions ``[T, D]``. None if T < 3."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    return _mean_squared_second_diff(a)


def compute_seam_jerk(
    actions: np.ndarray, action_is_pad: np.ndarray, env_id=None
) -> float | None:
    """Mean squared second-difference across consecutive chunk boundaries.

    ``env_id`` keeps every seam inside one env. Chunk rows are written in
    closing order, so with several envs two adjacent rows are usually two
    different trajectories, and a seam across them is not a seam.
    """
    a, pad = jerk_inputs(actions, action_is_pad)
    n = a.shape[0]
    order = np.arange(n)
    same_env = np.ones(max(n - 1, 0), dtype=bool)
    if env_id is not None:
        ids = np.asarray(env_id)
        if ids.shape != (n,):
            raise ValueError(f"expected env_id with shape ({n},), got {ids.shape}")
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        same_env = ids[:-1] == ids[1:]
    seams: list[float] = []
    for k in range(n - 1):
        if not same_env[k]:
            continue
        i, j = order[k], order[k + 1]
        tail = np.nonzero(~pad[i])[0]
        head = np.nonzero(~pad[j])[0]
        if tail.size < 2 or head.size < 1:
            continue
        window = np.stack([a[i, tail[-2]], a[i, tail[-1]], a[j, head[0]]])
        seams.append(_mean_squared_second_diff(window))
    return float(np.mean(seams)) if seams else None


def jerk_inputs(actions, action_is_pad) -> tuple[np.ndarray, np.ndarray]:
    """Shared validation for the two jerk functions."""
    a = np.asarray(actions)
    pad = np.asarray(action_is_pad, dtype=bool)
    if a.ndim != 3 or pad.shape != a.shape[:2]:
        raise ValueError(
            f"expected actions [N, H, D] and action_is_pad [N, H]; "
            f"got {a.shape} and {pad.shape}"
        )
    return a, pad


def compute_mean_chunk_jerk(actions: np.ndarray, action_is_pad: np.ndarray) -> float | None:
    """Average :func:`compute_action_jerk` over the executed prefixes of the chunks.
    Returns ``None`` when no chunk executed at least three steps.
    """
    a, pad = jerk_inputs(actions, action_is_pad)
    jerks: list[float] = []
    for chunk, mask in zip(a, pad):
        n_exec = int((~mask).sum())
        if n_exec >= 3:
            j = compute_action_jerk(chunk[:n_exec])
            if j is not None:
                jerks.append(j)
    return float(np.mean(jerks)) if jerks else None



def summarize_results(results: list[dict[str, Any]], seed: int = 0) -> dict[str, float]:
    """The mean, the median, the interval, and the count for one eval.

    Shared by both evaluators so a reported number means the same thing on the
    CPU path and on the GPU path.
    """
    returns = [float(r["return"]) for r in results if r.get("return") is not None]
    mean, low, high = bootstrap_interval(returns, seed=seed)
    finite = sum(1 for r in returns if np.isfinite(r))
    return {
        "return_mean": mean,
        "return_median": median_of(returns),
        "return_ci_low": low,
        "return_ci_high": high,
        "return_n": float(finite),
    }


class _BaseEvaluator:
    """The eval budget shared by the two evaluators. Holds no env logic."""

    def __init__(self, eval_seed: int, max_episodes: int, max_steps: int | None) -> None:
        _check_eval_counts(max_episodes, max_steps)
        self.eval_seed = eval_seed
        self.max_episodes = max_episodes
        self.max_steps = max_steps

    def summarize(
        self, results: list[dict[str, Any]], seed: int = 0
    ) -> dict[str, float]:
        """The mean, the median, the interval, and the count for one eval."""
        return summarize_results(results, seed=seed)


class Evaluator(_BaseEvaluator):
    """Deterministic evaluator for flat-obs (MuJoCo) envs."""

    def __init__(
        self,
        env: Any,
        eval_seed: int = 0,
        max_episodes: int = 12,
        max_steps: int | None = None,

    ):
        if env is None:
            raise ValueError("Evaluator requires a non-None env")
        if hasattr(env, "observation_space"):
            try:
                import gymnasium as gym
            except ImportError as e:
                raise ImportError(
                    "gymnasium is required for Evaluator obs-contract validation"
                ) from e
            if isinstance(env.observation_space, gym.spaces.Dict):
                raise AssertionError("Evaluator expects flat Box obs")
        self.env = env
        super().__init__(eval_seed, max_episodes, max_steps)

    def rollout_single_episode(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        seed: int,
        episode: int,
        step_budget: int | None,
    ) -> tuple[dict[str, Any], int]:
        """One deterministic episode. Returns ``(result, steps_taken)``."""
        obs, _ = self.env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        ep_length = 0
        ep_actions: list[np.ndarray] = []
        ep_success: float | None = None
        done = False
        while not done:
            if step_budget is not None and ep_length >= step_budget:
                break
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(terminated) or bool(truncated)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            if isinstance(info, dict) and info.get("success") is not None:
                ep_success = float(bool(info["success"]))
        jerk = compute_action_jerk(np.stack(ep_actions)) if ep_actions else None
        return (
            {
                "episode": episode,
                "return": ep_return,
                "length": ep_length,
                "success": ep_success,
                "jerk": jerk,
            },
            ep_length,
        )

    def evaluate(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        *,
        out_dir: str | Path | None = None,
        append_csv: bool = False,
    ) -> list[dict[str, Any]]:
        """Deterministic rollouts, optionally writing ``eval_history.csv``."""
        rng = np.random.default_rng(self.eval_seed)
        results: list[dict[str, Any]] = []
        total_steps = 0
        with _pinned_eval_mode(policy):
            for ep in range(self.max_episodes):
                if self.max_steps is not None and total_steps >= self.max_steps:
                    break
                seed = int(rng.integers(0, 2**31))
                budget = None if self.max_steps is None else self.max_steps - total_steps
                result, taken = self.rollout_single_episode(policy, seed, ep, budget)
                results.append(result)
                total_steps += taken
        return _finish_eval(out_dir, append_csv, results)

    def clone(self, max_episodes: int) -> Evaluator:
        """The same evaluator, with a different episode budget."""
        return Evaluator(
            self.env,
            eval_seed=self.eval_seed,
            max_episodes=max_episodes,
            max_steps=self.max_steps,
        )

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        """A uniform-random baseline policy for this env."""
        if seed is not None:
            env.action_space.seed(int(seed))
        return lambda _obs: env.action_space.sample()


class VectorEvaluator(_BaseEvaluator):
    """Deterministic batched evaluator for a torch vector env (ManiSkill GPU).

    Every row runs in lockstep and only stops at the time limit
    (``ignore_terminations=True`` on the env), so one rolling observation window
    is valid for the whole batch: what a per-env window would hold is the same
    frames for every row. Episodes start and end together, which is also what
    makes the episode metrics comparable row to row.

    ``max_steps`` caps the length of a single episode, not the whole eval: the
    budget is spent in rounds of ``num_envs`` episodes.
    """

    def __init__(
        self,
        env: Any,
        *,
        num_obs_steps: int,
        num_action_steps: int,
        action_horizon: int,
        eval_seed: int = 0,
        max_episodes: int = 12,
        max_steps: int | None = None,
    ):
        if env is None:
            raise ValueError("VectorEvaluator requires a non-None env")
        if num_obs_steps < 1:
            raise ValueError(f"num_obs_steps={num_obs_steps} must be >= 1")
        validate_num_action_steps(num_action_steps, action_horizon)
        self.env = env
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.action_horizon = action_horizon
        super().__init__(eval_seed, max_episodes, max_steps)
        self.action_low, self.action_high = resolve_action_bounds(
            env.single_action_space.low, env.single_action_space.high
        )

    def _plan(self, policy: Any, window: torch.Tensor) -> torch.Tensor:
        """One raw chunk from the policy, however it is packaged."""
        planner = getattr(policy, "plan", None)
        return planner(window) if planner is not None else policy(window)

    def _grab_frame(self) -> np.ndarray:
        """One RGB frame for the whole batch, tiled into a single image.

        ``render()`` returns every row, so a recording env must be built with a
        handful of rows: at the training width this would be gigabytes a frame.
        """
        frames = np.asarray(self.env.render())
        if frames.ndim == 3:
            return frames
        from mani_skill.utils.visualization.misc import tile_images

        return tile_images(frames, nrows=max(1, int(np.sqrt(len(frames)))))

    @staticmethod
    def _write_video(
        images: list[np.ndarray], video_path: str | Path, fps: int
    ) -> Path:
        """Encode the captured frames to ``video_path`` (a ``.mp4``)."""
        from mani_skill.utils.visualization.misc import images_to_video

        path = Path(video_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # images_to_video appends the suffix itself, so hand it the stem.
        images_to_video(images, str(path.parent), path.stem, fps=fps, verbose=False)
        return path

    def evaluate(
        self,
        policy: Any,
        *,
        out_dir: str | Path | None = None,
        append_csv: bool = False,
        video_path: str | Path | None = None,
        video_fps: int = 30,
    ) -> list[dict[str, Any]]:
        """Deterministic batched rollouts, optionally writing ``eval_history.csv``.

        ``video_path`` records the first round of episodes only: that is one
        full-length rollout of every row, which is what a video of the policy is
        for. Recording every round would write the same rollout over and over.
        """
        rng = np.random.default_rng(self.eval_seed)
        n = int(self.env.num_envs)
        device = torch.device(self.env.device)
        # A forked RNG keeps the noise-start sampling reproducible without
        # shifting the training loop's stream.
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        results: list[dict[str, Any]] = []
        with torch.random.fork_rng(devices=fork_devices), torch.inference_mode(), _pinned_eval_mode(policy):
            torch.manual_seed(self.eval_seed)
            while len(results) < self.max_episodes:
                seeds = [int(rng.integers(0, 2**31)) for _ in range(n)]
                obs, _ = self.env.reset(seed=seeds)
                if hasattr(policy, "reset"):
                    policy.reset()
                window = obs[:, None, :].repeat(1, self.num_obs_steps, 1)
                ep_return = torch.zeros(n, device=device)
                success = torch.zeros(n, dtype=torch.bool, device=device)
                saw_success = False
                frames: list[torch.Tensor] = []
                steps = 0
                # Only the first round is captured; one round is one episode of
                # every row, so later rounds would overwrite the same video.
                recording = video_path is not None and not results
                images: list[np.ndarray] = [self._grab_frame()] if recording else []
                chunk: torch.Tensor | None = None
                while True:
                    if chunk is None or steps % self.num_action_steps == 0:
                        planned = self._plan(policy, window)
                        unwrapped = getattr(policy, "policy", policy)
                        mode = getattr(
                            getattr(unwrapped, "config", None),
                            "action_processing_mode",
                            "tanh_squash",
                        )
                        chunk = to_env_action(
                            planned, self.action_low, self.action_high, mode=mode
                        )
                    action = chunk[:, steps % self.num_action_steps]
                    obs, reward, terminated, truncated, info = self.env.step_torch(action)
                    frames.append(action)
                    if recording:
                        images.append(self._grab_frame())
                    ep_return += reward
                    if isinstance(info, dict) and "success" in info:
                        success = success | info["success"]
                        saw_success = True
                    steps += 1
                    window = torch.cat((window[:, 1:], obs[:, None, :]), dim=1)
                    if bool(truncated.all()) or bool(terminated.all()):
                        break
                    if self.max_steps is not None and steps >= self.max_steps:
                        break
                actions = torch.stack(frames, dim=1).cpu().numpy()
                returns = ep_return.cpu().numpy()
                successes = success.cpu().numpy()
                if recording and images:
                    self._write_video(images, video_path, video_fps)
                for row in range(n):
                    if len(results) >= self.max_episodes:
                        break
                    results.append(
                        {
                            "episode": len(results),
                            "return": float(returns[row]),
                            "length": steps,
                            "success": float(successes[row]) if saw_success else None,
                            "jerk": compute_action_jerk(actions[row]),
                        }
                    )
        return _finish_eval(out_dir, append_csv, results)

    def clone(self, max_episodes: int) -> VectorEvaluator:
        """The same evaluator, with a different episode budget."""
        return VectorEvaluator(
            self.env,
            num_obs_steps=self.num_obs_steps,
            num_action_steps=self.num_action_steps,
            action_horizon=self.action_horizon,
            eval_seed=self.eval_seed,
            max_episodes=max_episodes,
            max_steps=self.max_steps,
        )

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        """A uniform-random baseline chunk policy for this env.

        The chunk is raw, so the evaluator's squash still bounds it. The torch
        RNG is seeded by :meth:`evaluate`, which is why ``seed`` is unused.
        """
        del seed

        def plan(_window: torch.Tensor) -> torch.Tensor:
            """Return one raw random chunk of ``action_horizon`` actions."""
            horizon = int(self.action_horizon)
            return env.random_chunk(horizon)

        return plan

