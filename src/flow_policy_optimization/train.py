"""flow_policy_optimization.train: evaluators, ManiSkill wrapper, checkpoint IO and the loop."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
import types
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flow_policy_optimization.config import FlowPolicyOptimizationConfig, resolve_action_bounds
from flow_policy_optimization.objective import to_env_action
from flow_policy_optimization.policy import FlowPolicyOptimizationPolicy
from flow_policy_optimization.rollout import _rollout_budget, is_torch_env, make_rollout_state, run_rollout
from flow_policy_optimization.update import make_optimizer, update

CSV_FIELDS = ["episode", "return", "length", "success", "jerk"]
MANISKILL_TASKS = {"MS-PickCube-v1": "PickCube-v1"}
MANISKILL_TASK_OPTIONS = {"MS-PickCube-v1": {"ignore_terminations": True}}
DEFAULT_OBS_MODE = "state"
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"

def compute_action_jerk(actions: np.ndarray) -> float | None:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 3:
        return None
    return float(np.square(a[2:] - 2.0 * a[1:-1] + a[:-2]).mean())
def write_eval_csv(results: list[dict[str, Any]], out_dir: str | Path, *, append: bool = False) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "eval_history.csv"
    mode = "a" if append and csv_path.is_file() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(results)
    return csv_path
def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Mean and median return of the finished episodes."""
    returns = np.asarray([float(r["return"]) for r in results if r.get("return") is not None], dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return {"return_mean": float("nan"), "return_median": float("nan"), "return_n": 0.0}
    return {"return_mean": float(returns.mean()), "return_median": float(np.median(returns)), "return_n": float(returns.size)}


@contextmanager
def _pinned_eval_mode(policy: Any):
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
class _BaseEvaluator:
    def __init__(self, eval_seed: int, max_episodes: int, max_steps: int | None) -> None:
        if max_episodes < 0 or (max_steps is not None and max_steps < 0):
            raise ValueError(f"invalid eval budget max_episodes={max_episodes}, max_steps={max_steps}")
        self.eval_seed, self.max_episodes, self.max_steps = eval_seed, max_episodes, max_steps
class Evaluator(_BaseEvaluator):
    """Deterministic evaluator for flat-obs CPU Gymnasium envs."""

    def __init__(self, env: Any, eval_seed: int = 0, max_episodes: int = 12, max_steps: int | None = None):
        if env is None:
            raise ValueError("Evaluator requires a non-None env")
        if hasattr(env, "observation_space"):
            import gymnasium as gym
            if isinstance(env.observation_space, gym.spaces.Dict):
                raise AssertionError("Evaluator expects flat Box obs")
        self.env = env
        super().__init__(eval_seed, max_episodes, max_steps)

    def rollout_single_episode(self, policy: Callable[[np.ndarray], np.ndarray], seed: int, episode: int, step_budget: int | None) -> tuple[dict[str, Any], int]:
        obs, _ = self.env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return, ep_length, ep_actions, ep_success, done = 0.0, 0, [], None, False
        while not done and (step_budget is None or ep_length < step_budget):
            action = policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(terminated) or bool(truncated)
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            if isinstance(info, dict) and info.get("success") is not None:
                ep_success = float(bool(info["success"]))
        jerk = compute_action_jerk(np.stack(ep_actions)) if ep_actions else None
        return {"episode": episode, "return": ep_return, "length": ep_length, "success": ep_success, "jerk": jerk}, ep_length

    def evaluate(self, policy: Callable[[np.ndarray], np.ndarray], *, out_dir: str | Path | None = None, append_csv: bool = False) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.eval_seed)
        results, total_steps = [], 0
        with torch.random.fork_rng(), _pinned_eval_mode(policy):
            torch.manual_seed(self.eval_seed)
            for ep in range(self.max_episodes):
                if self.max_steps is not None and total_steps >= self.max_steps:
                    break
                budget = None if self.max_steps is None else self.max_steps - total_steps
                res, taken = self.rollout_single_episode(policy, int(rng.integers(0, 2**31)), ep, budget)
                results.append(res)
                total_steps += taken
        if out_dir is not None:
            write_eval_csv(results, out_dir, append=append_csv)
        return results

    def clone(self, max_episodes: int) -> Evaluator:
        return Evaluator(self.env, eval_seed=self.eval_seed, max_episodes=max_episodes, max_steps=self.max_steps)

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        if seed is None:
            seed = self.eval_seed
        env.action_space.seed(int(seed))
        return lambda _obs: env.action_space.sample()
class VectorEvaluator(_BaseEvaluator):
    """Deterministic batched evaluator for ManiSkill GPU torch envs."""

    def __init__(self, env: Any, *, num_obs_steps: int, num_action_steps: int, action_horizon: int, eval_seed: int = 0, max_episodes: int = 12, max_steps: int | None = None):
        if env is None or num_obs_steps < 1 or not (1 <= num_action_steps <= action_horizon):
            raise ValueError("invalid VectorEvaluator configuration")
        self.env = env
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.action_horizon = action_horizon
        super().__init__(eval_seed, max_episodes, max_steps)
        low, high = resolve_action_bounds(env.single_action_space.low, env.single_action_space.high)
        dev = torch.device(env.device) if hasattr(env, "device") else torch.device("cpu")
        self.action_low, self.action_high = low.to(dev), high.to(dev)

    def evaluate(self, policy: Any, *, out_dir: str | Path | None = None, append_csv: bool = False) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.eval_seed)
        n, device = int(self.env.num_envs), torch.device(self.env.device)
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        results: list[dict[str, Any]] = []
        planner = getattr(policy, "plan", None)

        with torch.random.fork_rng(devices=fork_devices), torch.inference_mode(), _pinned_eval_mode(policy):
            torch.manual_seed(self.eval_seed)
            while len(results) < self.max_episodes:
                obs, _ = self.env.reset(seed=[int(rng.integers(0, 2**31)) for _ in range(n)])
                if hasattr(policy, "reset"):
                    policy.reset()
                window = obs[:, None, :].repeat(1, self.num_obs_steps, 1)
                ep_return = torch.zeros(n, device=device)
                success = torch.zeros(n, dtype=torch.bool, device=device)
                saw_success, frames, steps = False, [], 0
                chunk: torch.Tensor | None = None

                while True:
                    if chunk is None or steps % self.num_action_steps == 0:
                        planned = (planner(window) if planner is not None else policy(window)).to(device)
                        chunk = to_env_action(planned, self.action_low, self.action_high)
                    action = chunk[:, steps % self.num_action_steps]
                    obs, rew, term, trunc, info = self.env.step_torch(action)
                    frames.append(action)
                    ep_return += rew
                    if isinstance(info, dict) and "success" in info:
                        success |= info["success"]
                        saw_success = True
                    steps += 1
                    window = torch.cat((window[:, 1:], obs[:, None, :]), dim=1)
                    if bool(trunc.all()) or bool(term.all()) or (self.max_steps is not None and steps >= self.max_steps):
                        break

                actions_np = torch.stack(frames, dim=1).cpu().numpy()
                returns_np, succ_np = ep_return.cpu().numpy(), success.cpu().numpy()
                for row in range(n):
                    if len(results) >= self.max_episodes:
                        break
                    results.append({
                        "episode": len(results),
                        "return": float(returns_np[row]),
                        "length": steps,
                        "success": float(succ_np[row]) if saw_success else None,
                        "jerk": compute_action_jerk(actions_np[row]),
                    })
        if out_dir is not None:
            write_eval_csv(results, out_dir, append=append_csv)
        return results

    def clone(self, max_episodes: int) -> VectorEvaluator:
        return VectorEvaluator(
            self.env, num_obs_steps=self.num_obs_steps, num_action_steps=self.num_action_steps,
            action_horizon=self.action_horizon, eval_seed=self.eval_seed, max_episodes=max_episodes, max_steps=self.max_steps,
        )

    def random_policy(self, env: Any, seed: int | None = None) -> Callable:
        del seed  # shared signature with Evaluator.random_policy, which seeds the action space
        return lambda _w: env.random_chunk(int(self.action_horizon))
def is_maniskill_id(env_id: str) -> bool:
    return env_id in MANISKILL_TASKS
class ManiSkillTorchEnv:
    """GPU-parallel ManiSkill 3 env conforming to the SameStep collector contract."""

    def __init__(self, alias: str, num_envs: int, *, ignore_terminations: bool | None = None, obs_mode: str = DEFAULT_OBS_MODE, control_mode: str = DEFAULT_CONTROL_MODE, render_mode: str | None = None):
        if alias not in MANISKILL_TASKS or num_envs < 1:
            raise ValueError(f"invalid ManiSkillTorchEnv({alias!r}, num_envs={num_envs})")
        import gymnasium as gym
        from gymnasium.vector import AutoresetMode
        import mani_skill.envs  # noqa: F401
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        self.alias, self.task_id = alias, MANISKILL_TASKS[alias]
        self.obs_mode, self.control_mode, self.render_mode = obs_mode, control_mode, render_mode
        if ignore_terminations is None:
            ignore_terminations = bool(MANISKILL_TASK_OPTIONS.get(alias, {}).get("ignore_terminations", False))
        base = gym.make(self.task_id, num_envs=int(num_envs), obs_mode=obs_mode, control_mode=control_mode, render_mode=render_mode)
        self.vec = ManiSkillVectorEnv(base, auto_reset=True, ignore_terminations=ignore_terminations)
        self.num_envs = int(self.vec.num_envs)
        self.device = torch.device(self.vec.device)
        self.single_observation_space = self.observation_space = self.vec.single_observation_space
        self.single_action_space = self.action_space = self.vec.single_action_space
        self.autoreset_mode = AutoresetMode.SAME_STEP
        self.metadata = {"render_modes": []}
        self.spec = types.SimpleNamespace(id=alias)

    def reset(self, *, seed=None, options=None):
        return self.vec.reset(seed=seed, options=options)

    def step_torch(self, actions: torch.Tensor):
        return self.vec.step(actions.to(device=self.device, dtype=torch.float32))

    def step(self, actions):
        obs, rew, term, trunc, info = self.step_torch(torch.as_tensor(actions, dtype=torch.float32, device=self.device))
        return obs.cpu().numpy(), rew.cpu().numpy(), term.cpu().numpy(), trunc.cpu().numpy(), info

    def close(self) -> None:
        self.vec.close()

    def make_eval_env(self, *, num_envs: int | None = None, render_mode: str | None = None) -> ManiSkillTorchEnv:
        return ManiSkillTorchEnv(
            self.alias, self.num_envs if num_envs is None else int(num_envs),
            ignore_terminations=True, obs_mode=self.obs_mode, control_mode=self.control_mode, render_mode=render_mode,
        )

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() requires render_mode='rgb_array'")
        frames = self.vec.render()
        return np.asarray(frames.detach().cpu().numpy() if torch.is_tensor(frames) else frames)

    def random_chunk(self, horizon: int) -> torch.Tensor:
        return torch.empty((self.num_envs, int(horizon), int(self.action_space.shape[0])), device=self.device).uniform_(-1.0, 1.0)
def make_maniskill_env(alias: str, num_envs: int = 1, **kwargs) -> ManiSkillTorchEnv:
    return ManiSkillTorchEnv(alias, num_envs, **kwargs)
def register_gym_envs() -> None:
    import gymnasium as gym
    for alias, task_id in MANISKILL_TASKS.items():
        def _make(tid=task_id):
            import mani_skill.envs  # noqa: F401
            from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
            return CPUGymWrapper(gym.make(tid, num_envs=1, obs_mode=DEFAULT_OBS_MODE, control_mode=DEFAULT_CONTROL_MODE))
        try:
            gym.register(id=alias, entry_point=_make)
        except gym.error.Error:
            continue


# ---------------------------------------------------------------------------
# IO, Checkpoint, and Training Loop
def resolve_device() -> torch.device:
    want = os.environ.get("FLOW_POLICY_OPTIMIZATION_DEVICE", "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want not in {"auto", "cuda"}:
        raise ValueError(f"FLOW_POLICY_OPTIMIZATION_DEVICE={want!r} unknown; expected 'auto', 'cpu', or 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def save_checkpoint(out_dir, name: str, policy: FlowPolicyOptimizationPolicy, optimizer, global_step: int) -> str:
    ckpt = Path(out_dir) / "checkpoints" / name / "checkpoint.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "global_step": int(global_step),
        "config": policy.config.asdict(),
    }, ckpt)
    return str(ckpt)
def load_checkpoint(checkpoint_dir: str, policy: FlowPolicyOptimizationPolicy, optimizer=None) -> int:
    p = Path(checkpoint_dir)
    ckpt = p / "checkpoint.pt" if p.is_dir() else p
    if not ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint.pt at {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("global_step", 0))
def _shared_env(train_env, env) -> bool:
    """True when the collector env shares a sub-env with the evaluator env."""
    sub_envs = getattr(train_env, "envs", None)
    return any(e is env for e in sub_envs) if sub_envs is not None else train_env is env
def make_train_env(env, env_name: str, num_envs: int, record: dict | None = None):
    if is_torch_env(env):
        if record is not None:
            record["vector_env"] = "maniskill_gpu"
        return env
    import gymnasium as gym
    if num_envs <= 1:
        if record is not None:
            record["vector_env"] = "single"
        return gym.vector.SyncVectorEnv([lambda: env], autoreset_mode="SameStep")

    def _make_one():
        if env_name.startswith("Meta-World/"):
            import metaworld  # noqa: F401
            return gym.make("Meta-World/MT1", env_name=env_name[len("Meta-World/"):])
        if env_name.startswith("AdroitHand"):
            import gymnasium_robotics  # noqa: F401
        return gym.make(env_name)

    env_fns = [_make_one for _ in range(num_envs)]
    if env_name.startswith("Meta-World/"):
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    try:
        return gym.vector.AsyncVectorEnv(env_fns, context="forkserver", autoreset_mode="SameStep")
    except (ValueError, RuntimeError, ImportError):
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")
class _EvalAdapter:
    def __init__(self, policy: FlowPolicyOptimizationPolicy):
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs) -> np.ndarray:
        return np.atleast_1d(self.policy.select_action(obs).detach().cpu().numpy()).astype(np.float32)

    def plan(self, window) -> torch.Tensor:
        return self.policy.predict_action_chunk(window)
def _init_wandb(cfg: FlowPolicyOptimizationConfig, *, method: str, env_name: str, seed: int):
    if cfg.report_to == "none":
        return None
    import wandb
    wandb.init(
        project=method,
        name=f"{method}_{env_name}_s{seed}",
        config={**cfg.asdict(), "method": method, "seed": seed, "env": env_name},
    )
    return wandb
def run_flow_policy_optimization_training(
    env, out_dir, cfg: FlowPolicyOptimizationConfig | None = None, *, seed: int = 3407, total_steps: int = 2000,
    eval_steps: int = 1000, save_steps: int = 10**9, obs_dim: int | None = None,
    action_dim: int | None = None, resume_from_checkpoint: str | None = None,
    env_id: str | None = None, runtime: dict | None = None,
) -> dict:
    """Run one flow_policy_optimization training loop."""
    cfg = cfg or FlowPolicyOptimizationConfig()
    obs_dim = obs_dim or int(np.prod(env.observation_space.shape))
    action_dim = action_dim or int(np.prod(env.action_space.shape))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    env_name = env_id or (getattr(spec, "id", None) if spec is not None else "unknown") or "unknown"

    set_seed(seed)
    device = resolve_device()
    train_env = make_train_env(env, env_name, cfg.num_envs, record=runtime)
    if runtime is not None:
        (out_dir / "run_metadata.json").write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")

    low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))
    policy = FlowPolicyOptimizationPolicy(cfg, obs_dim, action_dim, low, high).to(device)
    optimizer = make_optimizer(policy)
    if resume_from_checkpoint is not None:
        step = load_checkpoint(resume_from_checkpoint, policy, optimizer)
        print(f"flow_policy_optimization: resumed from {resume_from_checkpoint} at step {step}", flush=True)

    eval_seed = 10000 + seed
    if hasattr(train_env, "make_eval_env"):
        eval_env = train_env.make_eval_env()
        evaluator = VectorEvaluator(
            eval_env, num_obs_steps=cfg.num_obs_steps, num_action_steps=cfg.num_action_steps,
            action_horizon=cfg.action_horizon, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes,
        )
    else:
        eval_env = env
        evaluator = Evaluator(env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes)

    wb = _init_wandb(cfg, method="flow_policy_optimization", env_name=env_name, seed=seed)
    artifacts: list[str] = []
    params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"flow_policy_optimization policy: actor=FlowMLP hidden={cfg.hidden_dim} params={params:,}", flush=True)

    n_envs = int(getattr(train_env, "num_envs", 1))
    full_chunks_per_env = max(1, int(cfg.chunks_per_update) // n_envs)
    rollout_state = make_rollout_state(obs_dim, action_dim, cfg, chunks_hint=full_chunks_per_env * n_envs)
    best_return: float | None = None
    steps_to_best = 0
    all_eval_returns: list[float] = []
    next_eval, next_save = eval_steps, save_steps
    t_start, elapsed, global_step = time.time(), 0.0, 0

    while global_step < total_steps:
        t0 = time.perf_counter()
        remaining_steps = max(0, total_steps - global_step)
        chunks_per_env = _rollout_budget(cfg, n_envs, remaining_steps)
        batch = run_rollout(
            policy, train_env, chunks_per_env, action_low=low, action_high=high,
            state=rollout_state,
        )
        collect_s = time.perf_counter() - t0
        metrics, grad_norm_actor, grad_norm_critic = update(policy, optimizer, batch, cfg)
        global_step += int(batch["step_rewards"].numel())

        if wb is not None:
            acts = batch["action"]
            wb.log({
                **{f"train/{k}": v for k, v in metrics.items()},
                "train/grad_norm_actor_before_clip": grad_norm_actor,
                "train/grad_norm_critic_before_clip": grad_norm_critic,
                "train/grad_norm_before_clip": math.hypot(grad_norm_actor, grad_norm_critic),
                "train/collect_s": collect_s,
                "train/policy_lr": optimizer.param_groups[0]["lr"],
                "train/critic_lr": optimizer.param_groups[-1]["lr"],
                "train/executed_action_abs_mean": float(acts.abs().mean().item()) if acts.numel() else 0.0,
                "train/executed_action_abs_max": float(acts.abs().max().item()) if acts.numel() else 0.0,
            }, step=global_step)

        if global_step >= next_eval or global_step >= total_steps:
            eval_results = evaluator.evaluate(_EvalAdapter(policy), out_dir=out_dir, append_csv=True)
            summary = summarize_results(eval_results)
            mean_ret = float(summary["return_mean"])
            all_eval_returns.append(mean_ret)
            if best_return is None or mean_ret > best_return:
                best_return, steps_to_best = mean_ret, global_step
                best_ckpt = save_checkpoint(out_dir, "best", policy, optimizer, global_step)
                if best_ckpt not in artifacts:
                    artifacts.append(best_ckpt)
            if wb is not None:
                successes = [float(r["success"]) for r in eval_results if r.get("success") is not None]
                wb.log({
                    "eval/return_mean": mean_ret,
                    "eval/return_median": summary["return_median"],
                    "eval/mean_length": float(np.mean([r["length"] for r in eval_results])),
                    "eval/episode_count": len(eval_results),
                    "eval/return_best": float(best_return),
                    "eval/steps_to_best": int(steps_to_best),
                    **({"eval/success_rate": float(np.mean(successes))} if successes else {}),
                }, step=global_step)
            next_eval = max(global_step + eval_steps, next_eval + eval_steps)
            if _shared_env(train_env, env):
                rollout_state.buffer.carry = None

        if global_step >= next_save:
            ckpt = save_checkpoint(out_dir, f"step_{global_step}", policy, optimizer, global_step)
            if ckpt not in artifacts:
                artifacts.append(ckpt)
            next_save = max(global_step + save_steps, next_save + save_steps)
        elapsed = time.time() - t_start

    fps = global_step / elapsed if elapsed > 0.0 else 0.0
    final_ckpt = save_checkpoint(out_dir, "final", policy, optimizer, global_step)
    if final_ckpt not in artifacts:
        artifacts.append(final_ckpt)
    if (out_dir / "eval_history.csv").exists():
        artifacts.append(str(out_dir / "eval_history.csv"))
    if wb is not None:
        wb.log({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step)
        wb.finish(exit_code=0)
    if train_env is not env:
        train_env.close()
    if eval_env is not env and eval_env is not train_env and hasattr(eval_env, "close"):
        eval_env.close()

    return {
        "method": "flow_policy_optimization",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": float(np.mean(all_eval_returns[-10:])) if all_eval_returns else 0.0,
        "reward_median_last10": float(np.median(all_eval_returns[-10:])) if all_eval_returns else 0.0,
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": next((a for a in artifacts if "best" in a), None),
        "best_return": best_return,
        "global_step": global_step,
    }
