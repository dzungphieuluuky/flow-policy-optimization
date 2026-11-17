"""Minimal PPO backend for ``--method ppo``.

A deliberately plain CPU-Gym PPO, kept in one file, for comparing against
myfpo on the same envs, seeds and eval protocol. Reference: CleanRL's
``ppo_continuous_action.py`` (MIT, https://github.com/vwxyzjn/cleanrl).

Design choices, and how this differs from myfpo on purpose:

- State-independent ``log_std`` Gaussian, tanh-squashed to the env bounds
  (SAC-style change-of-variables correction for the stored log-probs).
- 2-layer tanh MLPs (``cfg.hidden_dim`` wide) for actor mean and critic — the
  standard MuJoCo PPO shape. myfpo's actor is a much larger flow policy; the
  capacity gap is the comparison you asked for, not an accident to fix here.
- Time-limit truncations bootstrap the critic with the true final observation
  (``SameStep`` vector envs, ``info["final_obs"]``). CleanRL treats truncation
  as termination; we don't.
- One optimizer, two param groups (``policy_lr`` / ``critic_lr``), one total
  gradient-norm clip (``cfg.max_grad_norm``), mirroring CleanRL. myfpo clips
  actor and critic independently — not replicated, it is not part of PPO.
- Rollout-level advantage normalization over the whole batch (matches the
  "standard PPO contract" comment in ``MyfpoConfig``), not per minibatch.
- No value-loss clipping, no entropy bonus, no target KL, no EMA, no reward
  normalization, no divergence aborts, no ``torch.compile``. myfpo's CFM
  guards have no PPO analogue and are not simulated.
- Action processing: the network output is squashed on the way to the env;
  the stored actions are the pre-squash ``u``. No separate action normalizer,
  same contract as the flow policy.

Environment vectorization mirrors ``myfpo.myfpo.make_train_env`` (forkserver +
``SyncVectorEnv`` fallback, Meta-World synchronous) with ``SameStep``
autoreset instead of the default ``NextStep``.

Only these ``MyfpoConfig`` fields are read: ``num_envs``, ``hidden_dim``,
``policy_lr``, ``critic_lr``, ``adam_betas``, ``optimizer_weight_decay``,
``gamma``, ``gae_lambda``, ``clip_param``, ``num_learning_epochs``,
``chunks_per_update``, ``chunks_per_minibatch``, ``normalize_advantage``,
``value_loss_coef``, ``max_grad_norm``, ``report_to``, ``eval_max_episodes``.
Everything else in the config is ignored, including ``lr_schedule``.

Samples per update: ``chunks_per_update`` total env steps, split across
``num_envs`` vector workers — same meaning as in myfpo (2048 -> 32 steps per
env at ``num_envs=64``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from myfpo.myfpo import (
    Evaluator,
    RunningMeanStd,
    _init_wandb,
    compute_eval_summary_metrics,
    resolve_action_bounds,
    resolve_device,
    set_seed,
    summarize_results,
)

__all__ = ["run_ppo_training"]

def _random_baseline(evaluator, max_episodes: int = 30) -> float:
    """Mean return of a uniform-random policy, the divergence reference."""
    baseline = evaluator.clone(max_episodes=max_episodes)
    results = baseline.evaluate(baseline.random_policy(baseline.env), out_dir=None)
    if not results:
        return 0.0
    return float(np.mean([r["return"] for r in results]))

LOG_2 = math.log(2.0)


# ---------------------------------------------------------------------------
# env factory (SameStep autoreset; forkserver workers)
# ---------------------------------------------------------------------------

def _make_vec_env(env_name: str, num_envs: int, record: dict | None = None):
    """Vector env with ``SameStep`` autoreset; forkserver workers."""
    import gymnasium as gym

    def make_one():
        if env_name.startswith("Meta-World/"):
            import metaworld  # noqa: F401

            task = env_name[len("Meta-World/"):]
            return gym.make("Meta-World/MT1", env_name=task)
        if env_name.startswith("AdroitHand"):
            try:
                import gymnasium_robotics  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"{env_name} needs gymnasium-robotics: {exc}"
                ) from exc
        return gym.make(env_name)

    env_fns = [make_one for _ in range(num_envs)]
    if env_name.startswith("Meta-World/"):
        if record is not None:
            record["vector_env"] = "sync"
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    try:
        env = gym.vector.AsyncVectorEnv(env_fns, context="forkserver", autoreset_mode="SameStep")
        if record is not None:
            record["vector_env"] = "forkserver"
    except (ValueError, RuntimeError, ImportError):
        if record is not None:
            record["vector_env"] = "sync"
        return gym.vector.SyncVectorEnv(env_fns, autoreset_mode="SameStep")
    return env


# ---------------------------------------------------------------------------
# action squash (tanh to env bounds; log-prob keeps the correction)
# ---------------------------------------------------------------------------

def _squash(u: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.tanh(u) * scale + bias


def _tanh_logprob_correction(u: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Sum over dims of log|da/du|, a = scale * tanh(u) + bias."""
    return (log_scale + 2.0 * (LOG_2 - u - nn.functional.softplus(-2.0 * u))).sum(-1)


def _layer_init(layer: nn.Linear, std: float, bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PPOPolicy(nn.Module):
    """Gaussian actor (state-independent log_std) + single critic.

    Observations enter through a shared ``RunningMeanStd`` (frozen during the
    optimizer update, refreshed afterwards from the raw rollout frames). The
    action bounds live on the module as buffers, so ``deterministic_actions``
    yields env-ready actions.
    """

    def __init__(self, obs_dim: int, action_dim: int, action_low: np.ndarray,
                 action_high: np.ndarray, hidden_dim: int):
        super().__init__()
        self.obs_norm = RunningMeanStd(obs_dim)
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim), math.sqrt(2.0)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), math.sqrt(2.0)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, action_dim), 0.01),
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim), math.sqrt(2.0)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim), math.sqrt(2.0)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), 1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim, dtype=torch.float32))
        low, high = resolve_action_bounds(action_low, action_high)
        scale = ((high - low) * 0.5).reshape(1, -1)
        self.register_buffer("act_scale", scale)
        self.register_buffer("act_bias", ((high + low) * 0.5).reshape(1, -1))

    def _dist(self, obs: torch.Tensor):
        mean = self.actor(self.obs_norm(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs_norm(obs)).squeeze(-1)

    def sample(self, obs: torch.Tensor):
        dist = self._dist(obs)
        u = dist.rsample()
        logp = dist.log_prob(u).sum(-1) - _tanh_logprob_correction(
            u, torch.log(self.act_scale)
        )
        return u, logp, _squash(u, self.act_scale, self.act_bias)

    def evaluate_u(self, obs: torch.Tensor, u: torch.Tensor):
        """Recompute log-prob of stored pre-squash actions ``u``."""
        dist = self._dist(obs)
        logp = dist.log_prob(u).sum(-1) - _tanh_logprob_correction(
            u, torch.log(self.act_scale)
        )
        return logp, self.value(obs)

    @torch.no_grad()
    def env_action(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        obs = obs.float()
        if deterministic:
            return _squash(self.actor(self.obs_norm(obs)), self.act_scale, self.act_bias)
        _, _, env_a = self.sample(obs)
        return env_a


def _final_obs_from_info(info, num_envs):
    """Rows of the true final obs for envs that ended, aligned to env index.

    Returns ``(rows, mask)``: ``rows`` shape ``[k, obs_dim]`` stacked on the
    same device layout as the caller expects, ``mask`` shape ``[num_envs]``
    bool. Empty when the step ended nothing.
    """
    fo = info.get("final_obs") if isinstance(info, dict) else None
    mask = info.get("_final_obs") if isinstance(info, dict) else None
    if fo is None or mask is None:
        return None, torch.zeros(num_envs, dtype=torch.bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.any():
        return None, torch.zeros(num_envs, dtype=torch.bool)
    rows = [np.asarray(fo[i], dtype=np.float32) for i in range(mask.shape[0]) if mask[i]]
    return torch.as_tensor(np.stack(rows), dtype=torch.float32), torch.as_tensor(mask, dtype=torch.bool)


# ---------------------------------------------------------------------------
# checkpointing (minimal; no resume support by design)
# ---------------------------------------------------------------------------

def _save_checkpoint(out_dir: Path, step: int, policy, optimizer, name: str | None = None) -> str:
    ckpt_dir = out_dir / (name or f"checkpoint_{step}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": int(step),
            "policy_class": "PPOPolicy",
        },
        ckpt_dir / "checkpoint.pt",
    )
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# deterministic eval adapter (matches myfpo's Evaluator contract)
# ---------------------------------------------------------------------------

class _DeterministicPolicy:
    """``obs -> env action`` for :class:`Evaluator`; pins eval mode."""

    def __init__(self, policy: PPOPolicy, device: torch.device):
        self.policy = policy  # picked up by Evaluator's _pinned_eval_mode
        self.device = device

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy.env_action(x, deterministic=True)
        return action.squeeze(0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------

def run_ppo_training(
    env,
    out_dir,
    cfg,
    *,
    seed: int = 3407,
    total_steps: int = 2000,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    resume_from_checkpoint=None,
    runtime: dict | None = None,
    video_dir=None,
    env_id: str | None = None,
):
    """Train plain PPO. Env builds before any CUDA-side work."""
    if resume_from_checkpoint is not None:
        raise RuntimeError("--resume is myfpo-only; PPO cannot resume a flow checkpoint")
    if callable(getattr(env, "step_torch", None)):
        raise RuntimeError(
            "--method ppo supports CPU gymnasium envs (MuJoCo etc.) only; "
            "MS-* torch envs are not supported"
        )
    if video_dir is not None:
        print("ppo: --video requires a torch env; skipping video", flush=True)
        video_dir = None

    cfg = cfg or _default_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = getattr(getattr(env, "unwrapped", env), "spec", None)
    env_name = env_id or (getattr(spec, "id", None) if spec is not None else "unknown") or "unknown"

    if obs_dim is None or action_dim is None:
        raise ValueError("obs_dim and action_dim must be resolved by the caller (scripts/train.py)")

    set_seed(seed)
    device = resolve_device()

    # --- envs (SameStep so truncations expose the true final observation) ---
    train_env = _make_vec_env(env_name, cfg.num_envs, record=runtime) if cfg.num_envs > 1 else _make_vec_env(
        env_name, 1, record=runtime
    )
    if runtime is not None:
        (out_dir / "run_metadata.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8"
        )
    eval_seed = 10000 + seed
    evaluator = Evaluator(env, eval_seed=eval_seed, max_episodes=cfg.eval_max_episodes)

    action_low = np.atleast_1d(np.asarray(env.action_space.low, dtype=np.float32))
    action_high = np.atleast_1d(np.asarray(env.action_space.high, dtype=np.float32))

    policy = PPOPolicy(obs_dim, action_dim, action_low, action_high, cfg.hidden_dim).to(device)
    actor_params = [*policy.actor.parameters(), policy.log_std]
    critic_params = list(policy.critic.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": cfg.policy_lr},
            {"params": critic_params, "lr": cfg.critic_lr},
        ],
        betas=tuple(cfg.adam_betas),
        weight_decay=cfg.optimizer_weight_decay,
        eps=1e-5,
    )
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"ppo policy: 2-layer tanh MLP hidden={cfg.hidden_dim} params={num_params:,}", flush=True)

    r_rand = _random_baseline(evaluator)
    wandb_run = _init_wandb(cfg, method="ppo", env_name=env_name, seed=seed, total_steps=total_steps)
    artifacts: list[str] = []

    steps_per_env = max(1, int(cfg.chunks_per_update) // int(cfg.num_envs))

    global_step = 0
    best_return: float | None = None
    best_artifact: str | None = None
    steps_to_best = 0
    all_eval_returns: list[float] = []
    recent_eval_returns: list[float] = []
    next_eval = int(eval_steps)
    next_save = int(save_steps)

    obs, _ = train_env.reset(seed=[int(seed + i) for i in range(int(cfg.num_envs))])

    t_start = time.time()
    elapsed = 0.0
    while global_step < total_steps:
        steps_this_update_env = min(steps_per_env, math.ceil((total_steps - global_step) / int(cfg.num_envs)))
        t0 = time.perf_counter()

        obs_buf = torch.zeros((steps_this_update_env, cfg.num_envs, obs_dim), device=device)
        u_buf = torch.zeros((steps_this_update_env, cfg.num_envs, action_dim), device=device)
        env_a_buf = torch.zeros((steps_this_update_env, cfg.num_envs, action_dim), device=device)
        logp_buf = torch.zeros((steps_this_update_env, cfg.num_envs), device=device)
        val_buf = torch.zeros((steps_this_update_env, cfg.num_envs), device=device)
        rew_buf = torch.zeros((steps_this_update_env, cfg.num_envs), device=device)
        term_buf = torch.zeros((steps_this_update_env, cfg.num_envs), device=device)
        trunc_buf = torch.zeros((steps_this_update_env, cfg.num_envs), device=device)
        trunc_final_obs = torch.zeros((steps_this_update_env, cfg.num_envs, obs_dim), device=device)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            for t in range(steps_this_update_env):
                obs_buf[t] = obs_t
                u, logp, env_a = policy.sample(obs_t)
                val_buf[t] = policy.value(obs_t)
                u_buf[t] = u
                env_a_buf[t] = env_a
                logp_buf[t] = logp

                obs_np, rew, term, trunc, info = train_env.step(
                    env_a.detach().cpu().numpy().astype(np.float32)
                )
                rew_buf[t] = torch.as_tensor(rew, dtype=torch.float32, device=device)
                term_np = np.asarray(term, dtype=np.float32)
                trunc_np = np.asarray(trunc, dtype=np.float32)
                term_buf[t] = torch.as_tensor(term_np, device=device)
                trunc_buf[t] = torch.as_tensor(trunc_np, device=device)
                final_rows, final_mask = _final_obs_from_info(info, cfg.num_envs)
                if final_rows is not None:
                    trunc_final_obs[t][final_mask] = final_rows.to(device)
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        # --- GAE with correct bootstrap for terminated vs truncated ---
        with torch.no_grad():
            trunc_boot_mask = trunc_buf.bool()
            trunc_boot_values = torch.zeros_like(val_buf)
            if trunc_boot_mask.any():
                trunc_boot_values[trunc_boot_mask] = policy.value(
                    trunc_final_obs[trunc_boot_mask]
                )
            next_obs_value = policy.value(obs_t)  # obs after the last step
            nv = torch.empty_like(rew_buf)
            nv[:-1] = val_buf[1:]
            nv[-1] = next_obs_value
            nv = torch.where(term_buf.bool(), torch.zeros_like(nv), nv)
            nv = torch.where(trunc_boot_mask, trunc_boot_values, nv)

            advantages = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(steps_this_update_env)):
                delta = rew_buf[t] + cfg.gamma * nv[t] - val_buf[t]
                cont = 1.0 - (term_buf[t] + trunc_buf[t]).clamp(max=1.0)
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * cont * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_u = u_buf.reshape(-1, action_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        if cfg.normalize_advantage:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        # raw rollout frames feed the running obs normalizer AFTER the update
        raw_obs_for_normalizer = b_obs.detach().clone()

        batch_size = b_obs.shape[0]
        minibatch = max(1, int(cfg.chunks_per_minibatch))
        inds = np.arange(batch_size)
        actor_grad_norm = 0.0
        pg_loss = v_loss = torch.tensor(0.0)
        clipfrac = approx_kl = 0.0
        nclip = 0
        explained_var = float("nan")

        for _ in range(int(cfg.num_learning_epochs)):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch):
                mb = torch.as_tensor(inds[start:start + minibatch], dtype=torch.long, device=device)
                new_logp, new_value = policy.evaluate_u(b_obs[mb], b_u[mb])
                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl += float(((ratio - 1.0) - logratio).mean().item())
                    clipfrac += float(((ratio - 1.0).abs() > cfg.clip_param).float().mean().item())
                    nclip += 1

                mb_adv = b_adv[mb]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.reshape(-1)
                v_loss = 0.5 * (new_value - b_returns[mb]).pow(2).mean()
                loss = pg_loss + cfg.value_loss_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()), cfg.max_grad_norm
                )
                optimizer.step()

        with torch.no_grad():
            actor_grad_norm = grad_norm.item() if grad_norm is not None else 0.0
            var_y = returns.var(unbiased=False).item()
            explained_var = 1.0 - float((returns - val_buf).var(unbiased=False).item() / var_y) if var_y > 0 else float("nan")
            approx_kl = approx_kl / max(nclip, 1)
            clipfrac = clipfrac / max(nclip, 1)

        policy.obs_norm.update(raw_obs_for_normalizer)

        steps_done = int(steps_this_update_env * cfg.num_envs)
        global_step += steps_done
        collect_s = time.perf_counter() - t0
        elapsed = time.time() - t_start
        fps = global_step / elapsed if elapsed > 0.0 else 0.0

        if wandb_run is not None:
            executed_abs_max = float(env_a_buf.abs().max().item())
            executed_abs_mean = float(env_a_buf.abs().mean().item())
            wandb_run.log(
                {
                    "train/policy_loss": float(pg_loss.item()),
                    "train/value_loss": float(v_loss.item()),
                    "train/total_loss": float(pg_loss.item() + cfg.value_loss_coef * v_loss.item()),
                    "train/approx_kl": approx_kl,
                    "train/clipfrac": clipfrac,
                    "train/explained_var": explained_var,
                    "train/grad_norm_before_clip": actor_grad_norm,
                    "train/policy_lr": optimizer.param_groups[0]["lr"],
                    "train/critic_lr": optimizer.param_groups[1]["lr"],
                    "train/collect_s": collect_s,
                    "train/executed_action_abs_mean": executed_abs_mean,
                    "train/executed_action_abs_max": executed_abs_max,
                },
                step=global_step,
            )

        if global_step >= next_eval or global_step >= total_steps:
            results = evaluator.evaluate(
                _DeterministicPolicy(policy, device), out_dir=out_dir, append_csv=True
            )
            mean_return = float(
                summarize_results(results, seed=eval_seed)["return_mean"]
            ) if results else 0.0
            all_eval_returns.append(mean_return)
            recent_eval_returns.append(mean_return)
            if best_return is None or mean_return > best_return:
                best_return, steps_to_best = mean_return, global_step
                best_artifact = _save_checkpoint(out_dir, global_step, policy, optimizer, name="checkpoint_best")
                if best_artifact not in artifacts:
                    artifacts.append(best_artifact)
            if wandb_run is not None:
                summary = summarize_results(results, seed=eval_seed)
                wandb_run.log(
                    {
                        "eval/ema_active": 0.0,
                        "eval/return_ci_low": summary["return_ci_low"],
                        "eval/return_ci_high": summary["return_ci_high"],
                        "eval/return_ci_width": summary["return_ci_high"] - summary["return_ci_low"],
                        "eval/return_median": summary["return_median"],
                        "eval/return_n": summary["return_n"],
                        "eval/return_mean": mean_return,
                        "eval/return_best": float(best_return),
                        "eval/steps_to_best": float(steps_to_best),
                        "eval/random_policy_return": float(r_rand),
                    },
                    step=global_step,
                )
                wandb_run.log(
                    compute_eval_summary_metrics(
                        mean_return,
                        results,
                        all_eval_returns,
                        r_rand=r_rand,
                        best_return=best_return,
                        steps_to_best=steps_to_best,
                    ),
                    step=global_step,
                )
            next_eval = max(global_step + eval_steps, next_eval + eval_steps)

        if global_step >= next_save or global_step >= total_steps:
            ckpt = _save_checkpoint(out_dir, global_step, policy, optimizer)
            if ckpt not in artifacts:
                artifacts.append(ckpt)
            next_save = max(global_step + save_steps, next_save + save_steps)

    final_path = _save_checkpoint(out_dir, global_step, policy, optimizer)
    if final_path not in artifacts:
        artifacts.append(final_path)
    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        artifacts.append(str(csv_path))

    if wandb_run is not None:
        wandb_run.log({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=global_step)
        wandb_run.finish(exit_code=0)

    if train_env is not None:
        train_env.close()

    return {
        "method": "ppo",
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": float(np.mean(recent_eval_returns)) if recent_eval_returns else 0.0,
        "reward_median_last10": float(np.median(recent_eval_returns)) if recent_eval_returns else 0.0,
        "fps": fps,
        "artifacts": artifacts,
        "best_artifact": best_artifact or (artifacts[-1] if artifacts else None),
        "best_return": best_return,
        "global_step": global_step,
    }


def _default_config():
    from myfpo.myfpo import MyfpoConfig

    return MyfpoConfig()

