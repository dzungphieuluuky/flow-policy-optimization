"""skrl PPO/SAC baselines (plan tasks 11-12).

Wraps skrl 2.1.0 PPO/SAC into the same ``*_train_run`` interface as SJFPO's
``train_run`` (runner.py), sharing ``Evaluator`` and checkpointing.
Frame-stacking is identical (n_obs_steps=16 → flat ``[obs_dim*16]``).

Tensor conventions:
    obs_dim: flat per-step obs (MuJoCo)
    act_dim: action dim (env-dependent)
    n_obs_steps 16 → stacked obs ``[obs_dim*16]``, wrapped ``[N=1, obs_dim*16]`` in act()
    chunk_size 8 → action chunk ``[8, act_dim]`` DCT ``[8, act_dim]`` → IDCT

All fp32, no autocast/bf16/fp16. Provenance table is source-of-truth for hyperparameters.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import einx
import torch
import torch.nn as nn

from sjfpo.config import PPOConfig, SACConfig
from sjfpo.evaluator import Evaluator, check_divergence, compute_random_baseline
from sjfpo.wandb_logging import (
    init_wandb_run_for_method,
    log_eval_summary,
    log_metrics,
    finish_run,
)

# ---------------------------------------------------------------------------
# skrl imports (lazy to allow partial import when skrl is unavailable)
# ---------------------------------------------------------------------------

skrl_imported = False
PPO_CFG_cls = None
SAC_CFG_cls = None
PPO_cls = None
SAC_cls = None
RandomMemory_cls = None
wrap_env_fn = None
RunningStandardScaler_cls = None

# Model classes (created after import_skrl_dependencies runs)
PPOPolicy = None
PPOValue = None
SACPolicy = None
SACCritic = None


# ---------------------------------------------------------------------------
# Provenance table  (method, key, label) triples
# label: "default" = skrl default untouched; "paper" = from FPO++ paper;
#        "adaptation" = changed from skrl default for benchmark fairness
# ---------------------------------------------------------------------------

PROVENANCE_TABLE: list[tuple[str, str, str, str]] = [
    # (method, key, value_source, note)
    ("PPO", "rollouts", "2048", "default"),
    ("PPO", "learning_epochs", "10", "default"),
    ("PPO", "mini_batches", "8", "default"),
    ("PPO", "learning_rate", "3e-4", "paper"),
    ("PPO", "kl_threshold", "0.01", "default"),
    ("PPO", "grad_norm_clip", "1.0", "default"),
    ("PPO", "discount_factor", "0.99", "adaptation"),
    ("PPO", "gae_lambda", "0.95", "adaptation"),
    ("PPO", "ratio_clip", "0.2", "default"),
    ("SAC", "learning_rate", "3e-4", "default"),
    ("SAC", "batch_size", "256", "default"),
    ("SAC", "memory_size", "300000", "adaptation"),
    ("SAC", "gradient_steps", "1", "default"),
    ("SAC", "polyak", "0.005", "default"),
    ("SAC", "initial_random_steps", "5000", "adaptation"),
]


def build_mlp_layer(
    in_dim: int, hidden: tuple[int, ...], out_dim: int
) -> nn.Sequential:
    from sjfpo.networks import SwiGLU

    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(SwiGLU(prev, h, h))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def import_skrl_agent_classes():
    """Import skrl agent/memory/wrapper/preprocessor symbols (single home)."""
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.agents.torch.sac import SAC, SAC_CFG
    from skrl.memories.torch import RandomMemory
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    return (
        PPO_CFG,
        SAC_CFG,
        PPO,
        SAC,
        RandomMemory,
        wrap_env,
        RunningStandardScaler,
    )


def import_skrl_model_mixins():
    """Import skrl model mixin/base symbols for model-class creation."""
    from skrl.models.torch import GaussianMixin as GM, DeterministicMixin as DM, Model

    return GM, DM, Model


def create_gaussian_policy_class(GM, Model):
    """Shared Gaussian actor: MLP → mean + learnable log_std (PPO and SAC)."""

    class GaussianPolicyImpl(GM, Model):
        """Shared Gaussian actor: MLP → mean + learnable log_std (PPO and SAC)."""

        def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
            Model.__init__(
                self,
                observation_space=gym.spaces.Box(-1, 1, (obs_dim,)),
                action_space=gym.spaces.Box(-1, 1, (act_dim,)),
                device="cpu",
            )
            GM.__init__(
                self,
                clip_actions=True,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2,
                reduction="sum",
            )
            self.net = build_mlp_layer(obs_dim, hidden, act_dim)
            self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

        def compute(self, inputs: dict, role: str = "policy"):
            return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}

    return GaussianPolicyImpl


def create_ppo_value_class(DM, Model):
    """Deterministic value function for PPO: MLP → scalar."""

    class PpoValueImpl(DM, Model):
        """Deterministic value function for PPO: MLP → scalar."""

        def __init__(self, obs_dim: int, hidden: tuple[int, ...]):
            Model.__init__(
                self,
                observation_space=gym.spaces.Box(-1, 1, (obs_dim,)),
                device="cpu",
            )
            DM.__init__(self, clip_actions=False)
            self.net = build_mlp_layer(obs_dim, hidden, 1)

        def compute(self, inputs: dict, role: str = "value"):
            return self.net(inputs["observations"]), {}

    return PpoValueImpl


def create_sac_critic_class(DM, Model):
    """Q-critic for SAC: MLP over [obs; action] → scalar."""

    class SacCriticImpl(DM, Model):  # type: ignore[name-defined]
        """Q-critic for SAC: MLP over [obs; action] → scalar."""

        def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
            Model.__init__(
                self,
                observation_space=gym.spaces.Box(-1, 1, (obs_dim,)),
                action_space=gym.spaces.Box(-1, 1, (act_dim,)),
                device="cpu",
            )
            DM.__init__(self, clip_actions=False)
            self.net = build_mlp_layer(obs_dim + act_dim, hidden, 1)

        def compute(self, inputs: dict, role: str = "critic"):
            x = einx.id(
                "b o, b a -> b (o + a)", inputs["observations"], inputs["taken_actions"]
            )  # [B, obs+act]
            return self.net(x), {}

    return SacCriticImpl


def import_skrl_dependencies():
    global skrl_imported, PPO_CFG_cls, SAC_CFG_cls, PPO_cls, SAC_cls
    global RandomMemory_cls, wrap_env_fn, RunningStandardScaler_cls
    global PPOPolicy, PPOValue, SACPolicy, SACCritic

    if skrl_imported:
        return
    (
        PPO_CFG_cls,
        SAC_CFG_cls,
        PPO_cls,
        SAC_cls,
        RandomMemory_cls,
        wrap_env_fn,
        RunningStandardScaler_cls,
    ) = import_skrl_agent_classes()
    GM, DM, Model = import_skrl_model_mixins()
    PPOPolicy = create_gaussian_policy_class(GM, Model)
    PPOValue = create_ppo_value_class(DM, Model)
    SACPolicy = PPOPolicy
    SACCritic = create_sac_critic_class(DM, Model)
    skrl_imported = True


def seed_and_wrap_env(
    env: Any, seed: int, n_obs_steps: int
) -> tuple[Any, Any, int, int]:
    """Seed torch/numpy and wrap env with frame stacking + skrl wrapper."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    wrapped_env = FrameStackWrapper(env, n_obs_steps=n_obs_steps)
    skrl_env = wrap_env_fn(wrapped_env, wrapper="gymnasium", verbose=False)
    obs_dim = wrapped_env.observation_space.shape[0]
    act_dim = wrapped_env.action_space.shape[0]
    return wrapped_env, skrl_env, obs_dim, act_dim


def mean_action_eval_policy(agent: Any) -> Any:
    """Deterministic mean-action eval policy closure for a skrl agent."""

    def eval_policy(obs):
        obs_t = einx.id(
            "d -> 1 d", torch.as_tensor(obs, dtype=torch.float32)
        )  # [1, obs]
        with torch.no_grad():
            agent.enable_training_mode(False)
            actions, outputs = agent.act(obs_t, None, timestep=0, timesteps=1)
            agent.enable_training_mode(True)
        return einx.id("1 a -> a", outputs.get("mean_actions", actions)).cpu().numpy()

    return eval_policy

    # ---------------------------------------------------------------------------
    # Frame-stack wrapper (identical to SJFPO's n_obs_steps treatment)
    # ---------------------------------------------------------------------------


class FrameStackWrapper(gym.ObservationWrapper):
    """Stacks the last ``n_obs_steps`` observations along a new axis.

    The resulting observation is a flat vector of shape
    ``(obs_dim * n_obs_steps,)`` so that skrl's MLP sees the full history.
    """

    def __init__(self, env: gym.Env, n_obs_steps: int = 16):
        super().__init__(env)
        self.n_obs_steps = n_obs_steps
        obs_dim = int(np.prod(env.observation_space.shape))
        self.obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim * n_obs_steps,),
            dtype=np.float32,
        )
        self.frames: deque = deque(maxlen=n_obs_steps)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        flat = np.asarray(obs, dtype=np.float32).ravel()
        for _ in range(self.n_obs_steps):
            self.frames.append(flat.copy())
        return self.get_current_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        flat = np.asarray(obs, dtype=np.float32).ravel()
        self.frames.append(flat.copy())
        return self.get_current_observation(), reward, terminated, truncated, info

    def get_current_observation(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)

        # ---------------------------------------------------------------------------
        # Generic training loop (shared by PPO and SAC)
        # ---------------------------------------------------------------------------


@dataclasses.dataclass
class BaselineLoopState:
    """Mutable counters for one baseline_train_loop run."""

    t_start: float
    global_step: int = 0
    eval_count: int = 0
    consecutive_below: int = 0
    diverged: bool = False
    wandb_enabled: bool = False
    env_name: str = "unknown"
    all_eval_returns: list = dataclasses.field(default_factory=list)
    artifacts: list = dataclasses.field(default_factory=list)


def init_baseline_run(env, eval_env, out_dir, method_name, seed, total_steps, cfg_dict):
    """Create out_dir, evaluator, loop state, and the wandb run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(eval_env, eval_seed=10000 + seed, max_episodes=12)
    env_unwrapped = getattr(env, "unwrapped", env)
    env_spec = getattr(env_unwrapped, "spec", None)
    env_name_str = (
        getattr(env_spec, "id", "unknown") if env_spec is not None else "unknown"
    )
    wandb_enabled: bool = init_wandb_run_for_method(
        method=method_name,
        env_name=env_name_str,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=cfg_dict or {},
    )
    state = BaselineLoopState(
        t_start=time.time(), wandb_enabled=wandb_enabled, env_name=env_name_str
    )
    return evaluator, state


def resolve_eval_policy(agent, eval_policy_fn):
    """Fall back to raw agent.act when no eval policy is supplied."""
    if eval_policy_fn is not None:
        return eval_policy_fn
    return lambda obs: (
        agent.act(
            einx.id("d -> 1 d", torch.as_tensor(obs, dtype=torch.float32)),
            None,
            timestep=0,
            timesteps=1,
        )[0]
        .squeeze(0)
        .cpu()
        .numpy()
    )


def query_agent_actions(agent, observations, states, global_step, total_steps):
    """Run agent.act under no_grad; coerce the observation to a tensor."""
    with torch.no_grad():
        actions, outputs = agent.act(
            observations
            if torch.is_tensor(observations)
            else torch.as_tensor(observations, dtype=torch.float32),
            states,
            timestep=global_step,
            timesteps=total_steps,
        )
    return actions


def apply_env_step(env, actions):
    """Step the skrl env; return raw obs/reward/flags plus next states."""
    next_obs, reward, terminated, truncated, info = env.step(
        actions.float()
        if torch.is_tensor(actions)
        else torch.as_tensor(actions, dtype=torch.float32)
    )
    next_states = env.state() if hasattr(env, "state") else None
    return next_obs, reward, terminated, truncated, next_states


def record_agent_transition(
    agent,
    observations,
    states,
    actions,
    next_obs,
    reward,
    terminated,
    truncated,
    next_states,
    global_step,
    total_steps,
):
    """Coerce transition tensors; record and post-interaction with the agent."""
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
    reward_t = torch.as_tensor([reward], dtype=torch.float32)
    terminated_t = torch.as_tensor([terminated])
    truncated_t = torch.as_tensor([truncated])
    obs_t = (
        observations
        if torch.is_tensor(observations)
        else torch.as_tensor(observations, dtype=torch.float32)
    )
    agent.record_transition(
        observations=einx.id("d -> 1 d", obs_t) if obs_t.dim() == 1 else obs_t,
        states=states,
        actions=einx.id("a -> 1 a", actions) if actions.dim() == 1 else actions,
        rewards=reward_t,
        next_observations=einx.id("d -> 1 d", next_obs_t)
        if next_obs_t.dim() == 1
        else next_obs_t,
        next_states=next_states,
        terminated=terminated_t,
        truncated=truncated_t,
        infos=None,
        timestep=global_step,
        timesteps=total_steps,
    )
    agent.post_interaction(timestep=global_step, timesteps=total_steps)


def run_baseline_eval(
    agent, evaluator, eval_policy_fn, out_dir, state, divergence_baseline=None
):
    """Evaluate, track returns, log to wandb, and persist the eval CSV."""
    agent.enable_training_mode(False)
    eval_results = evaluator.evaluate(eval_policy_fn, out_dir=out_dir)
    agent.enable_training_mode(True)
    mean_return = np.mean([r["return"] for r in eval_results]) if eval_results else 0.0
    state.all_eval_returns.append(mean_return)
    state.eval_count += 1
    if state.wandb_enabled:
        log_eval_summary(
            mean_return,
            eval_results,
            state.all_eval_returns,
            state.eval_count,
            state.global_step,
            r_rand=divergence_baseline,
            consecutive_below=state.consecutive_below,
        )
    return mean_return


def check_baseline_divergence(mean_return, divergence_baseline, state):
    """Count sub-10% evals; flag diverged after 3 consecutive misses."""
    state.consecutive_below, diverged = check_divergence(
        mean_return, divergence_baseline, state.consecutive_below
    )
    if not diverged:
        return False
    state.diverged = True
    if state.wandb_enabled:
        log_metrics(
            {"eval/diverged": 1, "eval/diverge_step": state.global_step},
            step=state.global_step,
        )
    return state.diverged


def write_baseline_checkpoint(
    agent, out_dir, method_name, global_step, seed, artifacts, complete=False
):
    """Save the skrl agent plus a manifest; record the artifact path."""
    ckpt_dir = out_dir / (
        "checkpoint_final" if complete else f"checkpoint_{global_step}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(ckpt_dir / "skrl_agent.pt"))
    artifacts.append(str(ckpt_dir / "skrl_agent.pt"))
    manifest = {"method": method_name, "global_step": global_step, "seed": seed}
    if complete:
        manifest["complete"] = True
    with open(ckpt_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def finalize_baseline_run(env, agent, out_dir, method_name, seed, state, fps):
    """Write the final checkpoint and assemble the SJFPO-shaped result dict."""
    write_baseline_checkpoint(
        agent,
        out_dir,
        method_name,
        state.global_step,
        seed,
        state.artifacts,
        complete=True,
    )
    csv_path = out_dir / "eval_history.csv"
    if csv_path.exists():
        state.artifacts.append(str(csv_path))
    reward_mean_last10 = (
        float(np.mean(state.all_eval_returns[-10:])) if state.all_eval_returns else 0.0
    )
    env_name = "unknown"
    if (
        hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "spec")
        and env.unwrapped.spec
    ):
        env_name = env.unwrapped.spec.id
    return {
        "method": method_name,
        "env_name": env_name,
        "seed": seed,
        "reward_mean_last10": reward_mean_last10,
        "fps": fps,
        "artifacts": state.artifacts,
        "diverged": state.diverged,
        "global_step": state.global_step,
    }


def run_baseline_iteration(
    env,
    agent,
    evaluator,
    eval_policy_fn,
    out_dir,
    method_name,
    seed,
    divergence_baseline,
    eval_interval,
    ckpt_interval,
    total_steps,
    observations,
    states,
    state,
):
    """Run one act→step→record step plus scheduled eval/checkpoint; return (obs, states, diverged)."""
    agent.pre_interaction(timestep=state.global_step, timesteps=total_steps)
    actions = query_agent_actions(
        agent, observations, states, state.global_step, total_steps
    )
    next_obs, reward, terminated, truncated, next_states = apply_env_step(env, actions)
    record_agent_transition(
        agent,
        observations,
        states,
        actions,
        next_obs,
        reward,
        terminated,
        truncated,
        next_states,
        state.global_step,
        total_steps,
    )
    state.global_step += 1
    if terminated or truncated:
        observations, _ = env.reset()
        states = env.state() if hasattr(env, "state") else None
    else:
        observations = next_obs
        states = next_states
    if state.global_step % eval_interval == 0 or state.global_step >= total_steps:
        mean_return = run_baseline_eval(
            agent, evaluator, eval_policy_fn, out_dir, state, divergence_baseline
        )
        if check_baseline_divergence(mean_return, divergence_baseline, state):
            return observations, states, True
    if state.global_step % ckpt_interval == 0 or state.global_step >= total_steps:
        write_baseline_checkpoint(
            agent, out_dir, method_name, state.global_step, seed, state.artifacts
        )
    return observations, states, False


def baseline_train_loop(
    env,
    agent,
    *,
    eval_env,
    out_dir,
    seed,
    total_steps,
    eval_interval,
    ckpt_interval,
    method_name,
    eval_policy_fn,
    divergence_baseline=1.0,
    cfg_dict=None,
):
    """Run the skrl agent through a standard act→step→record→post_interaction loop.

    Returns the same result-dict shape as SJFPO's train_run.
    """
    evaluator, state = init_baseline_run(
        env, eval_env, out_dir, method_name, seed, total_steps, cfg_dict
    )
    agent.init()
    eval_policy_fn = resolve_eval_policy(agent, eval_policy_fn)
    agent.enable_training_mode(True)
    observations, _ = env.reset()
    states = env.state() if hasattr(env, "state") else None
    try:
        while state.global_step < total_steps:
            observations, states, diverged = run_baseline_iteration(
                env,
                agent,
                evaluator,
                eval_policy_fn,
                out_dir,
                method_name,
                seed,
                divergence_baseline,
                eval_interval,
                ckpt_interval,
                total_steps,
                observations,
                states,
                state,
            )
            if diverged:
                break
        elapsed = time.time() - state.t_start
        fps = state.global_step / max(elapsed, 1e-6)
        if state.wandb_enabled:
            log_metrics(
                {"timing/fps": fps, "timing/elapsed_s": elapsed}, step=state.global_step
            )
    finally:
        if state.wandb_enabled:
            finish_run(exit_code=1 if state.diverged else 0)
    return finalize_baseline_run(env, agent, out_dir, method_name, seed, state, fps)

    # ---------------------------------------------------------------------------
    # PPO train_run
    # ---------------------------------------------------------------------------


def build_ppo_skrl_config(cfg):
    """Build the skrl PPO_CFG from our PPOConfig (provenance in PROVENANCE_TABLE)."""
    return PPO_CFG_cls(
        rollouts=cfg.rollouts,
        learning_epochs=cfg.learning_epochs,
        mini_batches=cfg.mini_batches,
        discount_factor=cfg.discount,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.lr,
        learning_rate_scheduler=None,
        learning_rate_scheduler_kwargs={},
        observation_preprocessor=None,
        observation_preprocessor_kwargs={},
        state_preprocessor=None,
        state_preprocessor_kwargs={},
        value_preprocessor=None,
        value_preprocessor_kwargs={},
        grad_norm_clip=cfg.grad_norm_clip,
        ratio_clip=cfg.ratio_clip,
        value_clip=0.2,
        entropy_loss_scale=0.0,
        value_loss_scale=cfg.value_loss_scale,
        kl_threshold=cfg.kl_threshold,
        random_timesteps=0,
        learning_starts=0,
        time_limit_bootstrap=False,
        mixed_precision=False,
    )


def build_ppo_agent(policy, value, memory, skrl_env, ppo_cfg, device):
    """Assemble the skrl PPO agent from models, memory, and spaces."""
    return PPO_cls(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=skrl_env.observation_space,
        state_space=skrl_env.state_space,
        action_space=skrl_env.action_space,
        device=device,
        cfg=ppo_cfg,
    )


def launch_baseline_loop(
    skrl_env,
    agent,
    wrapped_env,
    out_dir,
    seed,
    total_steps,
    eval_interval,
    ckpt_interval,
    method_name,
    eval_policy,
    r_rand,
    cfg,
):
    """Invoke baseline_train_loop with the standard train_run arguments."""
    return baseline_train_loop(
        skrl_env,
        agent,
        eval_env=wrapped_env,
        out_dir=out_dir,
        seed=seed,
        total_steps=total_steps,
        eval_interval=eval_interval,
        ckpt_interval=ckpt_interval,
        method_name=method_name,
        eval_policy_fn=eval_policy,
        divergence_baseline=r_rand,
        cfg_dict=dataclasses.asdict(cfg),
    )


def ppo_train_run(
    env: Any,
    out_dir: str | Path,
    cfg: PPOConfig | None = None,
    *,
    seed: int = 0,
    total_steps: int = 2000,
    eval_interval: int = 1000,
    ckpt_interval: int = 1000,
    n_obs_steps: int = 16,
) -> dict[str, Any]:
    """Run a single PPO training run using skrl 2.1.0.

    Returns the same result-dict shape as SJFPO's train_run.
    """
    import_skrl_dependencies()

    cfg = cfg or PPOConfig()
    out_dir = Path(out_dir)
    device = "cpu"
    wrapped_env, skrl_env, obs_dim, act_dim = seed_and_wrap_env(env, seed, n_obs_steps)
    policy = PPOPolicy(obs_dim, act_dim, cfg.hidden)
    value = PPOValue(obs_dim, cfg.hidden)
    memory = RandomMemory_cls(memory_size=cfg.rollouts, num_envs=1, device=device)
    ppo_cfg = build_ppo_skrl_config(cfg)
    agent = build_ppo_agent(policy, value, memory, skrl_env, ppo_cfg, device)
    eval_policy = mean_action_eval_policy(agent)
    r_rand = compute_random_baseline(env, seed, n_obs_steps)
    return launch_baseline_loop(
        skrl_env,
        agent,
        wrapped_env,
        out_dir,
        seed,
        total_steps,
        eval_interval,
        ckpt_interval,
        "ppo",
        eval_policy,
        r_rand,
        cfg,
    )

    # ---------------------------------------------------------------------------
    # SAC train_run
    # ---------------------------------------------------------------------------


def build_sac_skrl_config(cfg):
    """Build the skrl SAC_CFG from our SACConfig (provenance in PROVENANCE_TABLE)."""
    return SAC_CFG_cls(
        gradient_steps=cfg.gradient_steps,
        batch_size=cfg.batch_size,
        discount_factor=cfg.discount,
        polyak=cfg.polyak,
        learning_rate=cfg.lr,
        learning_rate_scheduler=None,
        learning_rate_scheduler_kwargs={},
        observation_preprocessor=None,
        observation_preprocessor_kwargs={},
        state_preprocessor=None,
        state_preprocessor_kwargs={},
        random_timesteps=0,
        learning_starts=cfg.initial_random_steps,
        grad_norm_clip=0,
        learn_entropy=True,
        initial_entropy_value=0.2,
        target_entropy=None,
        mixed_precision=False,
    )


def build_sac_agent(
    policy,
    critic_1,
    critic_2,
    target_critic_1,
    target_critic_2,
    memory,
    skrl_env,
    sac_cfg,
    device,
):
    """Assemble the skrl SAC agent (policy + 2 critics + 2 target critics)."""
    return SAC_cls(
        models={
            "policy": policy,
            "critic_1": critic_1,
            "critic_2": critic_2,
            "target_critic_1": target_critic_1,
            "target_critic_2": target_critic_2,
        },
        memory=memory,
        observation_space=skrl_env.observation_space,
        state_space=skrl_env.state_space,
        action_space=skrl_env.action_space,
        device=device,
        cfg=sac_cfg,
    )


def sac_train_run(
    env: Any,
    out_dir: str | Path,
    cfg: SACConfig | None = None,
    *,
    seed: int = 0,
    total_steps: int = 2000,
    eval_interval: int = 1000,
    ckpt_interval: int = 1000,
    n_obs_steps: int = 16,
) -> dict[str, Any]:
    """Run a single SAC training run using skrl 2.1.0.

    Returns the same result-dict shape as SJFPO's train_run.
    """
    import_skrl_dependencies()

    cfg = cfg or SACConfig()
    out_dir = Path(out_dir)
    device = "cpu"
    wrapped_env, skrl_env, obs_dim, act_dim = seed_and_wrap_env(env, seed, n_obs_steps)
    policy = SACPolicy(obs_dim, act_dim, cfg.hidden)
    critic_1 = SACCritic(obs_dim, act_dim, cfg.hidden)
    critic_2 = SACCritic(obs_dim, act_dim, cfg.hidden)
    target_critic_1 = SACCritic(obs_dim, act_dim, cfg.hidden)
    target_critic_2 = SACCritic(obs_dim, act_dim, cfg.hidden)
    memory = RandomMemory_cls(memory_size=cfg.memory_size, num_envs=1, device=device)
    sac_cfg = build_sac_skrl_config(cfg)
    agent = build_sac_agent(
        policy,
        critic_1,
        critic_2,
        target_critic_1,
        target_critic_2,
        memory,
        skrl_env,
        sac_cfg,
        device,
    )
    eval_policy = mean_action_eval_policy(agent)
    r_rand = compute_random_baseline(env, seed, n_obs_steps)
    return launch_baseline_loop(
        skrl_env,
        agent,
        wrapped_env,
        out_dir,
        seed,
        total_steps,
        eval_interval,
        ckpt_interval,
        "sac",
        eval_policy,
        r_rand,
        cfg,
    )
