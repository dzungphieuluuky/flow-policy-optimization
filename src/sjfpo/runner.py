"""SJFPO end-to-end run driver (MuJoCo-only).

Glues collector → GAE → trainer → evaluator → checkpointing with
global-step accounting. MuJoCo-only, flat Box obs.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import einx
import gymnasium as gym

from sjfpo.checkpointing import save_checkpoint
from sjfpo.collector import collect_chunks
from sjfpo.config import SJFPOConfig
from sjfpo.dct_utils import dct, idct
from sjfpo.evaluator import Evaluator, check_divergence
from sjfpo.fpo_core import total_update_loss
from sjfpo.fpo_core import euler_sample
from sjfpo.gae import chunk_gae
from sjfpo.utils import is_bf16_available, resolve_device
from sjfpo.networks import (
    StateEncoder,
    SJFOPolicyModel,
    SpectralEncoder,
    Critic,
)
from sjfpo.trainer import SJFPOTrainer
from sjfpo.wandb_logging import (
    init_wandb_run_for_method,
    log_eval_summary,
    log_metrics,
    finish_run,
)


def decode_chunk_actions(x0: torch.Tensor) -> torch.Tensor:
    """IDCT-decode a [1, H, A] chunk to clamped [H, A] actions."""
    raw = idct(einx.id("1 h a -> h a", x0), dim=0)
    return raw.clamp(-1.0, 1.0)


def infer_chunk_actions(
    state: torch.Tensor,
    spectral,
    policy_head,
    cfg,
    noise: bool = False,
) -> torch.Tensor:
    """Shared forward path: [T, obs] state → spectral → Euler → [H, A].

    Single home for the ChunkPolicy and SjfpoEvalWrapper forward passes;
    wrappers own buffer/state plumbing, this owns only the forward math.
    Compute runs on the policy head's device; the returned chunk is
    always CPU (env stepping and transition storage stay on CPU).
    """
    dev = next(policy_head.parameters()).device
    with torch.no_grad():
        state_b = einx.id("t d -> 1 t d", state).to(dev)  # [1, T, obs]
        cond = spectral(state_b)
        act_dim = policy_head.out.out_features
        # Denoise in truncated coefficient space (top-p rows), then zero-pad
        # the time dim back to H: missing high-frequency bins are exactly 0.
        n_coeffs = max(1, round(cfg.top_p_spectral_tokens * cfg.chunk_size))
        shape = (1, n_coeffs, act_dim)
        eps = None if noise else torch.zeros(*shape, device=dev)
        # Autocast to bf16 only where the GPU supports it (else fp32 no-op).
        use_bf16 = dev.type == "cuda" and is_bf16_available()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            x0 = euler_sample(
                policy_head, shape, cond, steps=cfg.euler_eval, eps=eps, device=dev,
                state=state_b,
            )  # [1, Ka, Da]
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, cfg.chunk_size - n_coeffs))
        return decode_chunk_actions(x0).cpu()  # [H, Da]


class ChunkPolicy:
    """Explicit policy callable — no closure, flat implementation."""

    def __init__(
        self,
        policy_head: SJFOPolicyModel,
        spectral: SpectralEncoder,
        cfg: SJFPOConfig,
        noise: bool = False,
    ) -> None:
        self.policy_head = policy_head
        self.spectral = spectral
        self.cfg = cfg
        self.noise = noise

    def __call__(self, state) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = einx.id("d -> t d", state, t=self.cfg.n_obs_steps)
        return infer_chunk_actions(
            state,
            self.spectral,
            self.policy_head,
            self.cfg,
            noise=self.noise,
        )


class SjfpoEvalWrapper:
    """Chunk-aware eval wrapper for the SJFPO policy.

    The evaluator calls ``policy(obs)`` once per env step expecting a single
    action ``[act_dim]``.  SJFPO policies produce full ``[chunk_size, act_dim]``
    chunks that must be executed open-loop.  This wrapper maintains internal
    state (frame buffer + action buffer) so the single-step evaluator interface
    is satisfied while preserving chunk-level open-loop execution semantics.
    """

    def __init__(
        self,
        policy_head: SJFOPolicyModel,
        spectral: SpectralEncoder,
        cfg: SJFPOConfig,
    ):
        self.policy_head = policy_head
        self.spectral = spectral
        self.cfg = cfg
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=cfg.n_obs_steps)
        self.action_buffer: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        """Clear internal buffers (call at episode start)."""
        self.frame_buffer.clear()
        self.action_buffer.clear()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return a single action for the current obs."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # Warmup: fill buffer with first obs
        if not self.frame_buffer:
            for _ in range(self.cfg.n_obs_steps):
                self.frame_buffer.append(obs_t.clone())
        else:
            self.frame_buffer.append(obs_t)

        if not self.action_buffer:
            # Compute new chunk via the shared forward path (buffer-rotate stays here)
            state = torch.stack(tuple(self.frame_buffer), dim=0)  # [T, obs]
            actions = infer_chunk_actions(
                state,
                self.spectral,
                self.policy_head,
                self.cfg,
            )
            # Buffer only the executed prefix (reversed so popleft gives the
            # first action); the predicted tail is discarded, replan next.
            n_execute = max(
                1, round(self.cfg.executed_vs_predicted_ratio * actions.shape[0])
            )
            for i in range(n_execute - 1, -1, -1):
                self.action_buffer.append(actions[i])

        return self.action_buffer.popleft().numpy()


@dataclasses.dataclass
class TrainLoopState:
    """Mutable loop state threaded through the train_run helpers."""

    env: Any
    cfg: SJFPOConfig
    out_dir: Path
    env_name: str
    seed: int
    models: dict[str, Any]
    trainer: SJFPOTrainer
    evaluator: Evaluator
    train_policy: ChunkPolicy
    eval_policy: Any
    action_low: float
    action_high: float
    total_steps: int
    eval_interval: int
    ckpt_interval: int
    r_rand: float
    wandb_enabled: bool
    t_start: float
    global_step: int = 0
    eval_count: int = 0
    consecutive_below: int = 0
    diverged: bool = False
    artifacts: list = dataclasses.field(default_factory=list)
    all_eval_returns: list = dataclasses.field(default_factory=list)
    fps: float = 0.0


def resolve_env_name(env: Any) -> str:
    """Resolve the gym env id, falling back to unknown."""
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return getattr(spec, "id", "unknown") if spec is not None else "unknown"


def resolve_training_dims(env, obs_dim, act_dim):
    """Resolve flat-Box obs/act dims, defaulting from the env spaces."""
    if obs_dim is None or act_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
    return obs_dim, act_dim


def build_training_components(
    obs_dim: int, act_dim: int, cfg: SJFPOConfig, device=None
) -> dict:
    """Build flow policy, action-critic, raw-obs spectral tokens, critic trunk (all on device)."""
    dev = torch.device(device) if device is not None else resolve_device()
    hidden_dim = cfg.hidden_size
    policy_head = SJFOPolicyModel(act_dim, obs_dim, hidden_size=hidden_dim).to(dev)
    critic = Critic(hidden_size=hidden_dim, act_dim=act_dim).to(dev)
    spectral = SpectralEncoder(
        obs_dim,
        top_p_spectral_tokens=cfg.top_p_spectral_tokens,
        n_obs_steps=cfg.n_obs_steps,
    ).to(dev)
    critic_state_encoder = StateEncoder(
        obs_dim, n_obs_steps=cfg.n_obs_steps, hidden_size=hidden_dim
    ).to(dev)
    return {
        "policy_head": policy_head,
        "critic": critic,
        "spectral": spectral,
        "critic_state_encoder": critic_state_encoder,
    }


def create_run_policies(policy_head, spectral, cfg):
    """Create noisy train policy and chunk-aware eval wrapper."""
    train_policy = ChunkPolicy(policy_head, spectral, cfg, noise=True)
    eval_wrapper = SjfpoEvalWrapper(policy_head, spectral, cfg)
    return train_policy, eval_wrapper  # evaluator calls eval_wrapper(obs)


def compute_divergence_baseline(evaluator, env, seed) -> float:
    """Run random-policy episodes to get R_rand for divergence gating."""
    rng_baseline = np.random.default_rng(seed)
    random_results = evaluator.evaluate(
        lambda obs: env.action_space.sample(), out_dir=None
    )
    r_rand = np.mean([r["return"] for r in random_results]) if random_results else 1.0
    if r_rand == 0:
        r_rand = 1.0  # avoid div-by-zero
    return r_rand


def compute_chunk_values(critic_state_encoder, critic, chunks):
    """Compute V(s,a) chunk values and truncated-bootstrap next values."""
    dev = next(critic_state_encoder.parameters()).device
    with torch.no_grad():
        states = torch.stack([c["observation.state"] for c in chunks]).to(dev)
        actions = torch.stack([c["action"] for c in chunks]).to(dev)
        # Real-space actions for the critic: IDCT the stored DCT coefficients.
        values = critic(
            critic_state_encoder(states), idct(actions, dim=1)
        ).cpu().numpy()
        next_values = np.zeros(len(chunks))  # terminated/continue chunks keep 0
        for i, c in enumerate(chunks):
            if bool(c["next.truncated"]):  # truncated: bootstrap V(s',a')
                z_next = critic_state_encoder(
                    c["next.observation.state"].unsqueeze(0).to(dev)
                )
                next_values[i] = critic(
                    z_next, idct(c["next.action"].unsqueeze(0).to(dev), dim=1)
                ).item()
    return values, next_values


def collect_gae_inputs(chunks):
    """Flatten chunk rewards, lengths, and termination flags for chunk_gae."""
    all_rewards = []
    n_action_steps = []
    terminated = []
    for c in chunks:
        all_rewards.extend(c["next.reward"].tolist())
        n_action_steps.append(int(c["n_action_steps"]))
        terminated.append(bool(c["next.done"]))
    return all_rewards, n_action_steps, terminated


def compute_gae_targets(state, chunks):
    """Compute GAE advantages, returns, and values for collected chunks."""
    values, next_values = compute_chunk_values(
        state.models["critic_state_encoder"], state.models["critic"], chunks
    )
    all_rewards, n_action_steps, terminated = collect_gae_inputs(chunks)
    advantages, returns_np = chunk_gae(
        all_rewards,
        values,
        next_values,
        np.array(n_action_steps),
        np.array(terminated),
        gamma=state.cfg.gamma,
        lam=state.cfg.gae_lambda,
    )
    return advantages, returns_np, values


def log_trainer_diagnostics(state, diag, extra: dict | None = None) -> None:
    """Log per-step trainer diagnostics and the total update loss.

    Single commit per step (diag + total_loss + optional extra, e.g. gae/*)
    so train/* history stays dense like wandb system auto-logging.
    """
    cfg = state.cfg
    data = {f"train/{k}": v for k, v in diag.items()}
    data["train/total_loss"] = total_update_loss(
        diag["policy_loss"],
        diag["value_loss"],
        vf_coef=cfg.vf_coef,
    )
    if extra:
        data.update(extra)
    log_metrics(data, step=state.global_step)


def execute_trainer_update(state, chunks, advantages, returns_np, values):
    """Train on non-terminated chunks; terminated chunks keep GAE-only role."""
    train_idx = [i for i, c in enumerate(chunks) if not bool(c["next.done"])]
    if not train_idx:  # every chunk terminated: nothing to fit this round
        return None
    meta = {"advantages": torch.tensor(advantages[train_idx], dtype=torch.float32)}
    gae_data = (
        {
            "gae/advantages_mean": float(meta["advantages"].mean()),
            "gae/advantages_std": float(meta["advantages"].std()),
            "gae/returns_mean": float(returns_np[train_idx].mean()),
            "gae/values_mean": float(values[train_idx].mean()),
        }
        if state.wandb_enabled
        else None
    )
    diag = state.trainer.assemble_update([chunks[i] for i in train_idx], meta)
    if state.wandb_enabled:
        log_trainer_diagnostics(state, diag, extra=gae_data)
    return diag


def run_eval_and_record(state):
    """Evaluate the policy, log metrics, persist CSV; return mean return."""
    eval_results = state.evaluator.evaluate(state.eval_policy, out_dir=state.out_dir)
    mean_return = np.mean([r["return"] for r in eval_results])
    state.all_eval_returns.append(mean_return)
    state.eval_count += 1
    if state.wandb_enabled:
        log_eval_summary(
            mean_return,
            eval_results,
            state.all_eval_returns,
            state.eval_count,
            state.global_step,
            r_rand=state.r_rand,
            consecutive_below=state.consecutive_below,
        )
    return mean_return


def update_divergence_state(state, mean_return) -> bool:
    """Update divergence counters; return True when training must stop."""
    state.consecutive_below, diverged = check_divergence(
        mean_return, state.r_rand, state.consecutive_below
    )
    if not diverged:
        return False
    state.diverged = True
    if state.wandb_enabled:
        log_metrics(
            {"eval/diverged": 1, "eval/diverge_step": state.global_step},
            step=state.global_step,
        )
    print(
        f"DIVERGED at step {state.global_step}: mean_return={mean_return:.2f} < 0.10*R_rand={0.10 * state.r_rand:.2f}"
    )
    return True


def save_run_checkpoint(state, label) -> str:
    """Persist models, optimizer, and counters under out_dir/label."""
    ckpt_dir = state.out_dir / label
    models = state.models
    checkpoint_state = {
        "models": {
            "policy_head": models["policy_head"],
            "critic": models["critic"],
            "spectral": models["spectral"],
            "critic_state_encoder": models["critic_state_encoder"],
        },
        "optimizer": state.trainer.optimizer,
        "counters": {
            "global_step": state.global_step,
            "_step_count": state.trainer.step_count,
        },
    }
    save_checkpoint(ckpt_dir, checkpoint_state)
    return str(ckpt_dir / "checkpoint.pt")


def drive_training_loop(state) -> None:
    """Drive collect → GAE → train → eval → checkpoint until budget spent."""
    cfg = state.cfg
    while state.global_step < state.total_steps:
        steps_to_go = state.total_steps - state.global_step
        collect_steps = min(cfg.chunks_per_update * cfg.chunk_size, steps_to_go)
        chunks = collect_chunks(
            state.env,
            state.train_policy,
            collect_steps,
            state.action_low,
            state.action_high,
            n_obs_steps=cfg.n_obs_steps,
            # Training executes the full predicted chunk (H); partial execution
            # (receding horizon) applies at inference/eval only.
            n_action_steps=cfg.chunk_size,
        )
        if not chunks:
            break
        state.global_step += sum(int(c["n_action_steps"]) for c in chunks)
        advantages, returns_np, values = compute_gae_targets(state, chunks)
        execute_trainer_update(state, chunks, advantages, returns_np, values)
        if (
            state.global_step % state.eval_interval < cfg.chunk_size
            or state.global_step >= state.total_steps
        ):
            if update_divergence_state(state, run_eval_and_record(state)):
                break
        if (
            state.global_step % state.ckpt_interval < cfg.chunk_size
            or state.global_step >= state.total_steps
        ):
            state.artifacts.append(
                save_run_checkpoint(state, f"checkpoint_{state.global_step}")
            )


def record_timing_metrics(state) -> float:
    """Log fps/elapsed timing; return fps."""
    elapsed = time.time() - state.t_start
    fps = state.global_step / max(elapsed, 1e-6)
    if state.wandb_enabled:
        log_metrics(
            {"timing/fps": fps, "timing/elapsed_s": elapsed}, step=state.global_step
        )
    return fps


def write_final_checkpoint(state) -> None:
    """Write the final checkpoint and record CSV/ckpt artifacts."""
    state.artifacts.append(save_run_checkpoint(state, "checkpoint_final"))
    csv_path = state.out_dir / "eval_history.csv"
    if csv_path.exists():
        state.artifacts.append(str(csv_path))


def build_result_summary(state) -> dict[str, Any]:
    """Build the result dict with env name, reward mean, fps, artifacts."""
    reward_mean_last10 = (
        float(np.mean(state.all_eval_returns[-10:])) if state.all_eval_returns else 0.0
    )
    return {
        "method": "sjfpo",
        "env_name": (
            getattr(getattr(state.env, "unwrapped", state.env).spec, "id", "unknown")
            if hasattr(state.env, "unwrapped")
            and hasattr(state.env.unwrapped, "spec")
            and getattr(state.env.unwrapped, "spec", None) is not None
            else "unknown"
        ),
        "seed": state.seed,
        "reward_mean_last10": reward_mean_last10,
        "fps": state.fps,
        "artifacts": state.artifacts,
        "diverged": state.diverged,
        "global_step": state.global_step,
    }


def print_training_modules(parts: dict) -> None:
    """Print every nn.Module architecture once when training begins.

    Goes to stderr so CLI stdout stays JSON-clean (scripts/train.py prints
    the result dict; tests parse it).
    """
    for name, module in parts.items():
        print(f"--- {name} ---", file=sys.stderr)
        print(module, file=sys.stderr)


def assemble_run_state(
    env, out_dir, cfg, seed, total_steps, eval_interval, ckpt_interval, obs_dim, act_dim
) -> TrainLoopState:
    """Build networks, trainer, evaluator, policies, and baseline state."""
    out_dir = Path(out_dir)
    env_name_str = resolve_env_name(env)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim, act_dim = resolve_training_dims(env, obs_dim, act_dim)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device()  # CUDA if available, else CPU (SJFPO_DEVICE overrides)
    parts = build_training_components(obs_dim, act_dim, cfg, device=device)
    print_training_modules(parts)
    trainer = SJFPOTrainer(
        parts["policy_head"],
        parts["critic"],
        parts["spectral"],
        parts["critic_state_encoder"],
        cfg=cfg,
        device=device,
    )
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    train_policy, eval_policy = create_run_policies(
        parts["policy_head"], parts["spectral"], cfg
    )
    r_rand = compute_divergence_baseline(evaluator, env, seed)
    cfg_dict = dataclasses.asdict(cfg)
    wandb_enabled = init_wandb_run_for_method(
        method="sjfpo",
        env_name=env_name_str,
        seed=seed,
        total_steps=total_steps,
        cfg_dict=cfg_dict,
    )
    return TrainLoopState(
        env=env,
        cfg=cfg,
        out_dir=out_dir,
        env_name=env_name_str,
        seed=seed,
        models=parts,
        trainer=trainer,
        evaluator=evaluator,
        train_policy=train_policy,
        eval_policy=eval_policy,
        action_low=-1.0,
        action_high=1.0,
        total_steps=total_steps,
        eval_interval=eval_interval,
        ckpt_interval=ckpt_interval,
        r_rand=r_rand,
        wandb_enabled=wandb_enabled,
        t_start=time.time(),
    )


def train_run(
    env: Any,
    out_dir: str | Path,
    cfg: SJFPOConfig | None = None,
    *,
    seed: int = 0,
    total_steps: int = 2000,
    eval_interval: int = 1000,
    ckpt_interval: int = 1000,
    obs_dim: int | None = None,
    act_dim: int | None = None,
) -> dict[str, Any]:
    """Run a single SJFPO training run, returning summary with artifacts."""
    cfg = cfg or SJFPOConfig()
    state = assemble_run_state(
        env,
        out_dir,
        cfg,
        seed,
        total_steps,
        eval_interval,
        ckpt_interval,
        obs_dim,
        act_dim,
    )

    # --- Main loop ---
    try:
        drive_training_loop(state)
        state.fps = record_timing_metrics(state)
    finally:
        if state.wandb_enabled:
            finish_run(exit_code=1 if state.diverged else 0)

    write_final_checkpoint(state)
    return build_result_summary(state)
