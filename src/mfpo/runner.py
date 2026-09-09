"""MFPO end-to-end run driver (MuJoCo-only).

Glues collector → GAE → trainer → evaluator → checkpointing with
global-step accounting. MuJoCo-only, flat Box obs.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import einx
import gymnasium as gym

from mfpo.checkpointing import save_checkpoint
from mfpo.collector import collect_chunks
from mfpo.config import MFPOConfig
from mfpo.dct_utils import dct, idct
from mfpo.evaluator import Evaluator, check_divergence
from mfpo.fpo_core import total_update_loss
from mfpo.fpo_core import euler_sample
from mfpo.gae import chunk_gae
from mfpo.networks import (
    AuxStateFlowHead,
    ContextTrunk,
    EMAProjection,
    PolicyFlowHead,
    SpectralTokens,
    ValueCritic,
)
from mfpo.trainer import MFPOAborted, MFPOTrainer
from mfpo.wandb_logging import (
    init_wandb_run_for_method,
    log_eval_summary,
    log_metrics,
    finish_run,
)


def decode_chunk_actions(x0: torch.Tensor) -> torch.Tensor:
    """IDCT-decode a [1, H, A] chunk to clamped [H, A] actions."""
    raw = idct(einx.id("1 h a -> h a", x0), dim=0)
    return raw.clamp(-1.0, 1.0)


def infer_chunk_actions(context: torch.Tensor, trunk, spectral, policy_head, cfg, noise: bool = False) -> torch.Tensor:
    """Shared forward path: [T, obs] context → trunk → spectral → Euler → [H, A].

    Single home for the PolicyCallable and MpoEvalWrapper forward passes;
    wrappers own buffer/context plumbing, this owns only the forward math.
    """
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        z = trunk(einx.id("t d -> 1 t d", context))
        cond = spectral(z)
        act_dim = policy_head.out.out_features
        shape = (1, cfg.chunk_len, act_dim)
        eps = None if noise else torch.zeros(*shape)
        x0 = euler_sample(policy_head, shape, cond, steps=cfg.euler_eval, eps=eps)
        return decode_chunk_actions(x0)


class PolicyCallable:
    """Explicit policy callable — no closure, flat implementation."""

    def __init__(
        self,
        trunk: ContextTrunk,
        policy_head: PolicyFlowHead,
        spectral: SpectralTokens,
        cfg: MFPOConfig,
        noise: bool = False,
    ) -> None:
        self.trunk = trunk
        self.policy_head = policy_head
        self.spectral = spectral
        self.cfg = cfg
        self.noise = noise

    def __call__(self, context) -> torch.Tensor:
        if isinstance(context, np.ndarray):
            context = torch.as_tensor(context, dtype=torch.float32)
        if context.dim() == 1:
            context = einx.id("d -> t d", context, t=self.cfg.context_len)
        return infer_chunk_actions(context, self.trunk, self.spectral, self.policy_head, self.cfg, noise=self.noise)


class MpoEvalWrapper:
    """Chunk-aware eval wrapper for the MFPO policy.

    The evaluator calls ``policy(obs)`` once per env step expecting a single
    action ``[act_dim]``.  MFPO policies produce full ``[chunk_len, act_dim]``
    chunks that must be executed open-loop.  This wrapper maintains internal
    state (frame buffer + action buffer) so the single-step evaluator interface
    is satisfied while preserving chunk-level open-loop execution semantics.
    """

    def __init__(
        self,
        trunk: ContextTrunk,
        policy_head: PolicyFlowHead,
        spectral: SpectralTokens,
        cfg: MFPOConfig,
    ):
        self._trunk = trunk
        self._policy_head = policy_head
        self._spectral = spectral
        self.cfg = cfg
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=cfg.context_len)
        self._action_buffer: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        """Clear internal buffers (call at episode start)."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return a single action for the current obs."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # Warmup: fill buffer with first obs
        if not self._frame_buffer:
            for _ in range(self.cfg.context_len):
                self._frame_buffer.append(obs_t.clone())
        else:
            self._frame_buffer.append(obs_t)

        if not self._action_buffer:
            # Compute new chunk via the shared forward path (buffer-rotate stays here)
            context = torch.stack(tuple(self._frame_buffer), dim=0)  # [T, obs]
            actions = infer_chunk_actions(context, self._trunk, self._spectral, self._policy_head, self.cfg)
            # Push actions into buffer (reversed so popleft gives first action)
            for i in range(actions.shape[0] - 1, -1, -1):
                self._action_buffer.append(actions[i])

        return self._action_buffer.popleft().numpy()


@dataclasses.dataclass
class TrainLoopState:
    """Mutable loop state threaded through the train_run helpers."""

    env: Any
    cfg: MFPOConfig
    out_dir: Path
    env_name: str
    seed: int
    models: dict[str, Any]
    trainer: MFPOTrainer
    evaluator: Evaluator
    train_policy: PolicyCallable
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
    _unwrapped = getattr(env, "unwrapped", env)
    _spec = getattr(_unwrapped, "spec", None)
    return getattr(_spec, "id", "unknown") if _spec is not None else "unknown"


def resolve_training_dims(env, obs_dim, act_dim):
    """Resolve flat-Box obs/act dims, defaulting from the env spaces."""
    if obs_dim is None or act_dim is None:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
    return obs_dim, act_dim


def build_training_components(obs_dim: int, act_dim: int, cfg: MFPOConfig) -> dict:
    """Build trunk, flow heads, critic, EMA projection, spectral tokens."""
    dk = 64  # SpectralTokens projector output dim
    _Z = cfg.Z_dim
    trunk = ContextTrunk(obs_dim, context_len=cfg.context_len, z_dim=_Z)
    policy_head = PolicyFlowHead(act_dim, z_dim=_Z, cond_dims=(dk,))
    aux_head = AuxStateFlowHead(state_dim=_Z, z_dim=_Z, cond_dims=(dk,))
    critic = ValueCritic(z_dim=_Z)
    ema_proj = EMAProjection(obs_dim, latent=_Z, ema_decay=cfg.ema_decay)
    spectral = SpectralTokens(z_dim=_Z, k_spectral=cfg.K_spectral, context_len=cfg.context_len)
    return {"trunk": trunk, "policy_head": policy_head, "aux_head": aux_head, "critic": critic, "ema_proj": ema_proj, "spectral": spectral}


def create_run_policies(trunk, policy_head, spectral, cfg):
    """Create noisy train policy and chunk-aware eval wrapper."""
    train_policy = PolicyCallable(trunk, policy_head, spectral, cfg, noise=True)
    eval_wrapper = MpoEvalWrapper(trunk, policy_head, spectral, cfg)
    return train_policy, eval_wrapper  # evaluator calls eval_wrapper(obs)


def capture_spectral_stats(env, trunk, spectral, cfg, seed) -> None:
    """Capture frozen spectral stats from a short rollout."""
    trunk.eval()
    with torch.no_grad():
        obs, _ = env.reset(seed=seed)
        frames = deque(maxlen=cfg.context_len)
        first = torch.as_tensor(obs, dtype=torch.float32)
        for _ in range(cfg.context_len):
            frames.append(first.clone())
        ctx = torch.stack(list(frames), dim=0).unsqueeze(0)  # [1, T, obs]
        z_sample = trunk(ctx)  # [1, T, Z]
    spectral.capture_stats(z_sample.squeeze(0))  # [T, Z]: 3D->2D squeeze
    trunk.train()


def compute_divergence_baseline(evaluator, env, seed) -> float:
    """Run random-policy episodes to get R_rand for divergence gating."""
    rng_baseline = np.random.default_rng(seed)
    random_results = evaluator.evaluate(lambda obs: env.action_space.sample(), out_dir=None)
    r_rand = np.mean([r["return"] for r in random_results]) if random_results else 1.0
    if r_rand == 0:
        r_rand = 1.0  # avoid div-by-zero
    return r_rand


def compute_chunk_values(trunk, critic, chunks):
    """Compute chunk values and truncated-bootstrap next values."""
    with torch.no_grad():
        contexts = torch.stack([c.context for c in chunks])
        z = trunk(contexts)
        values = critic(z).numpy()
        next_values = np.zeros(len(chunks))  # terminated chunks keep 0
        for i, c in enumerate(chunks):
            if c.bootstrap:  # truncated: bootstrap from last frame
                next_obs = c.context[-1:].unsqueeze(0)
                z_next = trunk(next_obs)
                next_values[i] = critic(z_next).item()
    return values, next_values


def collect_gae_inputs(chunks):
    """Flatten chunk rewards, lengths, and termination flags for chunk_gae."""
    all_rewards = []
    chunk_lens = []
    terminated = []
    for c in chunks:
        all_rewards.extend(c.rewards)
        chunk_lens.append(c.exec_len)
        terminated.append(c.done == "terminated")
    return all_rewards, chunk_lens, terminated


def compute_gae_targets(state, chunks):
    """Compute GAE advantages, returns, and values for collected chunks."""
    values, next_values = compute_chunk_values(state.models["trunk"], state.models["critic"], chunks)
    all_rewards, chunk_lens, terminated = collect_gae_inputs(chunks)
    advantages, returns_np = chunk_gae(all_rewards, values, next_values, np.array(chunk_lens), np.array(terminated), gamma=state.cfg.gamma, lam=state.cfg.gae_lambda)
    return advantages, returns_np, values


def log_trainer_diagnostics(state, diag) -> None:
    """Log per-step trainer diagnostics and the total update loss."""
    cfg = state.cfg
    log_metrics({f"train/{k}": v for k, v in diag.items()}, step=state.global_step)
    log_metrics({"train/total_loss": total_update_loss(diag["policy_loss"], diag["value_loss"], diag["aux_loss"], diag["sigreg_loss"], c_value=cfg.c_value, lambda_aux=cfg.lambda_aux, lambda_sigreg=cfg.lambda_sigreg)}, step=state.global_step)


def execute_trainer_update(state, chunks, advantages, returns_np, values):
    """Train on full-length chunks only; return diagnostics or None."""
    cfg = state.cfg
    full_idx = [i for i, c in enumerate(chunks) if c.exec_len == cfg.chunk_len]
    if not full_idx:  # partial chunks can't be torch.stack'd
        return None
    meta = {"returns": torch.tensor(returns_np[full_idx], dtype=torch.float32), "advantages": torch.tensor(advantages[full_idx], dtype=torch.float32), "values": torch.tensor(values[full_idx], dtype=torch.float32)}
    if state.wandb_enabled:
        log_metrics({"gae/advantages_mean": float(meta["advantages"].mean()), "gae/advantages_std": float(meta["advantages"].std()), "gae/returns_mean": float(meta["returns"].mean()), "gae/values_mean": float(meta["values"].mean())}, step=state.global_step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        diag = state.trainer.assemble_update([chunks[i] for i in full_idx], meta)
    if state.wandb_enabled:
        log_trainer_diagnostics(state, diag)
    return diag


def run_eval_and_record(state):
    """Evaluate the policy, log metrics, persist CSV; return mean return."""
    eval_results = state.evaluator.evaluate(state.eval_policy, out_dir=state.out_dir)
    mean_return = np.mean([r["return"] for r in eval_results])
    state.all_eval_returns.append(mean_return)
    state.eval_count += 1
    if state.wandb_enabled:
        log_eval_summary(mean_return, eval_results, state.all_eval_returns, state.eval_count, state.global_step)
    return mean_return


def update_divergence_state(state, mean_return) -> bool:
    """Update divergence counters; return True when training must stop."""
    state.consecutive_below, diverged = check_divergence(mean_return, state.r_rand, state.consecutive_below)
    if not diverged:
        return False
    state.diverged = True
    if state.wandb_enabled:
        log_metrics({"eval/diverged": 1, "eval/diverge_step": state.global_step}, step=state.global_step)
    print(f"DIVERGED at step {state.global_step}: mean_return={mean_return:.2f} < 0.10*R_rand={0.10 * state.r_rand:.2f}")
    return True


def save_run_checkpoint(state, label) -> str:
    """Persist models, optimizer, and counters under out_dir/label."""
    ckpt_dir = state.out_dir / label
    models = state.models
    checkpoint_state = {"models": {"trunk": models["trunk"], "policy_head": models["policy_head"], "aux_head": models["aux_head"], "critic": models["critic"], "ema_proj": models["ema_proj"], "spectral": models["spectral"]}, "optimizer": state.trainer.optimizer, "counters": {"global_step": state.global_step, "_step_count": state.trainer._step_count}}
    save_checkpoint(ckpt_dir, checkpoint_state)
    return str(ckpt_dir / "checkpoint.pt")


def drive_training_loop(state) -> None:
    """Drive collect → GAE → train → eval → checkpoint until budget spent."""
    cfg = state.cfg
    while state.global_step < state.total_steps:
        steps_to_go = state.total_steps - state.global_step
        collect_steps = min(cfg.chunks_per_update * cfg.chunk_len, steps_to_go)
        chunks = collect_chunks(state.env, state.train_policy, collect_steps, state.action_low, state.action_high, context_len=cfg.context_len)
        if not chunks:
            break
        state.global_step += sum(c.exec_len for c in chunks)
        advantages, returns_np, values = compute_gae_targets(state, chunks)
        execute_trainer_update(state, chunks, advantages, returns_np, values)
        if state.global_step % state.eval_interval < cfg.chunk_len or state.global_step >= state.total_steps:
            if update_divergence_state(state, run_eval_and_record(state)):
                break
        if state.global_step % state.ckpt_interval < cfg.chunk_len or state.global_step >= state.total_steps:
            state.artifacts.append(save_run_checkpoint(state, f"checkpoint_{state.global_step}"))


def record_timing_metrics(state) -> float:
    """Log fps/elapsed timing; return fps."""
    elapsed = time.time() - state.t_start
    fps = state.global_step / max(elapsed, 1e-6)
    if state.wandb_enabled:
        log_metrics({"timing/fps": fps, "timing/elapsed_s": elapsed}, step=state.global_step)
    return fps


def write_final_checkpoint(state) -> None:
    """Write the final checkpoint and record CSV/ckpt artifacts."""
    state.artifacts.append(save_run_checkpoint(state, "checkpoint_final"))
    csv_path = state.out_dir / "eval_history.csv"
    if csv_path.exists():
        state.artifacts.append(str(csv_path))


def build_result_summary(state) -> dict[str, Any]:
    """Build the result dict with env name, reward mean, fps, artifacts."""
    reward_mean_last10 = float(np.mean(state.all_eval_returns[-10:])) if state.all_eval_returns else 0.0
    return {"method": "mfpo", "env_name": (getattr(getattr(state.env, "unwrapped", state.env).spec, "id", "unknown") if hasattr(state.env, "unwrapped") and hasattr(state.env.unwrapped, "spec") and getattr(state.env.unwrapped, "spec", None) is not None else "unknown"), "seed": state.seed, "reward_mean_last10": reward_mean_last10, "fps": state.fps, "artifacts": state.artifacts, "diverged": state.diverged, "global_step": state.global_step}


def assemble_run_state(env, out_dir, cfg, seed, total_steps, eval_interval, ckpt_interval, obs_dim, act_dim) -> TrainLoopState:
    """Build networks, trainer, evaluator, policies, and baseline state."""
    out_dir = Path(out_dir)
    env_name_str = resolve_env_name(env)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim, act_dim = resolve_training_dims(env, obs_dim, act_dim)
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = build_training_components(obs_dim, act_dim, cfg)
    capture_spectral_stats(env, parts["trunk"], parts["spectral"], cfg, seed)
    trainer = MFPOTrainer(parts["trunk"], parts["policy_head"], parts["aux_head"], parts["critic"], parts["ema_proj"], parts["spectral"], cfg=cfg)
    evaluator = Evaluator(env, eval_seed=10000 + seed, max_episodes=12)
    train_policy, eval_policy = create_run_policies(parts["trunk"], parts["policy_head"], parts["spectral"], cfg)
    r_rand = compute_divergence_baseline(evaluator, env, seed)
    cfg_dict = dataclasses.asdict(cfg)
    wandb_enabled = init_wandb_run_for_method(method="mfpo", env_name=env_name_str, seed=seed, total_steps=total_steps, cfg_dict=cfg_dict)
    return TrainLoopState(env=env, cfg=cfg, out_dir=out_dir, env_name=env_name_str, seed=seed, models=parts, trainer=trainer, evaluator=evaluator, train_policy=train_policy, eval_policy=eval_policy, action_low=-1.0, action_high=1.0, total_steps=total_steps, eval_interval=eval_interval, ckpt_interval=ckpt_interval, r_rand=r_rand, wandb_enabled=wandb_enabled, t_start=time.time())


def train_run(
    env: Any,
    out_dir: str | Path,
    cfg: MFPOConfig | None = None,
    *,
    seed: int = 0,
    total_steps: int = 2000,
    eval_interval: int = 1000,
    ckpt_interval: int = 1000,
    obs_dim: int | None = None,
    act_dim: int | None = None,
) -> dict[str, Any]:
    """Run a single MFPO training run, returning summary with artifacts."""
    cfg = cfg or MFPOConfig()
    state = assemble_run_state(
        env, out_dir, cfg, seed, total_steps, eval_interval, ckpt_interval,
        obs_dim, act_dim,
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
