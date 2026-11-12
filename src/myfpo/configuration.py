"""Single home for every hyperparameter."""

import torch
from pathlib import Path
import yaml
import os
import warnings
from dataclasses import dataclass

__all__ = [
    "ACTOR_ARCHITECTURES",
    "FLOW_STARTS",
    "RATIO_MODES",
    "MyfpoConfig",
]

# ---- from config.py ----
ENV_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "envs.yaml"

with ENV_REGISTRY_PATH.open() as f:
    _ENV_REGISTRY = yaml.safe_load(f) or {}

KNOWN_ENVS = tuple(_ENV_REGISTRY.keys())


def validate_num_action_steps(n: int, action_horizon: int) -> None:
    """Check that the executed span fits the chunk. Used by the config, the GPU evaluator, and the collector."""
    if not 1 <= n <= action_horizon:
        raise ValueError(
            f"num_action_steps={n} must satisfy "
            f"1 <= n <= action_horizon={action_horizon}"
        )


def get_observation_action_dim(env_id: str) -> tuple[int, int]:
    """Obs and action dims for ``env_id``, from ``configs/envs.yaml``."""
    if not _ENV_REGISTRY:
        raise KeyError(f"registry empty at {ENV_REGISTRY_PATH}")
    entry = _ENV_REGISTRY.get(env_id)
    if entry is None:
        raise KeyError(f"unknown env {env_id!r}; add it to {ENV_REGISTRY_PATH}")
    return int(entry["obs_dim"]), int(entry["action_dim"])


ACTOR_ARCHITECTURES = ("dit", "resmlp")


FLOW_STARTS = ("zero_start", "noise_start")


# ``cfm`` and ``cfm_raw`` read the stored CFM losses. See ``compute_policy_ratio``.
RATIO_MODES = ("cfm", "cfm_raw")


@dataclass
class MyfpoConfig:
    """myfpo hyperparameters."""

    # Filled by ``make_policy`` from the env. None means "not attached yet".
    obs_dim: int | None = None
    action_dim: int | None = None

    report_to: str = "all"
    num_obs_steps: int = 4
    action_horizon: int = 4
    num_action_steps: int = 4
    action_perturb_std: float = 0.02
    storage_action_noise_std: float = 0.0
    hidden_dim: int = 128

    num_envs: int = 64
    num_mc_samples: int = 16
    num_learning_epochs: int = 16
    chunks_per_update: int = 2048
    chunks_per_minibatch: int = 512
    grad_accum_steps: int = 1

    num_steps_zero_start: int = 10
    num_steps_noise_start: int = 10
    # Fresh policy chunks averaged per state for the V(s) advantage baseline.
    num_baseline_samples: int = 4
    min_sampled_timestep: float = 0.1
    max_sampled_timestep: float = 0.9
    timestep_sampling_mode: str = "uniform"
    noise_sampling_mode: str = "normal"
    timestep_inverse_cdf_beta: float = 1.0

    gamma: float = 0.99
    gae_lambda: float = 0.95
    group_mc_loss: bool = False
    mc_loss_group_size: int = 1
    clip_param: float = 0.05
    spo_clip_param: float = 0.01
    trust_region_mode: str = "aspo"
    cfm_loss_reduction: str = "sqrt"
    cfm_loss_target: str = "velocity"
    cfm_loss_weight_from_t: str = "constant"
    cfm_clamp: float = 20.0
    ratio_mode: str = "cfm"
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    cfm_diff_clamp_max: float = 10.0

    num_critics: int = 2
    critic_condition_on_exec_length: bool = True
    critic_hidden_dim: int = 512
    critic_num_layers: int = 2
    critic_dropout: float = 0.01
    normalize_advantage: bool = True
    # Off by default: the rollout-level normalization is the standard PPO
    # contract, and it is the only one that keeps one scale for the whole
    # update. This flag re-standardizes each minibatch on its own statistics,
    # which overrides the rollout-level pass and makes `normalize_advantage`
    # unreadable while it is on.
    normalize_advantage_per_minibatch: bool = False
    advantage_clamp_neg: float = 100.0
    advantage_clamp_pos: float = 100.0
    value_loss_coef: float = 1.0
    value_loss_beta: float = 10.0
    use_clipped_value_loss: bool = False
    value_clip_param: float = 0.2
    action_processing_mode: str = "linear_rescale"

    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    velocity_clip_value: float | None = None

    compile: bool = False
    compile_mode: str = "default"
    """torch.compile mode for the loss path: 'default' or 'reduce-overhead'.
    Enabled by default on CUDA via MYFPO_COMPILE=1; disabled on CPU unless
    MYFPO_COMPILE is set explicitly."""

    actor_arch: str = "dit"
    dit_depth: int = 4
    dit_heads: int = 2
    dit_mlp_ratio: float = 2.0
    resmlp_blocks: int = 2
    resmlp_units: int = 4
    resmlp_hidden_mult: float = 2.0
    time_embed_dim: int = 8

    policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500
    lr_warmup_steps: int = 10000
    lr_schedule: str = "fixed"
    normalize_rewards: bool = True
    # Fixed action scaling for the flow (reference actor_scale). The flow
    # operates in scaled space (target = action / actor_scale); inference
    # multiplies back. 1.0 is a no-op (raw actions).
    actor_scale: float = 1.0
    eval_max_episodes: int = 50
    eval_sampling: str = "noise_start"
    final_eval_sampling: str = "noise_start"

    # Costs one flow integration and one critic read on every rollout when a
    # backend logs.
    log_drift_diagnostics: bool = False

    # How often the policy measures the per-position gradient norms. 0 = never.
    temporal_probe_every: int = 0


    def __post_init__(self) -> None:
        """Set the derived fields, then run every validation check."""
        if self.num_action_steps is None:
            self.num_action_steps = self.action_horizon
        if os.environ.get("MYFPO_COMPILE") is not None:
            self.compile = bool(int(os.environ["MYFPO_COMPILE"])) and torch.cuda.is_available()
        self.validate()

    def validate(self) -> None:
        """Run every check. Each group is a method; the groups are independent."""
        self.validate_shape()
        self.validate_actor()
        self.validate_critic()
        self.validate_trust_region()
        self.validate_sampling()
        self.validate_optim_and_batching()
        self.validate_eval()

    def validate_shape(self) -> None:
        """Window, chunk, and width."""
        if self.temporal_probe_every < 0:
            raise ValueError(
                f"temporal_probe_every={self.temporal_probe_every} must be >= 0"
            )
        if self.num_obs_steps < 1:
            raise ValueError(f"num_obs_steps={self.num_obs_steps} must be >= 1")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon={self.action_horizon} must be >= 1")
        validate_num_action_steps(self.num_action_steps, self.action_horizon)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be >= 1")
        if self.action_perturb_std < 0:
            raise ValueError("action_perturb_std must be >= 0")
        if self.storage_action_noise_std < 0:
            raise ValueError("storage_action_noise_std must be >= 0")

    def validate_actor(self) -> None:
        """Actor widths, depths, the resmlp variant, and the EMA."""
        # Before any modulo: a zero head count divides by zero instead of
        # reporting the field.
        if self.dit_heads < 1:
            raise ValueError(f"dit_heads={self.dit_heads} must be >= 1")
        if self.hidden_dim % self.dit_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"dit_heads={self.dit_heads}"
            )
        if self.hidden_dim % (2 * self.dit_heads) != 0:
            raise ValueError(
                f"head_dim=hidden_dim//dit_heads={self.hidden_dim // self.dit_heads} "
                "must be even for RoPE, which rotates (even, odd) pairs"
            )
        if self.actor_arch not in ACTOR_ARCHITECTURES:
            raise ValueError(
                f"actor_arch={self.actor_arch!r} must be one of "
                f"{', '.join(repr(k) for k in ACTOR_ARCHITECTURES)}"
            )
        if self.dit_depth < 1:
            raise ValueError(f"dit_depth={self.dit_depth} must be >= 1")
        if self.dit_mlp_ratio <= 0:
            raise ValueError(f"dit_mlp_ratio={self.dit_mlp_ratio} must be > 0")
        for name, value in (
            ("resmlp_blocks", self.resmlp_blocks),
            ("resmlp_units", self.resmlp_units),
        ):
            if value < 1:
                raise ValueError(f"{name}={value} must be >= 1")
        if self.resmlp_hidden_mult <= 0:
            raise ValueError(
                f"resmlp_hidden_mult={self.resmlp_hidden_mult} must be > 0"
            )
        if self.time_embed_dim < 1:
            raise ValueError(f"time_embed_dim={self.time_embed_dim} must be >= 1")
        if self.cfm_loss_huber_delta <= 0:
            raise ValueError(
                f"cfm_loss_huber_delta={self.cfm_loss_huber_delta} must be > 0"
            )
        if (
            self.velocity_clip_value is not None
            and self.velocity_clip_value <= 0
        ):
            raise ValueError(
                f"velocity_clip_value={self.velocity_clip_value} must be > 0"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.ema_warmup_steps < 0:
            raise ValueError("ema_warmup_steps must be >= 0")

    def validate_critic(self) -> None:
        """Ensemble width, depth, dropout, and the critic learning rate."""
        if self.num_critics < 1:
            raise ValueError(f"num_critics={self.num_critics} must be >= 1")
        if self.critic_hidden_dim is not None and self.critic_hidden_dim < 1:
            raise ValueError(
                f"critic_hidden_dim={self.critic_hidden_dim} must be >= 1"
            )
        if self.critic_num_layers < 1:
            raise ValueError(f"critic_num_layers={self.critic_num_layers} must be >= 1")
        if not 0.0 <= self.critic_dropout < 1.0:
            raise ValueError(
                f"critic_dropout={self.critic_dropout} must be in [0, 1)"
            )
        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr={self.critic_lr} must be > 0")
        # The critic runs four attention heads over the state window, and
        # ``make_policy_networks`` passes ``hidden_dim`` as that window's width.
        # Each head width must be even for RoPE, so the width divides by eight.
        # The actor's rule (2 * dit_heads) is weaker and does not imply this.
        if self.hidden_dim % 8 != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by 8: the "
                "critic uses 4 attention heads on the state window, and each "
                "head width must be even"
            )

    def validate_trust_region(self) -> None:
        """Discounts, clipping, the ASPO mode, and the CFM ratio terms."""
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda={self.gae_lambda} must be in [0, 1]")
        if self.trust_region_mode in {"spo", "aspo"} and not 0.0 < self.spo_clip_param < 1.0:
            raise ValueError(
                f"spo_clip_param={self.spo_clip_param} must satisfy 0 < eps < 1 "
                f"when trust_region_mode={self.trust_region_mode!r}"
            )
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError(
                f"clip_param={self.clip_param} must satisfy 0 < eps < 1"
            )
        if self.trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError("trust_region_mode must be 'ppo', 'spo', or 'aspo'")
        if self.mc_loss_group_size == 0 or self.mc_loss_group_size < -1:
            raise ValueError(
                f"mc_loss_group_size={self.mc_loss_group_size} must be -1 "
                "(one group over every MC sample) or a positive divisor of "
                f"num_mc_samples={self.num_mc_samples}"
            )
        if (
            self.mc_loss_group_size != -1
            and self.num_mc_samples % self.mc_loss_group_size != 0
        ):
            raise ValueError(
                f"num_mc_samples={self.num_mc_samples} must divide evenly by "
                f"mc_loss_group_size={self.mc_loss_group_size}"
            )
        if self.cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if self.cfm_loss_target not in {"velocity", "action", "eps"}:
            raise ValueError(
                "cfm_loss_target must be 'velocity' (FPO++'s cfm_loss_mode='u'), "
                "'action' (='x0'), or 'eps'"
            )
        if self.cfm_loss_weight_from_t not in {
            "constant",
            "linear_0.1_to_1",
            "linear_0.01_to_1",
        }:
            raise ValueError(
                "cfm_loss_weight_from_t must be 'constant', 'linear_0.1_to_1', "
                "or 'linear_0.01_to_1'. A linear name states the weight at t=0 "
                "first, then the weight at t=1. Under this repo's path the "
                "noise sits at t=0, so a rising weight is full at the action end."
            )
        if self.ratio_mode not in RATIO_MODES:
            raise ValueError(
                f"ratio_mode={self.ratio_mode!r} must be one of "
                f"{', '.join(repr(mode) for mode in RATIO_MODES)}"
            )
        if self.cfm_clamp <= 0:
            raise ValueError(f"cfm_clamp={self.cfm_clamp} must be > 0")
        if self.cfm_loss_clamp_negative_advantages_max <= 0:
            raise ValueError("cfm_loss_clamp_negative_advantages_max must be > 0")
        if self.cfm_diff_clamp_max <= 0:
            raise ValueError(f"cfm_diff_clamp_max={self.cfm_diff_clamp_max} must be > 0")
        for name, value in (
            ("advantage_clamp_neg", self.advantage_clamp_neg),
            ("advantage_clamp_pos", self.advantage_clamp_pos),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} must be > 0")

    def validate_sampling(self) -> None:
        """The timestep law, the noise law, and the integration step counts."""
        if not (
            0.0
            <= self.min_sampled_timestep
            < self.max_sampled_timestep
            <= 1.0
        ):
            raise ValueError(
                "require 0 <= min_sampled_timestep < max_sampled_timestep <= 1, got "
                f"({self.min_sampled_timestep}, {self.max_sampled_timestep})"
            )
        if self.timestep_sampling_mode not in {"uniform", "fpo_beta"}:
            raise ValueError(
                "timestep_sampling_mode must be 'uniform' or 'fpo_beta'"
            )
        if self.noise_sampling_mode not in {"normal", "uniform"}:
            raise ValueError("noise_sampling_mode must be 'normal' or 'uniform'")
        if self.timestep_inverse_cdf_beta <= 0:
            raise ValueError("timestep_inverse_cdf_beta must be > 0")
        if self.timestep_inverse_cdf_beta == 1.0 and self.timestep_sampling_mode == "fpo_beta":
            # stacklevel=3: 2 would now land inside validate() itself.
            warnings.warn(
                "timestep_inverse_cdf_beta=1.0 makes timestep_sampling_mode="
                "'fpo_beta' draw the same law as 'uniform'. Choose a beta "
                "other than 1.0 to sample a non-uniform timestep law.",
                RuntimeWarning,
                stacklevel=3,
            )
        if self.num_steps_zero_start < 1:
            raise ValueError(
                f"num_steps_zero_start={self.num_steps_zero_start} must be >= 1"
            )
        if self.num_steps_noise_start < 1:
            raise ValueError(
                f"num_steps_noise_start={self.num_steps_noise_start} must be >= 1"
            )
        if self.num_baseline_samples < 1:
            raise ValueError(
                f"num_baseline_samples={self.num_baseline_samples} must be >= 1"
            )

    def validate_optim_and_batching(self) -> None:
        """Learning rates, the optimizer, the regularizers, and the batch shape."""
        if self.policy_lr <= 0:
            raise ValueError(f"policy_lr={self.policy_lr} must be > 0")
        if self.optimizer_weight_decay < 0:
            raise ValueError(
                f"optimizer_weight_decay={self.optimizer_weight_decay} must be >= 0"
            )
        if not all(0.0 < b < 1.0 for b in self.adam_betas):
            raise ValueError(f"adam_betas={self.adam_betas} must each be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm={self.max_grad_norm} must be > 0")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps={self.grad_accum_steps} must be >= 1")
        if self.chunks_per_update < 1:
            raise ValueError(f"chunks_per_update={self.chunks_per_update} must be >= 1")
        if self.chunks_per_minibatch < 1:
            raise ValueError(
                f"chunks_per_minibatch={self.chunks_per_minibatch} must be >= 1"
            )
        if self.chunks_per_minibatch > self.chunks_per_update:
            raise ValueError(
                f"chunks_per_minibatch={self.chunks_per_minibatch} must be <= "
                f"chunks_per_update={self.chunks_per_update}"
            )
        if self.num_learning_epochs < 1:
            raise ValueError(
                f"num_learning_epochs={self.num_learning_epochs} must be >= 1"
            )
        if self.num_envs < 1:
            raise ValueError(f"num_envs={self.num_envs} must be >= 1")
        # sample_eps draws half the count and mirrors it, so an odd count
        # changes the row count downstream.
        if self.num_mc_samples < 2 or self.num_mc_samples % 2:
            raise ValueError(
                f"num_mc_samples={self.num_mc_samples} must be even and >= 2"
            )
        if self.value_loss_coef < 0:
            raise ValueError(f"value_loss_coef={self.value_loss_coef} must be >= 0")
        if self.value_loss_beta <= 0:
            raise ValueError(f"value_loss_beta={self.value_loss_beta} must be > 0")
        if not 0.0 < self.value_clip_param < 1.0:
            raise ValueError(
                f"value_clip_param={self.value_clip_param} must satisfy 0 < eps < 1"
            )
        if self.action_processing_mode not in ("tanh_squash", "linear_rescale"):
            raise ValueError(
                f"action_processing_mode={self.action_processing_mode!r} must be "
                "'tanh_squash' or 'linear_rescale'"
            )
        if self.lr_schedule not in ("fixed", "cosine"):
            raise ValueError(
                f"lr_schedule={self.lr_schedule!r} must be 'fixed' or 'cosine'"
            )
        if self.compile_mode not in ("default", "reduce-overhead"):
            raise ValueError(
                f"compile_mode={self.compile_mode!r} must be 'default' or 'reduce-overhead'"
            )
        if self.lr_warmup_steps < 0:
            raise ValueError(
                f"lr_warmup_steps={self.lr_warmup_steps} must be >= 0"
            )

    def validate_eval(self) -> None:
        """The tracking backend and the two named flow starts."""
        if self.report_to not in ("wandb", "tensorboard", "all", "none"):
            raise ValueError(
                f"report_to={self.report_to!r} must be wandb, tensorboard, all or none"
            )
        if self.eval_max_episodes < 1:
            raise ValueError(f"eval_max_episodes={self.eval_max_episodes} must be >= 1")
        for name, value in (
            ("eval_sampling", self.eval_sampling),
            ("final_eval_sampling", self.final_eval_sampling),
        ):
            if value not in FLOW_STARTS:
                raise ValueError(
                    f"{name}={value!r} must be one of "
                    f"{', '.join(repr(k) for k in FLOW_STARTS)}"
                )

    def validate_dims(self) -> None:
        """Require the env dims. ``MyfpoPolicy.__init__`` calls this."""
        if self.obs_dim is None or self.obs_dim < 1:
            raise ValueError(
                f"obs_dim={self.obs_dim} must be set to a positive int before "
                "building the policy"
            )
        if self.action_dim is None or self.action_dim < 1:
            raise ValueError(
                f"action_dim={self.action_dim} must be set to a positive int "
                "before building the policy"
            )

    @property
    def num_minibatches(self) -> int:
        """The minibatch count in one update, at least 1."""
        return max(1, self.chunks_per_update // self.chunks_per_minibatch)
