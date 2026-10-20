"""Single home for every hyperparameter (LeRobot ``PreTrainedConfig``)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

__all__ = ["ACTOR_ARCHITECTURES", "FLOW_STARTS", "MyfpoConfig"]

ACTOR_ARCHITECTURES = ("dit", "resmlp")


FLOW_STARTS = ("zero_start", "noise_start")


@PreTrainedConfig.register_subclass("myfpo")
@dataclass
class MyfpoConfig(PreTrainedConfig):
    """myfpo hyperparameters."""

    report_to: str = "all"
    n_obs_steps: int = 16
    action_horizon: int = 8
    n_action_steps: int = 8
    action_perturb_std: float = 0.0
    hidden_dim: int = 128

    num_envs: int = 64
    num_mc_samples: int = 16
    num_learning_epochs: int = 10
    chunks_per_update: int = 1024
    batch_size: int = 128
    grad_accum_steps: int = 1

    num_steps_zero_start: int = 4
    num_steps_noise_start: int = 4
    min_sampled_timestep: float = 0.1
    max_sampled_timestep: float = 0.9
    timestep_sampling_mode: str = "uniform"
    noise_sampling_mode: str = "normal"
    timestep_inverse_cdf_beta: float = 2.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    gamma: float = 0.99
    gae_lambda: float = 0.95
    do_average_cfm_loss_in_chunk: bool = False
    cfm_average_group_size: int = 1
    clip_param: float = 0.01
    spo_clip_param: float = 0.01
    trust_region_mode: str = "aspo" 
    cfm_loss_reduction: str = "mean"
    cfm_loss_target: str = "velocity"
    cfm_loss_weight_from_t: str = "constant"
    cfm_clamp: float = 20.0
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    log_ratio_cap: float = 10.0

    num_critics: int = 2
    critic_condition_on_exec_length: bool = True
    critic_share_encoder: bool = False
    critic_hidden_dim: int = 512
    critic_num_layers: int = 2
    critic_dropout: float = 0.01
    normalize_advantage: bool = True
    advantage_clamp_neg: float = 100.0
    advantage_clamp_pos: float = 100.0
    value_loss_coef: float = 1.0
    sigreg_mode: str = "none"
    sigreg_coef: float = 0.0
    sigreg_sketch_dim: int = 64

    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    transported_clip_value: float | None = None

    compile: bool = False

    actor_arch: str = "dit"
    dit_depth: int = 4
    dit_heads: int = 2
    dit_mlp_ratio: float = 2.0
    resmlp_blocks: int = 2
    resmlp_units: int = 4
    resmlp_hidden_mult: float = 2.0
    time_embed_dim: int = 32

    policy_lr: float = 1e-5
    critic_lr: float = 1e-4
    actor_lr_warmup_steps: int = 5
    critic_lr_warmup_steps: int = 1
    optimizer_weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.99)
    max_grad_norm: float = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    eval_max_episodes: int = 50
    eval_sampling: str = "noise_start"
    stop_eval_sampling: str = "noise_start"

    log_drift_diagnostics: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        self.validate()

    def validate(self) -> None:
        """Run every check. Each group is a method; the groups are independent."""
        self._validate_shape()
        self._validate_actor()
        self._validate_critic()
        self._validate_trust_region()
        self._validate_sampling()
        self._validate_optim_and_batching()
        self._validate_eval()

    def _validate_shape(self) -> None:
        """Window, chunk, and width."""
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps={self.n_obs_steps} must be >= 1")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon={self.action_horizon} must be >= 1")
        if not 0 < self.n_action_steps <= self.action_horizon:
            raise ValueError(
                f"n_action_steps={self.n_action_steps} must satisfy "
                f"0 < n <= action_horizon={self.action_horizon}"
            )
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be >= 1")
        if self.action_perturb_std < 0:
            raise ValueError("action_perturb_std must be >= 0")

    def _validate_actor(self) -> None:
        """Actor widths, depths, the resmlp variant, and the EMA."""
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
        if self.dit_heads < 1:
            raise ValueError(f"dit_heads={self.dit_heads} must be >= 1")
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
            self.transported_clip_value is not None
            and self.transported_clip_value <= 0
        ):
            raise ValueError(
                f"transported_clip_value={self.transported_clip_value} must be > 0"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.ema_warmup_steps < 0:
            raise ValueError("ema_warmup_steps must be >= 0")

    def _validate_critic(self) -> None:
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
        if self.critic_lr_warmup_steps < 0:
            raise ValueError(
                f"critic_lr_warmup_steps={self.critic_lr_warmup_steps} must be >= 0"
            )

    def _validate_trust_region(self) -> None:
        """Discounts, clipping, the ASPO mode, and the CFM ratio terms."""
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda={self.gae_lambda} must be in [0, 1]")
        if not 0.0 < self.spo_clip_param < 1.0:
            raise ValueError(
                f"spo_clip_param={self.spo_clip_param} must satisfy 0 < eps < 1"
            )
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError(
                f"clip_param={self.clip_param} must satisfy 0 < eps < 1"
            )
        if self.trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError("trust_region_mode must be 'ppo', 'spo', or 'aspo'")
        if self.cfm_average_group_size == 0 or self.cfm_average_group_size < -1:
            raise ValueError(
                f"cfm_average_group_size={self.cfm_average_group_size} must be -1 "
                "(one group over every MC sample) or a positive divisor of "
                f"num_mc_samples={self.num_mc_samples}"
            )
        if (
            self.cfm_average_group_size != -1
            and self.num_mc_samples % self.cfm_average_group_size != 0
        ):
            raise ValueError(
                f"num_mc_samples={self.num_mc_samples} must divide evenly by "
                f"cfm_average_group_size={self.cfm_average_group_size}"
            )
        if self.cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if self.cfm_loss_target not in {"velocity", "action", "noise"}:
            raise ValueError(
                "cfm_loss_target must be 'velocity' (FPO++'s cfm_loss_mode='u'), "
                "'action' (='x0'), or 'noise' (='eps')"
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
        if self.cfm_clamp <= 0:
            raise ValueError(f"cfm_clamp={self.cfm_clamp} must be > 0")
        if self.cfm_loss_clamp_negative_advantages_max <= 0:
            raise ValueError("cfm_loss_clamp_negative_advantages_max must be > 0")
        if self.log_ratio_cap <= 0:
            raise ValueError(f"log_ratio_cap={self.log_ratio_cap} must be > 0")
        for name, value in (
            ("advantage_clamp_neg", self.advantage_clamp_neg),
            ("advantage_clamp_pos", self.advantage_clamp_pos),
        ):
            if value <= 0:
                raise ValueError(f"{name}={value} must be > 0")

    def _validate_sampling(self) -> None:
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
        if self.timestep_sampling_mode not in {"uniform", "normal", "beta", "fpo_beta"}:
            raise ValueError(
                "timestep_sampling_mode must be 'uniform', 'normal', 'beta', or 'fpo_beta'"
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
        if self.timestep_beta_alpha <= 0 or self.timestep_beta_beta <= 0:
            raise ValueError("timestep beta alpha/beta must be > 0")
        if self.num_steps_zero_start < 1:
            raise ValueError(
                f"num_steps_zero_start={self.num_steps_zero_start} must be >= 1"
            )
        if self.num_steps_noise_start < 1:
            raise ValueError(
                f"num_steps_noise_start={self.num_steps_noise_start} must be >= 1"
            )

    def _validate_optim_and_batching(self) -> None:
        """Learning rates, the optimiser, the regularisers, and the batch shape."""
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
        if self.actor_lr_warmup_steps < 0:
            raise ValueError(
                f"actor_lr_warmup_steps={self.actor_lr_warmup_steps} must be >= 0"
            )
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps={self.grad_accum_steps} must be >= 1")
        if self.chunks_per_update < 1:
            raise ValueError(f"chunks_per_update={self.chunks_per_update} must be >= 1")
        if self.batch_size < 1:
            raise ValueError(f"batch_size={self.batch_size} must be >= 1")
        if self.batch_size > self.chunks_per_update:
            raise ValueError(
                f"batch_size={self.batch_size} must be <= "
                f"chunks_per_update={self.chunks_per_update}"
            )
        if self.num_learning_epochs < 1:
            raise ValueError(
                f"num_learning_epochs={self.num_learning_epochs} must be >= 1"
            )
        if self.num_envs < 1:
            raise ValueError(f"num_envs={self.num_envs} must be >= 1")
        if self.num_mc_samples < 1:
            raise ValueError(f"num_mc_samples={self.num_mc_samples} must be >= 1")
        if self.value_loss_coef < 0:
            raise ValueError(f"value_loss_coef={self.value_loss_coef} must be >= 0")
        if self.sigreg_mode not in {"none", "weak", "decorr"}:
            raise ValueError("sigreg_mode must be 'none', 'weak', or 'decorr'")
        if self.sigreg_coef < 0:
            raise ValueError(f"sigreg_coef={self.sigreg_coef} must be >= 0")
        if self.sigreg_sketch_dim < 2:
            raise ValueError(
                f"sigreg_sketch_dim={self.sigreg_sketch_dim} must be >= 2: "
                "a covariance from one row is undefined"
            )

    def _validate_eval(self) -> None:
        """The tracking backend and the two named flow starts."""
        if self.report_to not in ("wandb", "tensorboard", "all", "none"):
            raise ValueError(
                f"report_to={self.report_to!r} must be wandb, tensorboard, all or none"
            )
        if self.eval_max_episodes < 1:
            raise ValueError(f"eval_max_episodes={self.eval_max_episodes} must be >= 1")
        for name, value in (
            ("eval_sampling", self.eval_sampling),
            ("stop_eval_sampling", self.stop_eval_sampling),
        ):
            if value not in FLOW_STARTS:
                raise ValueError(
                    f"{name}={value!r} must be one of "
                    f"{', '.join(repr(k) for k in FLOW_STARTS)}"
                )

    def validate_features(self) -> None:
        """Require the run-time features. ``MyfpoPolicy.__init__`` calls this."""
        if self.robot_state_feature is None:
            raise ValueError("myfpo requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("myfpo requires 'action' in output_features.")

    def get_optimizer_preset(self) -> "AdamWConfig":
        kwargs = dict(
            lr=self.policy_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.max_grad_norm,
        )
        return AdamWConfig(**kwargs)

    def get_scheduler_preset(self):
        return None

    @property
    def num_mini_batches(self) -> int:
        return max(1, self.chunks_per_update // self.batch_size)

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

