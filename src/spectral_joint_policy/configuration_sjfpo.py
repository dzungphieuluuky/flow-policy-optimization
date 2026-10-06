"""Single home for every SJFPO hyperparameter (LeRobot ``PreTrainedConfig``)."""

from __future__ import annotations
# Spectral profile: noise and proposed per-band trust region use the same 1/f^beta decay
# (pink-noise / JPEG-quantization / wavelet-threshold principle). See docs/AGENTS.md.


from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

__all__ = ["SJFPOConfig"]


@PreTrainedConfig.register_subclass("sjfpo")
@dataclass
class SJFPOConfig(PreTrainedConfig):
    """SJFPO hyperparameters.

    ``n_action_steps`` is the executed prefix at eval and defaults to
    ``action_horizon``. ``num_inference_steps`` is for zero-sampling,
    ``num_euler_steps`` for a sampled-noise start.
    """

    report_to: str = "all"
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16
    action_perturb_std: float = 0.02
    hidden_dim: int = 256

    num_envs: int = 64
    num_mc_samples: int = 16
    num_learning_epochs: int = 1
    chunks_per_update: int = 1024
    batch_size: int = 128
    grad_accum_steps: int = 1

    num_euler_steps: int = 4
    num_inference_steps: int = 4
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    # --- Spectral / frequency-domain settings ---
    # Noise scaling: white (uniform energy) or freq_weighted (pink noise, 1/f^beta).
    # DCT: orthonormal DCT-II analysis for spectral energy/collapse diagnostics.
    # Trust region: per-band clipping profile matches spectral decay (see docs).
    noise_scaling_mode: str = "white"
    noise_scaling_beta: float = 1.0
    # Spectral decay profile for per-frequency trust-region clipping (matches noise decay).
    trust_region_beta: float = 1.0
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    gamma: float = 0.99
    clip_param: float = 0.05
    trust_region_mode: str = "aspo"
    ratio_granularity: str = "per_sample"  # per_chunk | per_action | per_sample
    cfm_loss_reduction: str = "mean"
    cfm_clamp: float = 200.0
    cfm_mask_unexecuted: bool = True
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 200.0
    log_ratio_cap: float = 10.0

    num_critics: int = 2
    normalize_advantage: bool = True
    value_loss_coef: float = 1.0
    value_huber_delta: float = 1.0
    use_target_critic: bool = True
    critic_ema_decay: float = 0.05

    use_dct_transform: bool = True
    num_kept_tokens: int = 4

    dit_depth: int = 6
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    timestep_scale: float = 10000.0

    policy_lr: float = 1e-4
    critic_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.999, 0.999)
    max_grad_norm: float = 1.0
    critic_max_grad_norm: float | None = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    eval_max_episodes: int = 50
    stop_return_frac: float = 0.7
    stop_patience: int = 5
    lr_decay: float = 0.5
    max_lr_decays: int = 3

    log_drift_diagnostics: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        # Note: robot_state_feature / action_feature are read-only properties
        # derived from input_features / output_features (set above or via CLI/config).
        self.validate()
        # Features check included in validate()

    def validate(self) -> None:
        """All checks inline; no extra method names."""
        # Geometry / shape
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps={self.n_obs_steps} must be >= 1")
        if self.report_to not in ("wandb", "tensorboard", "all", "none"):
            raise ValueError(
                f"report_to={self.report_to!r} must be wandb, tensorboard, all or none"
            )
        if self.num_critics < 1:
            raise ValueError(f"num_critics={self.num_critics} must be >= 1")
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
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.ema_warmup_steps < 0:
            raise ValueError("ema_warmup_steps must be >= 0")
        if self.hidden_dim % self.dit_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"dit_heads={self.dit_heads}"
            )
        if self.dit_depth < 1:
            raise ValueError(f"dit_depth={self.dit_depth} must be >= 1")
        if self.dit_heads < 1:
            raise ValueError(f"dit_heads={self.dit_heads} must be >= 1")
        if self.dit_mlp_ratio <= 0:
            raise ValueError(f"dit_mlp_ratio={self.dit_mlp_ratio} must be > 0")
        if self.time_embed_dim < 1:
            raise ValueError(f"time_embed_dim={self.time_embed_dim} must be >= 1")
        if self.timestep_scale <= 0:
            raise ValueError(f"timestep_scale={self.timestep_scale} must be > 0")
        # Eval schedule
        if self.eval_max_episodes < 1:
            raise ValueError(f"eval_max_episodes={self.eval_max_episodes} must be >= 1")
        if not 0.0 < self.stop_return_frac <= 1.0:
            raise ValueError(
                f"stop_return_frac={self.stop_return_frac} must satisfy "
                f"0 < frac <= 1"
            )
        if self.stop_patience < 1:
            raise ValueError(f"stop_patience={self.stop_patience} must be >= 1")
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError(f"lr_decay={self.lr_decay} must satisfy 0 < factor <= 1")
        if self.max_lr_decays < 0:
            raise ValueError(f"max_lr_decays={self.max_lr_decays} must be >= 0")
        # Loss params
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError(
                f"clip_param={self.clip_param} must satisfy 0 < eps < 1"
            )
        if self.trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError("trust_region_mode must be 'ppo', 'spo', or 'aspo'")
        if self.ratio_granularity not in {"per_chunk", "per_action", "per_sample"}:
            raise ValueError(
                "ratio_granularity must be 'per_chunk' (FPO Eq 9), 'per_action' (FPO++ Eq 10), or 'per_sample'"
            )
        if self.cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if self.cfm_clamp <= 0:
            raise ValueError(f"cfm_clamp={self.cfm_clamp} must be > 0")
        if self.cfm_loss_clamp_negative_advantages_max <= 0:
            raise ValueError("cfm_loss_clamp_negative_advantages_max must be > 0")
        if self.log_ratio_cap <= 0:
            raise ValueError(f"log_ratio_cap={self.log_ratio_cap} must be > 0")
        if not 0.0 <= self.min_timestep < self.max_timestep <= 1.0:
            raise ValueError(
                "require 0 <= min_timestep < max_timestep <= 1, got "
                f"({self.min_timestep}, {self.max_timestep})"
            )
        if self.timestep_sampling_mode not in {"uniform", "normal", "beta", "fpo_beta"}:
            raise ValueError(
                "timestep_sampling_mode must be 'uniform', 'normal', 'beta', or 'fpo_beta'"
            )
        if self.noise_sampling_mode not in {"normal", "uniform"}:
            raise ValueError("noise_sampling_mode must be 'normal' or 'uniform'")
        if self.noise_scaling_mode not in {"white", "freq_weighted"}:
            raise ValueError("noise_scaling_mode must be 'white' or 'freq_weighted'")
        if self.noise_scaling_mode == "freq_weighted" and not self.use_dct_transform:
            raise ValueError(
                "noise_scaling_mode='freq_weighted' requires use_dct_transform=True"
            )
        if self.noise_scaling_beta < 0:
            raise ValueError(f"noise_scaling_beta must be >= 0, got {self.noise_scaling_beta}")
        if self.timestep_inverse_cdf_beta <= 0:
            raise ValueError("timestep_inverse_cdf_beta must be > 0")
        if self.timestep_beta_alpha <= 0 or self.timestep_beta_beta <= 0:
            raise ValueError("timestep beta alpha/beta must be > 0")
        if self.value_loss_coef < 0:
            raise ValueError(f"value_loss_coef={self.value_loss_coef} must be >= 0")
        if self.value_huber_delta <= 0:
            raise ValueError(f"value_huber_delta={self.value_huber_delta} must be > 0")
        # Optimizer
        if self.policy_lr <= 0:
            raise ValueError(f"policy_lr={self.policy_lr} must be > 0")
        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr={self.critic_lr} must be > 0")
        if self.optimizer_weight_decay < 0:
            raise ValueError(
                f"optimizer_weight_decay={self.optimizer_weight_decay} must be >= 0"
            )
        if not all(0.0 < b < 1.0 for b in self.adam_betas):
            raise ValueError(f"adam_betas={self.adam_betas} must each be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm={self.max_grad_norm} must be > 0")
        if self.critic_max_grad_norm is not None and self.critic_max_grad_norm <= 0:
            raise ValueError(
                f"critic_max_grad_norm={self.critic_max_grad_norm} must be > 0"
            )
        if not 0.0 < self.critic_ema_decay <= 1.0:
            raise ValueError(
                f"critic_ema_decay={self.critic_ema_decay} must be in (0, 1]"
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
        # Flags
        if self.num_euler_steps < 1:
            raise ValueError(f"num_euler_steps={self.num_euler_steps} must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} must be >= 1"
            )
        if not 1 <= self.num_kept_tokens <= self.action_horizon:
            raise ValueError(
                f"num_kept_tokens={self.num_kept_tokens} must satisfy "
                f"1 <= K <= action_horizon={self.action_horizon}"
            )
        # Features
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self, lr: float | None = None) -> "AdamWConfig":
        from lerobot.optim import AdamWConfig
        kwargs = dict(
            lr=self.policy_lr if lr is None else float(lr),
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
    def num_flow_tokens(self) -> int:
        return self.num_kept_tokens if self.use_dct_transform else self.action_horizon

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

