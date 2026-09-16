"""Single home for every SJFPO hyperparameter (LeRobot PreTrainedConfig).

Runs serialize/resume through HubMixin + draccus (``type = "sjfpo"``).
Legacy names (``config.py::SJFPOConfig``): horizon/chunk_size (now action_horizon),
optimizer_lr/lr, optimizer_weight_decay/wd,
grad_clip_norm/grad_clip. ``n_action_steps`` is the intended inference prefix
(``round(executed_vs_predicted_ratio * action_horizon)``); training executes the
full action_horizon open-loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("sjfpo")
@dataclass
class SJFPOConfig(PreTrainedConfig):
    """SJFPO hyperparameters (see field names; all self-explanatory)."""

    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 8
    hidden_dim: int = 128

    num_mc_samples: int = 16
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 128
    num_epochs: int = 10
    num_minibatches: int = 8
    num_envs: int = 8  # parallel rollout envs (AsyncVectorEnv); total env-steps
    # per update stay constant, so wall-clock collection scales ~linearly.
    # 8 fits a 16-core/32GB laptop (one MuJoCo proc is a few hundred MB); 1
    # reproduces the legacy single-env loop exactly.

    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    clip_range_vf: float = 10.0
    cfm_clamp: float = 20.0
    log_ratio_cap: float = 10.0
    min_timestep: float = 0.005
    max_timestep: float = 0.995

    dit_depth: int = 8
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    dit_time_dim: int = 256
    timestep_scale: float = 1000.0  # Flow time t in [0,1] is scaled by this before
    # the DDPM sinusoidal ladder (designed for integer steps 0-1000); without it
    # every channel sits at angle <= 1 and t is nearly unresolvable.

    vf_coef: float = 0.1
    kl_coef: float = 0.01
    ema_decay: float = 0.999
    grad_clip_norm: float = 1.0

    top_p_spectral_tokens: float = 0.3
    spectral_condition_mode: str = "attention"  # attention carries bin order via
    # frequency encoding; mean is permutation-invariant over bins by construction.
    executed_vs_predicted_ratio: float = 0.5

    optimizer_lr: float = 1e-4
    optimizer_lr_critic: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_8bit: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.num_euler_steps < 1:
            raise ValueError(f"num_euler_steps={self.num_euler_steps} must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps={self.num_inference_steps} must be >= 1")
        if not 0 < self.n_action_steps <= self.action_horizon:
            raise ValueError(
                f"n_action_steps={self.n_action_steps} must satisfy "
                f"0 < n <= action_horizon={self.action_horizon}"
            )
        r = self.executed_vs_predicted_ratio
        if not 0.0 < r <= 1.0:
            raise ValueError(f"executed_vs_predicted_ratio={r} must satisfy 0 < r <= 1")
        p = self.top_p_spectral_tokens
        if not 0.0 < p <= 1.0:
            raise ValueError(f"top_p_spectral_tokens={p} must satisfy 0 < p <= 1")
        if not 0.0 < self.epsilon_low <= self.epsilon_high < 1.0:
            raise ValueError(
                "require 0 < epsilon_low <= epsilon_high < 1, got "
                f"({self.epsilon_low}, {self.epsilon_high})"
            )
        if not 0.0 < self.min_timestep < self.max_timestep < 1.0:
            raise ValueError(
                "require 0 < min_timestep < max_timestep < 1, got "
                f"({self.min_timestep}, {self.max_timestep})"
            )
        if self.timestep_scale <= 0:
            raise ValueError(f"timestep_scale={self.timestep_scale} must be positive")
        if self.spectral_condition_mode not in ("mean", "attention"):
            raise ValueError(
                f"spectral_condition_mode={self.spectral_condition_mode!r} must be 'mean' or 'attention'"
            )
        if self.optimizer_lr <= 0 or self.optimizer_lr_critic <= 0:
            raise ValueError(
                "require optimizer_lr > 0 and optimizer_lr_critic > 0, got "
                f"({self.optimizer_lr}, {self.optimizer_lr_critic})"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay={self.ema_decay} must satisfy 0 <= d < 1")
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef={self.kl_coef} must be >= 0")
        if self.num_envs < 1:
            raise ValueError(f"num_envs={self.num_envs} must be >= 1")

    def validate_features(self) -> None:
        """Require observation.state + action features (base class skips this)."""
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Return a LRSchedulerConfig from lerobot.optim, or None."""
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Stacked past frames ending at the current step."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Action-chunk offsets the dataset loader returns."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """None: rollouts supply executed-step rewards directly."""
        return None
