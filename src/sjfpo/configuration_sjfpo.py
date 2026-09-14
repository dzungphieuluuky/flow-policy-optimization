"""Single home for every SJFPO hyperparameter (LeRobot PreTrainedConfig).

Runs serialize/resume through HubMixin + draccus (``type = "sjfpo"``).
Legacy names (``config.py::SJFPOConfig``): horizon/chunk_size,
hidden_dim/hidden_size, optimizer_lr/lr, optimizer_weight_decay/wd,
grad_clip_norm/grad_clip. ``n_action_steps`` is the executed inference prefix
(``round(executed_vs_predicted_ratio * horizon)``); training executes the
full horizon open-loop.
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
    horizon: int = 16
    n_action_steps: int = 8
    hidden_dim: int = 128

    num_mc_samples: int = 16
    euler_train: int = 16
    euler_eval: int = 16
    chunks_per_update: int = 128
    num_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    clip_range_vf: float = 10.0
    cfm_clamp: float = 20.0
    log_ratio_cap: float = 10.0
    tau_range: tuple[float, float] = (0.005, 0.995)

    dit_depth: int = 8
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    dit_time_dim: int = 256

    vf_coef: float = 0.1
    grad_clip_norm: float = 1.0

    top_p_spectral_tokens: float = 0.3
    spectral_pool: str = "mean"
    executed_vs_predicted_ratio: float = 0.5

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.n_action_steps <= self.horizon:
            raise ValueError(
                f"n_action_steps={self.n_action_steps} must satisfy "
                f"0 < n <= horizon={self.horizon}"
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
        if self.spectral_pool not in ("mean", "attention"):
            raise ValueError(
                f"spectral_pool={self.spectral_pool!r} must be 'mean' or 'attention'"
            )

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
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """None: rollouts supply executed-step rewards directly."""
        return None
