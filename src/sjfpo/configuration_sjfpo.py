"""Central SJFPO policy configuration (LeRobot-native).

Single home for every SJFPO hyperparameter, subclassing LeRobot's
``PreTrainedConfig`` so runs serialize / resume through the HubMixin +
draccus registry like any other LeRobot policy (``type = "sjfpo"``).

Legacy name mapping (``src/sjfpo/config.py::SJFPOConfig``):
    horizon <-> chunk_size, hidden_dim <-> hidden_size,
    optimizer_lr <-> lr, optimizer_weight_decay <-> wd,
    grad_clip_norm <-> grad_clip.
``n_action_steps`` is the executed prefix at inference
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
    """Configuration class for SJFPO (Spectral Joint Flow Policy Optimization).

    Args:
        n_obs_steps: Stacked obs frames conditioning each chunk (T).
        horizon: Predicted action-chunk length (H).
        n_action_steps: Executed prefix per replan at inference (<= horizon).
        hidden_dim: Fused latent dimension (D).
        num_mc_samples: (tau, eps) pairs per chunk for flow likelihoods.
        euler_train: Euler steps for training rollouts (stochastic).
        euler_eval: Euler steps for eval (deterministic zero-sampling).
        chunks_per_update: Chunks collected per update round.
        num_epochs: Gradient steps per collected batch.
        gamma: Discount factor.
        gae_lambda: GAE lambda for chunk semi-MDP advantages.
        clip_range: ASPO trust-region width (ratios).
        clip_range_vf: PPO-style value-loss clip (return units, not ratios).
        cfm_clamp: Symmetric clamp on CFM losses.
        log_ratio_cap: STE cap on log-ratio magnitude.
        tau_range: Flow time range (t_min, t_max).
        dit_depth: DiT blocks in the policy flow model.
        dit_heads: Attention heads per DiT block.
        dit_mlp_ratio: MLP width multiplier per DiT block.
        dit_time_dim: Sinusoidal timestep embedding dim for DiT.
        vf_coef: Value-loss coefficient.
        grad_clip_norm: Grad-norm clip (also forwarded to the optimizer preset).
        top_p_spectral_tokens: Kept low-frequency DCT fraction.
        executed_vs_predicted_ratio: Inference executed fraction (0 < r <= 1).
        optimizer_lr: AdamW learning rate.
        optimizer_weight_decay: AdamW weight decay.
    """

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
    clip_range: float = 0.05
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

    def validate_features(self) -> None:
        """Validate input/output feature compatibility.

        Call this explicitly from your policy's __init__ — the base class does not.
        SJFPO is state-only (MuJoCo): no images required.
        """
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
        """Relative timestep offsets the dataset loader provides per observation.

        16 stacked past frames ending at the current step.
        """
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns."""
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Env rollouts supply executed-step rewards directly; no dataset rewards."""
        return None
