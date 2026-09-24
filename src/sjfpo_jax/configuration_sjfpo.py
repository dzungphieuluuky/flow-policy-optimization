"""Single home for every SJFPO hyperparameter (LeRobot ``PreTrainedConfig``)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

__all__ = ["SJFPOConfig"]


@PreTrainedConfig.register_subclass("sjfpo_jax")
@dataclass
class SJFPOConfig(PreTrainedConfig):
    """SJFPO hyperparameters.

    Notes
    -----
    # ``n_action_steps`` must be set explicitly; ``action_horizon`` controls
      the full predicted chunk length (training) and the executed prefix length (eval).
    * ``num_inference_steps`` is used when sampling with ``noise=False``;
      ``num_euler_steps`` is used when sampling with ``noise=True``.
    """

    # -- Observation / action geometry -------------------------------------
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16  # must be set explicitly; no derived value
    hidden_dim: int = 128

    # -- Monte-Carlo / rollout ---------------------------------------------
    num_mc_samples: int = 16
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 256
    num_epochs: int = 4
    num_minibatches: int = 4
    num_envs: int = 64

    # -- Optimization ------------------------------------------------------
    gamma: float = 0.999
    gae_lambda: float = 0.995
    epsilon_clip: float = 0.05
    use_dct_transform: bool = True  # spectral chunk encoding; False = raw action chunk
    use_aspo: bool = True
    use_per_bin_clip: bool = False  # two-group epsilon over DCT bins
    epsilon_clip_lowfreq: float = 0.02  # tight: low bins carry task intent
    epsilon_clip_highfreq: float = 0.10  # loose: high bins are detail/exploration
    cfm_clamp: float = 20.0
    log_ratio_cap: float = 10.0
    advantage_clamp: float = 100.0
    use_clipped_value_loss: bool = True
    min_timestep: float = 0.005
    max_timestep: float = 0.995

    # -- DiT backbone ------------------------------------------------------
    dit_depth: int = 4
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    dit_time_dim: int = 128
    # Flow time t in [0, 1] is scaled by this before the DDPM sinusoidal ladder
    # (designed for integer steps 0–1000); without it every channel sits at
    # angle <= 1 and t is nearly unresolvable.
    timestep_scale: float = 1000.0

    # -- Losses / optimization --------------------------------------------
    vf_coef: float = 1.0  # skrl PPO default; 0.5 was too weak for critic
    grad_clip_norm: float = 0.5  # skrl PPO default; 1.0 was too loose

    # -- Chunk selection ---------------------------------------------------
    top_p_spectral_tokens: float = 1.0

    # -- Optimizer ---------------------------------------------------------
    policy_lr: float = 3e-4
    critic_lr: float = 9e-4  # 3x policy_lr for faster value tracking
    optimizer_weight_decay: float = 1e-2  # skrl AdamWConfig default; 1e-4 too weak for critic

    # -- Training loop -----------------------------------------------------
    # No separate warm-up phase; stats initialize naturally during first rollout.
    # -- Blackwell / GPU memory --------------------------------------------
    use_gradient_checkpointing: bool = False
    grad_accum_steps: int = 1  # skrl PPO default; >1 reduces GPU memory at cost of speed

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        self._resolve_n_action_steps()
        self._validate_geometry()
        self._validate_loss_params()
        self._validate_optimizer()
        self._validate_flags()

    def _resolve_n_action_steps(self) -> None:
        """If ``n_action_steps`` is None, use ``action_horizon`` as the default."""
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon

    def _validate_geometry(self) -> None:
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
        if self.dit_time_dim < 1:
            raise ValueError(f"dit_time_dim={self.dit_time_dim} must be >= 1")
        if self.timestep_scale <= 0:
            raise ValueError(f"timestep_scale={self.timestep_scale} must be > 0")

    def _validate_loss_params(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError(
                f"gae_lambda={self.gae_lambda} must satisfy 0 < lambda <= 1"
            )
        if not 0.0 < self.epsilon_clip < 1.0:
            raise ValueError(
                f"epsilon_clip={self.epsilon_clip} must satisfy 0 < eps < 1"
            )
        if self.use_per_bin_clip:
            lo, hi = self.epsilon_clip_lowfreq, self.epsilon_clip_highfreq
            if not 0.0 < lo <= hi < 1.0:
                raise ValueError(
                    "require 0 < epsilon_clip_lowfreq <= epsilon_clip_highfreq < 1, "
                    f"got ({lo}, {hi})"
                )
        if self.cfm_clamp <= 0:
            raise ValueError(f"cfm_clamp={self.cfm_clamp} must be > 0")
        if self.log_ratio_cap <= 0:
            raise ValueError(f"log_ratio_cap={self.log_ratio_cap} must be > 0")
        if self.advantage_clamp <= 0:
            raise ValueError(f"advantage_clamp={self.advantage_clamp} must be > 0")
        if not 0.0 <= self.min_timestep < self.max_timestep <= 1.0:
            raise ValueError(
                "require 0 <= min_timestep < max_timestep <= 1, got "
                f"({self.min_timestep}, {self.max_timestep})"
            )
        if self.vf_coef < 0:
            raise ValueError(f"vf_coef={self.vf_coef} must be >= 0")

    def _validate_optimizer(self) -> None:
        if self.policy_lr <= 0:
            raise ValueError(f"policy_lr={self.policy_lr} must be > 0")
        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr={self.critic_lr} must be > 0")
        if self.optimizer_weight_decay < 0:
            raise ValueError(
                f"optimizer_weight_decay={self.optimizer_weight_decay} must be >= 0"
            )
        if self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm={self.grad_clip_norm} must be > 0")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps={self.grad_accum_steps} must be >= 1")
        if self.num_minibatches < 1:
            raise ValueError(f"num_minibatches={self.num_minibatches} must be >= 1")
        if self.chunks_per_update < 1:
            raise ValueError(f"chunks_per_update={self.chunks_per_update} must be >= 1")
        if self.chunks_per_update < self.num_minibatches:
            raise ValueError(
                f"chunks_per_update={self.chunks_per_update} must be >= "
                f"num_minibatches={self.num_minibatches}"
            )
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs={self.num_epochs} must be >= 1")
        if self.num_envs < 1:
            raise ValueError(f"num_envs={self.num_envs} must be >= 1")
        if self.num_mc_samples < 1:
            raise ValueError(f"num_mc_samples={self.num_mc_samples} must be >= 1")

    def _validate_flags(self) -> None:
        if self.num_euler_steps < 1:
            raise ValueError(f"num_euler_steps={self.num_euler_steps} must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} must be >= 1"
            )
        p = self.top_p_spectral_tokens
        if not 0.0 < p <= 1.0:
            raise ValueError(f"top_p_spectral_tokens={p} must satisfy 0 < p <= 1")
        if self.use_per_bin_clip and not self.use_dct_transform:
            raise ValueError(
                "use_per_bin_clip=True requires use_dct_transform=True "
                "(per-bin epsilon needs frequency bins)"
            )

    # ----------------------------------------------------------------------
    # LeRobot hooks
    # ----------------------------------------------------------------------

    def validate_features(self) -> None:
        """Require observation.state + action features (base class skips this)."""
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return the standard FP32/bf16-compatible AdamW preset."""
        kwargs = dict(
            lr=self.policy_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )
        return AdamWConfig(**kwargs)

    def get_scheduler_preset(self):
        """No LR scheduler by default. Override in a subclass to return one."""
        return None

    # ----------------------------------------------------------------------
    # Dataset windows
    # ----------------------------------------------------------------------

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
