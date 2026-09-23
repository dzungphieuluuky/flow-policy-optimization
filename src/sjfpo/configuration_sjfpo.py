"""Single home for every SJFPO hyperparameter (LeRobot ``PreTrainedConfig``)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

__all__ = ["SJFPOConfig"]


@PreTrainedConfig.register_subclass("sjfpo")
@dataclass
class SJFPOConfig(PreTrainedConfig):
    """SJFPO hyperparameters.

    Notes
    -----
    * ``n_action_steps`` is derived from ``action_horizon * executed_vs_predicted_ratio``
      at construction and stored explicitly for consumers that read it directly.
      Set ``n_action_steps`` only when you want to override the derived value.
    * ``random_rollout_steps`` (formerly ``random_rollout_steps``) is the number of
      random-policy environment steps collected **before** live optimization
      starts. It is *not* LR warmup.
    * ``num_inference_steps`` is used when sampling with ``noise=False``;
      ``num_euler_steps`` is used when sampling with ``noise=True``.
    """

    # -- Observation / action geometry -------------------------------------
    n_obs_steps: int = 16
    action_horizon: int = 16
    # None -> derived from action_horizon * executed_vs_predicted_ratio.
    n_action_steps: int | None = None
    hidden_dim: int = 128

    # -- Monte-Carlo / rollout ---------------------------------------------
    num_mc_samples: int = 64  # raised from 16: halves ratio noise (1/sqrt(M))
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 128
    num_epochs: int = 16
    num_minibatches: int = 4
    num_envs: int = 2  # low for 8 GB Blackwell mobile VRAM

    # -- Optimization ------------------------------------------------------
    gamma: float = 0.999
    gae_lambda: float = 0.995
    epsilon_clip: float = 0.10  # widened: 5% was inside ratio noise floor
    use_dct_transform: bool = True    # spectral chunk encoding; False = raw action chunk
    use_coeff_space: bool = False     # coefficient-based flow; False = discrete token flow
    use_aspo: bool = False  # disabled for discriminator run; re-enable once confirmed             # piecewise ASPO (positive PPO / negative SPO)
    use_per_bin_clip: bool = False    # two-group epsilon over DCT bins
    epsilon_clip_lowfreq: float = 0.02   # tight: low bins carry task intent
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
    vf_coef: float = 1.0  # raised: critic needs equal footing; lower to 0.1 once explained_var > 0.3
    grad_clip_norm: float = 1.0

    # -- Chunk selection ---------------------------------------------------
    top_p_spectral_tokens: float = 1.0
    executed_vs_predicted_ratio: float = 0.5

    # -- Optimizer ---------------------------------------------------------
    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_8bit: bool = False

    # -- Training loop -----------------------------------------------------
    # Random-policy environment steps collected before live optimization.
    # Stats (RunningMeanStd) are initialized after this many steps.
    random_rollout_steps: int = 100_000

    # -- Blackwell / GPU memory --------------------------------------------
    use_gradient_checkpointing: bool = True
    grad_accum_steps: int = 4

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
        """If ``n_action_steps`` is None, derive it from the ratio."""
        if self.n_action_steps is None:
            derived = max(1, round(self.action_horizon * self.executed_vs_predicted_ratio))
            self.n_action_steps = min(derived, self.action_horizon)

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
            raise ValueError(f"gae_lambda={self.gae_lambda} must satisfy 0 < lambda <= 1")
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
        if self.optimizer_lr <= 0:
            raise ValueError(f"optimizer_lr={self.optimizer_lr} must be > 0")
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
        if self.random_rollout_steps < 0:
            raise ValueError(
                f"random_rollout_steps={self.random_rollout_steps} must be >= 0"
            )
        if self.num_mc_samples < 1:
            raise ValueError(f"num_mc_samples={self.num_mc_samples} must be >= 1")

    def _validate_flags(self) -> None:
        if self.num_euler_steps < 1:
            raise ValueError(f"num_euler_steps={self.num_euler_steps} must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} must be >= 1"
            )
        r = self.executed_vs_predicted_ratio
        if not 0.0 < r <= 1.0:
            raise ValueError(
                f"executed_vs_predicted_ratio={r} must satisfy 0 < r <= 1"
            )
        p = self.top_p_spectral_tokens
        if not 0.0 < p <= 1.0:
            raise ValueError(
                f"top_p_spectral_tokens={p} must satisfy 0 < p <= 1"
            )
        if self.use_coeff_space and not self.use_dct_transform:
            raise ValueError(
                "use_coeff_space=True requires use_dct_transform=True "
                "(coefficient flow consumes DCT coefficients)"
            )
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
        """AdamW preset, honoring ``optimizer_8bit`` when set.

        ``AdamWConfig`` in recent LeRobot exposes ``use_8bit``; if the installed
        version doesn't, this falls back to a plain fp32 AdamW and warns.
        """
        kwargs = dict(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )
        if self.optimizer_8bit:
            if "use_8bit" in AdamWConfig.__dataclass_fields__:
                kwargs["use_8bit"] = True
            else:
                import warnings

                warnings.warn(
                    "optimizer_8bit=True requested but AdamWConfig does not "
                    "expose 'use_8bit'; falling back to fp32 AdamW.",
                    RuntimeWarning,
                    stacklevel=2,
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