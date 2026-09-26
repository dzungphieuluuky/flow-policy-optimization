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
    # ``n_action_steps`` must be set explicitly; ``action_horizon`` controls
      the full predicted chunk length (training) and the executed prefix length (eval).
    * ``num_inference_steps`` is used when sampling with ``noise=False``;
      ``num_euler_steps`` is used when sampling with ``noise=True``.
    """

    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16  # must be set explicitly; no derived value
    action_perturb_std: float = 0.02
    hidden_dim: int = 128

    num_mc_samples: int = 16
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 256
    num_epochs: int = 16
    num_minibatches: int = 4
    num_envs: int = 64

    gamma: float = 0.999
    gae_lambda: float = 0.995
    epsilon_clip: float = 0.05
    use_dct_transform: bool = False  # spectral chunk encoding; False = raw action chunk
    # FPO++ trust region: ppo, spo, or aspo (official default).
    trust_region_mode: str = "aspo"
    use_per_bin_clip: bool = False  # legacy diagnostics-only bin clipping
    epsilon_clip_lowfreq: float = 0.02  # tight: low bins carry task intent
    epsilon_clip_highfreq: float = 0.10  # loose: high bins are detail/exploration
    # "mean" (E[(v-v*)^2] over the chunk) keeps the CFM loss O(1) -- ~1.25 for
    # a zero-velocity model on unit noise -- so cfm_clamp below is a real guard.
    # "sqrt" (FPO++'s default) scales the loss by sqrt(horizon*action_dim); at
    # 16x17 that starts the run above cfm_clamp, and clamp() has zero gradient
    # above its max, which silently pins rho at 1 and kills the policy gradient.
    cfm_loss_reduction: str = "mean"  # mean | sum | sqrt
    cfm_clamp: float = 20.0
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    log_ratio_cap: float = 10.0  # official cfm_diff_clamp_max (one-sided)
    advantage_clamp: float = 100.0
    normalize_advantage: bool = True
    use_clipped_value_loss: bool = False
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    dit_depth: int = 4
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    # Flow time t in [0, 1] is scaled by this before the DDPM sinusoidal ladder
    # (designed for integer steps 0–1000); without it every channel sits at
    # angle <= 1 and t is nearly unresolvable.
    timestep_scale: float = 1000.0

    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    vf_coef: float = 1.0  # 0.5 was too weak for the critic
    grad_clip_norm: float = 1.0

    top_p_spectral_tokens: float = 1.0

    policy_lr: float = 1e-4
    critic_lr: float = 1e-4  # legacy split-checkpoint compatibility
    optimizer_weight_decay: float = 1e-4

    # No separate warm-up phase; stats initialize naturally during first rollout.
    grad_accum_steps: int = 1  # >1 reduces GPU memory at cost of speed


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
        """Validate observation, action, and DiT shape parameters.

        Raises:
            ValueError: If any dimension is out of range, or ``hidden_dim`` is
                not divisible by ``dit_heads``.
        """
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

    def _validate_loss_params(self) -> None:
        """Validate discount, trust-region, clamp, and sampling parameters.

        Raises:
            ValueError: If any discount, epsilon, clamp bound, or enumerated
                mode is outside its allowed set.
        """
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
        if self.trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError("trust_region_mode must be 'ppo', 'spo', or 'aspo'")
        if self.cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if self.cfm_clamp <= 0:
            raise ValueError(f"cfm_clamp={self.cfm_clamp} must be > 0")
        if self.cfm_loss_clamp_negative_advantages_max <= 0:
            raise ValueError("cfm_loss_clamp_negative_advantages_max must be > 0")
        if self.log_ratio_cap <= 0:
            raise ValueError(f"log_ratio_cap={self.log_ratio_cap} must be > 0")
        if self.advantage_clamp <= 0:
            raise ValueError(f"advantage_clamp={self.advantage_clamp} must be > 0")
        if not 0.0 <= self.min_timestep < self.max_timestep <= 1.0:
            raise ValueError(
                "require 0 <= min_timestep < max_timestep <= 1, got "
                f"({self.min_timestep}, {self.max_timestep})"
            )
        if self.timestep_sampling_mode not in {
            "uniform",
            "normal",
            "beta",
            "fpo_beta",
        }:
            raise ValueError(
                "timestep_sampling_mode must be 'uniform', 'normal', 'beta', "
                "or 'fpo_beta'"
            )
        if self.noise_sampling_mode not in {"normal", "uniform"}:
            raise ValueError("noise_sampling_mode must be 'normal' or 'uniform'")
        if self.timestep_inverse_cdf_beta <= 0:
            raise ValueError("timestep_inverse_cdf_beta must be > 0")
        if self.timestep_beta_alpha <= 0 or self.timestep_beta_beta <= 0:
            raise ValueError("timestep beta alpha/beta must be > 0")
        if self.vf_coef < 0:
            raise ValueError(f"vf_coef={self.vf_coef} must be >= 0")

    def _validate_optimizer(self) -> None:
        """Validate learning rates, clipping, and batch-shaping parameters.

        Raises:
            ValueError: If a rate or count is non-positive, or if
                ``chunks_per_update`` is smaller than ``num_minibatches``.
        """
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
        """Validate step counts, spectral token fraction, and flag combinations.

        Raises:
            ValueError: If a step count is below one, if
                ``top_p_spectral_tokens`` is outside ``(0, 1]``, or if
                per-bin clipping is requested without the DCT transform.
        """
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


    def validate_features(self) -> None:
        """Require the ``observation.state`` and ``action`` features.

        The base class skips this check, so it is enforced here.

        Raises:
            ValueError: If either required feature is missing.
        """
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self, lr: float | None = None) -> AdamWConfig:
        """Build the AdamW preset used for the actor and critic.

        Args:
            lr: Override learning rate; ``policy_lr`` is used when omitted.

        Returns:
            An :class:`AdamWConfig` carrying the learning rate, weight decay,
            and gradient clip norm.
        """
        kwargs = dict(
            lr=self.policy_lr if lr is None else float(lr),
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )
        return AdamWConfig(**kwargs)

    def get_scheduler_preset(self):
        """Return the learning-rate scheduler, if any.

        Returns:
            None; training runs without a scheduler unless a subclass
            overrides this.
        """
        return None


    @property
    def observation_delta_indices(self) -> list[int] | None:
        """Observation window offsets fed to the policy.

        Returns:
            The ``n_obs_steps`` past-frame offsets ending at the current step.
        """
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Action-chunk offsets returned by the dataset loader.

        Returns:
            One offset per step of a full action chunk.
        """
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Reward offsets requested from the dataset loader.

        Returns:
            None, because rollouts supply executed-step rewards directly.
        """
        return None
