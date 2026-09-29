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
    * ``action_horizon`` is the full predicted chunk length (training) and the
      executed prefix length (eval); ``n_action_steps`` falls back to it.
    * ``num_inference_steps`` is used for zero-sampling; ``num_euler_steps`` is
      used when the flow starts from sampled noise.
    """

    report_to: str = "wandb"  # wandb | tensorboard | all | none
    num_critics: int = 1  # ensemble members; min over members feeds the GAE backup
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16  # falls back to action_horizon when None
    action_perturb_std: float = 0.02
    hidden_dim: int = 128

    num_mc_samples: int = 16
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 256
    num_learning_epochs: int = 4
    batch_size: int = 128  # chunks per minibatch; num_mini_batches is derived
    num_envs: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.05
    use_dct_transform: bool = False  # spectral chunk encoding; False = raw action chunk
    # FPO++ trust region: ppo, spo, or aspo (official default).
    trust_region_mode: str = "aspo"
    use_per_freq_clipping: bool = False
    clip_param_lowfreq: float = 0.02  # tight: low freqs carry task intent
    clip_param_highfreq: float = 0.10
    # "mean" divides the sum FPO++ uses (Eq. 8 sums CFM losses across chunk
    # timesteps) by tokens*action_dim, keeping the loss O(1) so cfm_clamp stays
    # a real guard. The cost is that the loss scale IS the trust-region
    # temperature, since ratio = exp(l_old - l_new), so clip_param is not
    # comparable to FPO++'s 0.05 and must be swept. Read clipfrac first: if it
    # sits near zero, clip_param is not constraining anything.
    cfm_loss_reduction: str = "mean"  # mean | sum | sqrt
    cfm_clamp: float = 20.0
    cfm_mask_unexecuted: bool = (
        True  # drop the unexecuted tail of a truncated chunk from the CFM loss
    )
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    log_ratio_cap: float = 10.0  # official cfm_diff_clamp_max (one-sided)
    normalize_advantage: bool = True
    use_clipped_value_loss: bool = (
        False  # clip the value update against the rollout snapshot, as rsl-rl does
    )
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    # Per-mode std across the token axis: "white" (1.0) or NEED's beta=1 1/f
    # coloring, which concentrates base variance in the low DCT modes.
    noise_scaling_mode: str = "white"
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    dit_depth: int = 2
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    timestep_scale: float = 10000.0

    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    value_loss_coef: float = 1.0  # rsl-rl's default; the target is raw returns
    max_grad_norm: float = 20.0  # critic_grad_norm_preclip runs ~150x over 1.0
    # Critic norm ran 45-170 against the actor's 0.14-3.7, so one shared clip of
    # 20 did nothing for the actor and shrank every critic step by 2-9x. None
    # disables the critic clip entirely, to measure what the guard was hiding.
    critic_max_grad_norm: float | None = 200.0

    num_kept_tokens: int = 16  # DCT coefficients the flow sees; inert when use_dct_transform=False

    policy_lr: float = 1e-3
    critic_lr: float = 1e-3  # legacy split-checkpoint compatibility
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
        """Validate observation, action, and DiT shape parameters."""
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

    def _validate_loss_params(self) -> None:
        """Validate discount, trust-region, clamp, and sampling parameters."""
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError(
                f"gae_lambda={self.gae_lambda} must satisfy 0 < lambda <= 1"
            )
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError(
                f"clip_param={self.clip_param} must satisfy 0 < eps < 1"
            )
        if self.use_per_freq_clipping:
            lo, hi = self.clip_param_lowfreq, self.clip_param_highfreq
            if not 0.0 < lo <= hi < 1.0:
                raise ValueError(
                    "require 0 < clip_param_lowfreq <= clip_param_highfreq < 1, "
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
        if self.noise_scaling_mode not in {"white", "freq_weighted"}:
            raise ValueError("noise_scaling_mode must be 'white' or 'freq_weighted'")
        if self.noise_scaling_mode == "freq_weighted" and not self.use_dct_transform:
            raise ValueError(
                "noise_scaling_mode='freq_weighted' requires "
                "use_dct_transform=True (the token axis is time, not frequency, "
                "without the DCT)"
            )
        if self.timestep_inverse_cdf_beta <= 0:
            raise ValueError("timestep_inverse_cdf_beta must be > 0")
        if self.timestep_beta_alpha <= 0 or self.timestep_beta_beta <= 0:
            raise ValueError("timestep beta alpha/beta must be > 0")
        if self.value_loss_coef < 0:
            raise ValueError(
                f"value_loss_coef={self.value_loss_coef} must be >= 0"
            )

    def _validate_optimizer(self) -> None:
        """Validate learning rates, clipping, and batch-shaping parameters."""
        if self.policy_lr <= 0:
            raise ValueError(f"policy_lr={self.policy_lr} must be > 0")
        if self.critic_lr <= 0:
            raise ValueError(f"critic_lr={self.critic_lr} must be > 0")
        if self.optimizer_weight_decay < 0:
            raise ValueError(
                f"optimizer_weight_decay={self.optimizer_weight_decay} must be >= 0"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm={self.max_grad_norm} must be > 0")
        if self.critic_max_grad_norm is not None and self.critic_max_grad_norm <= 0:
            raise ValueError(
                f"critic_max_grad_norm={self.critic_max_grad_norm} must be > 0"
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

    def _validate_flags(self) -> None:
        """Validate step counts, spectral token fraction, and flag combinations."""
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
        if self.use_per_freq_clipping and not self.use_dct_transform:
            raise ValueError(
                "use_per_freq_clipping=True requires use_dct_transform=True "
                "(per-frequency epsilon needs frequencies)"
            )


    def validate_features(self) -> None:
        """Require the ``observation.state`` and ``action`` features.

        The base class skips this check, so it is enforced here.
        """
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self, lr: float | None = None) -> AdamWConfig:
        """Build the AdamW preset used for the actor and critic.

        ``lr`` overrides ``policy_lr`` when given.
        """
        kwargs = dict(
            lr=self.policy_lr if lr is None else float(lr),
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.max_grad_norm,
        )
        return AdamWConfig(**kwargs)

    def get_scheduler_preset(self):
        """Return the learning-rate scheduler, if any.

        None; training runs without a scheduler unless a subclass overrides it.
        """
        return None


    @property
    def num_mini_batches(self) -> int:
        """Minibatches per epoch: ``chunks_per_update // batch_size``, floored at one.

        Computed on access rather than in ``__post_init__`` so it cannot go
        stale if ``chunks_per_update`` or ``batch_size`` is reassigned later.
        """
        return max(1, self.chunks_per_update // self.batch_size)

    @property
    def num_flow_tokens(self) -> int:
        """Token count the flow model operates on.

        ``num_kept_tokens`` with the DCT on, otherwise the full action horizon.
        ``num_kept_tokens`` is inert without the DCT rather than an error: the
        token axis is time there, so there is no frequency count to keep.
        """
        return self.num_kept_tokens if self.use_dct_transform else self.action_horizon

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """The ``n_obs_steps`` past-frame offsets ending at the current step."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """One offset per step of a full action chunk."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """None, because rollouts supply executed-step rewards directly."""
        return None
