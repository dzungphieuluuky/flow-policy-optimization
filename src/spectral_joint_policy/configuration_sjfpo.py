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

    # -- Observation / action geometry -----------------------------
    report_to: str = "all"  # wandb | tensorboard | all | none
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16
    action_perturb_std: float = 0.02
    hidden_dim: int = 128

    # -- Rollout and batching ---------------------------------------
    num_envs: int = 64
    num_mc_samples: int = 16
    # 1, not 2: reuse is what accelerates the feature-rank decay that precedes the
    # late-training collapse (PFO, NeurIPS 2024: more epochs => faster rank loss, and
    # rank loss breaks the trust region). Same samples per update, no reuse, so
    # rank decays at half the rate for the same env steps.
    num_learning_epochs: int = 1
    chunks_per_update: int = 1024
    batch_size: int = 128
    # Groups consecutive minibatches into one optimizer step. Does NOT change peak
    # memory: one microbatch's DiT activations are the peak either way, so
    # batch_size and num_mc_samples are the memory knobs.
    grad_accum_steps: int = 1

    # -- Flow sampling -----------------------------------------------
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    # Per-mode std over the token axis. "white" is 1.0. "freq_weighted" is NEED's
    # beta=1 coloring, which pushes base variance into the low DCT modes; it
    # requires use_dct_transform, since the token axis is time without the DCT.
    noise_scaling_mode: str = "white"
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    # -- Trust region and CFM loss -----------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # One epsilon for the whole chunk. Per-frequency clipping has nothing to bind to:
    # the "sample" reduction averages every token into one number per chunk.
    clip_param: float = 0.05
    trust_region_mode: str = "aspo"  # ppo | spo | aspo, FPO++'s default
    # "mean" divides FPO++'s sum (Eq. 8) by tokens*action_dim, keeping the loss O(1)
    # so cfm_clamp stays a real guard. The cost: the loss scale IS the trust-region
    # temperature, so clip_param is not comparable to FPO++'s 0.05 and must be swept.
    cfm_loss_reduction: str = "mean"  # mean | sum | sqrt
    cfm_clamp: float = 200.0
    cfm_mask_unexecuted: bool = True
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 200.0
    log_ratio_cap: float = 10.0  # official cfm_diff_clamp_max, one-sided

    # -- Critic and value targets -----------------------------------
    num_critics: int = 2
    normalize_advantage: bool = True
    value_loss_coef: float = 1.0  # the regression target is the raw return
    # Huber delta as a multiple of the batch return RMS, so the quadratic-to-linear
    # transition tracks the target's own scale (returns_rms runs ~2.9 on
    # InvertedPendulum, ~15 on Hopper). Bounds the per-sample gradient, so a few
    # large-residual chunks stop owning critic_grad_norm_preclip.
    value_huber_delta: float = 1.0
    use_clipped_value_loss: bool = True
    # gamma ** action_horizon puts 0.85 of the value target on the critic's own
    # output, the regime a target network is meant to hold fixed.
    use_target_critic: bool = True
    # qc and TD-JEPA use 0.005 but refresh over offline batches; online that left the
    # teacher tens of rollouts behind. 0.05 is a ~2.5-rollout time constant.
    critic_ema_decay: float = 0.05

    # -- Spectral chunk encoding ------------------------------------
    use_dct_transform: bool = True
    # At num_kept_tokens == action_horizon the transform is the full orthonormal DCT,
    # a rotation of the loss and a no-op up to numerics. Smaller truncates.
    num_kept_tokens: int = 16

    # -- DiT backbone -----------------------------------------------
    dit_depth: int = 2
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    timestep_scale: float = 10000.0

    # -- Optimizer --------------------------------------------------
    policy_lr: float = 1e-5
    critic_lr: float = 1e-5  # legacy split-checkpoint compatibility
    optimizer_weight_decay: float = 1e-4
    # Decay rates of Adam's two gradient-moment running averages. Equal by default:
    # Dohare et al. (arXiv:2306.13812) report Adam exacerbating loss of plasticity and
    # policy collapse being mitigated by using the same value for both moments, so the
    # asymmetric (0.9, 0.999) pair is the thing to suspect. Kept as the raw pair so any
    # asymmetry is still expressible.
    adam_betas: tuple[float, float] = (0.999, 0.999)
    # rsl-rl's default. Under AdamW the threshold does not set the step size; it
    # decides whether a transient spike may pollute the second-moment estimate.
    # Measured: a 1/170 rescale moved the trajectory 1.2e-04 per parameter, one
    # 1e4 spike moved it 9.0e-02.
    max_grad_norm: float = 1.0
    # Clipped separately from the actor: the critic's norm is two orders larger.
    critic_max_grad_norm: float | None = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    # -- Eval and early stopping ------------------------------------
    # The max over hundreds of evals is biased well above the truth, and too few
    # episodes puts the standard error near the size of the regressions these knobs
    # watch for.
    eval_max_episodes: int = 50
    # Stop once the return has sat below this fraction of its own peak for
    # stop_patience evals in a row, so the run ends on the peak region.
    stop_return_frac: float = 0.7
    stop_patience: int = 5
    # Each completed window of stop_patience bad evals cuts the lrs by
    # lr_decay; the run stops once max_lr_decays of them have been spent.
    lr_decay: float = 0.5
    max_lr_decays: int = 3

    # -- Diagnostics ------------------------------------------------
    # Each drift metric is a full critic forward over 2 * n_chunks windows. The
    # loop skips it whenever no tracker is enabled, so the flag is the manual
    # override for a report_to=none run that still wants the numbers.
    log_drift_diagnostics: bool = True


    def __post_init__(self) -> None:
        super().__post_init__()
        # n_action_steps is optional, so it resolves here rather than in the
        # validators that read it. Everything below can assume it is an int.
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        self.validate_geometry()
        self.validate_loss_params()
        self.validate_optimizer()
        self.validate_eval_schedule()
        self.validate_flags()

    def validate_geometry(self) -> None:
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

    def validate_eval_schedule(self) -> None:
        """Validate the eval budget and the plateau-driven stop and decay."""
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

    def validate_loss_params(self) -> None:
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
        if self.value_huber_delta <= 0:
            raise ValueError(
                f"value_huber_delta={self.value_huber_delta} must be > 0"
            )

    def validate_optimizer(self) -> None:
        """Validate learning rates, clipping, and batch-shaping parameters."""
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

    def validate_flags(self) -> None:
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
