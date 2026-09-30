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

    report_to: str = "all"  # wandb | tensorboard | all | none
    num_critics: int = 2  # ensemble members; min over members feeds the GAE backup
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int = 16  # falls back to action_horizon when None
    action_perturb_std: float = 0.02
    hidden_dim: int = 128

    num_mc_samples: int = 16
    num_euler_steps: int = 16
    num_inference_steps: int = 16
    chunks_per_update: int = 256
    # 4 epochs per rollout accelerated the feature-rank decay that precedes the
    # late-training collapse (PFO, NeurIPS 2024: more epochs per batch => faster
    # pre-activation-norm growth and rank loss, and rank loss breaks the trust
    # region). Same samples per update, half the reuse.
    num_learning_epochs: int = 2
    batch_size: int = 128  # chunks per minibatch; num_mini_batches is derived
    num_envs: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95
    # One epsilon for the whole chunk. No per-frequency clipping: the trust
    # region applies to the "sample" CFM reduction, which averages every token
    # into one number per chunk, so no per-frequency ratio survives into the
    # loss to hang a different epsilon on.
    clip_param: float = 0.05
    use_dct_transform: bool = False  # spectral chunk encoding; False = raw action chunk
    # FPO++ trust region: ppo, spo, or aspo (official default).
    trust_region_mode: str = "aspo"
    # "mean" divides FPO++'s sum (Eq. 8 sums CFM losses over chunk timesteps) by
    # tokens*action_dim, keeping the loss O(1) so cfm_clamp stays a real guard.
    # The cost: the loss scale IS the trust-region temperature, since
    # ratio = exp(l_old - l_new), so clip_param is not comparable to FPO++'s
    # 0.05 and must be swept. Read clipfrac first: near zero means clip_param
    # constrains nothing.
    cfm_loss_reduction: str = "mean"  # mean | sum | sqrt
    cfm_clamp: float = 200.0
    cfm_mask_unexecuted: bool = (
        True  # drop the unexecuted tail of a truncated chunk from the CFM loss
    )
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 200.0
    log_ratio_cap: float = 10.0  # official cfm_diff_clamp_max (one-sided)
    normalize_advantage: bool = True
    use_clipped_value_loss: bool = (
        True  # clip the value update against the rollout snapshot, as rsl-rl does
    )
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    # Per-mode std over the token axis: "white" (1.0), or NEED's beta=1 1/f
    # coloring, which pushes base variance into the low DCT modes.
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
    # Huber delta as a multiple of the batch return RMS, so the quadratic-to-
    # linear transition tracks the target's own scale (returns_rms runs ~2.9 on
    # InvertedPendulum, ~15 on Hopper) instead of a constant that is 5x off on one
    # of them. Bounds the per-sample gradient, which is what lets a few
    # large-residual chunks stop owning critic_grad_norm_preclip.
    value_huber_delta: float = 1.0
    # rsl-rl's default. These were 200.0, which never bound a joint norm of up
    # to 400. A global rescale is near-inert under AdamW, so the threshold does
    # not set the step size; it decides whether a transient spike is allowed to
    # pollute the second-moment estimate. Measured: a 1/170 rescale moved the
    # trajectory 1.2e-04 per parameter, one 1e4 spike moved it 9.0e-02.
    max_grad_norm: float = 1.0
    # Clipped separately from the actor: the critic's norm is two orders larger.
    critic_max_grad_norm: float | None = 1.0

    # qc and TD-JEPA use 0.005 but refresh over offline batches. This loop is
    # online and 0.005 left the teacher tens of rollouts behind; 0.05 is a
    # ~2.5-rollout time constant, damped within a rollout and still tracking.
    critic_ema_decay: float = 0.05
    # gamma ** action_horizon puts 0.85 of the value target on the critic's own
    # output, the regime a target network is meant to hold fixed.
    use_target_critic: bool = True
    # Each drift metric is a full critic forward over 2 * n_chunks windows, and
    # the loop runs it per grad-accumulation boundary. That is real compute for
    # numbers only the tracker reads, so the loop also skips it whenever no
    # tracker is enabled, and the flag is the manual override for a
    # report_to=none run that still wants the numbers.
    log_drift_diagnostics: bool = True

    num_kept_tokens: int = 16  # DCT coefficients the flow sees; inert when use_dct_transform=False

    policy_lr: float = 1e-3
    critic_lr: float = 1e-3  # legacy split-checkpoint compatibility
    optimizer_weight_decay: float = 1e-4

    # No separate warm-up. Stats initialize during the first rollout.
    # Groups consecutive minibatches into one optimizer step; it does NOT
    # change peak memory. Every microbatch is batch_size chunks x
    # num_mc_samples MC draws either way, and the DiT activations from one
    # microbatch are the peak (1.6 GiB saved for backward at the defaults,
    # batch_size=128 x num_mc_samples=16). The memory knobs are batch_size and
    # num_mc_samples; grad_accum_steps only trades optimizer steps for fewer.
    grad_accum_steps: int = 1

    # 12 episodes put the standard error of the mean near 60 return on
    # Humanoid, which is the same size as the regressions these knobs watch
    # for, and the max over hundreds of evals is biased well above the truth.
    eval_max_episodes: int = 50
    # Stop once the return has sat below this fraction of its own peak for
    # stop_patience evals in a row, so the run ends on the peak region rather
    # than on whatever the last rollout left behind.
    stop_return_frac: float = 0.7
    stop_patience: int = 5
    # Each completed window of stop_patience bad evals cuts the lrs by
    # lr_decay; the run stops once max_lr_decays of them have been spent.
    lr_decay: float = 0.5
    max_lr_decays: int = 3


    def __post_init__(self) -> None:
        super().__post_init__()
        self._resolve_n_action_steps()
        self._validate_geometry()
        self._validate_loss_params()
        self._validate_optimizer()
        self._validate_eval_schedule()
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

    def _validate_eval_schedule(self) -> None:
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
