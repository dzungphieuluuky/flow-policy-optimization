"""SJFPO hyperparameters, as a plain dataclass.

Mirror of ``spectral_joint_policy.configuration_sjfpo.SJFPOConfig`` **minus**
every LeRobot field, **plus** ``obs_dim`` and ``action_dim``.

Two differences from the torch original, both forced rather than chosen:

* **No LeRobot base class.** ``PreTrainedConfig`` carries ``repo_id``,
  ``push_to_hub``, ``input_features`` and friends, and importing lerobot would
  drag torch into the JAX process. Every field that exists only to serve the hub
  is gone, along with the properties that read ``input_features``.
* **Dims are plain ints.** torch reads them off ``PolicyFeature`` shapes
  (``modeling_sjfpo.py:205-206``); a dataclass cannot mirror that, and
  ``scripts/train.py`` already resolves both before dispatch.

The full validator suite of the torch config is **not** ported. Only the
invariants the learner itself depends on are checked here -- the ones whose
violation is a silent wrong answer rather than a clear error. CLI range errors
stay torch's until the trainer grows its own; see ``PLAN.md``.

Stays a dataclass: ``tracking.build_tracker`` calls ``dataclasses.asdict`` on it
(``tracking/__init__.py:39``), and the CLI parses it with draccus.
"""
from __future__ import annotations

import dataclasses

# Every LeRobot-only field, for the parity test that keeps this in sync.
LEROBOT_ONLY_FIELDS = frozenset(
    {
        "device",
        "input_features",
        "output_features",
        "license",
        "pretrained_path",
        "pretrained_revision",
        "private",
        "push_to_hub",
        "repo_id",
        "tags",
        "use_amp",
        "use_peft",
    }
)

# Properties that read input_features/output_features and so cannot exist here.
FEATURE_PROPERTIES = frozenset(
    {"action_feature", "env_state_feature", "image_features", "robot_state_feature", "type"}
)


@dataclasses.dataclass
class SJFPOConfig:
    """SJFPO hyperparameters. Field-for-field with the torch config otherwise."""

    # -- Dims, which torch reads off PolicyFeature shapes -------------------
    obs_dim: int = 0
    action_dim: int = 0

    # -- Observation / action geometry --------------------------------------
    report_to: str = "all"
    n_obs_steps: int = 16
    action_horizon: int = 16
    n_action_steps: int | None = 16
    action_perturb_std: float = 0.02
    hidden_dim: int = 256

    # -- Rollout and batching ----------------------------------------------
    num_envs: int = 64
    num_mc_samples: int = 16
    num_learning_epochs: int = 2
    chunks_per_update: int = 1024
    batch_size: int = 128
    grad_accum_steps: int = 1

    # -- Flow sampling ------------------------------------------------------
    num_euler_steps: int = 16
    num_inference_steps: int = 4
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    noise_scaling_mode: str = "white"
    noise_scaling_beta: float = 1.0
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    # -- Trust region and CFM loss -----------------------------------------
    gamma: float = 0.99
    clip_param: float = 0.05
    trust_region_mode: str = "aspo"
    cfm_loss_reduction: str = "mean"
    cfm_clamp: float = 200.0
    cfm_mask_unexecuted: bool = True
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 200.0
    log_ratio_cap: float = 10.0

    # -- Critic and value targets ------------------------------------------
    num_critics: int = 2
    normalize_advantage: bool = True
    value_loss_coef: float = 1.0
    value_huber_delta: float = 1.0
    use_clipped_value_loss: bool = True
    use_target_critic: bool = True
    critic_ema_decay: float = 0.05

    # -- Spectral chunk encoding -------------------------------------------
    use_dct_transform: bool = True
    num_kept_tokens: int = 4

    # -- DiT backbone -------------------------------------------------------
    dit_depth: int = 6
    dit_heads: int = 4
    dit_mlp_ratio: float = 4.0
    time_embed_dim: int = 128
    timestep_scale: float = 10000.0

    # -- Optimizer ----------------------------------------------------------
    policy_lr: float = 1e-5
    critic_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    critic_max_grad_norm: float | None = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    # -- Eval and early stopping --------------------------------------------
    eval_max_episodes: int = 50
    stop_return_frac: float = 0.7
    stop_patience: int = 5
    lr_decay: float = 0.5
    max_lr_decays: int = 3
    log_drift_diagnostics: bool = True

    def __post_init__(self) -> None:
        # n_action_steps is optional in the CLI, so it resolves here rather than
        # in the checks below.
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        self._check()

    def _check(self) -> None:
        """Only the invariants whose violation is a silent wrong answer.

        torch's config validates every range at construction. These four are the
        ones that quietly compute something else instead of raising: a token axis
        longer than the horizon, a minibatch bigger than the rollout, a hidden
        width the attention heads cannot split, and a discounted chunk of zero
        length.
        """
        if not 1 <= self.num_kept_tokens <= self.action_horizon:
            raise ValueError(
                f"num_kept_tokens={self.num_kept_tokens} must satisfy "
                f"1 <= K <= action_horizon={self.action_horizon}"
            )
        if self.batch_size > self.chunks_per_update:
            raise ValueError(
                f"batch_size={self.batch_size} must be <= "
                f"chunks_per_update={self.chunks_per_update}"
            )
        if self.hidden_dim % self.dit_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"dit_heads={self.dit_heads}"
            )
        if not 0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.min_timestep < self.max_timestep <= 1.0:
            raise ValueError(
                f"require 0 <= min_timestep < max_timestep <= 1, got "
                f"({self.min_timestep}, {self.max_timestep})"
            )
        if self.obs_dim < 1 or self.action_dim < 1:
            raise ValueError(
                f"obs_dim={self.obs_dim} and action_dim={self.action_dim} must be "
                "set; the JAX config cannot read them off a PolicyFeature"
            )

    @property
    def num_mini_batches(self) -> int:
        """Minibatches per epoch: ``chunks_per_update // batch_size``, at least one.

        Computed on access so it cannot go stale if the two are reassigned later.
        """
        return max(1, self.chunks_per_update // self.batch_size)

    @property
    def num_flow_tokens(self) -> int:
        """Token count the flow model operates on.

        ``num_kept_tokens`` with the DCT on, otherwise the full horizon. Without
        the DCT, ``num_kept_tokens`` is inert rather than an error: the token
        axis is time, so there is no frequency count to keep.
        """
        return self.num_kept_tokens if self.use_dct_transform else self.action_horizon

    @property
    def observation_delta_indices(self) -> list[int]:
        """The ``n_obs_steps`` past-frame offsets ending at the current step."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """One offset per step of a full action chunk."""
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        """None: rollouts supply executed-step rewards directly."""
        return None
