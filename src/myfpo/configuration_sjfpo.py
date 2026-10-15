"""Single home for every hyperparameter (LeRobot ``PreTrainedConfig``)."""

from __future__ import annotations


from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig

__all__ = ["SJFPOConfig"]


@PreTrainedConfig.register_subclass("sjfpo")
@dataclass
class SJFPOConfig(PreTrainedConfig):
    """SJFPO hyperparameters.

    ``n_action_steps`` is the executed prefix, default 4, half of
    ``action_horizon``. Set it to ``None`` and ``__post_init__`` fills in
    ``action_horizon`` -- that is the open-loop no-correction case.
    ``num_inference_steps`` is for zero-sampling, ``num_euler_steps`` for a
    sampled-noise start.
    """

    report_to: str = "all"
    n_obs_steps: int = 16
    action_horizon: int = 8
    # Execute half the chunk. Then replan. At action_horizon the chunk is
    # planned and executed fully open-loop. No closed-loop correction runs for 8
    # env steps. That is the single largest default weakness here.
    # The value is a default, not a settled result. Confirm it by sweeping
    # {2, 4, 8}.
    n_action_steps: int = 4
    action_perturb_std: float = 0.02
    hidden_dim: int = 128

    num_envs: int = 64
    num_mc_samples: int = 16
    num_learning_epochs: int = 10
    chunks_per_update: int = 1024
    batch_size: int = 128
    grad_accum_steps: int = 1

    num_euler_steps: int = 10
    num_inference_steps: int = 10
    min_timestep: float = 0.005
    max_timestep: float = 0.995
    timestep_sampling_mode: str = "fpo_beta"
    noise_sampling_mode: str = "normal"
    timestep_inverse_cdf_beta: float = 1.0
    timestep_beta_alpha: float = 2.0
    timestep_beta_beta: float = 2.0

    gamma: float = 0.99
    # 0.95 is the standard GAE trace. The chunk-level recursion runs above 0.
    # It gives an effective horizon of 1/(1 - gamma^h*lambda) chunks. Here that
    # is about 11 chunks, or 46 env steps. A bare QC backup gives 1 chunk, or 4
    # env steps.
    #
    # 0.0 is the opt-out. It reproduces QC exactly, with one backup per chunk.
    # It is not the default. This repo describes the chunk macro-MDP. Lambda
    # applies once per chunk, and that is the GAE of this macro-MDP. So the
    # formulation implies 0.95.
    gae_lambda: float = 0.95
    # These two knobs say which axes get folded into the ratio before exp.
    # FPO++ exposes them as two settings and not as an enum
    # (finetune_online_rl.py:92-95). The token axis comes first. Then the
    # MC-sample axis.
    #
    #   False, 1  gives one ratio per (chunk, MC pair, token).
    #   True,  1  gives one ratio per (chunk, MC pair). The tokens are averaged
    #             with a valid mask.
    #   True, -1  gives one ratio per chunk. That is FPO Eq 9. All the MC pairs
    #             sit in one group.
    #
    # FPO++'s released default is False/1.
    do_average_cfm_loss_in_chunk: bool = False
    # -1 means "one group holding every MC sample". Otherwise the sample count
    # must divide evenly by it.
    cfm_average_group_size: int = 1
    clip_param: float = 0.01
    # The SPO quadratic has its own width in the manipulation loop of FPO++.
    # The fields spo_clip_coef and clip_coef are separate. Both are 0.01
    # (finetune_online_rl.py:117-118). The Isaac Lab variant uses clip_param for
    # both. The behaviour here is the behaviour of manipulation.
    spo_clip_param: float = 0.01
    trust_region_mode: str = "aspo"
    # The sample-level reduction: "mean" (default), "sum", or "sqrt". The
    # default "mean" is not the "sum" of Eq. 8 in FPO. Under "sum" the function
    # compute_cfm_loss deliberately does not renormalise for padding, so a short
    # chunk's loss scales with its executed length.
    #
    # It reaches the policy gradient only when do_average_cfm_loss_in_chunk is
    # True. The default per-token path always averages over the action
    # dimension, so this flag does not change the loss there. It still feeds the
    # sample-level diagnostics and the above-clamp gate.
    # Both baselines pin "sum". See baselines.py.
    cfm_loss_reduction: str = "mean"
    # What the CFM loss compares against. This is the ``cfm_loss_mode`` of FPO++
    # (flow_model.py:592-631). Its "u" is "velocity" here. Its "x0" is "action".
    # Its "eps" is "noise".
    #
    # The reference names x0 the action endpoint and x1 the noise endpoint. The
    # time of this repo runs the other way, so t=1 is the action. The names below
    # name the endpoint and not the index.
    cfm_loss_target: str = "velocity"  # velocity | action | noise
    # The per-token weight of the CFM loss. FPO++ ships "constant" plus two
    # linear ramps (flow_model.py:648-661). It names them for their endpoints.
    # The weight is 1.0 at its t=0, the action end. The weight falls to 0.1 or
    # 0.01 at t=1, the noise end. The t=1 of this repo is the action end. So the
    # same schedules read here with the endpoints named
    # accordingly -- see :func:`myfpo.fpo_core.cfm_loss_weight`.
    cfm_loss_weight_from_t: str = "constant"
    cfm_clamp: float = 20.0
    cfm_loss_clamp_negative_advantages: bool = True
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    log_ratio_cap: float = 10.0

    num_critics: int = 2
    # A chunk is planned at action_horizon. Only `lengths` steps of it run. The
    # loop breaks on terminate or on truncate. So the tail of chunk_actions holds
    # a counterfactual plan. The critic used to see that tail. The critic had no
    # way to know that the tail never ran. Its regression target covered only
    # the executed
    # prefix. Set False to restore the old, tail-blind critic exactly.
    critic_condition_on_exec_length: bool = True
    # The chunk-level critic of FPO++ consumes the output of the observation
    # encoder of the actor itself (finetune_online_rl.py:1177). With this flag
    # on, the critic takes hidden_dim instead of the raw obs window. The encoder
    # then serves both networks.
    critic_share_encoder: bool = False
    # The critic was welded to the width of the actor. That choice is wrong
    # across the env ladder. The value obs_dim runs from 3 on Pendulum to 348 on
    # Humanoid-v5. So the first layer of the critic sees 56 or 5704 inputs. It
    # squeezes both counts into hidden_dim. The value None means "reuse
    # hidden_dim". It preserves the old construction exactly.
    critic_hidden_dim: int | None = None
    critic_num_layers: int = 2
    normalize_advantage: bool = True
    # (negative_clamp, positive_clamp), FPO++ rl_cfg.advantage_clamp.
    advantage_clamp: tuple[float, float] = (100.0, 100.0)
    value_loss_coef: float = 1.0

    # FPO++ FlowMatchingConfig.cfm_loss_use_huber. Elementwise, before the
    # reduction: the quadratic branch has no 0.5 factor, so it equals MSE for
    # |err| <= delta and only the tail becomes linear.
    cfm_loss_use_huber: bool = True
    cfm_loss_huber_delta: float = 0.5
    # Clip the actor's transported output (velocity here) to +/- this. None
    # disables. Applied on the training path and in the Euler sampler alike.
    transported_clip_value: float | None = None

    compile: bool = True

    # "dit" is the actor of this repo. "mlp" is the chunk-flattened MLP of FPO++
    # (flow_net_mlp.py). Both have the same forward signature. So the rest of the
    # code does not change.
    actor_arch: str = "dit"
    mlp_hidden_dims: tuple[int, ...] = (512, 512, 512)
    dit_depth: int = 3
    dit_heads: int = 2
    dit_mlp_ratio: float = 2.0
    time_embed_dim: int = 32

    # FPO++'s manipulation rates, tuned for fine-tuning a pretrained flow-BC
    # policy (finetune_online_rl.py:124-125). From scratch on MuJoCo they are
    # small. The sweep in the notebook is the way out.
    policy_lr: float = 1e-5
    critic_lr: float = 1e-4
    # Linear warmup on each param group, in optimizer steps. FPO++ defaults
    # are 5 and 1 (finetune_online_rl.py).
    actor_lr_warmup_steps: int = 5
    critic_lr_warmup_steps: int = 1
    optimizer_weight_decay: float = 1e-4
    # These are the betas of the manipulation actor (finetune_online_rl.py:128).
    # Isaac Lab ships (0.9, 0.999). The old value here, (0.999, 0.999), matched
    # neither release.
    adam_betas: tuple[float, float] = (0.9, 0.99)
    # One clip over every parameter, actor and critic together.
    max_grad_norm: float = 1.0
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500

    eval_max_episodes: int = 50
    stop_return_frac: float = 0.7
    stop_patience: int = 5
    lr_decay: float = 0.5
    max_lr_decays: int = 3

    log_drift_diagnostics: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps is None:
            self.n_action_steps = self.action_horizon
        self.validate()

    def validate(self) -> None:
        """Run every check inline. The function calls no other check method.

        This function does not check the feature shapes. The fields
        ``input_features`` and ``output_features`` are empty on a bare config.
        ``build_policy`` fills them in at run time. It takes the dimensions from
        the env registry. Then ``train.py`` compares those dimensions with the
        live observation and action spaces.

        The requirement is enforced by ``validate_features`` instead. That
        function runs from ``SJFPOPolicy.__init__``, after the features are
        attached.
        """
        # Geometry / shape
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps={self.n_obs_steps} must be >= 1")
        if self.report_to not in ("wandb", "tensorboard", "all", "none"):
            raise ValueError(
                f"report_to={self.report_to!r} must be wandb, tensorboard, all or none"
            )
        if self.num_critics < 1:
            raise ValueError(f"num_critics={self.num_critics} must be >= 1")
        if self.critic_hidden_dim is not None and self.critic_hidden_dim < 1:
            raise ValueError(
                f"critic_hidden_dim={self.critic_hidden_dim} must be >= 1"
            )
        if self.critic_num_layers < 1:
            raise ValueError(f"critic_num_layers={self.critic_num_layers} must be >= 1")
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
        if self.actor_arch not in {"dit", "mlp"}:
            raise ValueError("actor_arch must be 'dit' or 'mlp'")
        if not self.mlp_hidden_dims or any(d < 1 for d in self.mlp_hidden_dims):
            raise ValueError("mlp_hidden_dims must be a non-empty tuple of >= 1")
        if self.dit_depth < 1:
            raise ValueError(f"dit_depth={self.dit_depth} must be >= 1")
        if self.dit_heads < 1:
            raise ValueError(f"dit_heads={self.dit_heads} must be >= 1")
        if self.dit_mlp_ratio <= 0:
            raise ValueError(f"dit_mlp_ratio={self.dit_mlp_ratio} must be > 0")
        if self.time_embed_dim < 1:
            raise ValueError(f"time_embed_dim={self.time_embed_dim} must be >= 1")
        if self.cfm_loss_huber_delta <= 0:
            raise ValueError(
                f"cfm_loss_huber_delta={self.cfm_loss_huber_delta} must be > 0"
            )
        if (
            self.transported_clip_value is not None
            and self.transported_clip_value <= 0
        ):
            raise ValueError(
                f"transported_clip_value={self.transported_clip_value} must be > 0"
            )
        if len(self.advantage_clamp) != 2:
            raise ValueError(
                f"advantage_clamp={self.advantage_clamp} must be "
                "(negative_clamp, positive_clamp)"
            )
        if any(c <= 0 for c in self.advantage_clamp):
            raise ValueError(
                f"advantage_clamp={self.advantage_clamp} entries must be > 0"
            )
        # Eval schedule
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
        # Loss params
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must satisfy 0 < gamma <= 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda={self.gae_lambda} must be in [0, 1]")
        if not 0.0 < self.spo_clip_param < 1.0:
            raise ValueError(
                f"spo_clip_param={self.spo_clip_param} must satisfy 0 < eps < 1"
            )
        if not 0.0 < self.clip_param < 1.0:
            raise ValueError(
                f"clip_param={self.clip_param} must satisfy 0 < eps < 1"
            )
        if self.trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError("trust_region_mode must be 'ppo', 'spo', or 'aspo'")
        if self.cfm_average_group_size == 0 or self.cfm_average_group_size < -1:
            raise ValueError(
                f"cfm_average_group_size={self.cfm_average_group_size} must be -1 "
                "(one group over every MC sample) or a positive divisor of "
                f"num_mc_samples={self.num_mc_samples}"
            )
        if (
            self.cfm_average_group_size != -1
            and self.num_mc_samples % self.cfm_average_group_size != 0
        ):
            raise ValueError(
                f"num_mc_samples={self.num_mc_samples} must divide evenly by "
                f"cfm_average_group_size={self.cfm_average_group_size}"
            )
        if self.cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError("cfm_loss_reduction must be 'mean', 'sum', or 'sqrt'")
        if self.cfm_loss_target not in {"velocity", "action", "noise"}:
            raise ValueError(
                "cfm_loss_target must be 'velocity' (FPO++'s cfm_loss_mode='u'), "
                "'action' (='x0'), or 'noise' (='eps')"
            )
        if self.cfm_loss_weight_from_t not in {
            "constant",
            "linear_1_to_0.1",
            "linear_1_to_0.01",
        }:
            raise ValueError(
                "cfm_loss_weight_from_t must be 'constant', 'linear_1_to_0.1', "
                "or 'linear_1_to_0.01'"
            )
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
        if self.timestep_sampling_mode not in {"uniform", "normal", "beta", "fpo_beta"}:
            raise ValueError(
                "timestep_sampling_mode must be 'uniform', 'normal', 'beta', or 'fpo_beta'"
            )
        if self.noise_sampling_mode not in {"normal", "uniform"}:
            raise ValueError("noise_sampling_mode must be 'normal' or 'uniform'")
        if self.timestep_inverse_cdf_beta <= 0:
            raise ValueError("timestep_inverse_cdf_beta must be > 0")
        if self.timestep_beta_alpha <= 0 or self.timestep_beta_beta <= 0:
            raise ValueError("timestep beta alpha/beta must be > 0")
        if self.value_loss_coef < 0:
            raise ValueError(f"value_loss_coef={self.value_loss_coef} must be >= 0")
        # Optimizer
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
        if self.actor_lr_warmup_steps < 0:
            raise ValueError(
                f"actor_lr_warmup_steps={self.actor_lr_warmup_steps} must be >= 0"
            )
        if self.critic_lr_warmup_steps < 0:
            raise ValueError(
                f"critic_lr_warmup_steps={self.critic_lr_warmup_steps} must be >= 0"
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
        # Flags
        if self.num_euler_steps < 1:
            raise ValueError(f"num_euler_steps={self.num_euler_steps} must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} must be >= 1"
            )

    def validate_features(self) -> None:
        """Require the run-time features. Called from SJFPOPolicy.__init__."""
        if self.robot_state_feature is None:
            raise ValueError("SJFPO requires 'observation.state' in input_features.")
        if self.action_feature is None:
            raise ValueError("SJFPO requires 'action' in output_features.")

    def get_optimizer_preset(self, lr: float | None = None) -> "AdamWConfig":
        from lerobot.optim import AdamWConfig
        kwargs = dict(
            lr=self.policy_lr if lr is None else float(lr),
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.max_grad_norm,
        )
        return AdamWConfig(**kwargs)

    def get_scheduler_preset(self):
        return None

    @property
    def num_mini_batches(self) -> int:
        return max(1, self.chunks_per_update // self.batch_size)

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

