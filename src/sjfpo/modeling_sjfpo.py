"""SJFPO policy: flow head + state conditioner + critic.

Batch contract: ``[B, T, obs]`` states, ``[B, H, action_dim]`` action chunks,
frozen GAE advantages/returns from the rollout.

Performance notes
-----------------
* Diagnostics returned by ``forward`` are batched into a single device→host
  transfer (one ``.tolist()``) instead of ~20 ``float()`` syncs per call, and
  the whole diagnostic pass is gated on ``compute_info=True``.
* ``ratio_token`` is computed once per forward, not once per branch.
* ``snapshot_behavior_policy`` copies weights in-place into the frozen
  snapshot instead of ``copy.deepcopy``-ing it.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import dataclasses

import einx
import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import RolloutBuffer, collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import (
    velocity_to_action_steps,
    extract_dct_coeffs,
    velocity_to_action_steps,
)
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_monte_carlo_sampling,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo.gae import compute_chunk_advantages
from sjfpo.networks import (
    Critic,
    RunningMeanStd,
    SJFPOFlowModel,
    StateConditioner,
)
from sjfpo.processor_sjfpo import make_sjfpo_pre_post_processors
from sjfpo.utils import (
    autocast_context,
    resolve_storage_dtype,
    to_device_async,
    to_tensor_1d,
)

class SJFPOAborted(RuntimeError):
    """Non-finite training loss (message names the offending term)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _gae_diagnostics(
    env_id: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    advantages: torch.Tensor,
    raw_returns: torch.Tensor,
    budget_truncated: torch.Tensor | None = None,
) -> dict[str, float]:
    """Trace-liveness and scale probes for the chunk-level backup.

    ``linked_frac`` is the one that matters: it is the share of chunk pairs the
    GAE recursion actually joined. Chunks are stored in wavefront order, so an
    adjacency-based mask silently joins nothing and every advantage collapses
    to a bare 1-step delta, with no error raised. A linked_frac of 0 means the
    trace is dead regardless of what gae_lambda is set to.
    """
    n = env_id.numel()
    linked = 0
    if n > 1:
        order = torch.argsort(env_id.to(torch.long), stable=True)
        e = env_id.to(torch.long)[order]
        t = terminated.to(torch.bool)[order]
        u = truncated.to(torch.bool)[order]
        linked = int(((e[1:] == e[:-1]) & ~(t[:-1] | u[:-1])).sum())
    return {
        "linked_frac": linked / max(n - 1, 1),
        "advantage_std": float(advantages.float().std()) if n > 1 else 0.0,
        "return_rms": float(raw_returns.float().pow(2).mean().sqrt()) if n else 0.0,
        # A = R - V, so Var(A) <= Var(R) with equality only when V is 0. This is
        # the direct check that the baseline reached the advantage: it sits at 1.0
        # when the critic is inert and falls toward 1 - R^2 as the critic improves.
        "advantage_std_fraction": (
            float(advantages.float().std() / raw_returns.float().std().clamp_min(1e-8))
            if n > 1
            else 0.0
        ),
        # The budget cut is reported on its own so it cannot be confused with the
        # env's time-limit truncation, which is the flag GAE reads. Not
        # truncated_chunk_frac: that one is derived from action_is_pad in
        # _compute_info and means "the chunk left action rows unexecuted".
        "budget_truncated_chunk_frac": (
            float(budget_truncated.to(torch.bool).float().mean())
            if n and budget_truncated is not None
            else 0.0
        ),
    }


def _sample_spectral_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw Monte-Carlo noise matching the configured token layout.

    Returns noise of shape ``[n, num_mc_samples, tokens, action_dim]``.
    """
    shape = (n, cfg.num_mc_samples, cfg.num_flow_tokens, action_dim)
    return sample_noise(shape, cfg, device, None)


def build_optimizer(policy: SJFPOPolicy) -> torch.optim.Optimizer:
    """One AdamW over actor and critic, kept in separate param groups.

    The groups stay split per role so grad clipping can be applied per role: the
    critic's value gradient is orders of magnitude larger than the policy's, and
    one shared clip would let it own the actor's step. rsl-rl uses this same
    single-optimizer, per-role-group layout.
    """
    kwargs = dataclasses.asdict(policy.config.get_optimizer_preset())
    kwargs.pop("grad_clip_norm", None)  # AdamW preset field; clipping is per role
    if next(policy.parameters()).is_cuda:
        kwargs["fused"] = True
    groups = [
        *policy.get_actor_optim_params()["params"],
        *policy.get_critic_optim_params()["params"],
    ]
    return torch.optim.AdamW(groups, **kwargs)


def _mc_x_t(
    flatten_mc_timestep: torch.Tensor,
    flatten_mc_noise: torch.Tensor,
    flatten_mc_action: torch.Tensor,
    flow_shape: tuple[int, ...],
) -> torch.Tensor:
    """Interpolate between noise and action to form the flow point x_t.

    Returns the flow point, of shape ``[-1, *flow_shape]``.
    """
    t = flatten_mc_timestep.expand(-1, -1, flow_shape[-1])
    noise = flatten_mc_noise.reshape(-1, *flow_shape)
    action = flatten_mc_action.reshape(-1, *flow_shape)
    return (1.0 - t) * noise + t * action


class SJFPOPolicy(PreTrainedPolicy):
    config_class = SJFPOConfig
    name = "sjfpo"

    def __init__(
        self, config: SJFPOConfig, dataset_stats: dict[str, Any] | None = None
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        obs_dim = int(config.robot_state_feature.shape[0])
        action_dim = int(config.action_feature.shape[0])

        self.actor = SJFPOFlowModel(
            action_dim,
            config.action_horizon,
            config.hidden_dim,
            config.dit_depth,
            config.dit_heads,
            config.dit_mlp_ratio,
            config.time_embed_dim,
            config.timestep_scale,
        )
        self.conditioner = StateConditioner(obs_dim, config.hidden_dim)
        self.critic = Critic(
            obs_dim, config.hidden_dim, config.n_obs_steps, config.num_critics
        )
        self.obs_normalizer = RunningMeanStd(obs_dim)

        self.register_buffer(
            "action_low", torch.full((action_dim,), -1.0), persistent=False
        )
        self.register_buffer(
            "action_high", torch.full((action_dim,), 1.0), persistent=False
        )


        self.old_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.old_conditioner = (
            copy.deepcopy(self.conditioner).requires_grad_(False).eval()
        )
        self._ema_shadow: dict[str, torch.Tensor] = {}
        self._ema_initialized = False
        self._rollout_buffer: RolloutBuffer | None = None
        self._device_rollout_buffer: RolloutBuffer | None = None
        self._rollout_h2d_event: torch.cuda.Event | None = None

        self.cfm_above_clamp_frac = torch.zeros(())

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        self.preprocessor, self.postprocessor = make_sjfpo_pre_post_processors(self)
        del dataset_stats


    def train(self, mode: bool = True):
        """Set the module mode, keeping the frozen snapshot in eval.

        Returns this policy, so the call chains.
        """
        super().train(mode)
        self.old_actor.eval()
        self.old_conditioner.eval()
        return self

    def reset(self):
        """Clear the cached observation and action buffers between episodes."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    @torch.inference_mode()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        """Update the live running statistics from raw post-step observations.

        Raises ``ValueError`` if the shape is not ``[N, obs_dim]``.
        """
        if (
            observations.ndim != 2
            or observations.shape[-1] != self.obs_normalizer.mean.numel()
        ):
            raise ValueError(
                "observations must have shape [N, observation_dim], got "
                f"{tuple(observations.shape)}"
            )
        self.obs_normalizer.update(observations)

    def set_action_bounds(
        self,
        low: np.ndarray | torch.Tensor,
        high: np.ndarray | torch.Tensor,
    ) -> None:
        """Replace the env action bounds used by the postprocessor.

        Raises ``ValueError`` if the bounds disagree in shape or fail the
        ``high > low`` check.
        """
        storage_dtype = resolve_storage_dtype(self.action_low.device)
        low_t = to_tensor_1d(
            low, dtype=storage_dtype, device=self.action_low.device
        ).reshape(-1)
        high_t = to_tensor_1d(
            high, dtype=storage_dtype, device=self.action_high.device
        ).reshape(-1)
        if (
            low_t.shape != self.action_low.shape
            or high_t.shape != self.action_high.shape
        ):
            raise ValueError(
                f"action bounds must match action_dim={self.action_low.numel()}: "
                f"low={tuple(low_t.shape)}, high={tuple(high_t.shape)}"
            )
        if not bool((high_t > low_t).all()):
            raise ValueError("action_high must be elementwise >= action_low")

        self.action_low.copy_(low_t)
        self.action_high.copy_(high_t)
        self.postprocessor.set_action_bounds(self.action_low, self.action_high)

    def snapshot_behavior_policy(self) -> None:
        """Refresh the frozen π_old snapshot in place.

        Copies weights into the existing modules rather than
        ``copy.deepcopy``-ing the live ones over them.
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_conditioner.load_state_dict(self.conditioner.state_dict())

        self.old_actor.requires_grad_(False)
        self.old_actor.eval()
        self.old_conditioner.requires_grad_(False)
        self.old_conditioner.eval()

    @torch.inference_mode()
    def update_ema(self, global_step: int) -> None:
        """Update the FPO++ actor EMA after an optimizer update.

        The shadow is seeded on the first call and hard-copied at the end of warmup,
        so only properly warmed weights enter the average. ``global_step`` is compared
        against ``ema_warmup_steps``.
        """
        decay = self.config.ema_decay
        if decay <= 0:
            return
        actor = self.actor
        current = {
            name: param
            for name, param in actor.named_parameters()
            if param.requires_grad
        }
        if not self._ema_shadow:
            self._ema_shadow = {
                name: param.detach().clone() for name, param in current.items()
            }
        if not self._ema_initialized and global_step >= self.config.ema_warmup_steps:
            # >= and gated on _ema_initialized, not ==. global_step advances by a
            # whole rollout's executed steps, so == ema_warmup_steps was never
            # observed with the defaults (500 vs a ~4096 step delta) and the
            # shadow was never initialised, leaving ema_ready() false forever.
            # The gate keeps this a one-shot: without it every later call would
            # re-hard-copy and discard the running average.
            for name, shadow in self._ema_shadow.items():
                shadow.copy_(current[name].detach())
            self._ema_initialized = True
            return
        if global_step > self.config.ema_warmup_steps:
            for name, shadow in self._ema_shadow.items():
                shadow.mul_(decay).add_(current[name].detach(), alpha=1.0 - decay)

    def ema_ready(self, global_step: int) -> bool:
        """Report whether the EMA shadow may now be used for evaluation.
        """
        return (
            self.config.ema_decay > 0
            and self._ema_initialized
            and global_step > self.config.ema_warmup_steps
        )

    def ema_state_dict(self) -> dict:
        """Export the EMA shadow for checkpointing.

        Returns:
            Dict with the decay, the initialisation flag, and the shadow
            parameters.
        """
        return {
            "decay": self.config.ema_decay,
            "initialized": self._ema_initialized,
            "shadow_params": self._ema_shadow,
        }

    def load_ema_state_dict(self, state: dict) -> None:
        """Restore an EMA shadow exported by :meth:`ema_state_dict`.
        """
        self._ema_shadow = {
            name: value.detach().clone()
            for name, value in state["shadow_params"].items()
        }
        self._ema_initialized = bool(state.get("initialized", True))

    def get_actor_optim_params(self) -> dict:
        """Build the actor optimizer's param groups.

        adaLN modulation and the output layer are excluded from weight decay,
        and the conditioner shares the policy learning rate.

        Returns:
            Optimizer param-group dict.
        """
        decay, no_decay = [], []
        for name, p in self.actor.named_parameters():
            if "adaLN" in name or name.startswith("out."):
                no_decay.append(p)
            else:
                decay.append(p)
        policy_lr = self.config.policy_lr
        return {
            "params": [
                {"params": decay, "lr": policy_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": policy_lr},
                {"params": list(self.conditioner.parameters()), "lr": policy_lr},
            ]
        }

    def get_critic_optim_params(self) -> dict:
        """Build the critic optimizer's param groups at the critic learning rate.

        Returns:
            Optimizer param-group dict.
        """
        return {
            "params": [
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.config.critic_lr,
                }
            ]
        }

    def get_optim_params(self) -> dict:
        """Build a single FPO++-style group over all trainable weights.

        Implements the abstract base-class method. The training path does not
        use it: :func:`build_optimizer` composes the per-role groups from
        :meth:`get_actor_optim_params` and :meth:`get_critic_optim_params` so
        grad clipping stays separable. Kept because the base class requires it.

        Returns:
            Optimizer param-group dict.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        return {
            "params": [
                {
                    "params": params,
                    "lr": self.config.policy_lr,
                    "weight_decay": self.config.optimizer_weight_decay,
                }
            ]
        }


    @staticmethod
    def compute_target_velocity(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Compute the flow-matching target velocity.

        ``v* = a - eps``, broadcast over the Monte-Carlo draws, mapping
        ``[B, *flow]`` against ``[B, m, *flow]`` to ``[B, m, *flow]``.
        """
        return a.unsqueeze(1) - eps

    def compute_predicted_velocity(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        condition: torch.Tensor,
        *,
        old: bool = False,
    ) -> torch.Tensor:
        """Run the actor on a flow point against a prepared conditioning.

        ``condition`` is the state conditioner's output, already broadcast over the
        Monte-Carlo draws, so both the frozen and live paths share the one
        forward-conditioning pass their caller already performed. ``old`` selects the
        frozen behavior snapshot. Returns velocities in the actor's own dtype.
        """
        net = self.old_actor if old else self.actor
        return net(x_t, tau, condition)


    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce observations to the ``[B, T, obs]`` layout the policy expects.

        Returns a storage-dtype tensor of shape ``[B, T, obs]``. Raises
        ``ValueError`` if the rank is not 1, 2, or 3.
        """
        states = to_tensor(states)
        states = states.to(dtype=resolve_storage_dtype(states.device))
        if states.dim() == 3:
            return states
        if states.dim() == 2:
            if states.shape[0] == self.config.n_obs_steps:
                return einx.id("t d -> 1 t d", states)
            return einx.id("d -> 1 t d", states, t=self.config.n_obs_steps)
        raise ValueError(
            f"observation.state must be [B, T, obs], [T, obs] or [B, obs]; "
            f"got {tuple(states.shape)}"
        )


    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Predict a full action chunk.

        Action perturbation is applied before postprocessing, matching FPO++, so the
        noise lands in the policy's normalized action space. ``zero_sampling=True``
        integrates from eps=0 over ``num_inference_steps``; ``zero_sampling=False``
        integrates from sampled noise over ``num_euler_steps``, honouring
        ``noise_sampling_mode`` and ``noise_scaling_mode``. Returns a chunk of shape
        ``[B, H, action_dim]``.
        """
        zero_sampling = kwargs.get("zero_sampling", True)
        cfg = self.config
        device = next(self.actor.parameters()).device

        states = self.standardize_obs_tensor(batch[OBS_STATE])
        states = to_device_async(states, device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        num_spectral_tokens_kept = cfg.num_flow_tokens
        action_dim = int(self.config.action_feature.shape[0])
        actor = self.actor

        with torch.inference_mode(), autocast_context(device):
            condition = self.conditioner(states)
            shape = (states.shape[0], num_spectral_tokens_kept, action_dim)
            # zero_sampling starts the flow at eps=0, which is what FPO++ uses
            # for eval and for the reported metrics. Otherwise start from
            # sampled noise, drawn through sample_noise so noise_scaling_mode
            # applies; the old path called torch.randn here and skipped it,
            # so freq_weighted trained pink and rolled out white.
            if zero_sampling:
                eps = torch.zeros(*shape, device=device)
                num_steps = cfg.num_inference_steps
            else:
                eps = sample_noise(shape, cfg, device, None)
                num_steps = cfg.num_euler_steps
            x0 = sample_action(
                lambda x, t, c: actor(x, t, c),
                shape,
                condition,
                num_inference_steps=num_steps,
                eps=eps,
                device=device,
            ).float()

        if cfg.use_dct_transform:
            # velocity_to_action_steps reads the coefficients back on the half-open grid
            # t_j = j*H/H. linspace(0, 1, K) instead queries j = 0, 1.067, ..., K,
            # which time-warps every executed action by K/(K-1).
            scaled = velocity_to_action_steps(x0, cfg.action_horizon).clamp(-1.0, 1.0)
        else:
            scaled = x0[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)

        if self.training and cfg.action_perturb_std > 0:
            scaled = scaled + cfg.action_perturb_std * torch.randn_like(scaled)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the next action, receding through a cached chunk.

        A chunk is predicted once and its executed prefix is buffered, so consecutive
        calls do not re-run the actor. ``batch`` holds ``observation.state``, shaped
        ``[obs]`` for a single frame or ``[T, obs]`` for a full window; extra kwargs
        are forwarded to :meth:`predict_action_chunk`. Raises ``ValueError`` if the
        observation rank is neither 1 nor 2.
        """
        obs = to_tensor(batch[OBS_STATE])
        obs = obs.to(dtype=resolve_storage_dtype(obs.device))
        if obs.dim() == 1:
            if not self._frame_buffer:
                for _ in range(self.config.n_obs_steps):
                    self._frame_buffer.append(obs.detach().clone())
            else:
                self._frame_buffer.append(obs.detach())
            window = torch.stack(tuple(self._frame_buffer), dim=0)
        elif obs.dim() == 2:
            window = obs
        else:
            raise ValueError(
                f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}"
            )
        if not self._action_buffer:
            chunk = self.predict_action_chunk({OBS_STATE: window}, **kwargs)
            n_execute = max(1, min(self.config.n_action_steps, chunk.shape[1]))
            for i in range(n_execute - 1, -1, -1):
                self._action_buffer.append(chunk[0, i])
        return self._action_buffer.popleft()


    @staticmethod
    def get_token_count_for_cfm_loss(cfg: SJFPOConfig) -> int:
        """How many tokens the CFM loss reduces over.

        The action-step grid when the loss operates on rows, the frequencies when
        it does not. :meth:`prepare_cfm_loss_inputs` picks the layout and the
        ``token_mean`` reduction inherits its length, so the two must agree or
        the ratio compares different-width vectors.
        """
        on_action_grid = not cfg.use_dct_transform or cfg.cfm_mask_unexecuted
        return cfg.action_horizon if on_action_grid else cfg.num_flow_tokens

    def prepare_cfm_loss_inputs(
        self,
        cfg: SJFPOConfig,
        action_is_pad: torch.Tensor,
        *velocities: torch.Tensor,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        """Return the execution mask and the velocities it can be applied to.

        ``action_is_pad`` marks the action steps a chunk never ran, so it only
        indexes a velocity laid out over the ``action_horizon`` steps. DCT off,
        that is already the layout. DCT on, the velocity is
        ``num_flow_tokens`` frequency coefficients and every step maps to every
        coefficient, so expand the coefficients back onto the action steps first;
        the same mask then drops the tail, and a chunk that terminated early
        contributes no gradient from the steps it never ran instead of a
        wrong-signed one.

        Expanding each velocity is exact because the inverse DCT is linear, so
        ``grid(v_hat) - grid(v_star) == grid(v_hat - v_star)`` and the loss still
        measures the same residual. At ``num_kept_tokens == action_horizon`` the
        basis is orthonormal, so this is a rotation and the loss is unchanged.

        The mask and the layout are returned together on purpose: they are only
        consistent with each other, and at ``num_kept_tokens == action_horizon``
        the two spaces have the same shape, so returning a mask with unexpanded
        velocities applies it in the wrong space and fails silently.

        Both the ``l_old`` and ``l_new`` reductions must come through here, or the
        importance ratio would compare means over different positions.
        """
        if cfg.use_dct_transform and cfg.cfm_mask_unexecuted:
            steps = self.get_token_count_for_cfm_loss(cfg)
            return action_is_pad, tuple(
                velocity_to_action_steps(v, steps) for v in velocities
            )
        # DCT off, the velocities already are the action steps. DCT on with the
        # mask off, the mask would index the frequency axis, so drop it.
        mask = None if cfg.use_dct_transform else action_is_pad
        return mask, velocities

    def _behavior_cfm_losses(
        self,
        normalized_states: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the frozen behavior policy's CFM loss for one minibatch.

        The behavior weights are the snapshot taken at the start of the rollout, so
        this is the ``l_old`` term of the importance ratio. Both reductions are masked
        identically to the current-policy losses they get compared against, so the
        ratio never mixes means over different positions. Returns the ``sample`` and
        ``token_mean`` reductions.
        """
        cfg = self.config
        token_count = eps.shape[2]
        if cfg.use_dct_transform:
            spectral_action = extract_dct_coeffs(actions, M=token_count - 1)
        else:
            spectral_action = actions
        flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
            run_monte_carlo_sampling(spectral_action, eps, tau)
        )
        x_t = _mc_x_t(flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape)
        actor_tau = flatten_timestep.squeeze(-1)
        with torch.inference_mode(), autocast_context(actions.device):
            condition = self.old_conditioner(normalized_states)
            condition_flat = einx.id(
                "n d -> n m d", condition, m=num_mc
            ).reshape(num_mc * condition.shape[0], condition.shape[-1])
            v_old = self.compute_predicted_velocity(
                x_t, actor_tau, condition_flat, old=True
            ).float()
        with torch.autocast(device_type=actions.device.type, enabled=False):
            v_star = (spectral_action[:, None] - eps).float()
            v_old = v_old.reshape(eps.shape[0], num_mc, *flow_shape)
            loss_mask, (v_old, v_star) = self.prepare_cfm_loss_inputs(
                cfg, action_is_pad, v_old, v_star
            )
            return (
                compute_cfm_loss(v_old, v_star, cfg, "sample", loss_mask),
                compute_cfm_loss(v_old, v_star, cfg, "token_mean", loss_mask),
            )

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop action chunks and build a training batch.

        A chunk may execute only one step before termination, so the allocation-free
        upper bound on chunk count is one chunk per requested step. Each chunk is one
        macro-transition: the backup sums the rewards of its executed steps, discounts
        by ``gamma ** h``, and the GAE recursion runs between consecutive chunks of
        the same env (see :mod:`sjfpo.gae`). Advantages are normalised over chunks.

        The batch carries one raw post-step frame per executed step, kept separate from
        the overlapping observation windows so the training loop can fold them into the
        live normalizer without double counting, plus independent per-chunk/Monte-Carlo
        CFM draws that the policy loss reuses for both the old and new losses.

        Raises ``ValueError`` if ``total_steps`` is not positive, or if a collected
        transition is inconsistent.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        def propose(state: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(state, device)}, zero_sampling=not noisy
            )[0]
            return chunk.cpu()

        def propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(states, device)}, zero_sampling=not noisy
            )
            return chunk.cpu()

        if self._rollout_h2d_event is not None:
            self._rollout_h2d_event.synchronize()
            self._rollout_h2d_event = None

        num_envs = int(getattr(env, "num_envs", 1))
        action_dim = int(self.config.action_feature.shape[0])
        obs_dim = int(self.config.robot_state_feature.shape[0])
        max_chunks = max(1, total_steps)
        buffer_key = (
            max_chunks,
            total_steps,
            cfg.n_obs_steps,
            obs_dim,
            cfg.action_horizon,
            action_dim,
        )
        if (
            self._rollout_buffer is None
            # [:6] because capacity_key() also carries pin_memory, which is a
            # property of the device rather than of this call's shape. Comparing
            # the full tuple can never match, so the buffer was reallocated on
            # every rollout, defeating the reuse the docstring promises.
            or self._rollout_buffer.capacity_key()[:6] != buffer_key
        ):
            self._rollout_buffer = RolloutBuffer(
                max_chunks=max_chunks,
                max_steps=total_steps,
                obs_steps=cfg.n_obs_steps,
                obs_dim=obs_dim,
                action_horizon=cfg.action_horizon,
                action_dim=action_dim,
                pin_memory=device.type == "cuda",
            )
        buffer = self._rollout_buffer
        buffer.reset()

        if num_envs > 1:
            collect_chunks_vec(
                env,
                propose_batch,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                chunk_size=cfg.action_horizon,
                buffer=buffer,
            )
        else:
            collect_chunks(
                env,
                propose,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                chunk_size=cfg.action_horizon,
                buffer=buffer,
            )

        if device.type == "cuda":
            if (
                self._device_rollout_buffer is None
                or self._device_rollout_buffer.capacity_key()[:6] != buffer.capacity_key()[:6]
            ):
                self._device_rollout_buffer = RolloutBuffer(
                    max_chunks=max_chunks,
                    max_steps=total_steps,
                    obs_steps=cfg.n_obs_steps,
                    obs_dim=obs_dim,
                    action_horizon=cfg.action_horizon,
                    action_dim=action_dim,
                    device=device,
                )
            buffer.copy_to(self._device_rollout_buffer, non_blocking=True)
            self._rollout_h2d_event = torch.cuda.Event()
            self._rollout_h2d_event.record()
            active = self._device_rollout_buffer.active()
        else:
            active = buffer.active()
        n_chunks = buffer.n_chunks
        n_steps = buffer.n_steps
        num_spectral_tokens_kept = cfg.num_flow_tokens

        lengths = active["chunk_lengths"]
        chunk_starts = active["chunk_starts"]
        step_states = active["step_states"]
        step_next_states = active["step_next_states"]
        all_rewards = active["step_rewards"]
        actions = active["chunk_actions"]
        states = step_states.index_select(0, chunk_starts)

        # Only the chunk's first and post-chunk states are ever read by the
        # backup, so the critic and the preprocessor see 2 * n_chunks windows
        # rather than 2 * n_steps. The obs normalizer is updated separately
        # from raw frames via normalizer_observations.
        with torch.inference_mode():
            last_step = chunk_starts + lengths - 1
            next_states = step_next_states.index_select(0, last_step)
            all_chunk_states_n = self.preprocessor(
                {OBS_STATE: torch.cat([states, next_states], dim=0)}
            )[OBS_STATE]
            normalized_states = all_chunk_states_n[:n_chunks]
            with autocast_context(device):
                all_chunk_values = self.critic(all_chunk_states_n).float()
            # min over the ensemble, for the backup only. The value loss trains
            # every member separately; min has no justification inside a V loss
            # because there is no max operator to counteract, so it would only
            # bias V low and push advantages up.
            chunk_value = all_chunk_values[:, :n_chunks].amin(dim=0)
            chunk_next_value = all_chunk_values[:, n_chunks:].amin(dim=0)

        # One chunk is one macro-transition, so the backup is taken at chunk
        # granularity: reward summed over the chunk, discount gamma ** h.
        # One space throughout, as rsl-rl: the critic predicts raw returns, GAE
        # subtracts that from raw rewards, and the value target is the raw
        # return. No return normalizer, so V(s) is the true expected return and
        # the regression target does not drift as the return scale grows.
        # Per-batch advantage whitening below is the only normalization, and it
        # happens after the backup.
        raw_advantages, raw_returns = compute_chunk_advantages(
            all_rewards,
            chunk_value,
            chunk_next_value,
            chunk_starts,
            lengths,
            chunk_terminated=active["chunk_terminated"],
            chunk_truncated=active["chunk_truncated"],
            chunk_env_id=active["chunk_env_id"],
            chunk_size=cfg.action_horizon,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        raw_values = chunk_value.detach()
        normalized_advantages = raw_advantages
        if cfg.normalize_advantage and raw_advantages.numel() > 0:
            normalized_advantages = (raw_advantages - raw_advantages.mean()) / (
                raw_advantages.std() + 1e-8
            )
        gae_diagnostics = _gae_diagnostics(
            active["chunk_env_id"][:n_chunks],
            active["chunk_terminated"][:n_chunks],
            active["chunk_truncated"][:n_chunks],
            raw_advantages,
            raw_returns,
            active.get("chunk_budget_truncated")[:n_chunks]
            if "chunk_budget_truncated" in active
            else None,
        )
        raw_advantages = raw_advantages.to(storage_dtype)
        normalized_advantages = normalized_advantages.to(storage_dtype)
        # raw_returns and raw_values stay fp32. They are the critic's regression
        # target and its clip snapshot, and they are the only tensors here whose
        # magnitude is large rather than O(1): returns_rms runs to 35 and the
        # values reach the hundreds. bf16 carries 8 mantissa bits, so 1234.5
        # round-trips to 1232.0 and the target is quantised before the critic
        # ever sees it. They are 256 elements each, so fp32 costs nothing, and
        # the buffers holding step_states and chunk_actions are already fp32
        # (collector._OBS_DTYPE), which means storage_dtype never saved
        # meaningful memory in the first place.

        action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        tau = sample_timesteps(
            (n_chunks, cfg.num_mc_samples, num_spectral_tokens_kept),
            cfg,
            device,
            None,
        ).unsqueeze(-1)
        eps = _sample_spectral_eps(cfg, n_chunks, action_dim, device)

        expected_eps_shape = (n_chunks, cfg.num_mc_samples, num_spectral_tokens_kept, action_dim)
        if tuple(eps.shape) != expected_eps_shape:
            raise ValueError(
                f"MC noise shape {tuple(eps.shape)} != expected "
                f"{expected_eps_shape}"
            )

        chunk_clip_counts = active["chunk_clip_counts"][:n_chunks]

        behavior_cfm_loss, behavior_cfm_loss_token = self._behavior_cfm_losses(
            normalized_states, actions, tau, eps, action_is_pad
        )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": action_is_pad,
            "raw_advantages": raw_advantages,
            "normalized_advantages": normalized_advantages,
            "raw_returns": raw_returns,
            "raw_values": raw_values,
            "normalizer_observations": step_next_states[:, -1, :].detach(),
            "mc_tau": tau,
            "mc_eps": eps,
            "chunk_clip_counts": chunk_clip_counts,
            "gae_diagnostics": gae_diagnostics,
            "behavior_cfm_loss": behavior_cfm_loss,
            "behavior_cfm_loss_token": behavior_cfm_loss_token,
        }


    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        compute_info: bool = True,
        check_finite: bool = True,
    ) -> tuple[torch.Tensor, dict | None]:
        """Combine the ASPO policy and value losses for one minibatch.

        Each selected action chunk is treated as one macro-action for the policy loss.
        The unexecuted suffix is still part of the chunk the behavior policy selected,
        so it must contribute to the FPO ratio when receding-horizon execution is
        enabled; ``action_is_pad`` records execution bookkeeping only, and just the
        chunks with no executed action are invalid for the update.

        Spectral encoding keeps the first ``M+1`` DCT frequencies, giving a token count of
        ``M+1`` derived from ``top_p * H``. The condition is fused from ``[n, D]`` to
        ``[n*m, D]`` so one conditioning vector covers every Monte-Carlo draw. The CFM
        losses are computed under target velocities in fp32 for stability, and the
        value loss is a mean squared error, as the official FPO++ does.

        On info-requesting steps the modulation MLPs are re-run on a single row to
        collect conditioner statistics, which is cheap relative to attention.

        When ``compute_info=False`` the diagnostic reductions are skipped entirely and
        the returned info is ``None``. Training runs without wandb should pass
        ``compute_info=False`` -- the info pass is ~25 small reductions per minibatch
        that would otherwise be discarded.

        Returns ``(loss, info)``, where info is None when ``compute_info`` is false.
        Raises ``ValueError`` if a batch tensor is missing or malformed, if ``mc_tau``
        and ``mc_eps`` disagree about being present, or if ``reduction`` is not
        ``mean`` or ``none``; and ``SJFPOAborted`` if ``check_finite`` is set and the
        loss is non-finite.
        """
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        states = to_device_async(
            to_tensor(batch[OBS_STATE]), device
        ).to(dtype=storage_dtype)
        actions = to_device_async(to_tensor(batch[ACTION]), device).to(
            dtype=storage_dtype
        )
        normalized_advantages = to_tensor(batch["normalized_advantages"]).to(
            device=device, dtype=storage_dtype
        )
        raw_returns = to_tensor(batch["raw_returns"]).to(
            device=device, dtype=torch.float32
        )
        action_is_pad = to_tensor(
            batch["action_is_pad"], dtype=torch.bool, device=device
        )

        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )
        expected_action_mask_shape = (states.shape[0], cfg.action_horizon)
        if tuple(action_is_pad.shape) != expected_action_mask_shape:
            raise ValueError(
                f"action_is_pad shape {tuple(action_is_pad.shape)} != "
                f"{(states.shape[0], cfg.action_horizon)}"
            )

        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        num_spectral_tokens_kept = cfg.num_flow_tokens
        M_modes = num_spectral_tokens_kept - 1
        n = states.shape[0]
        action_dim = actions.shape[-1]

        loss_mask: torch.Tensor = action_is_pad

        spectral_action_tokens = (
            extract_dct_coeffs(actions, M=M_modes) if cfg.use_dct_transform else actions
        )

        has_mc_tau = "mc_tau" in batch
        has_mc_eps = "mc_eps" in batch
        if has_mc_tau != has_mc_eps:
            raise ValueError("mc_tau and mc_eps must be provided together")
        if has_mc_tau:
            tau = to_tensor(batch["mc_tau"], dtype=storage_dtype, device=device)
            expected_tau = (n, cfg.num_mc_samples, num_spectral_tokens_kept, 1)
            if tuple(tau.shape) != expected_tau:
                raise ValueError(
                    f"mc_tau shape {tuple(tau.shape)} != {expected_tau}"
                )
            eps = to_tensor(batch["mc_eps"], dtype=storage_dtype, device=device)
            expected_eps = (n, cfg.num_mc_samples, num_spectral_tokens_kept, action_dim)
            if tuple(eps.shape) != expected_eps:
                raise ValueError(
                    f"stored MC noise shape {tuple(eps.shape)} != {expected_eps}"
                )
        else:
            tau = sample_timesteps(
                (n, cfg.num_mc_samples, num_spectral_tokens_kept),
                cfg,
                device,
                None,
            ).unsqueeze(-1)
            eps = _sample_spectral_eps(cfg, n, action_dim, device)

        flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
            run_monte_carlo_sampling(spectral_action_tokens, eps, tau)
        )
        x_t_flat = _mc_x_t(
            flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape
        )
        actor_tau = flatten_timestep.squeeze(-1)

        with autocast_context(device):
            cond_flat = einx.id(
                "n m d -> (n m) d",
                einx.id("n d -> n m d", self.conditioner(states), m=num_mc),
            )
        if "behavior_cfm_loss" not in batch or "behavior_cfm_loss_token" not in batch:
            raise KeyError(
                "forward requires 'behavior_cfm_loss' and 'behavior_cfm_loss_token' "
                "from generate_rollout. They are computed once per rollout against "
                "the frozen snapshot and must not be recomputed here."
            )
        l_old = to_tensor(
            batch["behavior_cfm_loss"], dtype=torch.float32, device=device
        )
        l_old_token = to_tensor(
            batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device
        )
        if tuple(l_old.shape) != (n, num_mc):
            raise ValueError(
                f"behavior_cfm_loss shape {tuple(l_old.shape)} != {(n, num_mc)}"
            )
        # The token axis is the action-step grid when the mask moved the loss onto
        # rows, and the frequencies otherwise, so it is not always flow_shape[0].
        expected_tokens = self.get_token_count_for_cfm_loss(cfg)
        if tuple(l_old_token.shape) != (n, num_mc, expected_tokens):
            raise ValueError(
                f"behavior_cfm_loss_token shape {tuple(l_old_token.shape)} != "
                f"{(n, num_mc, expected_tokens)}"
            )

        with autocast_context(device):
            v_new = self.compute_predicted_velocity(
                x_t_flat, actor_tau, cond_flat
            ).float()

        cond_stats = None
        if compute_info:
            with torch.inference_mode(), autocast_context(device):
                _, cond_stats = self.actor(
                    x_t_flat[:1], actor_tau[:1], cond_flat[:1], return_stats=True
                )

        with torch.autocast(device_type=device.type, enabled=False):
            v_star_f = self.compute_target_velocity(spectral_action_tokens, eps).float()
            v_new_r_f = v_new.reshape(n, num_mc, *flow_shape)
            loss_mask, (v_new_r_f, v_star_f) = self.prepare_cfm_loss_inputs(
                cfg, action_is_pad, v_new_r_f, v_star_f
            )

            adv_mc = normalized_advantages.float().unsqueeze(1)
            l_new_token = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "token_mean", loss_mask)
            ratio_token = compute_policy_ratio(l_old_token, l_new_token, cfg)

            l_new = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "sample", loss_mask)
            ratio = compute_policy_ratio(l_old, l_new, cfg, advantage=adv_mc)
            self.cfm_above_clamp_frac = (l_new > cfg.cfm_clamp).float().mean().detach()

            surrogate = compute_aspo_surrogate(ratio, adv_mc, cfg, cfg.clip_param)
            policy_per_chunk = -surrogate.mean(dim=1)

        with autocast_context(device):
            raw_value_pred = self.critic(states).float()  # [num_critics, n]
        raw_returns_flat = raw_returns.reshape(-1)
        if raw_value_pred.shape[-1] != raw_returns_flat.shape[0]:
            raise ValueError(
                f"value/return shape mismatch: raw_value_pred="
                f"{tuple(raw_value_pred.shape)}, raw_returns="
                f"{tuple(raw_returns_flat.shape)}"
            )
        # rsl-rl's value loss verbatim: squared error against the raw return, and
        # when clipping is on, the worse of the clipped and unclipped prediction.
        # Broadcasting against the [n, B] member axis trains every member; the
        # mean over both axes is the per-chunk loss.
        value_loss_per_chunk = torch.square(raw_value_pred - raw_returns_flat).mean(
            dim=0
        )
        if cfg.use_clipped_value_loss and "raw_values" in batch:
            rollout_values = to_tensor(
                batch["raw_values"], dtype=torch.float32, device=device
            )
            clipped_value_pred = rollout_values.unsqueeze(0) + (
                raw_value_pred - rollout_values.unsqueeze(0)
            ).clamp(-cfg.clip_param, cfg.clip_param)
            value_loss_per_chunk = torch.maximum(
                value_loss_per_chunk,
                torch.square(clipped_value_pred - raw_returns_flat).mean(dim=0),
            )

        if reduction == "mean":
            policy_loss = policy_per_chunk.mean()
            value_loss = value_loss_per_chunk.mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss
        elif reduction == "none":
            loss = policy_per_chunk + cfg.value_loss_coef * value_loss_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        if check_finite and not bool(torch.isfinite(loss).all()):
            raise SJFPOAborted(
                f"Non-finite loss: "
                f"policy_loss={float(policy_per_chunk.mean().detach())}, "
                f"value_loss={float(value_loss_per_chunk.mean().detach())}"
            )

        if not compute_info:
            return loss, None

        info = self._compute_info(
            l_old=l_old,
            l_new=l_new,
            ratio=ratio,
            ratio_token=ratio_token,
            action_is_pad=action_is_pad,
            v_star_f=v_star_f,
            raw_value_pred=raw_value_pred,
            raw_returns_flat=raw_returns_flat,
            policy_per_chunk=policy_per_chunk,
            value_loss_per_chunk=value_loss_per_chunk,
            normalized_advantages=normalized_advantages,
            cond_stats=cond_stats,
            cfg=cfg,
        )
        return loss, info


    @torch.inference_mode()
    def _compute_info(
        self,
        *,
        l_old,
        l_new,
        ratio,
        ratio_token,
        action_is_pad,
        v_star_f,
        raw_value_pred,
        raw_returns_flat,
        policy_per_chunk,
        value_loss_per_chunk,
        normalized_advantages,
        cond_stats,
        cfg,
    ) -> dict[str, float]:
        """Compute every diagnostic reduction, batched into a single sync.

        The log ratio is taken on the raw delta rather than the clamped one, so the
        reported figure is never an artefact of the clamp. The reported
        clamp-saturation fraction matters because samples above ``cfm_clamp``
        contribute no gradient at all, so a run where it stays high is training the
        critic only. Every scalar is packed into one tensor and transferred in a single
        sync.

        Only called when ``forward(compute_info=True)``; the training loop can skip it
        entirely when nothing will consume the result.
        """
        delta = l_old - l_new
        log_ratio_abs = delta.abs()
        log_ratio_max = log_ratio_abs.max()
        log_ratio_clip_frac = (log_ratio_abs > cfg.log_ratio_cap).float().mean()
        cfm_above_clamp_frac = (l_new > cfg.cfm_clamp).float().mean()

        ratio_mean = ratio.mean()
        ratio_std = ratio.std(correction=0)

        ratio_token_mean = ratio_token.mean(dim=(0, 1))
        num_tokens = ratio_token_mean.numel()
        split = max(1, num_tokens // 2)
        ratio_token_range = ratio_token_mean.max() - ratio_token_mean.min()
        ratio_token_low = ratio_token_mean[:split].mean()
        ratio_token_high = (
            ratio_token_mean[split:].mean() if num_tokens > 1 else ratio_token_mean.mean()
        )
        ratio_token_outside = (
            ((ratio_token < 1.0 - cfg.clip_param) | (ratio_token > 1.0 + cfg.clip_param))
            .float()
            .mean()
        )
        clip_low = cfg.clip_param_lowfreq if cfg.use_per_freq_clipping else cfg.clip_param
        clip_high = (
            cfg.clip_param_highfreq if cfg.use_per_freq_clipping else cfg.clip_param
        )
        low_slice = ratio_token[..., :split]
        ratio_token_low_outside = (
            ((low_slice < 1.0 - clip_low) | (low_slice > 1.0 + clip_low)).float().mean()
        )
        if num_tokens > 1:
            high_slice = ratio_token[..., split:]
            ratio_token_high_outside = (
                ((high_slice < 1.0 - clip_high) | (high_slice > 1.0 + clip_high))
                .float()
                .mean()
            )
        else:
            ratio_token_high_outside = ratio_token_low_outside

        # Does the trust region ever bind? If clipfrac sits near zero the ratio
        # is moving well inside the band and clip_param is not constraining
        # anything, whatever its value. Read it before trusting clip_param.
        clipfrac = (
            (ratio < 1.0 - cfg.clip_param) | (ratio > 1.0 + cfg.clip_param)
        ).float().mean()
        # Share of chunks with any never-executed action position. Those rows
        # still enter the DCT target, so this bounds the contamination.
        truncated_chunk_frac = action_is_pad.any(dim=1).float().mean()
        # Share of action rows the chunk actually ran. With cfm_mask_unexecuted
        # on, this is the fraction of the CFM loss that survives masking, so the
        # two together say how much of the batch the row mask discards.
        executed_action_step_frac = (~action_is_pad).float().mean()

        # explained_var is measured against the member mean, not the min the
        # backup uses. min is a downward-biased selection, so scoring it here
        # would overstate critic quality, and the mean is what the loss trains.
        # The absolute value is therefore not comparable to runs that had a
        # single critic, though the ratio trend across the run still is.
        mean_value_pred = raw_value_pred.mean(dim=0)
        returns_var = raw_returns_flat.var(correction=0).clamp_min(1e-8)
        explained_var = 1.0 - (raw_returns_flat - mean_value_pred).var(
            correction=0
        ) / returns_var
        value_huber_max = mean_value_pred.sub(raw_returns_flat).abs().max()
        value_pred_max = mean_value_pred.abs().max()
        v_star_rms = v_star_f.pow(2).mean().sqrt()

        l_old_std = l_old.std(correction=0)
        l_new_std = l_new.std(correction=0)
        corr_denom = l_old_std * l_new_std + 1e-8
        l_old_new_corr = (
            (l_old - l_old.mean()) * (l_new - l_new.mean())
        ).mean() / corr_denom

        cond_keys = list(cond_stats.keys()) if cond_stats else []
        scalars = [
            policy_per_chunk.mean(),
            value_loss_per_chunk.mean(),
            explained_var,
            delta.mean(),
            log_ratio_max,
            log_ratio_clip_frac,
            clipfrac,
            value_huber_max,
            value_pred_max,
            ratio_mean,
            ratio_std,
            ratio_token_range,
            ratio_token_low,
            ratio_token_high,
            ratio_token_outside,
            ratio_token_low_outside,
            ratio_token_high_outside,
            v_star_rms,
            l_old.mean(),
            l_new.mean(),
            l_old_std,
            l_new_std,
            cfm_above_clamp_frac,
            l_old_new_corr,
            truncated_chunk_frac,
            executed_action_step_frac,
            raw_returns_flat.std(correction=0),
            raw_value_pred.std(correction=0),        ]
        scalars.extend(
            cond_stats[k].detach()
            if torch.is_tensor(cond_stats[k])
            else torch.tensor(float(cond_stats[k]), device=delta.device)
            for k in cond_keys
        )
        packed = torch.stack([s.detach().reshape(()) for s in scalars]).float().cpu()
        vals = packed.tolist()

        keys = [
            "policy_loss",
            "value_loss",
            "explained_var",
            "mean_log_ratio",
            "log_ratio_max",
            "log_ratio_clip_frac",
            "clipfrac",
            "value_huber_max",
            "value_pred_max",
            "flow_ratio_mean",
            "flow_ratio_std",
            "flow_ratio_token_range",
            "flow_ratio_token_lowfreq",
            "flow_ratio_token_highfreq",
            "flow_ratio_token_outside_frac",
            "flow_ratio_token_low_outside_frac",
            "flow_ratio_token_high_outside_frac",
            "flow_v_star_rms",
            "l_old_mean",
            "l_new_mean",
            "l_old_std",
            "l_new_std",
            "cfm_above_clamp_frac",
            "l_old_new_corr",
            "truncated_chunk_frac",
            "executed_action_step_frac",
            "value_target_std",
            "value_pred_std",
        ]
        info = dict(zip(keys, vals))
        for i, k in enumerate(cond_keys):
            info[f"cond_{k}"] = vals[len(keys) + i]
        return info