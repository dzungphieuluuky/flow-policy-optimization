"""SJFPO policy: flow head + state conditioner + critic.

Batch contract: ``[B, T, obs]`` states, ``[B, H, action_dim]`` chunks, frozen
GAE advantages/returns from the rollout.

Speed
-----
* ``forward`` diagnostics batch into one device-to-host transfer (one
  ``.tolist()``) instead of ~20 ``float()`` syncs, and the whole pass is gated
  on ``compute_info=True``.
* ``ratio_token`` is computed once per forward, not once per branch.
* ``snapshot_behavior_policy`` copies weights in place rather than
  ``copy.deepcopy``-ing.
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

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.collector import RolloutBuffer, collect_chunks, collect_chunks_vec
from spectral_joint_policy.dct_utils import (
    velocity_to_action_steps,
    extract_dct_coeffs,
    compute_spectral_energy_retained,
)
from spectral_joint_policy.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_monte_carlo_sampling,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from spectral_joint_policy.gae import compute_chunk_advantages
from spectral_joint_policy.networks import (
    Critic,
    RunningMeanStd,
    SJFPOFlowModel,
    StateConditioner,
)
from spectral_joint_policy.processor_sjfpo import make_sjfpo_pre_post_processors
from spectral_joint_policy.utils import (
    autocast_context,
    resolve_storage_dtype,
    rms,
    to_device_async,
    to_tensor_1d,
)

class SJFPOAborted(RuntimeError):
    """Non-finite training loss. The message names the offending term."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def compute_value_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Elementwise Huber penalty, broadcasting ``target`` over the member axis.

    ``delta`` is a 0-dim tensor so the threshold can track the batch return RMS.
    """
    return torch.nn.functional.huber_loss(
        pred, target.expand_as(pred), delta=delta, reduction="none"
    )


# Pack order of ``compute_gae_diagnostics``'s return dict, and the pack order of its
# one device-to-host transfer. The trainer reads these by key.
_GAE_DIAGNOSTIC_KEYS = (
    "linked_frac",
    "advantage_std",
    "return_rms",
    "advantage_std_fraction",
    "budget_truncated_chunk_frac",
)


def compute_gae_diagnostics(
    env_id: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    advantages: torch.Tensor,
    raw_returns: torch.Tensor,
    budget_truncated: torch.Tensor | None = None,
) -> dict[str, float]:
    """Trace-liveness and scale probes for the chunk backup.

    ``linked_frac`` is the one that matters: the share of chunk pairs the
    recursion joined. Chunks arrive in wavefront order, so raw adjacency joins
    nothing and the env_id sort is what makes the chain live.
    """
    n = env_id.numel()
    if n == 0:
        return dict.fromkeys(_GAE_DIAGNOSTIC_KEYS, 0.0)
    order = torch.argsort(env_id.to(torch.long), stable=True)
    e = env_id.to(torch.long)[order]
    ends = terminated.to(torch.bool)[order][:-1] | truncated.to(torch.bool)[order][:-1]
    ret = raw_returns.float()
    # Each field keeps the fp32 arithmetic it always had and is widened only
    # here, so the packed values match the Python floats they replace exactly.
    # The stack is float64; 5 numbers, so the wider transfer costs nothing.
    zero = ret.double().new_zeros(())
    # Population std, matching the advantage normaliser, returns_var and
    # compute_info. The unbiased default is NaN at a single chunk and
    # inflates by sqrt(n/(n-1)) elsewhere. correction=0 is defined at n == 1,
    # so the n > 1 guard the unbiased form needed is gone: a lone chunk now
    # falls out of clamp_min as 0/1e-8 = 0, the same 0.0 the guard produced.
    adv_std = advantages.float().std(correction=0)
    ret_std = ret.std(correction=0).clamp_min(1e-8)
    fields = [
        # linked_frac: chunks the recursion joined, over the n-1 possible links.
        ((e[1:] == e[:-1]) & ~ends).sum().double() / max(n - 1, 1),
        adv_std,
        rms(ret).double(),
        # A = R - V, so Var(A) <= Var(R), equal only when V is 0. 1.0 means the
        # critic is inert; it falls toward 1 - R^2 as the critic improves.
        adv_std / ret_std,
        # Budget cut, reported apart from chunk_truncated, the flag GAE reads.
        # Not truncated_chunk_frac: that one is derived from action_is_pad and
        # means "the chunk left action rows unexecuted".
        budget_truncated.to(torch.bool).float().mean().double()
        if budget_truncated is not None else zero,
    ]
    # One transfer, not one per field: float() on a device tensor is a blocking
    # sync, and this runs once per rollout.
    return dict(zip(_GAE_DIAGNOSTIC_KEYS, torch.stack(fields).tolist()))


def sample_spectral_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """MC noise matching the configured token layout.

    ``[n, num_mc_samples, tokens, action_dim]``.
    """
    shape = (n, cfg.num_mc_samples, cfg.num_flow_tokens, action_dim)
    return sample_noise(shape, cfg, device, None)


def build_optimizer(policy: SJFPOPolicy) -> torch.optim.Optimizer:
    """One AdamW over actor and critic, in separate param groups.

    - groups split per role, so clipping splits per role
    - the critic's gradient is orders of magnitude larger
    - one shared clip would let it own the actor's step
    - rsl-rl uses this same single-optimizer, per-role-group layout
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


def compute_mc_xt(
    flatten_mc_timestep: torch.Tensor,
    flatten_mc_noise: torch.Tensor,
    flatten_mc_action: torch.Tensor,
    flow_shape: tuple[int, ...],
) -> torch.Tensor:
    """Interpolate noise to action to form the flow point.

    Returns ``[-1, *flow_shape]``.
    """
    t = flatten_mc_timestep.expand(-1, -1, flow_shape[-1])
    noise = flatten_mc_noise.reshape(-1, *flow_shape)
    action = flatten_mc_action.reshape(-1, *flow_shape)
    # lerp, not (1-t)*noise + t*action: same value, one kernel instead of four.
    return torch.lerp(noise, action, t)


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
        # Polyak target for the GAE bootstrap, following qc and TD-JEPA. Kept even
        # when use_target_critic is False so toggling the flag does not change
        # which weights exist in the module.
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self._target_critic_initialised = False
        self._ema_shadow: dict[str, torch.Tensor] = {}
        self._ema_initialised = False
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
        """Set the module mode, frozen snapshot stays in eval. Chains."""
        super().train(mode)
        self.old_actor.eval()
        self.old_conditioner.eval()
        return self

    def reset(self):
        """Drop the cached obs and action buffers between episodes."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    @torch.inference_mode()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        """Update live running stats from raw post-step obs.

        ``ValueError`` if the shape is not ``[N, obs_dim]``.
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
        """Set the env action bounds the postprocessor uses.

        ``ValueError`` if shapes disagree or ``high > low`` fails.
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
        """Refresh the frozen pi_old snapshot in place.

        Copies weights into the existing modules instead of
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
        """Update the FPO++ actor EMA after an optimizer step.

        Seeded on the first call, hard-copied at warmup end. Only warmed weights
        enter the average.
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
        if not self._ema_initialised and global_step >= self.config.ema_warmup_steps:
            # >= and gated on _ema_initialised, not ==. global_step moves by a
            # whole rollout, so == 500 was never hit and ema_ready() stayed
            # false forever. The gate makes it one-shot; without it every later
            # call re-hard-copied and threw away the running average.
            names = list(self._ema_shadow)
            torch._foreach_copy_(
                [self._ema_shadow[name] for name in names],
                [current[name].detach() for name in names],
            )
            self._ema_initialised = True
            return
        if global_step > self.config.ema_warmup_steps:
            # One multi-tensor pass over the shadow rather than one per tensor.
            names = list(self._ema_shadow)
            torch._foreach_mul_([self._ema_shadow[name] for name in names], decay)
            torch._foreach_add_(
                [self._ema_shadow[name] for name in names],
                [current[name].detach() for name in names],
                alpha=1.0 - decay,
            )

    @torch.no_grad()
    def update_target_critic(self) -> None:
        """Polyak the target critic toward the live one.

        ``gamma ** action_horizon`` puts 0.85 of every value target on the live
        critic's own output, and targets then sit for a whole rollout while the
        critic trains. A slow teacher is the same guard qc (``tau=0.005``) and
        TD-JEPA use.

        First call hard-copies. Later calls move the target by
        ``critic_ema_decay`` per optimizer step.
        """
        if not self._target_critic_initialised:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self._target_critic_initialised = True
            return
        decay = self.config.critic_ema_decay
        # Multi-tensor ops: the critic is 26 parameters, so a Python loop is 78
        # launches per optimizer step against foreach's 28.
        target_params = list(self.target_critic.parameters())
        torch._foreach_mul_(target_params, 1.0 - decay)
        torch._foreach_add_(
            target_params, [p.detach() for p in self.critic.parameters()], alpha=decay
        )
        # The critic registers no buffers today (RMSNorm keeps a Parameter, and
        # RunningMeanStd hangs off the policy), so this is usually empty --
        # foreach rejects an empty list where the old zip() loop was a silent
        # no-op. Kept so a future buffer follows the teacher.
        target_buffers = list(self.target_critic.buffers())
        if target_buffers:
            torch._foreach_copy_(target_buffers, list(self.critic.buffers()))

    def ema_ready(self, global_step: int) -> bool:
        """Can the EMA shadow be used for evaluation yet?"""
        return (
            self.config.ema_decay > 0
            and self._ema_initialised
            and global_step > self.config.ema_warmup_steps
        )

    @torch.no_grad()
    def swap_in_ema(self, global_step: int) -> dict[str, torch.Tensor]:
        """Copy the EMA shadow into the actor, returning the live weights.

        The caller holds the return and hands it back to
        :meth:`swap_out_ema` after evaluating, so the live weights are never
        lost. No-op returning ``{}`` when the shadow is not ready. Takes
        ``global_step``, not ``step_count``: the shadow is warmed on the env-step
        clock, so the optimizer-step counter would call it ready long before it
        is.
        """
        if not self.ema_ready(global_step):
            return {}
        params = dict(self.actor.named_parameters())
        names = list(self._ema_shadow)
        displaced = {name: params[name].detach().clone() for name in names}
        torch._foreach_copy_(
            [params[name] for name in names],
            [self._ema_shadow[name] for name in names],
        )
        return displaced

    @torch.no_grad()
    def swap_out_ema(self, displaced: dict[str, torch.Tensor]) -> None:
        """Put the weights :meth:`swap_in_ema` displaced back into the actor."""
        if not displaced:
            return
        params = dict(self.actor.named_parameters())
        names = list(displaced)
        torch._foreach_copy_(
            [params[name] for name in names], [displaced[name] for name in names]
        )

    def ema_state_dict(self) -> dict:
        """Export the EMA shadow. Returns decay, init flag, shadow params."""
        return {
            "decay": self.config.ema_decay,
            "initialised": self._ema_initialised,
            "shadow_params": self._ema_shadow,
        }

    def load_ema_state_dict(self, state: dict) -> None:
        """Restore an EMA shadow from :meth:`ema_state_dict`."""
        self._ema_shadow = {
            name: value.detach().clone()
            for name, value in state["shadow_params"].items()
        }
        self._ema_initialised = bool(state.get("initialised", True))

    def get_actor_optim_params(self) -> dict:
        """Actor optimizer param groups.

        adaLN modulation and the output layer skip weight decay. The conditioner
        shares the policy lr. Returns an optimizer param-group dict.
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
        """Critic optimizer param groups at the critic lr, without weight decay.

        Decay pulls the value head toward zero, which shows up as a drifting
        value scale and a wandering grad norm; the actor keeps its decay.

        Returns an optimizer param-group dict.
        """
        return {
            "params": [
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.config.critic_lr,
                    "weight_decay": 0.0,
                }
            ]
        }

    def get_optim_params(self) -> dict:
        """One FPO++-style group over all trainable weights.

        Implements the abstract base method. Training does not use it:
        :func:`build_optimizer` composes per-role groups from
        :meth:`get_actor_optim_params` and :meth:`get_critic_optim_params` so
        grad clipping stays separable. Kept because the base class needs it.
        Returns an optimizer param-group dict.
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
        """Flow-matching target velocity.

        ``v* = a - eps``, broadcast over the MC draws: ``[B, *flow]`` against
        ``[B, m, *flow]`` -> ``[B, m, *flow]``.
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
        """Run the actor on a flow point against prepared conditioning.

        ``condition`` is already the conditioner's output broadcast over the MC
        draws, so the frozen and live paths share the one forward-conditioning
        pass the caller already did. ``old`` picks the frozen snapshot. Returns
        velocities in the actor's dtype.
        """
        net = self.old_actor if old else self.actor
        return net(x_t, tau, condition)


    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce obs to the ``[B, T, obs]`` layout the policy wants.

        Returns a storage-dtype tensor. ``ValueError`` if rank is not 1, 2, or 3.
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

        Action perturbation runs before postprocessing, matching FPO++, so the
        noise lands in normalised action space. ``zero_sampling=True`` integrates
        from ``eps=0`` over ``num_inference_steps``; ``False`` integrates from
        sampled noise over ``num_euler_steps``, honouring
        ``noise_sampling_mode`` and ``noise_scaling_mode``. Returns
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
            # velocity_to_action_steps reads the coeffs back on the half-open grid
            # t_j = j*H/H. linspace(0, 1, K) instead queries j = 0, 1.067, ..., K,
            # which time-warps every executed action by K/(K-1).
            scaled = velocity_to_action_steps(x0, cfg.action_horizon).clamp(-1.0, 1.0)
        else:
            scaled = x0[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)

        if self.training and cfg.action_perturb_std > 0:
            scaled = scaled + cfg.action_perturb_std * torch.randn_like(scaled)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Next action, receding through a cached chunk.

        The chunk is predicted once and its executed prefix is buffered, so
        back-to-back calls do not re-run the actor. ``batch`` holds
        ``observation.state``: ``[obs]`` for one frame, ``[T, obs]`` for a full
        window. Extra kwargs go to :meth:`predict_action_chunk`. ``ValueError``
        if obs rank is neither 1 nor 2.
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

        The action-step grid when the loss runs on rows, the frequencies when it
        does not. :meth:`prepare_cfm_loss_inputs` picks the layout and
        ``token_mean`` inherits its length, so the two must agree or the ratio
        compares different-width vectors.
        """
        on_action_grid = not cfg.use_dct_transform or cfg.cfm_mask_unexecuted
        return cfg.action_horizon if on_action_grid else cfg.num_flow_tokens

    def prepare_cfm_loss_inputs(
        self,
        cfg: SJFPOConfig,
        action_is_pad: torch.Tensor,
        *velocities: torch.Tensor,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        """Execution mask plus velocities the mask can index.

        ``action_is_pad`` marks action steps a chunk never ran. A mask only
        indexes a velocity laid out over the ``action_horizon`` steps.

        - DCT off: already that layout
        - DCT on: ``num_flow_tokens`` freq coefficients, every step mapping to
          every coeff, so expand back onto the action steps first
        - the mask drops the tail
        - so a short chunk gets no gradient from unrun steps
        - instead of a wrong-signed one

        Expanding is exact. The inverse DCT is linear. At
        ``num_kept_tokens == action_horizon`` the basis is orthonormal, so it is
        a rotation and the loss is unchanged.

        Mask and layout return together on purpose:

        - they are only consistent with each other
        - at ``num_kept_tokens == action_horizon`` both spaces share a shape
        - a mask with unexpanded velocities lands on the wrong axis
        - and that fails silently

        Both ``l_old`` and ``l_new`` must come through here, else the ratio
        compares means over different positions.
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

    def compute_behavior_cfm_losses(
        self,
        normalised_states: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen behavior policy's CFM loss for one minibatch.

        Behavior weights are the snapshot taken at rollout start, so this is the
        ``l_old`` term of the ratio. Both reductions are masked exactly like the
        current-policy losses they get compared against, so the ratio never
        mixes means over different positions. Returns the ``sample`` and
        ``token_mean`` reductions.
        """
        cfg = self.config
        token_count = eps.shape[2]
        if cfg.use_dct_transform:
            spectral_action = extract_dct_coeffs(actions, num_modes=token_count)
        else:
            spectral_action = actions
        flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
            run_monte_carlo_sampling(spectral_action, eps, tau)
        )
        x_t = compute_mc_xt(flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape)
        actor_tau = flatten_timestep.squeeze(-1)
        with torch.inference_mode(), autocast_context(actions.device):
            condition = self.old_conditioner(normalised_states)
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
            return compute_cfm_loss(v_old, v_star, cfg, "both", loss_mask)

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop chunks, build a training batch.

        A chunk can execute one step before terminating, so the chunk count can
        hit one per requested step. Each chunk is one macro-transition: the
        backup sums its executed rewards, discounts by ``gamma ** h``, and the
        GAE recursion runs between consecutive chunks of the same env, see
        :mod:`sjfpo.gae`. Advantages normalise over chunks.

        The batch carries one raw post-step frame per executed step, separate
        from the overlapping obs windows, so the loop can fold them into the
        live normalizer without double counting. Plus per-chunk/MC CFM draws the
        policy loss reuses for both the old and new losses.

        ``ValueError`` if ``total_steps`` is not positive, or a transition is
        inconsistent.
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
            # [:6] drops pin_memory, a device property not part of this key. The
            # full tuple never matches, so the buffer was reallocated every
            # rollout.
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
        num_spectral_tokens_kept = cfg.num_flow_tokens

        lengths = active["chunk_lengths"]
        chunk_starts = active["chunk_starts"]
        all_rewards = active["step_rewards"]
        actions = active["chunk_actions"]
        states = active["chunk_state"]

        # Only the chunk's first and post-chunk states are read by the backup, so
        # the critic and preprocessor see 2 * n_chunks windows, not 2 * n_steps.
        # The obs normalizer updates separately from raw frames via
        # normalizer_observations.
        with torch.inference_mode():
            next_states = active["chunk_next_state"]
            all_chunk_normalised_states = self.preprocessor(
                {OBS_STATE: torch.cat([states, next_states], dim=0)}
            )[OBS_STATE]
            normalised_states = all_chunk_normalised_states[:n_chunks]
            with autocast_context(device):
                all_chunk_values = self.critic(all_chunk_normalised_states).float()
            # min over the ensemble, backup only. The loss trains every member
            # separately: min has no case inside a V loss, no max to counteract,
            # so it would only bias V low.
            chunk_value = all_chunk_values[:, :n_chunks].amin(dim=0)
            if cfg.use_target_critic:
                with torch.inference_mode(), autocast_context(device):
                    boot_values = self.target_critic(all_chunk_normalised_states[n_chunks:]).float()
                chunk_next_value = boot_values.amin(dim=0)
            else:
                chunk_next_value = all_chunk_values[:, n_chunks:].amin(dim=0)

        # One chunk is one macro-transition: sum the chunk's rewards, discount
        # gamma ** h. One space throughout, as rsl-rl: the critic predicts raw
        # returns, GAE subtracts them from raw rewards, the value target is the
        # raw return. No return normalizer, so V is the true expected return and
        # the target stops drifting as the return scale grows. The only
        # normalisation is the per-batch advantage whitening after the backup.
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
        normalised_advantages = raw_advantages
        # Population std, matching returns_var and compute_info. The unbiased
        # default is NaN at a single chunk, which a short final rollout
        # produces, and inflates by sqrt(n/(n-1)) elsewhere. clamp_min is the
        # torch guard; `+ 1e-8` only happened to agree when std > 0.
        if cfg.normalize_advantage and raw_advantages.numel() > 0:
            normalised_advantages = (raw_advantages - raw_advantages.mean()) / (
                raw_advantages.std(correction=0).clamp_min(1e-8)
            )
        gae_diagnostics = compute_gae_diagnostics(
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
        normalised_advantages = normalised_advantages.to(storage_dtype)
        # raw_returns and raw_values stay fp32: the critic's target and clip
        # snapshot, and the only stored tensors that are not O(1) (returns_rms
        # runs to 35). bf16 has 8 mantissa bits, so 1234.5 round-trips to
        # 1232.0. Each is 256 elements, so fp32 is free, and chunk_state and
        # chunk_actions are already fp32, so storage_dtype never saved memory.

        action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        tau = sample_timesteps(
            (n_chunks, cfg.num_mc_samples, num_spectral_tokens_kept),
            cfg,
            device,
            None,
        ).unsqueeze(-1)
        eps = sample_spectral_eps(cfg, n_chunks, action_dim, device)

        expected_eps_shape = (n_chunks, cfg.num_mc_samples, num_spectral_tokens_kept, action_dim)
        if tuple(eps.shape) != expected_eps_shape:
            raise ValueError(
                f"MC noise shape {tuple(eps.shape)} != expected "
                f"{expected_eps_shape}"
            )

        chunk_clip_counts = active["chunk_clip_counts"][:n_chunks]

        # 1.0 at num_kept_tokens == action_horizon (the transform is then a
        # rotation and this is a no-op), falling as the spectrum truncates.
        # None when the DCT is off, so the tracker can skip the key.
        spectral_energy = None
        if cfg.use_dct_transform:
            spectral_energy = float(
                compute_spectral_energy_retained(actions, cfg.num_flow_tokens)
            )

        behavior_cfm_loss, behavior_cfm_loss_token = self.compute_behavior_cfm_losses(
            normalised_states, actions, tau, eps, action_is_pad
        )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": action_is_pad,
            "raw_advantages": raw_advantages,
            "normalised_advantages": normalised_advantages,
            "raw_returns": raw_returns,
            "raw_values": raw_values,
            "normalizer_observations": active["step_frames"].detach(),
            # Diagnostics: how far the live critic moved from the bootstrap value
            # the targets were built with. With a target critic that gap is the
            # price of the slow teacher; without one it is the target's drift.
            "boot_observations": all_chunk_normalised_states[n_chunks:].to(storage_dtype),
            "boot_values_start": chunk_next_value.unsqueeze(1),
            "mc_tau": tau,
            "mc_eps": eps,
            "chunk_clip_counts": chunk_clip_counts,
            "gae_diagnostics": gae_diagnostics,
            "compute_spectral_energy_retained": spectral_energy,
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
        """ASPO policy and value losses for one minibatch.

        One chunk is one macro-action. Only executed steps count;
        ``action_is_pad`` masks the rest out of the CFM loss, so a short chunk
        is never fitted on unrun steps.

        - spectral encoding keeps ``num_flow_tokens`` DCT freqs
        - the condition fuses ``[n, D]`` to ``[n*m, D]``, one vector per draw
        - CFM losses run under target velocities in fp32
        - value loss is mean squared error against the raw return (FPO++)

        Info steps re-run the modulation MLPs on one row. Cheap next to
        attention.

        ``compute_info=False`` skips the reductions and returns ``info=None``.
        Pass it when nothing will consume the result, or one reduction per
        minibatch is wasted.

        Returns ``(loss, info)``, ``info`` None when ``compute_info`` is off.
        ``ValueError`` on a missing or malformed batch tensor, on ``mc_tau`` and
        ``mc_eps`` disagreeing about presence, or on ``reduction`` outside
        ``mean``/``none``. ``SJFPOAborted`` on a non-finite loss when
        ``check_finite`` is set.
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
        normalised_advantages = to_tensor(batch["normalised_advantages"]).to(
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
        n = states.shape[0]
        action_dim = actions.shape[-1]

        loss_mask: torch.Tensor = action_is_pad

        dct_action_tokens = (
            extract_dct_coeffs(actions, num_modes=num_spectral_tokens_kept)
            if cfg.use_dct_transform
            else actions
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
            eps = sample_spectral_eps(cfg, n, action_dim, device)

        flatten_timestep, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = (
            run_monte_carlo_sampling(dct_action_tokens, eps, tau)
        )
        x_t_flat = compute_mc_xt(
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
        # The token axis is the action-step grid when the mask moved the loss
        # onto rows, and the frequencies otherwise. Not always flow_shape[0].
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
            v_star_f = self.compute_target_velocity(dct_action_tokens, eps).float()
            v_new_r_f = v_new.reshape(n, num_mc, *flow_shape)
            loss_mask, (v_new_r_f, v_star_f) = self.prepare_cfm_loss_inputs(
                cfg, action_is_pad, v_new_r_f, v_star_f
            )

            adv_mc = normalised_advantages.float().unsqueeze(1)
            l_new, l_new_token = compute_cfm_loss(
                v_new_r_f, v_star_f, cfg, "both", loss_mask
            )
            ratio_token = compute_policy_ratio(l_old_token, l_new_token, cfg)
            ratio = compute_policy_ratio(l_old, l_new, cfg, advantage=adv_mc)
            self.cfm_above_clamp_frac = (l_new > cfg.cfm_clamp).float().mean().detach()

            surrogate = compute_aspo_surrogate(ratio, adv_mc, cfg)
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
        # rsl-rl's value loss, with squared error replaced by Huber: quadratic
        # inside delta, linear outside, so one large-residual chunk cannot
        # dominate the gradient. The clipped branch is the same penalty on the
        # clipped prediction, and max() still takes the worse of the two.
        # Broadcasting against the [n, B] member axis trains every member; the
        # mean over both axes is the per-chunk loss.
        returns_rms = rms(raw_returns_flat)
        huber_delta = cfg.value_huber_delta * returns_rms.clamp_min(1e-6)
        value_loss_per_chunk = compute_value_huber_loss(
            raw_value_pred, raw_returns_flat, huber_delta
        ).mean(dim=0)
        if cfg.use_clipped_value_loss and "raw_values" in batch:
            rollout_values = to_tensor(
                batch["raw_values"], dtype=torch.float32, device=device
            )
            clipped_value_pred = rollout_values.unsqueeze(0) + (
                raw_value_pred - rollout_values.unsqueeze(0)
            ).clamp(-cfg.clip_param, cfg.clip_param)
            value_loss_per_chunk = torch.maximum(
                value_loss_per_chunk,
                compute_value_huber_loss(
                    clipped_value_pred, raw_returns_flat, huber_delta
                ).mean(dim=0),
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

        info = self.compute_info(
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
            normalised_advantages=normalised_advantages,
            cond_stats=cond_stats,
            cfg=cfg,
        )
        return loss, info


    @torch.inference_mode()
    def compute_info(
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
        normalised_advantages,
        cond_stats,
        cfg,
    ) -> dict[str, float]:
        """Every diagnostic reduction, batched into one sync.

        The log ratio is taken on the raw delta, not the clamped one, so the
        figure is never a clamp artefact. The clamp-saturation fraction matters
        because samples above ``cfm_clamp`` contribute no gradient at all, so a
        run that stays high is training the critic only. Every scalar packs into
        one tensor, one transfer.

        Only called under ``forward(compute_info=True)``. The loop can skip it
        when nothing consumes the result.
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
        # One epsilon for the whole token axis, the same bound the surrogate
        # uses. A per-frequency epsilon had nothing to bind to: the loss runs the
        # "sample" reduction, averaging every token into one number per chunk, so
        # per-frequency ratios exist only in this diagnostic.
        low_slice = ratio_token[..., :split]
        ratio_token_low_outside = (
            ((low_slice < 1.0 - cfg.clip_param) | (low_slice > 1.0 + cfg.clip_param))
            .float()
            .mean()
        )
        if num_tokens > 1:
            high_slice = ratio_token[..., split:]
            ratio_token_high_outside = (
                ((high_slice < 1.0 - cfg.clip_param) | (high_slice > 1.0 + cfg.clip_param))
                .float()
                .mean()
            )
        else:
            ratio_token_high_outside = ratio_token_low_outside

        # Does the trust region ever bind? clipfrac near zero means the ratio
        # stays well inside the band and clip_param constrains nothing, whatever
        # its value. Read it before trusting clip_param.
        clipfrac = (
            (ratio < 1.0 - cfg.clip_param) | (ratio > 1.0 + cfg.clip_param)
        ).float().mean()
        # Chunks with any never-executed action row. Those rows still enter the
        # DCT target, so this bounds the contamination.
        truncated_chunk_frac = action_is_pad.any(dim=1).float().mean()
        # Share of action rows the chunk ran. With cfm_mask_unexecuted on, this
        # is the fraction of CFM loss surviving the mask, so the two together say
        # how much of the batch the row mask drops.
        executed_action_step_frac = (~action_is_pad).float().mean()

        # explained_var scores the member mean, not the min the backup uses: min
        # is biased low, so scoring it would flatter the critic. Not comparable
        # in absolute terms to single-critic runs, but the trend is.
        mean_value_pred = raw_value_pred.mean(dim=0)
        returns_var = raw_returns_flat.var(correction=0).clamp_min(1e-8)
        explained_var = 1.0 - (raw_returns_flat - mean_value_pred).var(
            correction=0
        ) / returns_var
        value_huber_max = mean_value_pred.sub(raw_returns_flat).abs().max()
        value_pred_max = mean_value_pred.abs().max()
        v_star_rms = rms(v_star_f)

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