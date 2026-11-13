"""Owns: policy architecture, EMA shadow, loss computation. NOT: training loop or rollout. Key: MyfpoPolicy, make_optimizer.
The class holds the flow head, the state conditioner, and the EMA shadow used for evaluation.
"""

from __future__ import annotations

import copy
from collections import deque

import dataclasses
import os

import numpy as np
import torch
from torch import nn

from myfpo.config import MyfpoConfig
from myfpo.flow import (
    broadcast_condition,
    cfm_loss_weight,
    cfm_target,
    clamp_saturation,
    clip_velocity,
    compute_trust_region_surrogate,
    compute_cfm_log_ratio,
    compute_cfm_loss,
    compute_policy_ratio,
    group_cfm_loss,
    expand_mc_samples,
    sample_action,
    sample_noise,
    velocity_to_target_param,
)
from myfpo.flow import (
    compute_policy_info,
    compute_value_loss,
)
from myfpo.nn import Critic, FlowMLP
from myfpo.env import RunningMeanStd
from myfpo.nn import StateEncoder
from myfpo.actions import bound_for_storage, to_critic_action, to_env_action
from myfpo.collect import RolloutBuffers, run_rollout
from myfpo.env import resolve_action_bounds
from myfpo.nn import (
    autocast_context,
    cudagraph_safe,
    resolve_storage_dtype,
    to_device_async,
)

class TrainingAborted(RuntimeError):
    """Non-finite training loss. The message names the offending term."""




def make_optimizer(policy: MyfpoPolicy) -> torch.optim.Optimizer:
    """Build one AdamW. Split the actor and the critic into separate groups."""
    cfg = policy.config
    kwargs = dict(
        lr=cfg.policy_lr,
        betas=cfg.adam_betas,
        eps=1e-8,
        weight_decay=cfg.optimizer_weight_decay,
    )
    if next(policy.actor.parameters()).is_cuda:
        kwargs["fused"] = True
    groups = [
        *policy.get_actor_optim_params()["params"],
        *policy.get_critic_optim_params()["params"],
    ]
    return torch.optim.AdamW(groups, **kwargs)


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    """One spelling of the batch shape guard: raise when ``tensor`` is not ``expected``."""
    got = tuple(tensor.shape)
    if got != tuple(expected):
        raise ValueError(f"{name} shape {got} != {tuple(expected)}")


@dataclasses.dataclass
class TrainingBatch:
    """The tensors one training step reads. Loaded and shape-checked."""

    states: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_is_pad: torch.Tensor
    chunk_truncated: torch.Tensor
    timesteps: torch.Tensor
    eps: torch.Tensor
    x_t: torch.Tensor
    flat_timesteps: torch.Tensor
    flow_shape: tuple[int, ...]
    n: int
    num_mc: int
    condition: torch.Tensor
    condition_flat: torch.Tensor
    cfm_loss_old: torch.Tensor
    cfm_loss_old_token: torch.Tensor

@dataclasses.dataclass
class PolicyTerms:
    """The policy loss, and the terms the metrics read."""

    policy_per_chunk: torch.Tensor
    cfm_loss_new: torch.Tensor
    ratio: torch.Tensor
    ratio_token: torch.Tensor
    delta: torch.Tensor
    target_velocity: torch.Tensor
    cfm_above_clamp_frac: torch.Tensor
    cfm_both_clamped_frac: torch.Tensor


# ARCHITECTURE

def make_policy_networks(config: MyfpoConfig):
    """Build the actor, the state encoder, and the critic from the config, and return the dims."""
    config.validate_dims()
    obs_dim = int(config.obs_dim)
    action_dim = int(config.action_dim)
    actor = FlowMLP(
        action_dim, config.action_horizon, config.hidden_dim, config.time_embed_dim,
    )
    encoder = StateEncoder(obs_dim, config.hidden_dim, config.num_obs_steps)
    critic_width = config.critic_hidden_dim if config.critic_hidden_dim is not None else config.hidden_dim
    critic = Critic(
        obs_dim, critic_width, action_dim, config.num_obs_steps, config.action_horizon,
        config.num_critics, condition_on_exec_length=config.critic_condition_on_exec_length,
        num_layers=config.critic_num_layers,
        dropout=config.critic_dropout, state_hidden=config.hidden_dim,
    )
    return actor, encoder, critic, obs_dim, action_dim

def init_ema_buffers(obj, config: MyfpoConfig):
    """Create the EMA state, the rollout buffers, the diagnostics counters, and the inference deques."""
    obj.ema_shadow = {}
    obj.ema_initialized = False
    obj.rollout_buffers = RolloutBuffers()
    obj._ema_param_cache = None
    obj.cfm_above_clamp_frac = torch.zeros(())
    obj.cfm_both_clamped_frac = torch.zeros(())
    obj.probe_counter = 0
    obj.actor_step_s = 0.0
    obj.critic_step_s = 0.0
    obj.frame_buffer = deque(maxlen=config.num_obs_steps)
    obj.action_buffer = deque()
    obj.step_count = 0

class MyfpoPolicy(nn.Module):
    """The flow actor, the state encoder, the critic pair, and the EMA shadow."""

    def __init__(self, config: MyfpoConfig):
        super().__init__()
        self.config = config
        self.actor, self.encoder, self.critic, obs_dim, action_dim = make_policy_networks(config)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.critic_target.eval()
        self.reward_normalizer = RunningMeanStd(1)
        self.obs_normalizer = RunningMeanStd(obs_dim)
        self.register_buffer("action_low", torch.full((action_dim,), -1.0), persistent=False)
        self.register_buffer("action_high", torch.full((action_dim,), 1.0), persistent=False)
        init_ema_buffers(self, config)

    def train(self, mode: bool = True):
        """Set the module mode, and keep the target critic in eval.

        ``critic_target`` is the bootstrap read. Dropout there would corrupt
        the target values, so it never trains.
        """
        super().train(mode)
        self.critic_target.eval()
        return self

    def reset(self):
        """Drop the cached obs and action buffers."""
        self.frame_buffer.clear()
        self.action_buffer.clear()

    @torch.inference_mode()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        """Update live stats from raw post-step observations."""
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
        """Set the env action bounds that ``select_action`` squashes into.

        The bounds keep the buffer's dtype. Rounding them to the
        compute dtype would quantise the bound the env receives.
        """
        low_t, high_t = resolve_action_bounds(
            low, high, dtype=self.action_low.dtype
        )
        if (
            low_t.shape != self.action_low.shape
            or high_t.shape != self.action_high.shape
        ):
            raise ValueError(
                f"action bounds must match action_dim={self.action_low.numel()}: "
                f"low={tuple(low_t.shape)}, high={tuple(high_t.shape)}"
            )

        self.action_low.copy_(low_t)
        self.action_high.copy_(high_t)

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        """List every parameter that the EMA shadow covers.

        The names carry the module prefix, so the shadow can be matched back to
        a live parameter without holding a second reference to the modules. The
        parameter set is fixed after construction, so the list is built once.
        """
        cache = self._ema_param_cache
        if cache is None:
            cache = []
            for prefix, module in (("actor", self.actor), ("encoder", self.encoder)):
                cache.extend(
                    (f"{prefix}.{name}", param)
                    for name, param in module.named_parameters()
                    if param.requires_grad
                )
            self._ema_param_cache = cache
        return dict(cache)


    def _live_params(self, names) -> list[torch.nn.Parameter]:
        """The live parameters for ``names``, in order."""
        params = self.ema_named_params()
        return [params[name] for name in names]

    @torch.inference_mode()
    def update_ema(self, global_step: int) -> None:
        """Update the actor EMA after an optimizer step."""
        decay = self.config.ema_decay
        if decay <= 0:
            return
        current = self.ema_named_params()
        if not self.ema_shadow:
            self.ema_shadow = {
                name: param.detach().clone() for name, param in current.items()
            }
        names = list(self.ema_shadow)
        shadow = [self.ema_shadow[name] for name in names]
        values = [current[name].detach() for name in names]
        if not self.ema_initialized and global_step >= self.config.ema_warmup_steps:
            torch._foreach_copy_(shadow, values)
            self.ema_initialized = True
            return
        if global_step > self.config.ema_warmup_steps:
            torch._foreach_mul_(shadow, decay)
            torch._foreach_add_(shadow, values, alpha=1.0 - decay)


    def ema_ready(self, global_step: int) -> bool:
        """Can the EMA shadow be used for evaluation yet?"""
        return (
            self.config.ema_decay > 0
            and self.ema_initialized
            and global_step > self.config.ema_warmup_steps
        )


    @torch.no_grad()
    def swap_in_ema(self, global_step: int) -> dict[str, torch.Tensor]:
        """Copy the EMA shadow in, and return the displaced live weights."""
        if not self.ema_ready(global_step):
            return {}
        names = list(self.ema_shadow)
        shadow = [self.ema_shadow[name] for name in names]
        values = self._live_params(names)
        values = [value.detach() for value in values]
        displaced = {name: value.clone() for name, value in zip(names, values)}
        torch._foreach_copy_(values, shadow)
        return displaced

    @torch.no_grad()
    def swap_out_ema(self, displaced: dict[str, torch.Tensor]) -> None:
        """Put back the weights :meth:`swap_in_ema` displaced."""
        if not displaced:
            return
        names = list(displaced)
        torch._foreach_copy_(
            self._live_params(names), [displaced[name] for name in names]
        )

    def ema_state_dict(self) -> dict:
        """Export the EMA shadow: decay, init flag, params."""
        return {
            "decay": self.config.ema_decay,
            "initialised": self.ema_initialized,
            "shadow_params": self.ema_shadow,
        }

    def load_ema_state_dict(self, state: dict) -> bool:
        """Restore the shadow. Return False if the function re-seeded it instead."""
        if "shadow_params" not in state or "initialised" not in state:
            raise ValueError("EMA state needs shadow_params and initialised")
        stored = state["shadow_params"]
        initialised = state["initialised"]
        # A default here hid a malformed payload. Read the flag and fail loudly.
        if stored is None or initialised is None:
            raise ValueError("EMA state carries a None value")
        current = self.ema_named_params()
        if set(stored) != set(current):
            self.ema_shadow = {}
            self.ema_initialized = False
            return False
        # The state is loaded onto the CPU, so the shadow has to follow the
        # live parameters onto their device, or the EMA update mixes them.
        self.ema_shadow = {
            name: value.detach().clone().to(current[name].device)
            for name, value in stored.items()
        }
        # Encoder updated with actor; ratio compares encoder_old vs encoder_new.
        self.ema_initialized = bool(initialised)
        return True

    def get_actor_optim_params(self) -> dict:
        """Actor optimizer param groups."""
        decay, no_decay = [], []
        for name, p in self.actor.named_parameters():
            bare = name.removeprefix("_orig_mod.")
            if "adaLN" in bare or bare.startswith("out."):
                no_decay.append(p)
            else:
                decay.append(p)
        policy_lr = self.config.policy_lr
        return {
            "params": [
                {"params": decay, "lr": policy_lr},
                {"params": no_decay, "weight_decay": 0.0, "lr": policy_lr},
                {"params": list(self.encoder.parameters()), "lr": policy_lr},
            ]
        }

    def get_critic_optim_params(self) -> dict:
        """Critic groups at critic lr, no weight decay."""
        return {
            "params": [
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.config.critic_lr,
                    "weight_decay": 0.0,
                }
            ]
        }
    @torch.no_grad()
    def update_target_critic(self, soft_update_rate: float = 0.005):
        """Blend the target critic toward the live critic at ``soft_update_rate``."""
        # The flow time parameter was also spelled ``tau``; this method
        # keeps the paper letter ``t`` in its argument name.
        for p_target, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            p_target.copy_(soft_update_rate * p + (1.0 - soft_update_rate) * p_target)

    def compute_predicted_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Actor on a flow point."""
        cfg = self.config
        return clip_velocity(self.actor(x_t, t, condition), cfg.velocity_clip_value)


    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        """The actor's condition for a batch of observation windows.

        The window is normalized and encoded once. A caller that samples
        several chunks from the same window calls this once and reuses the
        condition, so the encoder does not run again per chunk.
        """
        cfg = self.config
        device = next(self.actor.parameters()).device
        # The storage dtype of the compute device, not of the input. The
        # training pass normalizes in that dtype, so the rollout must give
        # the actor the same condition the training pass does.
        states = states.to(dtype=resolve_storage_dtype(device))
        states = to_device_async(states, device)
        states = self.obs_normalizer(states)
        with torch.inference_mode(), autocast_context(device):
            return cudagraph_safe(self.encoder(states), cfg.compile)

    def sample_chunk(
        self, condition: torch.Tensor, *, start_from_zero: bool = True
    ) -> torch.Tensor:
        """Sample one action chunk from a condition. ``[B, H, action_dim]``.

        The values are unbounded: they are what the flow was trained to
        produce. The env-facing action is the squash of this chunk, applied
        by the collector on the way to the env.
        """
        cfg = self.config
        device = next(self.actor.parameters()).device
        shape = (condition.shape[0], cfg.action_horizon, int(cfg.action_dim))
        with torch.inference_mode(), autocast_context(device):
            if start_from_zero:
                eps = torch.zeros(*shape, device=device)
                num_steps = cfg.num_steps_zero_start
            else:
                eps = sample_noise(shape, device, None)
                num_steps = cfg.num_steps_noise_start
            action_chunk = sample_action(
                self.compute_predicted_velocity,
                shape,
                condition,
                num_steps=num_steps,
                eps=eps,
                device=device,
                t_min=cfg.min_sampled_timestep,
                t_max=cfg.max_sampled_timestep,
            ).float()

        # The flow operates in scaled space (target = action / actor_scale);
        # multiply back to raw actions on the way to the env.
        action_chunk = self._from_flow_space(action_chunk)

        proposal = action_chunk
        if self.training and cfg.action_perturb_std > 0:
            proposal = proposal + cfg.action_perturb_std * torch.randn_like(proposal)
        # Stored actions are bounded in [-1, 1] under linear processing; only
        # bound outliers. Rescaling here would assume env scale and
        # double-rescale, collapsing actions toward 0 over training. The
        # chunk here is raw (scaled-space flow output x actor_scale); the
        # env-facing action is produced on the way to the env, not here.
        return bound_for_storage(proposal, cfg.action_processing_mode)

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], *, start_from_zero: bool = True
    ) -> torch.Tensor:
        """Predict ``[B, H, action_dim]``, in the policy's raw action space.

        The values are unbounded: they are what the flow was trained to
        produce. The env-facing action is the squash of this chunk, applied
        by the collector on the way to the env.
        """
        cfg = self.config
        states = torch.as_tensor(batch["observation.state"])
        if states.dim() == 2 and states.shape[0] == cfg.num_obs_steps:
            states = states.unsqueeze(0)
        elif states.dim() != 3:
            raise ValueError(
                f"observation.state must be [B, T, obs] or [T, obs]; "
                f"got {tuple(states.shape)}. A bare [obs] frame goes through "
                "select_action, which buffers it into a window."
            )
        condition = self.encode_states(states)
        return self.sample_chunk(condition, start_from_zero=start_from_zero)

    def select_action(self, batch: dict[str, torch.Tensor], *, start_from_zero: bool = True) -> torch.Tensor:
        """Next action, receding through a cached chunk.

        The action handed out is the env-facing one: the raw chunk the policy
        planned, squashed into the env's bounds.
        """
        obs = torch.as_tensor(batch["observation.state"])
        obs = obs.to(dtype=resolve_storage_dtype(obs.device))
        if obs.dim() == 1:
            if not self.frame_buffer:
                for _ in range(self.config.num_obs_steps):
                    self.frame_buffer.append(obs.detach().clone())
            else:
                self.frame_buffer.append(obs.detach())
            window = torch.stack(tuple(self.frame_buffer), dim=0)
        elif obs.dim() == 2:
            window = obs
        else:
            raise ValueError(
                f"select_action expects [obs] or [T, obs]; got {tuple(obs.shape)}"
            )
        if not self.action_buffer:
            chunk = self.predict_action_chunk({"observation.state": window}, start_from_zero=start_from_zero)
            n_execute = max(1, min(self.config.num_action_steps, chunk.shape[1]))
            env_chunk = to_env_action(
                chunk, self.action_low, self.action_high,
                mode=self.config.action_processing_mode,
            )
            for i in range(n_execute):
                self.action_buffer.append(env_chunk[0, i])
        return self.action_buffer.popleft()


    def _to_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        """The CFM regression frame: the raw action divided by ``actor_scale``."""
        return actions / self.config.actor_scale

    def _from_flow_space(self, actions: torch.Tensor) -> torch.Tensor:
        """Undo :meth:`_to_flow_space`."""
        return actions * self.config.actor_scale

    def exec_length_frac(self, executed_steps: torch.Tensor) -> torch.Tensor:
        """Normalise an executed-step count to ``[0, 1]``."""
        return executed_steps.to(torch.float32) / float(self.config.action_horizon)

    def _token_cfm_loss(
        self,
        v: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        mask: torch.Tensor,
        *,
        n: int,
        num_mc: int,
        flow_shape,
        device_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The shared token-level CFM loss chain.

        Behavior and training read the same flow point, so the prologue is
        identical and lives here once: parametrize the velocity, form the
        target, and score both at sample and token level. ``v`` is the
        parametrized velocity (``v_old`` at rollout, ``v_new`` at training).
        Callers keep their outer autocast/no-grad contexts; this opens the
        fp32 window inside.
        """
        cfg = self.config
        with torch.autocast(device_type=device_type, enabled=False):
            v_param = velocity_to_target_param(
                v.reshape(n, num_mc, *flow_shape),
                x_t.reshape(n, num_mc, *flow_shape),
                t,
                cfg,
            )
            target = cfm_target(actions, eps, cfg).float()
            loss, loss_token = compute_cfm_loss(
                v_param, target, cfg, "sample_and_token", mask, cfm_loss_weight(t, cfg)
            )
        return loss, loss_token, target, v_param


    def compute_behavior_cfm_losses(
        self,
        normalized_states: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The CFM loss of the behavior policy.

        That is the ``cfm_loss_old`` of the FPO ratio, evaluated against the
        frozen snapshot at rollout time. It is computed once per rollout and
        must not be recomputed here.
        """
        # The behavior loss and the training loss are two readings of the same
        # flow point, so they must see the same numbers. The training pass
        # reloads the action in fp32 and the noise and timestep in the storage
        # dtype, so do the same here. Rounding the action to the storage dtype
        # would move the CFM target the ratio divides by.
        storage_dtype = resolve_storage_dtype(actions.device)
        eps = eps.to(storage_dtype)
        t = t.to(storage_dtype)
        # Flow operates in scaled space (target = action / actor_scale); keep
        # the behavior losses in the same frame the training pass will use.
        actions = self._to_flow_space(actions)
        x_t, flat_timesteps, flow_shape, num_mc = expand_mc_samples(actions, eps, t)
        with torch.no_grad():
            with autocast_context(actions.device):
                condition = self.encoder(normalized_states)
                condition_flat = broadcast_condition(condition, num_mc)
                v_old = self.compute_predicted_velocity(
                    x_t, flat_timesteps, condition_flat
                ).float()
            loss, loss_token, _target, _v_old = self._token_cfm_loss(
                v_old, x_t, t, actions, eps, action_is_pad,
                n=eps.shape[0], num_mc=num_mc, flow_shape=flow_shape,
                device_type=actions.device.type,
            )
            return loss, loss_token

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
        need_bootstrap: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop chunks and build a training batch.

        ``need_bootstrap`` adds the diagnostic-only bootstrap values. Without
        them the batch carries no ``boot_actions`` and no
        ``boot_values_start``.
        """
        return run_rollout(
            self,
            self.rollout_buffers,
            env,
            total_steps,
            action_low=action_low,
            action_high=action_high,
            noisy=noisy,
            need_bootstrap=need_bootstrap,
        )


    @staticmethod
    def to_device_tensor(
        batch: dict[str, torch.Tensor],
        key: str,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """One spelling of 'coerce to device and dtype'."""
        return torch.as_tensor(batch[key], dtype=dtype, device=device)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        check_finite: bool = True,
    ) -> torch.Tensor:
        """Surrogate policy loss + MSE value loss. See ``loss_with_info`` for diagnostics."""
        loss, _ = self.loss_and_info(batch, reduction, check_finite, compute_info=False)
        return loss

    def loss_with_info(self, batch, reduction="mean", check_finite=True):
        return self.loss_and_info(batch, reduction, check_finite)

    def to_device_batch(
        self,
        batch: dict[str, torch.Tensor],
        device,
        storage_dtype: torch.dtype,
        grad_inputs: bool = False,
    ) -> TrainingBatch:
        """Load the batch, check its shapes, and draw the MC flow points.

        ``grad_inputs`` makes the obs window and the chunk leaves that carry a
        gradient, so the probe can ask how much the value depends on each
        frame and each chunk token. The values do not change.
        """
        cfg = self.config
        # No pinned async copies on the loss path: under torch.compile the H2D
        # is a graph node, and pin_memory has no place inside a CUDA graph.
        # The regression target is built from the stored fp32 action (scaled by
        # actor_scale). A bf16 rounding would put a 0.4% error on the target the
        # flow is fit to.
        states = self.to_device_tensor(batch, "observation.state", device, storage_dtype)
        actions = self.to_device_tensor(batch, "action", device, torch.float32)
        advantages = self.to_device_tensor(
            batch, "normalized_advantages", device, storage_dtype
        )
        returns = self.to_device_tensor(batch, "raw_returns", device, torch.float32)
        action_is_pad = self.to_device_tensor(
            batch, "action_is_pad", device, torch.bool
        )
        chunk_truncated = self.to_device_tensor(
            batch, "chunk_truncated", device, torch.bool
        )

        if states.shape[-2] != cfg.num_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != num_obs_steps {cfg.num_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )
        expected_action_mask_shape = (states.shape[0], cfg.action_horizon)
        _require_shape("action_is_pad", action_is_pad, expected_action_mask_shape)

        states = self.obs_normalizer(states)
        # Flow operates in scaled space (target = action / actor_scale), same
        # frame as the behavior losses. The critic still reads the env-space
        # squashed action downstream.
        actions = self._to_flow_space(actions)
        if grad_inputs:
            states = states.detach().requires_grad_(True)
            actions = actions.detach().requires_grad_(True)

        horizon = cfg.action_horizon
        n = states.shape[0]
        action_dim = actions.shape[-1]

        if "mc_t" not in batch or "mc_eps" not in batch:
            raise ValueError("Mandatory paired reuse: batch must contain 'mc_t' and 'mc_eps' (coupled draws for both old/new CFM loss)")
        t = torch.as_tensor(batch["mc_t"], dtype=storage_dtype, device=device)
        expected_t = (n, cfg.num_mc_samples, horizon, 1)
        _require_shape("mc_t", t, expected_t)
        eps = torch.as_tensor(batch["mc_eps"], dtype=storage_dtype, device=device)
        expected_eps = (n, cfg.num_mc_samples, horizon, action_dim)
        _require_shape("mc_eps", eps, expected_eps)

        x_t, flat_timesteps, flow_shape, num_mc = expand_mc_samples(actions, eps, t)

        with autocast_context(device):
            condition = cudagraph_safe(self.encoder(states), self.config.compile)
            condition_flat = broadcast_condition(condition, num_mc)

        cfm_loss_old, cfm_loss_old_token = self.to_device_behavior_losses(
            batch, device, n, num_mc
        )

        return TrainingBatch(
            states=states,
            actions=actions,
            advantages=advantages,
            returns=returns,
            action_is_pad=action_is_pad,
            chunk_truncated=chunk_truncated,
            timesteps=t,
            eps=eps,
            x_t=x_t,
            flat_timesteps=flat_timesteps,
            flow_shape=flow_shape,
            n=n,
            num_mc=num_mc,
            condition=condition,
            condition_flat=condition_flat,
            cfm_loss_old=cfm_loss_old,
            cfm_loss_old_token=cfm_loss_old_token,
        )

    def to_device_behavior_losses(
        self,
        batch: dict[str, torch.Tensor],
        device,
        n: int,
        num_mc: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The frozen behavior CFM losses, computed once per rollout."""
        cfg = self.config
        required = (
            "behavior_cfm_loss",
            "behavior_cfm_loss_token",
        )
        if any(key not in batch for key in required):
            raise KeyError(
                f"forward requires {', '.join(repr(key) for key in required)} "
                "from generate_rollout. They are computed once per rollout against "
                "the frozen snapshot and must not be recomputed here."
            )
        cfm_loss_old = torch.as_tensor(
            batch["behavior_cfm_loss"], dtype=torch.float32, device=device
        )
        cfm_loss_old_token = torch.as_tensor(
            batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device
        )
        _require_shape("behavior_cfm_loss", cfm_loss_old, (n, num_mc))
        expected_tokens = cfg.action_horizon
        _require_shape("behavior_cfm_loss_token", cfm_loss_old_token, (n, num_mc, expected_tokens))
        return cfm_loss_old, cfm_loss_old_token

    def policy_loss(self, b: TrainingBatch, v_new: torch.Tensor, device) -> PolicyTerms:
        """The trust-region surrogate loss, one value per chunk."""
        cfg = self.config
        n, num_mc, flow_shape = b.n, b.num_mc, b.flow_shape
        with torch.autocast(device_type=device.type, enabled=False):
            loss_mask = b.action_is_pad
            cfm_loss_new, cfm_loss_new_token, target_velocity, _v_new_r_f = self._token_cfm_loss(
                v_new, b.x_t, b.timesteps, b.actions, b.eps, loss_mask,
                n=n, num_mc=num_mc, flow_shape=flow_shape,
                device_type=device.type,
            )

            adv = b.advantages.float()
            adv_mc = adv.unsqueeze(1)
            ratio_token = compute_policy_ratio(
                b.cfm_loss_old_token, cfm_loss_new_token, cfg,
            )

            if cfg.group_mc_loss:
                ratio_loss_old = group_cfm_loss(
                    b.cfm_loss_old, cfg.mc_loss_group_size
                )
                ratio_loss_new = group_cfm_loss(
                    cfm_loss_new, cfg.mc_loss_group_size
                )
                adv_ratio = adv.reshape(-1, 1)
                ratio = compute_policy_ratio(
                    ratio_loss_old, ratio_loss_new, cfg
                )
            else:
                ratio_loss_old = b.cfm_loss_old_token
                ratio_loss_new = cfm_loss_new_token
                adv_ratio = adv_mc.unsqueeze(-1)
                ratio = ratio_token
            # The diagnostics must read the same losses that built the
            # ratio, or they describe a trust region the update never used.
            cfm_above_clamp_frac = (
                ratio_loss_new > cfg.cfm_clamp
            ).float().mean().detach()
            cfm_both_clamped_frac = clamp_saturation(
                ratio_loss_old, ratio_loss_new, cfg
            ).detach()
            delta = compute_cfm_log_ratio(
                ratio_loss_old, ratio_loss_new, cfg
            )
            surrogate = compute_trust_region_surrogate(ratio, adv_ratio, cfg)
            if surrogate.dim() == 3:
                keep = (~b.action_is_pad).to(surrogate.dtype)
                surrogate = surrogate * keep[:, None, :]
                n_executed = keep.sum(dim=-1).clamp_min(1.0)
                policy_per_chunk = -(
                    surrogate.sum(dim=(1, 2)) / (n_executed * surrogate.shape[1])
                )
            else:
                policy_per_chunk = -surrogate.flatten(1).mean(dim=1)

        return PolicyTerms(
            policy_per_chunk=policy_per_chunk,
            cfm_loss_new=cfm_loss_new,
            ratio=ratio,
            ratio_token=ratio_token,
            delta=delta,
            target_velocity=target_velocity,
            cfm_above_clamp_frac=cfm_above_clamp_frac,
            cfm_both_clamped_frac=cfm_both_clamped_frac,
        )

    def value_loss(
        self, b: TrainingBatch, device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score the critic on the executed chunk. Return the loss, the values, and the returns."""
        with autocast_context(device):
            chunk_exec_frac = self.exec_length_frac((~b.action_is_pad).sum(dim=-1))
            pad_mask = (~b.action_is_pad).unsqueeze(-1).float()
            # b.actions is in flow (scaled) space for the CFM target; the critic
            # reads the raw policy output to match the rollout bootstrap (which
            # uses the stored raw action). Recover it before the env mapping.
            raw_actions = self._from_flow_space(b.actions)
            b_actions_for_critic = to_critic_action(
                raw_actions, self.action_low, self.action_high,
                mode=self.config.action_processing_mode, mask=pad_mask,
            )
            raw_value_pred = self.critic(b.states, b_actions_for_critic, chunk_exec_frac).float()  # [num_critics, n]
        raw_returns_flat = b.returns.reshape(-1)
        if raw_value_pred.shape[-1] != raw_returns_flat.shape[0]:
            raise ValueError(
                f"value/return shape mismatch: raw_value_pred="
                f"{tuple(raw_value_pred.shape)}, raw_returns="
                f"{tuple(raw_returns_flat.shape)}"
            )
        cfg = self.config
        value_loss_per_chunk = compute_value_loss(
            raw_value_pred, raw_returns_flat, cfg
        )
        value_loss_per_chunk = value_loss_per_chunk.mean(dim=0)
        return value_loss_per_chunk, raw_value_pred, raw_returns_flat

# LOSS
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        *,
        grad_inputs: bool = False,
        timers=None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The pure loss path. Tensor ops only: no host transfer, no float().

        Returns ``(loss, aux)`` where ``aux`` carries the tensors the metrics
        and the probes read. ``grad_inputs`` makes the obs and action leaves
        require a gradient for the temporal probe. ``timers`` is an optional
        tuple of CUDA events ``(a0, a1, c0, c1)``; the compiled path is called
        without them.
        """
        cfg = self.config
        device = next(self.actor.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        b = self.to_device_batch(batch, device, storage_dtype, grad_inputs=grad_inputs)

        if timers is not None:
            timers[0].record()
        with autocast_context(device):
            v_new = self.compute_predicted_velocity(
                b.x_t, b.flat_timesteps, b.condition_flat
            ).float()
        if timers is not None:
            timers[1].record()

        terms = self.policy_loss(b, v_new, device)
        if timers is not None:
            timers[2].record()
        value_loss_per_chunk, raw_value_pred, raw_returns_flat = self.value_loss(
            b, device
        )
        if timers is not None:
            timers[3].record()

        if reduction == "mean":
            policy_loss = terms.policy_per_chunk.mean()
            value_loss = value_loss_per_chunk.mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss
        elif reduction == "none":
            loss = terms.policy_per_chunk + cfg.value_loss_coef * value_loss_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        aux = {
            "cfm_loss_old": b.cfm_loss_old.detach(),
            "cfm_loss_new": terms.cfm_loss_new.detach(),
            "ratio": terms.ratio.detach(),
            "ratio_token": terms.ratio_token.detach(),
            "delta": terms.delta.detach(),
            "target_velocity": terms.target_velocity.detach(),
            "action_is_pad": b.action_is_pad,
            "chunk_truncated": b.chunk_truncated,
            "policy_per_chunk": terms.policy_per_chunk.detach(),
            "value_loss_per_chunk": value_loss_per_chunk.detach(),
            # raw_value_pred keeps its grad: the temporal probe backprops
            # through it. The metrics pass calls detach() internally.
            "raw_value_pred": raw_value_pred,
            "raw_returns_flat": raw_returns_flat.detach(),
            "policy_loss": terms.policy_per_chunk.mean().detach(),
            "value_loss": value_loss_per_chunk.mean().detach(),
            "cfm_above_clamp_frac": terms.cfm_above_clamp_frac,
            "cfm_both_clamped_frac": terms.cfm_both_clamped_frac,
            # The probes read these. They stay tensors; no host transfer here.
            "x_t": b.x_t,
            "flat_timesteps": b.flat_timesteps,
            "condition_flat": b.condition_flat,
            "states": b.states,
            "actions": b.actions,
        }
        return loss, aux

    def _loss_impl(self):
        """Return ``(impl, compiled)`` for the loss path under the current config."""
        cfg = self.config
        if not cfg.compile:
            return self.compute_loss, False
        fn = getattr(self, "_compiled_compute_loss", None)
        if fn is None:
            fn = torch.compile(self.compute_loss, mode=cfg.compile_mode)
            self._compiled_compute_loss = fn
        return fn, True

    def compute_loss_and_probes(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str,
        check_finite: bool,
        *,
        compute_info: bool = True,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        """Forward, probes, and the clamp fractions. No host transfer.

        Returns ``(loss, staged)``. ``staged`` holds the tensors that
        :meth:`pack_metrics` reads, so the one device-to-host move happens
        after ``backward``. ``staged`` is None when ``compute_info`` is off.
        """
        cfg = self.config
        device = next(self.actor.parameters()).device

        self.probe_counter += 1
        probe_on = (
            cfg.temporal_probe_every > 0
            and self.probe_counter % cfg.temporal_probe_every == 0
            and torch.is_grad_enabled()
        )
        timing = os.environ.get("MYFPO_TIMING") == "1"

        # The temporal probe backprops leaves; it always runs eager.
        if cfg.compile and not probe_on:
            impl, compiled = self._loss_impl()
            timers = None
        else:
            impl, compiled = self.compute_loss, False
            timers = (
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                if timing and device.type == "cuda"
                else None
            )

        loss, aux = impl(batch, reduction, grad_inputs=probe_on, timers=timers)

        if compiled:
            # A compiled graph hands out pool-owned outputs; clone before any
            # consumer keeps them.
            aux = {
                k: (cudagraph_safe(v, True) if torch.is_tensor(v) else v)
                for k, v in aux.items()
            }

        if timers is not None:
            self.actor_step_s = timers[0].elapsed_time(timers[1]) / 1e3
            self.critic_step_s = timers[2].elapsed_time(timers[3]) / 1e3
        else:
            self.actor_step_s = 0.0
            self.critic_step_s = 0.0

        self.cfm_above_clamp_frac = aux["cfm_above_clamp_frac"]
        self.cfm_both_clamped_frac = aux["cfm_both_clamped_frac"]

        cond_stats = None
        if compute_info:
            with torch.inference_mode(), autocast_context(device):
                probe = max(1, cfg.hidden_dim // cfg.action_horizon)
                _, cond_stats = self.actor(
                    aux["x_t"][:probe],
                    aux["flat_timesteps"][:probe],
                    aux["condition_flat"][:probe],
                    collect_stats=True,
                )

        sensitivity = (
            self.temporal_sensitivity(aux["raw_value_pred"], aux["states"], aux["actions"])
            if probe_on
            else {}
        )

        if check_finite and not bool(torch.isfinite(loss).all()):
            raise TrainingAborted(
                f"Non-finite loss: "
                f"policy_loss={float(aux['policy_loss'])}, "
                f"value_loss={float(aux['value_loss'])}"
            )

        if not compute_info:
            return loss, None

        staged = {
            "above_clamp_frac": self.cfm_above_clamp_frac,
            "both_clamped_frac": self.cfm_both_clamped_frac,
            "cfm_loss_old": aux["cfm_loss_old"],
            "cfm_loss_new": aux["cfm_loss_new"],
            "ratio": aux["ratio"],
            "ratio_token": aux["ratio_token"],
            "delta": aux["delta"],
            "action_is_pad": aux["action_is_pad"],
            "chunk_truncated": aux["chunk_truncated"],
            "target_velocity": aux["target_velocity"],
            # Detached: the metrics read values, never gradients, so the staged
            # row must not keep the graph alive across backward.
            "raw_value_pred": aux["raw_value_pred"].detach(),
            "raw_returns_flat": aux["raw_returns_flat"],
            "policy_per_chunk": aux["policy_per_chunk"],
            "value_loss_per_chunk": aux["value_loss_per_chunk"],
            "cond_stats": cond_stats,
            "sensitivity": sensitivity,
        }
        return loss, staged

    def pack_metrics(self, staged: dict[str, object]) -> dict[str, float]:
        """Pack the staged tensors into one device-to-host transfer."""
        info = compute_policy_info(
            above_clamp_frac=staged["above_clamp_frac"],
            both_clamped_frac=staged["both_clamped_frac"],
            cfm_loss_old=staged["cfm_loss_old"],
            cfm_loss_new=staged["cfm_loss_new"],
            ratio=staged["ratio"],
            ratio_token=staged["ratio_token"],
            delta=staged["delta"],
            action_is_pad=staged["action_is_pad"],
            chunk_truncated=staged["chunk_truncated"],
            target_velocity=staged["target_velocity"],
            raw_value_pred=staged["raw_value_pred"],
            raw_returns_flat=staged["raw_returns_flat"],
            policy_per_chunk=staged["policy_per_chunk"],
            value_loss_per_chunk=staged["value_loss_per_chunk"],
            cond_stats=staged["cond_stats"],
            cfg=self.config,
        )
        info.update(staged["sensitivity"])
        return info

    def loss_and_info(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str,
        check_finite: bool,
        *,
        compute_info: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        """The loss, then its metrics in one host transfer. Kept for the pins."""
        loss, staged = self.compute_loss_and_probes(
            batch, reduction, check_finite, compute_info=compute_info
        )
        if staged is None:
            return loss, None
        return loss, self.pack_metrics(staged)

    @torch.compiler.disable
    def temporal_sensitivity(
        self,
        raw_value_pred: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, float]:
        """Diagnostic: Q gradient wrt chunk tokens / obs frames.
        Measures temporal filter quality; prices full plan, not executed prefix."""
        # ``grad_inputs=True`` makes the two leaves require a gradient. What
        # the probe needs is that both leaves still require a gradient.
        if not (actions.requires_grad and states.requires_grad):
            import warnings
            warnings.warn(
                "temporal_sensitivity: the probe leaves do not require a "
                "gradient; build the batch with grad_inputs=True.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        out = {}
        grads = torch.autograd.grad(
            raw_value_pred.sum(), actions, retain_graph=True
        )[0]
        for i, value in enumerate(grads.norm(dim=-1).mean(dim=0).tolist()):
            out[f"critic_action_grad_norm_pos{i}"] = value
        grads = torch.autograd.grad(
            raw_value_pred.sum(), states, retain_graph=True
        )[0]
        for j, value in enumerate(grads.norm(dim=-1).mean(dim=0).tolist()):
            out[f"obs_frame_grad_norm_pos{j}"] = value
        return out



