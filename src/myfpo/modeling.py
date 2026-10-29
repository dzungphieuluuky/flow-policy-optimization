"""Owns: policy architecture, EMA shadow, loss computation. NOT: training loop or rollout. Key: MyfpoPolicy, build_optimizer.
The class holds the flow head, the state conditioner, and the EMA shadow used for evaluation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import dataclasses
import time

import numpy as np
import torch

import warnings
from lerobot.policies import PreTrainedPolicy
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.configuration import MyfpoConfig
from myfpo.fpo_core import (
    broadcast_condition,
    cfm_loss_weight,
    cfm_target,
    clamp_saturation,
    clip_velocity,
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    group_cfm_loss,
    mc_samples,
    sample_action,
    sample_noise,
    velocity_to_target_param,
)
from myfpo.fpo_core import (
    compute_policy_info,
    compute_value_loss,
)
from myfpo.networks import Critic, MyfpoFlowModel, MyfpoFlowResMLP
from myfpo.processor import RunningMeanStd, StateEncoder, squash_to_env_bounds
from myfpo.processor import make_pre_post_processors
from myfpo.rollout import RolloutBuffers, run_rollout
from myfpo.layers import (
    autocast_context,
    cudagraph_safe,
    resolve_storage_dtype,
    to_device_async,
    to_tensor_1d,
)

class MyfpoAborted(RuntimeError):
    """Non-finite training loss. The message names the offending term."""




def build_optimizer(policy: MyfpoPolicy) -> torch.optim.Optimizer:
    """Build one AdamW. Split the actor and the critic into separate groups."""
    kwargs = dataclasses.asdict(policy.config.get_optimizer_preset())
    kwargs.pop("grad_clip_norm", None)  # AdamW would clip all groups as one. See split_grad_params.
    kwargs["betas"] = policy.config.adam_betas
    if next(policy.actor.parameters()).is_cuda:
        kwargs["fused"] = True
    groups = [
        *policy.get_actor_optim_params()["params"],
        *policy.get_critic_optim_params()["params"],
    ]
    return torch.optim.AdamW(groups, **kwargs)

@dataclasses.dataclass
class Batch:
    """The tensors one training step reads. Loaded and shape-checked."""

    states: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_is_pad: torch.Tensor
    t: torch.Tensor
    eps: torch.Tensor
    x_t: torch.Tensor
    actor_t: torch.Tensor
    flow_shape: tuple[int, ...]
    n: int
    num_mc: int
    condition: torch.Tensor
    condition_flat: torch.Tensor
    cfm_loss_old: torch.Tensor
    cfm_loss_old_token: torch.Tensor
    velocity_old_token: torch.Tensor


@dataclasses.dataclass
class PolicyTerms:
    """The policy loss, and the terms the metrics read."""

    policy_per_chunk: torch.Tensor
    cfm_loss_new: torch.Tensor
    ratio: torch.Tensor
    ratio_token: torch.Tensor
    delta: torch.Tensor
    v_star_f: torch.Tensor


# ARCHITECTURE

def build_policy_head(config: MyfpoConfig):
    obs_dim = int(config.robot_state_feature.shape[0])
    action_dim = int(config.action_feature.shape[0])
    if config.actor_arch == "resmlp":
        actor = MyfpoFlowResMLP(
            action_dim, config.action_horizon, config.hidden_dim,
            config.resmlp_blocks, config.resmlp_units, config.resmlp_hidden_mult,
            config.dit_heads, config.time_embed_dim,
        )
    else:
        actor = MyfpoFlowModel(
            action_dim, config.action_horizon, config.hidden_dim,
            config.dit_depth, config.dit_heads, config.dit_mlp_ratio, config.time_embed_dim,
        )
    encoder = StateEncoder(obs_dim, config.hidden_dim, config.n_obs_steps)
    critic_width = config.critic_hidden_dim if config.critic_hidden_dim is not None else config.hidden_dim
    critic = Critic(
        obs_dim, critic_width, action_dim, config.n_obs_steps, config.action_horizon,
        config.num_critics, condition_on_exec_length=config.critic_condition_on_exec_length,
        num_layers=config.critic_num_layers,
        dropout=config.critic_dropout, state_hidden=config.hidden_dim,
    )
    return actor, encoder, critic, obs_dim, action_dim

def init_ema_buffers(obj, config: MyfpoConfig):
    obj.ema_shadow = {}
    obj.ema_initialised = False
    obj.rollout_buffers = RolloutBuffers()
    obj.cfm_above_clamp_frac = torch.zeros(())
    obj.cfm_both_clamped_frac = torch.zeros(())
    obj.probe_counter = 0
    obj.actor_step_s = 0.0
    obj.critic_step_s = 0.0
    obj.frame_buffer = deque(maxlen=config.n_obs_steps)
    obj.action_buffer = deque()
    obj.step_count = 0

class MyfpoPolicy(PreTrainedPolicy):
    config_class = MyfpoConfig
    name = "myfpo"

    def __init__(
        self, config: MyfpoConfig, dataset_stats: dict[str, Any] | None = None
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.actor, self.encoder, self.critic, obs_dim, action_dim = build_policy_head(config)
        import copy
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.critic_target.eval()
        self.reward_normalizer = RunningMeanStd(1)
        self.obs_normalizer = RunningMeanStd(obs_dim)
        self.register_buffer("action_low", torch.full((action_dim,), -1.0), persistent=False)
        self.register_buffer("action_high", torch.full((action_dim,), 1.0), persistent=False)
        init_ema_buffers(self, config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(self)
        del dataset_stats

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
        """Set the env action bounds the postprocessor uses."""
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
            raise ValueError("action_high must be strictly greater than action_low")

        self.action_low.copy_(low_t)
        self.action_high.copy_(high_t)
        self.postprocessor.set_action_bounds(self.action_low, self.action_high)

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        """List every parameter that the EMA shadow covers.

        The names carry the module prefix, so the shadow can be matched back to
        a live parameter without holding a second reference to the modules.
        """
        out: dict[str, torch.nn.Parameter] = {}
        for prefix, module in (("actor", self.actor), ("encoder", self.encoder)):
            for name, param in module.named_parameters():
                if param.requires_grad:
                    out[f"{prefix}.{name}"] = param
        return out

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
        if not self.ema_initialised and global_step >= self.config.ema_warmup_steps:
            torch._foreach_copy_(shadow, values)
            self.ema_initialised = True
            return
        if global_step > self.config.ema_warmup_steps:
            torch._foreach_mul_(shadow, decay)
            torch._foreach_add_(shadow, values, alpha=1.0 - decay)


    def ema_ready(self, global_step: int) -> bool:
        """Can the EMA shadow be used for evaluation yet?"""
        return (
            self.config.ema_decay > 0
            and self.ema_initialised
            and global_step > self.config.ema_warmup_steps
        )


    @torch.no_grad()
    def swap_in_ema(self, global_step: int) -> dict[str, torch.Tensor]:
        """Copy the EMA shadow in, and return the displaced live weights."""
        if not self.ema_ready(global_step):
            return {}
        params = self.ema_named_params()
        names = list(self.ema_shadow)
        shadow = [self.ema_shadow[name] for name in names]
        values = [params[name].detach() for name in names]
        displaced = {name: value.clone() for name, value in zip(names, values)}
        torch._foreach_copy_(values, shadow)
        return displaced

    @torch.no_grad()
    def swap_out_ema(self, displaced: dict[str, torch.Tensor]) -> None:
        """Put back the weights :meth:`swap_in_ema` displaced."""
        if not displaced:
            return
        params = self.ema_named_params()
        names = list(displaced)
        torch._foreach_copy_(
            [params[name] for name in names], [displaced[name] for name in names]
        )

    def ema_state_dict(self) -> dict:
        """Export the EMA shadow: decay, init flag, params."""
        return {
            "decay": self.config.ema_decay,
            "initialised": self.ema_initialised,
            "shadow_params": self.ema_shadow,
        }

    def load_ema_state_dict(self, state: dict) -> bool:
        """Restore the shadow. Return False if the function re-seeded it instead."""
        stored = state.get("shadow_params", {})
        current = self.ema_named_params()
        if set(stored) != set(current):
            self.ema_shadow = {}
            self.ema_initialised = False
            return False
        # The state is loaded onto the CPU, so the shadow has to follow the
        # live parameters onto their device, or the EMA update mixes them.
        self.ema_shadow = {
            name: value.detach().clone().to(current[name].device)
            for name, value in stored.items()
        }
        # Encoder updated with actor; ratio compares encoder_old vs encoder_new.
        self.ema_initialised = bool(state.get("initialised", True))
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
        # The flow time parameter was also spelled ``tau``; this method
        # keeps the paper letter ``t`` in its argument name.
        for p_target, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            p_target.copy_(soft_update_rate * p + (1.0 - soft_update_rate) * p_target)

    def get_optim_params(self) -> dict:
        """Required by ``PreTrainedPolicy``. Returns what the optimizer actually uses.

        The split is delegated rather than restated, so it cannot drift away
        from :meth:`build_optimizer`.
        """
        return {"params": [
            *self.get_actor_optim_params()["params"],
            *self.get_critic_optim_params()["params"],
        ]}


    def compute_predicted_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Actor on a flow point."""
        cfg = self.config
        return clip_velocity(self.actor(x_t, t, condition), cfg.transported_clip_value)


    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Predict ``[B, H, action_dim]``, in the policy's raw action space.

        The values are unbounded: they are what the flow was trained to
        produce. The env-facing action is the squash of this chunk, applied
        by the collector on the way to the env.
        """
        start_from_zero = kwargs.get("start_from_zero", True)
        cfg = self.config
        device = next(self.actor.parameters()).device

        states = to_tensor(batch[OBS_STATE])
        states = states.to(dtype=resolve_storage_dtype(states.device))
        if states.dim() == 2 and states.shape[0] == cfg.n_obs_steps:
            states = states.unsqueeze(0)
        elif states.dim() != 3:
            raise ValueError(
                f"observation.state must be [B, T, obs] or [T, obs]; "
                f"got {tuple(states.shape)}. A bare [obs] frame goes through "
                "select_action, which buffers it into a window."
            )
        states = to_device_async(states, device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        horizon = cfg.action_horizon
        action_dim = int(self.config.action_feature.shape[0])

        with torch.inference_mode(), autocast_context(device):
            condition = cudagraph_safe(self.encoder(states), self.config.compile)
            shape = (states.shape[0], horizon, action_dim)
            if start_from_zero:
                eps = torch.zeros(*shape, device=device)
                num_steps = cfg.num_steps_zero_start
            else:
                eps = sample_noise(shape, cfg, device, None)
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

        proposal = action_chunk
        if self.training and cfg.action_perturb_std > 0:
            proposal = proposal + cfg.action_perturb_std * torch.randn_like(proposal)
        # The raw chunk, in the policy's own action space. The env-facing
        # action is the squash of this, and the squash is applied on the way
        # to the env, not here: the flow is trained on the raw value.
        return proposal

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Next action, receding through a cached chunk.

        The action handed out is the env-facing one: the raw chunk the policy
        planned, squashed into the env's bounds.
        """
        obs = to_tensor(batch[OBS_STATE])
        obs = obs.to(dtype=resolve_storage_dtype(obs.device))
        if obs.dim() == 1:
            if not self.frame_buffer:
                for _ in range(self.config.n_obs_steps):
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
            chunk = self.predict_action_chunk({OBS_STATE: window}, **kwargs)
            n_execute = max(1, min(self.config.n_action_steps, chunk.shape[1]))
            squashed = self.postprocessor({ACTION: chunk})[ACTION]
            for i in range(n_execute):
                self.action_buffer.append(squashed[0, i])
        return self.action_buffer.popleft()


    def exec_length_frac(self, executed_steps: torch.Tensor) -> torch.Tensor:
        """Normalise an executed-step count to ``[0, 1]``."""
        return executed_steps.to(torch.float32) / float(self.config.action_horizon)

    def compute_behavior_cfm_losses(
        self,
        normalised_states: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The CFM loss of the behavior policy.

        That is the ``cfm_loss_old`` of the FPO ratio, evaluated against the
        frozen snapshot at rollout time. It is computed once per rollout and
        must not be recomputed here.
        """
        cfg = self.config
        # The behavior loss and the training loss are two readings of the same
        # flow point, so they must see the same numbers. The rollout samples
        # the noise and the timestep in fp32, while the training pass reloads
        # them in the storage dtype, so the triple is aligned here, in the
        # storage dtype, before anything is computed from it.
        storage_dtype = resolve_storage_dtype(actions.device)
        actions = actions.to(storage_dtype)
        eps = eps.to(storage_dtype)
        t = t.to(storage_dtype)
        x_t, actor_t, flow_shape, num_mc = mc_samples(actions, eps, t)
        with torch.no_grad():
            with autocast_context(actions.device):
                condition = self.encoder(normalised_states)
                condition_flat = broadcast_condition(condition, num_mc)
                v_old = self.compute_predicted_velocity(
                    x_t, actor_t, condition_flat
                ).float()
            with torch.autocast(device_type=actions.device.type, enabled=False):
                v_star = cfm_target(actions, eps, cfg).float()
                v_old = v_old.reshape(eps.shape[0], num_mc, *flow_shape)
                v_old = velocity_to_target_param(
                    v_old,
                    x_t.reshape(eps.shape[0], num_mc, *flow_shape),
                    t,
                    cfg,
                )
                loss_mask = action_is_pad
                loss, loss_token = compute_cfm_loss(
                    v_old, v_star, cfg, "sample_and_token", loss_mask, cfm_loss_weight(t, cfg)
                )
                return loss, loss_token, v_old

    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noisy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Roll out open-loop chunks and build a training batch."""
        return run_rollout(
            self,
            self.rollout_buffers,
            env,
            total_steps,
            action_low=action_low,
            action_high=action_high,
            noisy=noisy,
        )


    @staticmethod
    def load_tensor(
        batch: dict[str, torch.Tensor],
        key: str,
        device,
        dtype: torch.dtype,
        *,
        async_: bool = False,
    ) -> torch.Tensor:
        """One spelling of 'coerce to device and dtype'."""
        tensor = to_tensor(batch[key], dtype=dtype, device=device)
        return to_device_async(tensor, device) if async_ else tensor

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        check_finite: bool = True,
    ) -> torch.Tensor:
        """ASPO policy loss + MSE value loss. See ``loss_with_info`` for diagnostics."""
        loss, _ = self.loss_and_info(
            batch, reduction, check_finite, compute_info=False
        )
        return loss

    def loss_with_info(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        check_finite: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The loss and the metric dict. The loss is identical to :meth:`forward`."""
        loss, info = self.loss_and_info(
            batch, reduction, check_finite, compute_info=True
        )
        return loss, info

    def load_batch(
        self,
        batch: dict[str, torch.Tensor],
        device,
        storage_dtype: torch.dtype,
        grad_inputs: bool = False,
    ) -> Batch:
        """Load the batch, check its shapes, and draw the MC flow points.

        ``grad_inputs`` makes the obs window and the chunk leaves that carry a
        gradient, so the probe can ask how much the value depends on each
        frame and each chunk token. The values do not change.
        """
        cfg = self.config
        states = self.load_tensor(batch, OBS_STATE, device, storage_dtype, async_=True)
        # The action is the flow's raw output, and it is the CFM regression
        # target. It stays fp32: the raw values are unbounded, so a bf16
        # rounding would put a 0.4% error on the target the flow is fit to.
        actions = self.load_tensor(batch, ACTION, device, torch.float32, async_=True)
        advantages = self.load_tensor(
            batch, "normalised_advantages", device, storage_dtype
        )
        returns = self.load_tensor(batch, "raw_returns", device, torch.float32)
        action_is_pad = self.load_tensor(
            batch, "action_is_pad", device, torch.bool
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
        if grad_inputs:
            states = states.detach().requires_grad_(True)
            actions = actions.detach().requires_grad_(True)

        horizon = cfg.action_horizon
        n = states.shape[0]
        action_dim = actions.shape[-1]

        if "mc_t" not in batch or "mc_eps" not in batch:
            raise ValueError("Mandatory paired reuse: batch must contain 'mc_t' and 'mc_eps' (coupled draws for both old/new CFM loss)")
        t = to_tensor(batch["mc_t"], dtype=storage_dtype, device=device)
        expected_t = (n, cfg.num_mc_samples, horizon, 1)
        if tuple(t.shape) != expected_t:
            raise ValueError(f"mc_t shape {tuple(t.shape)} != {expected_t}")
        eps = to_tensor(batch["mc_eps"], dtype=storage_dtype, device=device)
        expected_eps = (n, cfg.num_mc_samples, horizon, action_dim)
        if tuple(eps.shape) != expected_eps:
            raise ValueError(f"stored MC noise shape {tuple(eps.shape)} != {expected_eps}")

        x_t, actor_t, flow_shape, num_mc = mc_samples(actions, eps, t)

        with autocast_context(device):
            condition = cudagraph_safe(self.encoder(states), self.config.compile)
            condition_flat = broadcast_condition(condition, num_mc)

        cfm_loss_old, cfm_loss_old_token, velocity_old_token = (
            self.load_behavior_losses(batch, device, n, num_mc, action_dim)
        )

        return Batch(
            states=states,
            actions=actions,
            advantages=advantages,
            returns=returns,
            action_is_pad=action_is_pad,
            t=t,
            eps=eps,
            x_t=x_t,
            actor_t=actor_t,
            flow_shape=flow_shape,
            n=n,
            num_mc=num_mc,
            condition=condition,
            condition_flat=condition_flat,
            cfm_loss_old=cfm_loss_old,
            cfm_loss_old_token=cfm_loss_old_token,
            velocity_old_token=velocity_old_token,
        )

    def load_behavior_losses(
        self,
        batch: dict[str, torch.Tensor],
        device,
        n: int,
        num_mc: int,
        action_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The frozen behavior CFM losses and velocity, computed once per rollout."""
        cfg = self.config
        required = (
            "behavior_cfm_loss",
            "behavior_cfm_loss_token",
            "behavior_velocity_token",
        )
        if any(key not in batch for key in required):
            raise KeyError(
                f"forward requires {', '.join(repr(key) for key in required)} "
                "from generate_rollout. They are computed once per rollout against "
                "the frozen snapshot and must not be recomputed here."
            )
        cfm_loss_old = to_tensor(
            batch["behavior_cfm_loss"], dtype=torch.float32, device=device
        )
        cfm_loss_old_token = to_tensor(
            batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device
        )
        velocity_old_token = to_tensor(
            batch["behavior_velocity_token"], dtype=torch.float32, device=device
        )
        if tuple(cfm_loss_old.shape) != (n, num_mc):
            raise ValueError(
                f"behavior_cfm_loss shape {tuple(cfm_loss_old.shape)} != {(n, num_mc)}"
            )
        expected_tokens = cfg.action_horizon
        if tuple(cfm_loss_old_token.shape) != (n, num_mc, expected_tokens):
            raise ValueError(
                f"behavior_cfm_loss_token shape {tuple(cfm_loss_old_token.shape)} != "
                f"{(n, num_mc, expected_tokens)}"
            )
        expected_velocity = (n, num_mc, expected_tokens, action_dim)
        if tuple(velocity_old_token.shape) != expected_velocity:
            raise ValueError(
                f"behavior_velocity_token shape {tuple(velocity_old_token.shape)} != "
                f"{expected_velocity}"
            )
        return cfm_loss_old, cfm_loss_old_token, velocity_old_token

    def policy_loss(self, b: Batch, v_new: torch.Tensor, device) -> PolicyTerms:
        """The ASPO surrogate loss, one value per chunk."""
        cfg = self.config
        n, num_mc, flow_shape = b.n, b.num_mc, b.flow_shape
        with torch.autocast(device_type=device.type, enabled=False):
            v_new_r_f = velocity_to_target_param(
                v_new.reshape(n, num_mc, *flow_shape),
                b.x_t.reshape(n, num_mc, *flow_shape),
                b.t,
                cfg,
            )
            v_star_f = cfm_target(b.actions, b.eps, cfg).float()
            loss_mask = b.action_is_pad

            adv = b.advantages.float()
            adv_mc = adv.unsqueeze(1)
            weight = cfm_loss_weight(b.t, cfg)
            cfm_loss_new, cfm_loss_new_token = compute_cfm_loss(
                v_new_r_f, v_star_f, cfg, "sample_and_token", loss_mask, weight,
            )
            ratio_token = compute_policy_ratio(
                b.cfm_loss_old_token, cfm_loss_new_token, cfg,
                advantage=adv_mc.unsqueeze(-1),
            )

            if cfg.do_average_cfm_loss_in_chunk:
                loss_old = group_cfm_loss(b.cfm_loss_old, cfg.cfm_average_group_size)
                loss_new = group_cfm_loss(cfm_loss_new, cfg.cfm_average_group_size)
                ratio = compute_policy_ratio(
                    loss_old, loss_new, cfg, advantage=adv.reshape(-1, 1)
                )
            else:
                ratio = ratio_token
                loss_old = group_cfm_loss(
                    b.cfm_loss_old_token, cfg.cfm_average_group_size
                )
                loss_new = group_cfm_loss(
                    cfm_loss_new_token, cfg.cfm_average_group_size
                )
            self.cfm_above_clamp_frac = (
                (loss_new > cfg.cfm_clamp).float().mean().detach()
            )
            self.cfm_both_clamped_frac = clamp_saturation(
                loss_old, loss_new, cfg
            ).detach()
            delta = loss_old - loss_new
            adv_ratio = adv.reshape(-1, *([1] * (delta.dim() - 1)))
            surrogate = compute_aspo_surrogate(ratio, adv_ratio, cfg)
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
            v_star_f=v_star_f,
        )

    def value_loss(
        self, b: Batch, device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with autocast_context(device):
            chunk_exec_frac = self.exec_length_frac((~b.action_is_pad).sum(dim=-1))
            b_actions_squashed = squash_to_env_bounds(b.actions, self.action_low, self.action_high)
            # Zero out padded steps so critic doesn't condition on arbitrary values.
            pad_mask = (~b.action_is_pad).unsqueeze(-1).float()
            b_actions_squashed = b_actions_squashed * pad_mask
            raw_value_pred = self.critic(b.states, b_actions_squashed, chunk_exec_frac).float()  # [num_critics, n]
        raw_returns_flat = b.returns.reshape(-1)
        if raw_value_pred.shape[-1] != raw_returns_flat.shape[0]:
            raise ValueError(
                f"value/return shape mismatch: raw_value_pred="
                f"{tuple(raw_value_pred.shape)}, raw_returns="
                f"{tuple(raw_returns_flat.shape)}"
            )
        value_loss_per_chunk = compute_value_loss(
            raw_value_pred, raw_returns_flat
        ).mean(dim=0)
        return value_loss_per_chunk, raw_value_pred, raw_returns_flat

# LOSS
    def loss_and_info(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str,
        check_finite: bool,
        *,
        compute_info: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        """The shared body. ``compute_info`` picks the metrics pass."""
        cfg = self.config
        device = next(self.actor.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        self.probe_counter += 1
        probe_on = (
            cfg.temporal_probe_every > 0
            and self.probe_counter % cfg.temporal_probe_every == 0
            and torch.is_grad_enabled()
        )

        b = self.load_batch(batch, device, storage_dtype, grad_inputs=probe_on)

        t0 = time.perf_counter()
        with autocast_context(device):
            v_new = self.compute_predicted_velocity(
                b.x_t, b.actor_t, b.condition_flat
            ).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        actor_s = time.perf_counter() - t0

        cond_stats = None
        if compute_info:
            with torch.inference_mode(), autocast_context(device):
                probe = max(1, cfg.hidden_dim // cfg.action_horizon)
                _, cond_stats = self.actor.forward_with_stats(
                    b.x_t[:probe],
                    b.actor_t[:probe],
                    b.condition_flat[:probe],
                )

        terms = self.policy_loss(b, v_new, device)
        t0 = time.perf_counter()
        value_loss_per_chunk, raw_value_pred, raw_returns_flat = self.value_loss(
            b, device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        critic_s = time.perf_counter() - t0

        sensitivity = (
            self.temporal_sensitivity(raw_value_pred, b) if probe_on else {}
        )

        if reduction == "mean":
            policy_loss = terms.policy_per_chunk.mean()
            value_loss = value_loss_per_chunk.mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss
        elif reduction == "none":
            loss = terms.policy_per_chunk + cfg.value_loss_coef * value_loss_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        if check_finite and not bool(torch.isfinite(loss).all()):
            raise MyfpoAborted(
                f"Non-finite loss: "
                f"policy_loss={float(terms.policy_per_chunk.mean().detach())}, "
                f"value_loss={float(value_loss_per_chunk.mean().detach())}"
            )

        if not compute_info:
            return loss, None

        info = compute_policy_info(
            above_clamp_frac=self.cfm_above_clamp_frac,
            both_clamped_frac=self.cfm_both_clamped_frac,
            cfm_loss_old=b.cfm_loss_old,
            cfm_loss_new=terms.cfm_loss_new,
            ratio=terms.ratio,
            ratio_token=terms.ratio_token,
            delta=terms.delta,
            action_is_pad=b.action_is_pad,
            v_star_f=terms.v_star_f,
            raw_value_pred=raw_value_pred,
            raw_returns_flat=raw_returns_flat,
            policy_per_chunk=terms.policy_per_chunk,
            value_loss_per_chunk=value_loss_per_chunk,
            cond_stats=cond_stats,
            cfg=cfg,
        )
        self.actor_step_s = actor_s
        self.critic_step_s = critic_s
        info.update(sensitivity)
        return loss, info

    def temporal_sensitivity(
        self, raw_value_pred: torch.Tensor, b: Batch
    ) -> dict[str, float]:
        """Diagnostic: Q gradient wrt chunk tokens / obs frames.
        Measures temporal filter quality; prices full plan, not executed prefix."""
        if b.actions.grad_fn is None and b.actions.requires_grad:
            import warnings
            warnings.warn(
                "temporal_sensitivity: b.actions requires_grad=True but grad_fn is None; "
                "autograd graph is dead. Check backward() was not called before this probe.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        out = {}
        grads = torch.autograd.grad(
            raw_value_pred.sum(), b.actions, retain_graph=True
        )[0]
        for i, value in enumerate(grads.norm(dim=-1).mean(dim=0).tolist()):
            out[f"critic_action_grad_norm_pos{i}"] = value
        grads = torch.autograd.grad(
            raw_value_pred.sum(), b.states, retain_graph=True
        )[0]
        for j, value in enumerate(grads.norm(dim=-1).mean(dim=0).tolist()):
            out[f"obs_frame_grad_norm_pos{j}"] = value
        return out



