"""myfpo policy. The class holds the flow head, the state conditioner, and the
EMA shadow used for evaluation.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import dataclasses

import numpy as np
import torch

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
    mc_draws,
    sample_action,
    sample_noise,
    velocity_to_target_param,
)
from myfpo.metrics import (
    compute_policy_info,
    compute_value_loss,
)
from myfpo.critic import Critic
from myfpo.encoder import RunningMeanStd, StateEncoder
from myfpo.flow import MyfpoFlowModel, MyfpoFlowResMLP
from myfpo.processor import make_pre_post_processors
from myfpo.rollout import RolloutBuffers, run_rollout
from myfpo.utils import (
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
    cfg = policy.config
    kwargs = dataclasses.asdict(policy.config.get_optimizer_preset())
    kwargs.pop("grad_clip_norm", None)  # AdamW would clip all groups as one. See split_grad_params.
    kwargs["betas"] = policy.config.adam_betas
    if next(policy.parameters()).is_cuda:
        kwargs["fused"] = True
    groups = [
        *policy.get_actor_optim_params()["params"],
        *policy.get_critic_optim_params()["params"],
    ]
    optimizer = torch.optim.AdamW(groups, **kwargs)
    warmups = [cfg.actor_lr_warmup_steps] * len(
        policy.get_actor_optim_params()["params"]
    ) + [cfg.critic_lr_warmup_steps] * len(
        policy.get_critic_optim_params()["params"]
    )
    for group, warmup in zip(optimizer.param_groups, warmups):
        group["lr_warmup_steps"] = int(warmup)
        group["base_lr"] = float(group["lr"])
    return optimizer




class MyfpoPolicy(PreTrainedPolicy):
    config_class = MyfpoConfig
    name = "myfpo"

    def __init__(
        self, config: MyfpoConfig, dataset_stats: dict[str, Any] | None = None
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        obs_dim = int(config.robot_state_feature.shape[0])
        action_dim = int(config.action_feature.shape[0])

        if config.actor_arch == "resmlp":
            self.actor = MyfpoFlowResMLP(
                action_dim,
                config.action_horizon,
                config.hidden_dim,
                config.resmlp_blocks,
                config.resmlp_units,
                config.resmlp_hidden_mult,
                config.dit_heads,
                config.time_embed_dim,
            )
        else:
            self.actor = MyfpoFlowModel(
                action_dim,
                config.action_horizon,
                config.hidden_dim,
                config.dit_depth,
                config.dit_heads,
                config.dit_mlp_ratio,
                config.time_embed_dim,
            )
        self.encoder = StateEncoder(obs_dim, config.hidden_dim, config.n_obs_steps)
        critic_width = (
            config.critic_hidden_dim
            if config.critic_hidden_dim is not None
            else config.hidden_dim
        )
        self.critic = Critic(
            obs_dim,
            critic_width,
            action_dim,
            config.n_obs_steps,
            config.action_horizon,
            config.num_critics,
            condition_on_exec_length=config.critic_condition_on_exec_length,
            num_layers=config.critic_num_layers,
            condition_dim=config.hidden_dim if config.critic_share_encoder else None,
            dropout=config.critic_dropout,
        )
        self.obs_normalizer = RunningMeanStd(obs_dim)

        self.register_buffer(
            "action_low", torch.full((action_dim,), -1.0), persistent=False
        )
        self.register_buffer(
            "action_high", torch.full((action_dim,), 1.0), persistent=False
        )

        self._ema_shadow: dict[str, torch.Tensor] = {}
        self._ema_initialised = False
        self.rollout_buffers = RolloutBuffers()

        self.cfm_above_clamp_frac = torch.zeros(())
        self.cfm_both_clamped_frac = torch.zeros(())

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        self.preprocessor, self.postprocessor = make_pre_post_processors(self)
        del dataset_stats

    def reset(self):
        """Drop the cached obs and action buffers."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

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

    def _ema_named_params(self) -> dict[str, torch.nn.Parameter]:
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

    def _ema_foreach(
        self, values: dict[str, torch.nn.Parameter]
    ) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
        """Names, live shadow tensors, detached value tensors."""
        names = list(self._ema_shadow)
        shadow = [self._ema_shadow[name] for name in names]
        return names, shadow, [values[name].detach() for name in names]

    @torch.inference_mode()
    def update_ema(self, global_step: int) -> None:
        """Update the actor EMA after an optimizer step."""
        decay = self.config.ema_decay
        if decay <= 0:
            return
        current = self._ema_named_params()
        if not self._ema_shadow:
            self._ema_shadow = {
                name: param.detach().clone() for name, param in current.items()
            }
        if not self._ema_initialised and global_step >= self.config.ema_warmup_steps:
            _, shadow, values = self._ema_foreach(current)
            torch._foreach_copy_(shadow, values)
            self._ema_initialised = True
            return
        if global_step > self.config.ema_warmup_steps:
            _, shadow, values = self._ema_foreach(current)
            torch._foreach_mul_(shadow, decay)
            torch._foreach_add_(shadow, values, alpha=1.0 - decay)

    def ema_ready(self, global_step: int) -> bool:
        """Can the EMA shadow be used for evaluation yet?"""
        return (
            self.config.ema_decay > 0
            and self._ema_initialised
            and global_step > self.config.ema_warmup_steps
        )

    @torch.no_grad()
    def swap_in_ema(self, global_step: int) -> dict[str, torch.Tensor]:
        """Copy the EMA shadow in, and return the displaced live weights."""
        if not self.ema_ready(global_step):
            return {}
        params = self._ema_named_params()
        names, shadow, values = self._ema_foreach(params)
        displaced = {name: value.clone() for name, value in zip(names, values)}
        torch._foreach_copy_(values, shadow)
        return displaced

    @torch.no_grad()
    def swap_out_ema(self, displaced: dict[str, torch.Tensor]) -> None:
        """Put back the weights :meth:`swap_in_ema` displaced."""
        if not displaced:
            return
        params = self._ema_named_params()
        names = list(displaced)
        torch._foreach_copy_(
            [params[name] for name in names], [displaced[name] for name in names]
        )

    def ema_state_dict(self) -> dict:
        """Export the EMA shadow: decay, init flag, params."""
        return {
            "decay": self.config.ema_decay,
            "initialised": self._ema_initialised,
            "shadow_params": self._ema_shadow,
        }

    def load_ema_state_dict(self, state: dict) -> bool:
        """Restore the shadow. Return False if the function re-seeded it instead."""
        stored = state.get("shadow_params", {})
        current = self._ema_named_params()
        if set(stored) != set(current):
            self._ema_shadow = {}
            self._ema_initialised = False
            return False
        # The state is loaded onto the CPU, so the shadow has to follow the
        # live parameters onto their device, or the EMA update mixes them.
        self._ema_shadow = {
            name: value.detach().clone().to(current[name].device)
            for name, value in stored.items()
        }
        self._ema_initialised = bool(state.get("initialised", True))
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
        actor = self.actor

        with torch.inference_mode(), autocast_context(device):
            condition = cudagraph_safe(self.encoder(states))
            shape = (states.shape[0], horizon, action_dim)
            if start_from_zero:
                eps = torch.zeros(*shape, device=device)
                num_steps = cfg.num_steps_zero_start
            else:
                eps = sample_noise(shape, cfg, device, None)
                num_steps = cfg.num_steps_noise_start
            action_chunk = sample_action(
                lambda x, t, c: clip_velocity(
                    actor(x, t, c), cfg.transported_clip_value
                ),
                shape,
                condition,
                num_steps=num_steps,
                eps=eps,
                device=device,
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
            squashed = self.postprocessor({ACTION: chunk})[ACTION]
            for i in range(n_execute):
                self._action_buffer.append(squashed[0, i])
        return self._action_buffer.popleft()


    def critic_inputs(
        self, normalised_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """What the critic reads: the obs window, or the shared encoder's vector."""
        return condition if self.config.critic_share_encoder else normalised_states

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        x_t, actor_t, flow_shape, num_mc = mc_draws(actions, eps, t)
        with torch.inference_mode(), autocast_context(actions.device):
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
            return compute_cfm_loss(
                v_old, v_star, cfg, "sample_and_token", loss_mask, cfm_loss_weight(t, cfg)
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
    def _load(
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
        """Compute the ASPO policy loss and the MSE value loss.

        Seven phases, in order:
          1. load and shape-check the batch
          2. encode the observation window
          3. load the frozen behavior CFM loss from the rollout
          4. the new velocity, and the flow targets under no autocast
          5. ratio, ASPO surrogate, chunk-masked policy loss
          6. the critic and the value loss
          7. assemble, then optionally raise on a non-finite loss

        Use :meth:`loss_with_info` for the diagnostics; they cost a second
        forward pass, so they are not computed here.
        """
        loss, _ = self._loss_and_info(
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
        loss, info = self._loss_and_info(
            batch, reduction, check_finite, compute_info=True
        )
        return loss, info

    def _loss_and_info(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str,
        check_finite: bool,
        *,
        compute_info: bool,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        """The shared body. ``compute_info`` picks the metrics pass."""
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        states = self._load(batch, OBS_STATE, device, storage_dtype, async_=True)
        # The action is the flow's raw output, and it is the CFM regression
        # target. It stays fp32: the raw values are unbounded, so a bf16
        # rounding would put a 0.4% error on the target the flow is fit to.
        actions = self._load(batch, ACTION, device, torch.float32, async_=True)
        normalised_advantages = self._load(
            batch, "normalised_advantages", device, storage_dtype
        )
        raw_returns = self._load(batch, "raw_returns", device, torch.float32)
        action_is_pad = self._load(
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

        x_t_flat, actor_t, flow_shape, num_mc = mc_draws(actions, eps, t)

        with autocast_context(device):
            cond = cudagraph_safe(self.encoder(states))
            cond_flat = broadcast_condition(cond, num_mc)
        if "behavior_cfm_loss" not in batch or "behavior_cfm_loss_token" not in batch:
            raise KeyError(
                "forward requires 'behavior_cfm_loss' and 'behavior_cfm_loss_token' "
                "from generate_rollout. They are computed once per rollout against "
                "the frozen snapshot and must not be recomputed here."
            )
        cfm_loss_old = to_tensor(
            batch["behavior_cfm_loss"], dtype=torch.float32, device=device
        )
        cfm_loss_old_token = to_tensor(
            batch["behavior_cfm_loss_token"], dtype=torch.float32, device=device
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

        with autocast_context(device):
            v_new = self.compute_predicted_velocity(
                x_t_flat, actor_t, cond_flat
            ).float()

        cond_stats = None
        if compute_info:
            with torch.inference_mode(), autocast_context(device):
                probe = max(1, cfg.hidden_dim // cfg.action_horizon)
                _, cond_stats = self.actor.forward_with_stats(
                    x_t_flat[:probe],
                    actor_t[:probe],
                    cond_flat[:probe],
                )

        with torch.autocast(device_type=device.type, enabled=False):
            v_new_r_f = velocity_to_target_param(
                v_new.reshape(n, num_mc, *flow_shape),
                x_t_flat.reshape(n, num_mc, *flow_shape),
                t,
                cfg,
            )
            v_star_f = cfm_target(actions, eps, cfg).float()
            loss_mask = action_is_pad

            adv = normalised_advantages.float()
            adv_mc = adv.unsqueeze(1)
            cfm_loss_new, cfm_loss_new_token = compute_cfm_loss(
                v_new_r_f, v_star_f, cfg, "sample_and_token", loss_mask,
                cfm_loss_weight(t, cfg),
            )
            ratio_token = compute_policy_ratio(
                cfm_loss_old_token, cfm_loss_new_token, cfg,
                advantage=adv_mc.unsqueeze(-1),
            )

            if cfg.do_average_cfm_loss_in_chunk:
                loss_old, loss_new = cfm_loss_old, cfm_loss_new
            else:
                loss_old, loss_new = cfm_loss_old_token, cfm_loss_new_token
            loss_old = group_cfm_loss(loss_old, cfg.cfm_average_group_size)
            loss_new = group_cfm_loss(loss_new, cfg.cfm_average_group_size)
            self.cfm_above_clamp_frac = (
                (loss_new > cfg.cfm_clamp).float().mean().detach()
            )
            self.cfm_both_clamped_frac = clamp_saturation(
                loss_old, loss_new, cfg
            ).detach()
            delta = loss_old - loss_new
            adv_ratio = adv.reshape(-1, *([1] * (delta.dim() - 1)))
            ratio = (
                ratio_token
                if loss_old is cfm_loss_old_token
                else compute_policy_ratio(loss_old, loss_new, cfg, advantage=adv_ratio)
            )
            surrogate = compute_aspo_surrogate(ratio, adv_ratio, cfg)
            if surrogate.dim() == 3:
                keep = (~action_is_pad).to(surrogate.dtype)
                surrogate = surrogate * keep[:, None, :]
                n_executed = keep.sum(dim=-1).clamp_min(1.0)
                policy_per_chunk = -(
                    surrogate.sum(dim=(1, 2)) / (n_executed * surrogate.shape[1])
                )
            else:
                policy_per_chunk = -surrogate.flatten(1).mean(dim=1)

        with autocast_context(device):
            chunk_exec_frac = self.exec_length_frac((~action_is_pad).sum(dim=-1))
            critic_in = self.critic_inputs(states, cond)
            raw_value_pred = self.critic(critic_in, actions, chunk_exec_frac).float()  # [num_critics, n]
        raw_returns_flat = raw_returns.reshape(-1)
        if raw_value_pred.shape[-1] != raw_returns_flat.shape[0]:
            raise ValueError(
                f"value/return shape mismatch: raw_value_pred="
                f"{tuple(raw_value_pred.shape)}, raw_returns="
                f"{tuple(raw_returns_flat.shape)}"
            )
        value_loss_per_chunk = compute_value_loss(
            raw_value_pred, raw_returns_flat
        ).mean(dim=0)

        if reduction == "mean":
            policy_loss = policy_per_chunk.mean()
            value_loss = value_loss_per_chunk.mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss
        elif reduction == "none":
            loss = policy_per_chunk + cfg.value_loss_coef * value_loss_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        if check_finite and not bool(torch.isfinite(loss).all()):
            raise MyfpoAborted(
                f"Non-finite loss: "
                f"policy_loss={float(policy_per_chunk.mean().detach())}, "
                f"value_loss={float(value_loss_per_chunk.mean().detach())}"
            )

        if not compute_info:
            return loss, None

        info = compute_policy_info(
            above_clamp_frac=self.cfm_above_clamp_frac,
            both_clamped_frac=self.cfm_both_clamped_frac,
            cfm_loss_old=cfm_loss_old,
            cfm_loss_new=cfm_loss_new,
            ratio=ratio,
            ratio_token=ratio_token,
            delta=delta,
            action_is_pad=action_is_pad,
            v_star_f=v_star_f,
            raw_value_pred=raw_value_pred,
            raw_returns_flat=raw_returns_flat,
            policy_per_chunk=policy_per_chunk,
            value_loss_per_chunk=value_loss_per_chunk,
            cond_stats=cond_stats,
            cfg=cfg,
        )
        return loss, info



