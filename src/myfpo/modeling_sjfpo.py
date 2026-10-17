"""SJFPO policy. The class holds the flow head, the state conditioner, and the
critic.

Batch contract. The states are ``[B, T, obs]``. The chunks are
``[B, H, action_dim]``. The GAE advantages and the returns come frozen from the
rollout.

Diagnostics batch into one device-to-host transfer. ``float()`` on a device
tensor is a blocking sync.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import dataclasses

import einx
import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, OBS_STATE

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.collector import (
    RolloutBuffer,
    collect_chunks_vec,
    resolve_action_bounds,
    scale_and_clip_chunk,
)
from myfpo.fpo_core import (
    cfm_loss_weight,
    group_cfm_loss,
    velocity_to_target_param,
    cfm_target,
    clamp_saturation,
    compute_aspo_surrogate,
    compute_cfm_loss,
    clip_velocity,
    compute_policy_ratio,
    sample_action,
    sample_noise,
    sigreg_decorr_loss,
    sigreg_weak_loss,
)
from myfpo.gae import compute_chunk_advantages
from myfpo.metrics import (
    broadcast_condition,
    compute_cfm_metrics,
    compute_gae_diagnostics,
    compute_ratio_metrics,
    compute_value_loss,
    compute_value_metrics,
    mc_draws,
    sample_eps,
    sample_mc_timesteps,
)
from myfpo.critic import Critic
from myfpo.encoder import RunningMeanStd, StateEncoder
from myfpo.flow import SJFPOFlowModel, SJFPOFlowResMLP
from myfpo.processor_sjfpo import make_sjfpo_pre_post_processors
from myfpo.utils import (
    autocast_context,
    resolve_storage_dtype,
    to_device_async,
    to_tensor_1d,
)

class SJFPOAborted(RuntimeError):
    """Non-finite training loss. The message names the offending term."""




def build_optimizer(policy: SJFPOPolicy) -> torch.optim.Optimizer:
    """Build one AdamW. Split the actor and the critic into separate groups.
    The layout comes from rsl-rl.

    The groups split per role. So each role keeps its own learning rate and its
    own warmup. There is still a single optimizer. The clip is joint over both
    roles: ``clip_grad_norm_(optimizer.param_groups)``.

    FPO++ does the same (fpo.py). One optimizer. One ``max_grad_norm`` over
    every parameter.
    """
    cfg = policy.config
    kwargs = dataclasses.asdict(policy.config.get_optimizer_preset())
    kwargs.pop("grad_clip_norm", None)  # Clipping is joint. The caller applies it.
    kwargs["betas"] = policy.config.adam_betas
    if next(policy.parameters()).is_cuda:
        kwargs["fused"] = True
    groups = [
        *policy.get_actor_optim_params()["params"],
        *policy.get_critic_optim_params()["params"],
    ]
    optimizer = torch.optim.AdamW(groups, **kwargs)
    # Record each group's own warmup and its pre-warmup lr, so the schedule is
    # recomputed from the bases rather than compounded in place.
    warmups = [cfg.actor_lr_warmup_steps] * len(
        policy.get_actor_optim_params()["params"]
    ) + [cfg.critic_lr_warmup_steps] * len(
        policy.get_critic_optim_params()["params"]
    )
    for group, warmup in zip(optimizer.param_groups, warmups):
        group["lr_warmup_steps"] = int(warmup)
        group["base_lr"] = float(group["lr"])
    return optimizer




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

        if config.actor_arch == "resmlp":
            self.actor = SJFPOFlowResMLP(
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
            self.actor = SJFPOFlowModel(
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
        self._rollout_buffer: RolloutBuffer | None = None
        self._device_rollout_buffer: RolloutBuffer | None = None
        self._rollout_h2d_event: torch.cuda.Event | None = None

        self.cfm_above_clamp_frac = torch.zeros(())
        self.cfm_both_clamped_frac = torch.zeros(())
        self.action_pre_clamp_frac = torch.zeros(())

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        self.preprocessor, self.postprocessor = make_sjfpo_pre_post_processors(self)
        del dataset_stats

    def reset(self):
        """Drop the cached obs and action buffers."""
        self._frame_buffer.clear()
        self._action_buffer.clear()

    @torch.inference_mode()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        """Update live stats from raw post-step observations.

        ``ValueError`` unless shape is ``[N, obs_dim]``.
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

        ``ValueError`` on shape mismatch or when ``high <= low``.
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
            raise ValueError("action_high must be strictly greater than action_low")

        self.action_low.copy_(low_t)
        self.action_high.copy_(high_t)
        self.postprocessor.set_action_bounds(self.action_low, self.action_high)

    def _ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        """List every parameter that the EMA shadow covers. The names carry the
        module prefix.

        The list holds the actor and the encoder together. The two drift as one.
        The encoder has no loss of its own. It gets a gradient only from the CFM
        loss of the actor. Its output reaches the actor through adaLN. It does
        not reach the actor through attention.

        A swap of one of the two without the other evaluates a pairing that never
        existed during training.
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
        """Names, live shadow tensors, detached value tensors.

        The shadow list is live memory, never a copy. So the caller writes
        through it.
        """
        names = list(self._ema_shadow)
        shadow = [self._ema_shadow[name] for name in names]
        return names, shadow, [values[name].detach() for name in names]

    @torch.inference_mode()
    def update_ema(self, global_step: int) -> None:
        """Update the actor EMA after an optimizer step.

        The code seeds the shadow on the first call. It hard-copies the shadow
        at the end of warmup. Only warmed weights enter the average.
        """
        decay = self.config.ema_decay
        if decay <= 0:
            return
        current = self._ema_named_params()
        if not self._ema_shadow:
            self._ema_shadow = {
                name: param.detach().clone() for name, param in current.items()
            }
        if not self._ema_initialised and global_step >= self.config.ema_warmup_steps:
            # The gate makes the hard copy one-shot.
            # global_step moves by a whole rollout, so an ==
            # ema_warmup_steps test never fires.
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
        """Copy the EMA shadow in, and return the displaced live weights.

        The caller hands them back to :meth:`swap_out_ema`. The function is a
        no-op that returns ``{}`` until the shadow is warm. It takes
        ``global_step``, not ``step_count``, because the shadow warms on the
        env-step clock.
        """
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
        """Restore the shadow. Return False if the function re-seeded it instead.

        A shadow written before the encoder was covered names only the actor
        parameters. Applying half of it would restore the actor and encoder
        mismatch that this function closes. So the function drops a mismatched
        shadow. The caller then reseeds from the live weights.
        """
        stored = state.get("shadow_params", {})
        if set(stored) != set(self._ema_named_params()):
            self._ema_shadow = {}
            self._ema_initialised = False
            return False
        self._ema_shadow = {
            name: value.detach().clone() for name, value in stored.items()
        }
        self._ema_initialised = bool(state.get("initialised", True))
        return True

    def get_actor_optim_params(self) -> dict:
        """Actor optimizer param groups.

        adaLN modulation and the output layer skip weight decay.
        The conditioner shares the policy lr.
        """
        decay, no_decay = [], []
        for name, p in self.actor.named_parameters():
            # torch.compile prefixes every name with "_orig_mod.".
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
        """Critic groups at critic lr, no weight decay.

        Decay pulls the value head toward zero, which reads
        as a drifting value scale.
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
        """Return one group over all the trainable weights.

        The base class requires this. Training does not use it. Training uses
        :func:`build_optimizer` so the clipping stays per role.
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


    def compute_predicted_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Actor on a flow point.

        ``condition`` arrives already broadcast over the MC draws, so one
        conditioning pass covers every draw.
        """
        cfg = self.config
        return clip_velocity(self.actor(x_t, t, condition), cfg.transported_clip_value)


    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Predict ``[B, H, action_dim]``.

        The value ``start_from_zero`` starts at ``eps=0`` and runs
        ``num_steps_zero_start``. Without it, the call starts from sampled noise
        and runs ``num_steps_noise_start``. Both runs use the same Euler loop.
        The two counts differ only in the start point.

        Perturbation runs before the postprocessing. This matches FPO++. So the
        noise lands in the normalised action space.
        """
        start_from_zero = kwargs.get("start_from_zero", True)
        cfg = self.config
        device = next(self.actor.parameters()).device

        # [B, T, obs] passes through. A [T, obs] window gains the batch axis.
        # A bare [obs] frame belongs to select_action.
        states = to_tensor(batch[OBS_STATE])
        states = states.to(dtype=resolve_storage_dtype(states.device))
        if states.dim() == 2 and states.shape[0] == cfg.n_obs_steps:
            states = einx.id("t d -> 1 t d", states)
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
            condition = self.encoder(states)
            shape = (states.shape[0], horizon, action_dim)
            if start_from_zero:
                eps = torch.zeros(*shape, device=device)
                num_steps = cfg.num_steps_zero_start
            else:
                eps = sample_noise(shape, cfg, device, None)
                num_steps = cfg.num_steps_noise_start
            # The chain runs t=0 (noise) to t=1 (action), so this is the END of
            # the chain. The old name x0 read as the START.
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

        scaled = action_chunk.clamp(-1.0, 1.0)
        # The share of coordinates this clamp had to move. This is the
        # projection the CFM target is built on, so it belongs in the log.
        # ``chunk_clip_counts`` cannot see it: the collector's
        # ``scale_and_clip_chunk`` receives a value that is already inside the
        # env bounds, so that count is structurally zero.
        #
        # The value is the last proposal of the rollout, not a rollout average.
        # It is a 0-dim tensor, so reading it costs the caller one sync.
        self.action_pre_clamp_frac = (action_chunk.abs() > 1.0).float().mean().detach()

        if self.training and cfg.action_perturb_std > 0:
            scaled = scaled + cfg.action_perturb_std * torch.randn_like(scaled)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Next action, receding through a cached chunk.

        ``batch`` holds ``observation.state``. That value is ``[obs]`` for one
        frame or ``[T, obs]`` for a window. Kwargs pass to
        :meth:`predict_action_chunk`.

        The buffer is FIFO. The left value is next out. The code appends the
        chunk in plan order: ``chunk[0, 0]`` first, ``chunk[0, n_execute - 1]``
        last. The collector uses the direct index ``plan[..., pos_t]``, so this
        buffer only serves eval and play.
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
            for i in range(n_execute):
                self._action_buffer.append(chunk[0, i])
        return self._action_buffer.popleft()


    def critic_inputs(
        self, normalised_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """What the critic reads: the obs window, or the shared encoder's vector.

        With ``critic_share_encoder`` the critic is fed the same embedding the
        actor conditions on, which is FPO++'s arrangement
        (``finetune_online_rl.py:1177``). Off by default, so the critic keeps its
        own view of the observation.
        """
        return condition if self.config.critic_share_encoder else normalised_states

    def exec_length_frac(self, executed_steps: torch.Tensor) -> torch.Tensor:
        """Normalise an executed-step count to ``[0, 1]``.

        The function always returns, even when the caller does not use the
        value. So a caller never branches on the flag.
        """
        return executed_steps.to(torch.float32) / float(self.config.action_horizon)

    def compute_behavior_cfm_losses(
        self,
        normalised_states: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The CFM loss of the behavior policy. That is the ``cfm_loss_old`` of
        the ratio.

        The behavior policy is the live one. This function runs inside
        ``generate_rollout``, and no optimizer step runs until the rollout
        returns. So the live weights are already frozen. A second copy of the
        actor would hold the same numbers.

        The code masks both reductions in the same way as the live losses. So the
        ratio never mixes means over different positions.
        """
        cfg = self.config
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
            # Same target as the live pass, so the ratio compares like with like.
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
        """Roll out open-loop chunks, build a training batch.

        See :mod:`myfpo.gae`. A chunk can run one step and then terminate. So a
        count can reach one.

        The batch carries one raw post-step frame per executed step. The frames
        stay separate from the overlapping obs windows. So the normalizer does
        not count a frame twice.

        The batch also carries the per-chunk and per-MC CFM draws. The policy
        loss reuses those draws for the old value and for the new value.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        def propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(states, device)}, start_from_zero=not noisy
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

        def make_buffer(**extra) -> RolloutBuffer:
            """Host and device buffers share every shape argument."""
            return RolloutBuffer(
                max_chunks=max_chunks,
                max_steps=total_steps,
                obs_steps=cfg.n_obs_steps,
                obs_dim=obs_dim,
                action_horizon=cfg.action_horizon,
                action_dim=action_dim,
                **extra,
            )

        if (
            self._rollout_buffer is None
            or self._rollout_buffer.shape_key() != buffer_key
        ):
            self._rollout_buffer = make_buffer(pin_memory=device.type == "cuda")
        buffer = self._rollout_buffer
        buffer.reset()

        collect_kwargs = dict(
            total_steps=total_steps,
            action_low=action_low,
            action_high=action_high,
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.n_action_steps,
            buffer=buffer,
        )
        # A single env becomes a 1-env vec env. One collector, so the chunk
        # contract cannot drift between the n==1 and n>1 paths.
        if num_envs == 1:
            import gymnasium as gym

            envs = gym.vector.SyncVectorEnv([lambda: env])
            try:
                collect_chunks_vec(envs, propose_batch, **collect_kwargs)
            finally:
                envs.close()
        else:
            collect_chunks_vec(env, propose_batch, **collect_kwargs)

        if device.type == "cuda":
            if (
                self._device_rollout_buffer is None
                or self._device_rollout_buffer.shape_key() != buffer.shape_key()
            ):
                self._device_rollout_buffer = make_buffer(device=device)
            buffer.copy_to(self._device_rollout_buffer, non_blocking=True)
            self._rollout_h2d_event = torch.cuda.Event()
            self._rollout_h2d_event.record()
            active = self._device_rollout_buffer.active()
        else:
            active = buffer.active()
        n_chunks = buffer.n_chunks
        horizon = cfg.action_horizon

        lengths = active["chunk_lengths"]
        chunk_starts = active["chunk_starts"]
        all_rewards = active["step_rewards"]
        actions = active["chunk_actions"]
        states = active["chunk_state"]

        # Two obs windows per chunk, the start and the end. The preprocessor
        # sees 2 * n_chunks, not 2 * n_steps.
        # The normalizer updates from normalizer_observations.
        # The bootstrap a' is drawn fresh at the end state of each chunk.
        # That end state is the target of QC. A terminated row keeps the chunk.
        with torch.inference_mode():
            next_states = active["chunk_next_state"]
            # Seeded from the chunk, so no row is garbage.
            # The backup scales by ~terminated. 0 * NaN is NaN.
            boot_actions = actions.clone()
            # Only a live chunk needs a proposal. A terminated chunk keeps the
            # cloned chunk actions. Nothing reads those values. The backup masks
            # them with (1 - terminated).
            needs_proposal = ~active["chunk_terminated"]
            if bool(needs_proposal.any()):
                # The call predict_action_chunk returns the env bounds. The
                # tensor chunk_actions is scaled to [-1, 1]. The critic reads
                # the scaled form. So the proposal takes the same affine
                # transform.
                proposed = self.predict_action_chunk(
                    {OBS_STATE: to_device_async(next_states[needs_proposal], device)},
                    start_from_zero=False,
                )
                low_t, range_t = resolve_action_bounds(action_low, action_high)
                boot_actions[needs_proposal] = scale_and_clip_chunk(
                    proposed, low_t.to(proposed.device), range_t.to(proposed.device)
                )[1]

            all_chunk_normalised_states = self.preprocessor(
                {OBS_STATE: torch.cat([states, next_states], dim=0)}
            )[OBS_STATE]
            normalised_states = all_chunk_normalised_states[:n_chunks]
            normalised_next_states = all_chunk_normalised_states[n_chunks:]
            # The bootstrap chunk only ever executes n_action_steps, so its
            # fraction is n_action_steps/H, NOT H/H. Passing H/H
            # reintroduces the mismatch this feature removes.
            exec_length_frac = self.exec_length_frac(lengths)
            boot_exec_frac = self.exec_length_frac(
                torch.full_like(lengths, cfg.n_action_steps)
            )
            # amin across members. That is for the backup only, never in the
            # loss.
            # The value loss averages the same members.
            # The shared encoder runs here so the critic can read it. One pass
            # covers the chunk and its bootstrap. The weights are the weights of
            # the actor.
            critic_cond = (
                self.encoder(all_chunk_normalised_states)
                if cfg.critic_share_encoder
                else None
            )
            chunk_cond = critic_cond[:n_chunks] if critic_cond is not None else None
            next_cond = critic_cond[n_chunks:] if critic_cond is not None else None
            with autocast_context(device):
                q_current_chunk = self.critic(
                    self.critic_inputs(normalised_states, chunk_cond),
                    actions,
                    exec_length_frac,
                ).float().amin(dim=0)
            # The code is already inside the outer inference_mode. Autocast is
            # the only context this block adds.
            with autocast_context(device):
                q_bootstrap = self.critic(
                    self.critic_inputs(normalised_next_states, next_cond),
                    boot_actions,
                    boot_exec_frac,
                ).float().amin(dim=0)

        # One chunk is one macro-transition. Sum the rewards, then discount by
        # gamma**h.
        # This is the rsl-rl space. Q predicts raw returns, and the target is the
        # raw return, so Q is the true expected return.
        raw_advantages, raw_returns = compute_chunk_advantages(
            all_rewards,
            q_current_chunk,
            q_bootstrap,
            chunk_starts,
            lengths,
            chunk_terminated=active["chunk_terminated"],
            chunk_size=cfg.action_horizon,
            gamma=cfg.gamma,
            # The code reads both values only when gae_lambda > 0. QC ignores
            # both.
            # The value chunk_truncated is the ENV limit, not the step budget.
            # The two limits are separate. So the code never reads a live
            # episode as finished.
            chunk_truncated=active["chunk_truncated"],
            chunk_env_id=active["chunk_env_id"],
            gae_lambda=cfg.gae_lambda,
        )
        normalised_advantages = raw_advantages
        # The rationale for correction=0 is in compute_gae_diagnostics.
        if cfg.normalize_advantage and raw_advantages.numel() > 0:
            normalised_advantages = (raw_advantages - raw_advantages.mean()) / (
                raw_advantages.std(correction=0).clamp_min(1e-8)
            )
        # FPO++ clamps inside the minibatch loop, right before the surrogate.
        # This code clamps here instead. The clamp is a fixed elementwise bound,
        # so one clamp per rollout gives the same numbers in one fewer pass.
        # The order is after normalisation, as theirs is.
        normalised_advantages = normalised_advantages.clamp(
            -cfg.advantage_clamp_neg, cfg.advantage_clamp_pos
        )
        gae_diagnostics = compute_gae_diagnostics(
            active["chunk_env_id"][:n_chunks],
            active["chunk_terminated"][:n_chunks],
            active["chunk_truncated"][:n_chunks],
            raw_advantages,
            raw_returns,
        )
        raw_advantages = raw_advantages.to(storage_dtype)
        normalised_advantages = normalised_advantages.to(storage_dtype)
        # raw_returns stays fp32. It is the critic target, and it is the only
        # stored tensor that is not O(1) (returns_rms runs to 35). There, the 8
        # mantissa bits of bf16 lose real precision.

        action_is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        t = sample_mc_timesteps(cfg, n_chunks, device)
        eps = sample_eps(cfg, n_chunks, action_dim, device)

        chunk_clip_counts = active["chunk_clip_counts"][:n_chunks]

        behavior_cfm_loss, behavior_cfm_loss_token = self.compute_behavior_cfm_losses(
            normalised_states, actions, t, eps, action_is_pad
        )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": action_is_pad,
            "raw_advantages": raw_advantages,
            "normalised_advantages": normalised_advantages,
            "raw_returns": raw_returns,
            "normalizer_observations": active["step_frames"].detach(),
            # Critic drift from the bootstrap value that the targets used.
            # Target critic. That is the price of the slow teacher.
            # Without one, the target's own drift.
            "boot_observations": all_chunk_normalised_states[n_chunks:].to(storage_dtype),
            "boot_actions": boot_actions.to(storage_dtype),
            "boot_values_start": q_bootstrap.unsqueeze(1),
            "mc_t": t,
            "mc_eps": eps,
            "chunk_clip_counts": chunk_clip_counts,
            "gae_diagnostics": gae_diagnostics,
            "behavior_cfm_loss": behavior_cfm_loss,
            "behavior_cfm_loss_token": behavior_cfm_loss_token,
        }


    @staticmethod
    def _load(
        batch: dict[str, torch.Tensor],
        key: str,
        device,
        dtype: torch.dtype,
        *,
        async_: bool = False,
    ) -> torch.Tensor:
        """One spelling of 'coerce to device and dtype'.

        The flag ``async_`` pins the transfer. Pinning pays only for the two
        big tensors. On a small tensor it costs time and gains nothing.
        """
        tensor = to_tensor(batch[key], dtype=dtype, device=device)
        return to_device_async(tensor, device) if async_ else tensor

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        compute_info: bool = True,
        check_finite: bool = True,
    ) -> tuple[torch.Tensor, dict | None]:
        """Compute the ASPO policy loss and the MSE value loss. Return
        ``(loss, info)``.

        One chunk is one macro-action. The tensor ``action_is_pad`` masks the
        steps that never ran. The CFM losses run in fp32 under the target
        velocity.

        The flags ``do_average_cfm_loss_in_chunk`` and
        ``cfm_average_group_size`` pick the granularity of the trust region, as
        FPO++ does. The token axis comes first. Then the MC-sample axis. The
        value ``True/-1`` is FPO Eq 9. It gives one ratio per chunk.

        The value ``compute_info=False`` skips the reductions. It returns
        ``info=None``.
        """
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        # states and actions go async. They are the two large tensors here.
        # The rest are small enough that the pinned transfer costs more.
        states = self._load(batch, OBS_STATE, device, storage_dtype, async_=True)
        actions = self._load(batch, ACTION, device, storage_dtype, async_=True)
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

        has_mc_t = "mc_t" in batch
        has_mc_eps = "mc_eps" in batch
        if has_mc_t != has_mc_eps:
            raise ValueError("mc_t and mc_eps must be provided together")
        if has_mc_t:
            t = to_tensor(batch["mc_t"], dtype=storage_dtype, device=device)
            expected_t = (n, cfg.num_mc_samples, horizon, 1)
            if tuple(t.shape) != expected_t:
                raise ValueError(
                    f"mc_t shape {tuple(t.shape)} != {expected_t}"
                )
            eps = to_tensor(batch["mc_eps"], dtype=storage_dtype, device=device)
            expected_eps = (n, cfg.num_mc_samples, horizon, action_dim)
            if tuple(eps.shape) != expected_eps:
                raise ValueError(
                    f"stored MC noise shape {tuple(eps.shape)} != {expected_eps}"
                )
        else:
            t = sample_mc_timesteps(cfg, n, device)
            eps = sample_eps(cfg, n, action_dim, device)

        x_t_flat, actor_t, flow_shape, num_mc = mc_draws(actions, eps, t)

        with autocast_context(device):
            # One encoder pass. The actor gets the MC-broadcast copy. The critic
            # gets the unbroadcast one when it shares this encoder.
            cond = self.encoder(states)
            cond_flat = broadcast_condition(cond, num_mc)
        # This is representation health, not the trust region. The value "none"
        # skips all new math. The raw value logs even at coef 0.0, for
        # calibration.
        sigreg_raw = torch.zeros((), device=device)
        if cfg.sigreg_mode == "weak":
            sigreg_raw = sigreg_weak_loss(cond.float(), cfg.sigreg_sketch_dim)
        elif cfg.sigreg_mode == "decorr":
            sigreg_raw = sigreg_decorr_loss(cond.float(), cfg.sigreg_sketch_dim)
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
                # Rank probe rows scale with token width.
                probe = max(1, cfg.hidden_dim // cfg.action_horizon)
                _, cond_stats = self.actor(
                    x_t_flat[:probe],
                    actor_t[:probe],
                    cond_flat[:probe],
                    return_stats=True,
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
            # Both ratios take the advantage. Without it, the A<0 clamp is
            # skipped.
            # The token loss is [n, num_mc, H], so the advantage needs the extra
            # axis. [n, 1] would not line up with it.
            ratio_token = compute_policy_ratio(
                cfm_loss_old_token, cfm_loss_new_token, cfg,
                advantage=adv_mc.unsqueeze(-1),
            )

            # Token axis, then MC-sample axis. ``ratio_token`` doubles as the loss
            # ratio when the token axis is kept, so the code does not recompute
            # the expensive per-token pass.
            if cfg.do_average_cfm_loss_in_chunk:
                loss_old, loss_new = cfm_loss_old, cfm_loss_new
            else:
                loss_old, loss_new = cfm_loss_old_token, cfm_loss_new_token
            loss_old = group_cfm_loss(loss_old, cfg.cfm_average_group_size)
            loss_new = group_cfm_loss(loss_new, cfg.cfm_average_group_size)
            # Measure the pair the ratio clamps, not cfm_loss_new. The two axes
            # cross, so the old reading was wrong in both directions.
            self.cfm_above_clamp_frac = (
                (loss_new > cfg.cfm_clamp).float().mean().detach()
            )
            # Measured after grouping, because these are the two tensors the
            # surviving ratio clamps. Measuring the token pair would read the
            # ungrouped level when cfm_average_group_size averages the draws.
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
                # The token axis survived. So the padded tokens are still in the
                # average. A padded token has cfm_loss 0, so its ratio is 1.0.
                # That gives a zero gradient. The surrogate still evaluates to A,
                # though. So the code masks the value, then divides by the
                # executed token count, not by the full horizon.
                keep = (~action_is_pad).to(surrogate.dtype)
                surrogate = surrogate * keep[:, None, :]
                n_executed = keep.sum(dim=-1).clamp_min(1.0)
                policy_per_chunk = -(
                    surrogate.sum(dim=(1, 2)) / (n_executed * surrogate.shape[1])
                )
            else:
                policy_per_chunk = -surrogate.flatten(1).mean(dim=1)

        with autocast_context(device):
            # The tensor action_is_pad marks the unexecuted steps. Count the
            # complement.
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
            loss = (
                policy_loss
                + cfg.value_loss_coef * value_loss
                + cfg.sigreg_coef * sigreg_raw
            )
        elif reduction == "none":
            loss = (
                policy_per_chunk
                + cfg.value_loss_coef * value_loss_per_chunk
                + cfg.sigreg_coef * sigreg_raw
            )
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
        info["sigreg_raw"] = float(sigreg_raw.detach())
        return loss, info


    @torch.inference_mode()
    def compute_info(
        self,
        *,
        cfm_loss_old,
        cfm_loss_new,
        ratio,
        ratio_token,
        delta,
        action_is_pad,
        v_star_f,
        raw_value_pred,
        raw_returns_flat,
        policy_per_chunk,
        value_loss_per_chunk,
        cond_stats,
        cfg,
    ) -> dict[str, float]:
        """Merge the metric groups into one device transfer.

        The call costs one sync, not about 20. Plain floats are accepted next
        to the tensors. NaN marks a metric that does not apply.
        """
        scalars = {
            **compute_ratio_metrics(
                ratio=ratio,
                ratio_token=ratio_token,
                delta=delta,
                action_is_pad=action_is_pad,
                cfg=cfg,
            ),
            **compute_value_metrics(
                raw_value_pred=raw_value_pred,
                raw_returns_flat=raw_returns_flat,
                policy_per_chunk=policy_per_chunk,
                value_loss_per_chunk=value_loss_per_chunk,
            ),
            **compute_cfm_metrics(
                cfm_loss_old=cfm_loss_old,
                cfm_loss_new=cfm_loss_new,
                v_star_f=v_star_f,
                above_clamp_frac=self.cfm_above_clamp_frac,
                both_clamped_frac=self.cfm_both_clamped_frac,
            ),
        }
        if cond_stats:
            scalars.update(
                {
                    f"cond_{k}": torch.as_tensor(cond_stats[k], device=ratio.device)
                    for k in cond_stats
                }
            )
        packed = [
            s.detach().reshape(())
            if isinstance(s, torch.Tensor)
            else torch.as_tensor(float(s))
            for s in scalars.values()
        ]
        return dict(zip(scalars.keys(), torch.stack(packed).float().cpu().tolist()))

