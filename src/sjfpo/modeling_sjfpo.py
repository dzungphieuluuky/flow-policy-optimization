"""SJFPO policy: flow head + state conditioner + critic.

Batch contract: ``[B, T, obs]`` states, ``[B, H, action_dim]`` action chunks,
frozen GAE advantages/returns from the rollout.

Performance notes
-----------------
* Diagnostics returned by ``forward`` are batched into a single device→host
  transfer (one ``.tolist()``) instead of ~20 ``float()`` syncs per call, and
  the whole diagnostic pass is gated on ``compute_info=True``.
* ``rho_bin`` is computed once per forward, not once per branch.
* Gradient checkpointing is off by default for shallow DiT stacks; when on,
  it actually wraps the actor call.
* ``snapshot_behavior_policy`` copies weights in-place into the frozen
  snapshot instead of ``copy.deepcopy``-ing a ``torch.compile`` wrapper.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import einx
import numpy as np
import torch
import torch.utils.checkpoint as cp

from lerobot.policies import PreTrainedPolicy
from lerobot.processor.converters import to_tensor
from lerobot.utils.constants import ACTION, OBS_STATE

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import RolloutBuffer, collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import (
    decode_coeffs_to_velocity,
    extract_dct_coeffs,
)
from sjfpo.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_mc_simulation,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo.gae import compute_step_advantages_torch
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


def _num_spectral_tokens_kept(cfg: SJFPOConfig) -> int:
    """Count the spectral tokens retained per action chunk.

    Args:
        cfg: Active configuration.

    Returns:
        ``M+1`` derived from ``top_p_spectral_tokens`` when the DCT transform
        is on, otherwise the full action horizon.
    """
    return max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon)) if cfg.use_dct_transform else cfg.action_horizon


def _sample_spectral_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw Monte-Carlo noise matching the configured token layout.

    Args:
        cfg: Active configuration.
        n: Number of chunks.
        action_dim: Feature dimension, one per action dimension.
        device: Device the noise is allocated on.

    Returns:
        Noise of shape ``[n, num_mc_samples, tokens, action_dim]``.
    """
    shape = (
        n,
        cfg.num_mc_samples,
        (
            _num_spectral_tokens_kept(cfg)
            if cfg.use_dct_transform
            else cfg.action_horizon
        ),
        action_dim,
    )
    return sample_noise(shape, cfg, device, None)


def _mc_x_t(
    flatten_mc_timestep: torch.Tensor,
    flatten_mc_noise: torch.Tensor,
    flatten_mc_action: torch.Tensor,
    flow_shape: tuple[int, ...],
) -> torch.Tensor:
    """Interpolate between noise and action to form the flow point x_t.

    Args:
        flatten_mc_timestep: Flattened timesteps, either ``[B*m]`` or
            ``[B*m, K, 1]``.
        flatten_mc_noise: Flattened noise of shape ``[B*m, d]``.
        flatten_mc_action: Flattened action of shape ``[B*m, d]``.
        flow_shape: Per-sample flow shape the result is reshaped to.

    Returns:
        The flow point, of shape ``[-1, *flow_shape]``.
    """
    if flatten_mc_timestep.dim() == 1:
        flatten_mc_timestep = flatten_mc_timestep[:, None].expand(
            -1, flatten_mc_noise.shape[-1]
        )
        flatten_mc_noise = flatten_mc_noise.reshape(flatten_mc_timestep.shape[0], -1)
        flatten_mc_action = flatten_mc_action.reshape(flatten_mc_timestep.shape[0], -1)
    else:
        flatten_mc_timestep = flatten_mc_timestep.expand(-1, -1, flow_shape[-1])
        flatten_mc_noise = flatten_mc_noise.reshape(
            flatten_mc_timestep.shape[0], flow_shape[0], flow_shape[1]
        )
        flatten_mc_action = flatten_mc_action.reshape(
            flatten_mc_timestep.shape[0], flow_shape[0], flow_shape[1]
        )
    return (
        (1.0 - flatten_mc_timestep) * flatten_mc_noise
        + flatten_mc_timestep * flatten_mc_action
    ).reshape(-1, *flow_shape)


def _actor_call(actor, x_t, tau, cond, *, use_ckpt: bool):
    """Run the actor, optionally under gradient checkpointing.

    Args:
        actor: Actor module to call.
        x_t: Flow point.
        tau: Per-token flow times.
        cond: Conditioning.
        use_ckpt: Whether to wrap the call in a checkpointed region.

    Returns:
        The actor's predicted velocity.
    """
    if use_ckpt:
        return cp.checkpoint(actor, x_t, tau, cond, use_reentrant=False)
    return actor(x_t, tau, cond)


def _unwrap_compiled(m: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a ``torch.compile`` module.

    Args:
        m: Module that may be an ``OptimizedModule``.

    Returns:
        The inner eager module, or ``m`` itself when it is not compiled.
    """
    return getattr(m, "_orig_mod", m)


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
        self.critic = Critic(obs_dim, config.hidden_dim)
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

        Args:
            mode: Training mode to switch to.

        Returns:
            This policy, so the call chains.
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

        Args:
            observations: Raw frames of shape ``[N, obs_dim]``.

        Raises:
            ValueError: If the shape is not ``[N, obs_dim]``.
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

        Args:
            low: Per-dimension lower bounds.
            high: Per-dimension upper bounds.

        Raises:
            ValueError: If the bounds disagree in shape or fail the
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
        if not bool((high_t >= low_t).all()):
            raise ValueError("action_high must be elementwise >= action_low")

        self.action_low.copy_(low_t)
        self.action_high.copy_(high_t)
        self.postprocessor.set_action_bounds(self.action_low, self.action_high)

    def snapshot_behavior_policy(self) -> None:
        """Refresh the frozen π_old snapshot in place.

        Avoids ``copy.deepcopy`` on ``torch.compile`` wrappers
        (``OptimizedModule``), which clones the FX graph + guards and forces a
        fresh compile warmup on the snapshot every rollout. Copying weights
        into the existing modules instead keeps the compiled graph intact.
        """
        actor_old = _unwrap_compiled(self.old_actor)
        actor_live = _unwrap_compiled(self.actor)
        actor_old.load_state_dict(actor_live.state_dict())

        cond_old = _unwrap_compiled(self.old_conditioner)
        cond_live = _unwrap_compiled(self.conditioner)
        cond_old.load_state_dict(cond_live.state_dict())

        self.old_actor.requires_grad_(False)
        self.old_actor.eval()
        self.old_conditioner.requires_grad_(False)
        self.old_conditioner.eval()

    @torch.inference_mode()
    def update_ema(self, global_step: int) -> None:
        """Update the FPO++ actor EMA after an optimizer update.

        The shadow is seeded on the first call and hard-copied at the end of
        warmup, so only properly warmed weights enter the average.

        Args:
            global_step: Current environment-step count, compared against
                ``ema_warmup_steps``.
        """
        decay = self.config.ema_decay
        if decay <= 0:
            return
        actor = _unwrap_compiled(self.actor)
        current = {
            name: param
            for name, param in actor.named_parameters()
            if param.requires_grad
        }
        if not self._ema_shadow:
            self._ema_shadow = {
                name: param.detach().clone() for name, param in current.items()
            }
        if global_step == self.config.ema_warmup_steps:
            for name, shadow in self._ema_shadow.items():
                shadow.copy_(current[name].detach())
            self._ema_initialized = True
            return
        if global_step > self.config.ema_warmup_steps:
            for name, shadow in self._ema_shadow.items():
                shadow.mul_(decay).add_(current[name].detach(), alpha=1.0 - decay)

    def ema_ready(self, global_step: int) -> bool:
        """Report whether the EMA shadow may now be used for evaluation.

        Args:
            global_step: Current environment-step count.

        Returns:
            True once the warmup has passed and the shadow is initialised.
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

        Args:
            state: Previously exported EMA state.
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
            short = name.split("_orig_mod.")[-1]
            if "adaLN" in short or short.startswith("out."):
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

        Retained for legacy single-optimizer checkpoints; new runs use
        :meth:`get_actor_optim_params` and :meth:`get_critic_optim_params`.

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

        Args:
            a: Action chunk.
            eps: Monte-Carlo noise of the matching shape.

        Returns:
            Target velocities of shape ``[B, m, *flow]``.
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

        ``condition`` is the state conditioner's output, already broadcast over
        the Monte-Carlo draws, so both the frozen and live paths share the one
        forward-conditioning pass their caller already performed. Gradient
        checkpointing is applied here for the live actor, matching the training
        loop's configuration.

        Args:
            x_t: Flow point, of shape ``[B, K, action_dim]``.
            tau: Per-token flow times.
            condition: Conditioning of shape ``[B, D]`` or ``[B*m, D]``.
            old: Whether to use the frozen behavior snapshot.

        Returns:
            Predicted velocities, in the actor's own dtype.
        """
        net = self.old_actor if old else self.actor
        if old or not self.config.use_gradient_checkpointing:
            return net(x_t, tau, condition)
        return _actor_call(net, x_t, tau, condition, use_ckpt=True)


    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
        """Coerce observations to the ``[B, T, obs]`` layout the policy expects.

        Args:
            states: Observations in any of the accepted layouts.

        Returns:
            A storage-dtype tensor of shape ``[B, T, obs]``.

        Raises:
            ValueError: If the rank is not 1, 2, or 3.
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

        Action perturbation is applied before postprocessing, matching FPO++,
        so the noise lands in the policy's normalized action space.

        Args:
            batch: Batch holding ``observation.state``.
            **kwargs: ``noise=True`` samples the chunk's first token; the
                default returns zeros at t=0.

        Returns:
            Action chunk of shape ``[B, H, action_dim]``.
        """
        noise = kwargs.get("noise", False)
        cfg = self.config
        device = next(self.actor.parameters()).device

        states = self.standardize_obs_tensor(batch[OBS_STATE])
        states = to_device_async(states, device)
        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        num_spectral_tokens_kept = _num_spectral_tokens_kept(cfg)
        action_dim = int(self.config.action_feature.shape[0])
        actor = _unwrap_compiled(self.actor)

        with torch.inference_mode(), autocast_context(device):
            condition = self.conditioner(states)
            shape = (states.shape[0], num_spectral_tokens_kept, action_dim)
            eps = None if noise else torch.zeros(*shape, device=device)
            num_steps = cfg.num_euler_steps if noise else cfg.num_inference_steps
            x0 = sample_action(
                lambda x, t, c: actor(x, t, c),
                shape,
                condition,
                num_inference_steps=num_steps,
                eps=eps,
                generator=None,
                device=device,
            ).float()

        if cfg.use_dct_transform:
            tau_grid = torch.linspace(
                0.0, 1.0, cfg.action_horizon, device=x0.device, dtype=torch.float32
            )
            scaled = decode_coeffs_to_velocity(
                x0, tau=tau_grid, T=1.0, K_original=cfg.action_horizon
            )[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)
        else:
            scaled = x0[:, : cfg.action_horizon, :].clamp(-1.0, 1.0)

        if self.training and cfg.action_perturb_std > 0:
            scaled = scaled + cfg.action_perturb_std * torch.randn_like(scaled)
        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the next action, receding through a cached chunk.

        A chunk is predicted once and its executed prefix is buffered, so
        consecutive calls do not re-run the actor.

        Args:
            batch: Batch holding ``observation.state``, shaped ``[obs]`` for a
                single frame or ``[T, obs]`` for a full window.
            **kwargs: Forwarded to :meth:`predict_action_chunk`.

        Returns:
            The next action to execute.

        Raises:
            ValueError: If the observation rank is neither 1 nor 2.
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


    def _initial_cfm_losses(
        self,
        normalized_states: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the initial old and new CFM losses for one minibatch.

        Args:
            normalized_states: Normalised observation window.
            actions: Action chunk in ``[-1, 1]``.
            tau: Flow times.
            eps: Monte-Carlo noise matching the token layout.

        Returns:
            Tuple of the frozen-policy and current-policy CFM losses.
        """
        cfg = self.config
        token_count = eps.shape[2]
        if cfg.use_dct_transform:
            spectral_action = extract_dct_coeffs(actions, M=token_count - 1)
        else:
            spectral_action = actions
        tau_bn, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = run_mc_simulation(
            spectral_action, eps, tau
        )
        x_t = _mc_x_t(tau_bn, flatten_mc_noise, flatten_mc_action, flow_shape)
        actor_tau = tau_bn if tau_bn.dim() == 1 else tau_bn.squeeze(-1)
        with torch.inference_mode(), autocast_context(actions.device):
            condition = self.conditioner(normalized_states)
            condition_flat = einx.id(
                "n d -> n m d", condition, m=num_mc
            ).reshape(num_mc * condition.shape[0], condition.shape[-1])
            v_initial = self.compute_predicted_velocity(
                x_t, actor_tau, condition_flat
            ).float()
        with torch.autocast(device_type=actions.device.type, enabled=False):
            v_star = (spectral_action[:, None] - eps).float()
            v_initial = v_initial.reshape(
                eps.shape[0], num_mc, *flow_shape
            )
            return (
                compute_cfm_loss(v_initial, v_star, cfg, "sample"),
                compute_cfm_loss(v_initial, v_star, cfg, "bin"),
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

        A chunk may execute only one step before termination, so the
        allocation-free upper bound on chunk count is one chunk per requested
        step. Advantages are normalised over every executed environment step,
        matching rsl_rl/FPO++ storage semantics, before the first executed
        step's step-GAE target is exposed as the chunk-level policy target.

        The batch carries one raw post-step frame per executed step, kept
        separate from the overlapping observation windows so the training loop
        can fold them into the live normalizer without double counting, plus
        independent per-chunk/Monte-Carlo CFM draws that the policy loss reuses
        for both the old and new losses.

        Args:
            env: Environment to roll out.
            total_steps: Environment-step budget for the rollout.
            action_low: Scalar lower bound of the env action space.
            action_high: Scalar upper bound of the env action space.
            noisy: Whether the policy samples rather than integrating from
                zero.

        Returns:
            A forward-ready batch of per-chunk tensors.

        Raises:
            ValueError: If ``total_steps`` is not positive, or if a collected
                transition is inconsistent.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config
        device = next(self.parameters()).device
        storage_dtype = resolve_storage_dtype(device)

        def propose(state: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(state, device)}, noise=noisy
            )[0]
            return chunk.cpu()

        def propose_batch(states: torch.Tensor) -> torch.Tensor:
            chunk = self.predict_action_chunk(
                {OBS_STATE: to_device_async(states, device)}, noise=noisy
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
            or self._rollout_buffer.capacity_key() != buffer_key
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
        num_spectral_tokens_kept = _num_spectral_tokens_kept(cfg)

        lengths = active["chunk_lengths"]
        chunk_starts = active["chunk_starts"]
        step_states = active["step_states"]
        step_next_states = active["step_next_states"]
        all_rewards = active["step_rewards"]
        actions = active["chunk_actions"]
        step_terminated = active["step_terminated"]
        step_truncated = active["step_truncated"]
        states = step_states.index_select(0, chunk_starts)

        with torch.inference_mode():
            all_step_states_n = self.preprocessor(
                {OBS_STATE: torch.cat([step_states, step_next_states], dim=0)}
            )[OBS_STATE]
            normalized_states = all_step_states_n.index_select(0, chunk_starts)
            with autocast_context(device):
                all_step_values = self.critic(all_step_states_n).float()
            total_executed = n_steps
            step_values = all_step_values[:total_executed]
            step_next_values = all_step_values[total_executed:]
            values = step_values.index_select(0, chunk_starts)

        step_advantages, step_returns = compute_step_advantages_torch(
            all_rewards,
            step_values,
            step_next_values,
            step_terminated,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            truncated=step_truncated,
        )
        if cfg.normalize_advantage and step_advantages.numel() > 0:
            step_advantages = (step_advantages - step_advantages.mean()) / (
                step_advantages.std() + 1e-8
            )
        if step_advantages.numel() > 0 and cfg.advantage_clamp is not None:
            step_advantages = step_advantages.clamp(
                -cfg.advantage_clamp, cfg.advantage_clamp
            )

        advantages = step_advantages.index_select(0, chunk_starts)
        returns = step_returns.index_select(0, chunk_starts)
        advantages = advantages.to(storage_dtype)
        returns = returns.to(storage_dtype)

        is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
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

        initial_cfm_loss, initial_cfm_loss_bin = self._initial_cfm_losses(
            normalized_states, actions, tau, eps
        )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": is_pad,
            "advantages": advantages,
            "returns": returns,
            "values": values.detach(),
            "normalizer_observations": step_next_states[:, -1, :].detach(),
            "mc_tau": tau,
            "mc_eps": eps,
            "initial_cfm_loss": initial_cfm_loss,
            "initial_cfm_loss_bin": initial_cfm_loss_bin,
        }


    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        compute_info: bool = True,
        check_finite: bool = True,
    ) -> tuple[torch.Tensor, dict | None]:
        """Combine the ASPO policy and value losses for one minibatch.

        Each selected action chunk is treated as one macro-action for the
        policy loss. The unexecuted suffix is still part of the chunk the
        behavior policy selected, so it must contribute to the FPO ratio when
        receding-horizon execution is enabled; ``action_is_pad`` records
        execution bookkeeping only, and just the chunks with no executed action
        are invalid for the update.

        Spectral encoding keeps the first ``M+1`` DCT bins, giving a token
        count of ``M+1`` derived from ``top_p * H``. The condition is fused
        from ``[n, D]`` to ``[n*m, D]`` so one conditioning vector covers every
        Monte-Carlo draw. The CFM losses are computed under target velocities
        in fp32 for stability, and the value loss is a mean squared error, as
        the official FPO++ does.

        On info-requesting steps the modulation MLPs are re-run on a single row
        to collect conditioner statistics, which is cheap relative to
        attention.

        When ``compute_info=False`` the diagnostic reductions are skipped
        entirely and the returned info is ``None``. Training runs without wandb
        should pass ``compute_info=False`` — the info pass is ~25 small
        reductions per minibatch that would otherwise be discarded.

        Args:
            batch: Batch produced by :meth:`generate_rollout`.
            reduction: ``"mean"`` to return a scalar loss, ``"none"`` to return
                the per-chunk losses.
            compute_info: Whether to compute the diagnostic info dict.
            check_finite: Whether to abort on a non-finite loss.

        Returns:
            Tuple of the loss and the info dict, which is None when
            ``compute_info`` is false.

        Raises:
            ValueError: If a batch tensor is missing or malformed, if
                ``mc_tau`` and ``mc_eps`` disagree about being present, or if
                ``reduction`` is not ``mean`` or ``none``.
            SJFPOAborted: If ``check_finite`` is set and the loss is non-finite.
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
        advantages = to_tensor(batch["advantages"]).to(
            device=device, dtype=storage_dtype
        )
        returns = to_tensor(batch["returns"]).to(device=device, dtype=storage_dtype)
        action_is_pad = to_tensor(
            batch["action_is_pad"], dtype=torch.bool, device=device
        )
        sample_mask = None
        if "sample_mask" in batch:
            sample_mask = to_tensor(
                batch["sample_mask"], dtype=torch.bool, device=device
            )
            if sample_mask.shape != (states.shape[0],):
                raise ValueError(
                    f"sample_mask must have shape {(states.shape[0],)}, "
                    f"got {tuple(sample_mask.shape)}"
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

        num_spectral_tokens_kept = _num_spectral_tokens_kept(cfg)
        M_modes = num_spectral_tokens_kept - 1
        n = states.shape[0]
        action_dim = actions.shape[-1]

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

        tau_bn, flatten_mc_noise, flatten_mc_action, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
        )
        x_t_flat = _mc_x_t(tau_bn, flatten_mc_noise, flatten_mc_action, flow_shape)
        actor_tau = tau_bn if tau_bn.dim() == 1 else tau_bn.squeeze(-1)

        with autocast_context(device):
            cond_flat = einx.id(
                "n m d -> (n m) d",
                einx.id("n d -> n m d", self.conditioner(states), m=num_mc),
            )
        stored_l_old = "initial_cfm_loss" in batch
        if stored_l_old:
            l_old = to_tensor(
                batch["initial_cfm_loss"], dtype=torch.float32, device=device
            )
            l_old_bin = to_tensor(
                batch["initial_cfm_loss_bin"], dtype=torch.float32, device=device
            )
            if tuple(l_old.shape) != (n, num_mc):
                raise ValueError(
                    f"initial_cfm_loss shape {tuple(l_old.shape)} != {(n, num_mc)}"
                )
            if tuple(l_old_bin.shape) != (n, num_mc, flow_shape[0]):
                raise ValueError(
                    f"initial_cfm_loss_bin shape {tuple(l_old_bin.shape)} != "
                    f"{(n, num_mc, flow_shape[0])}"
                )
        else:
            with torch.inference_mode(), autocast_context(device):
                old_cond_flat = einx.id(
                    "n m d -> (n m) d",
                    einx.id("n d -> n m d", self.old_conditioner(states), m=num_mc),
                )
                v_old = self.compute_predicted_velocity(
                    x_t_flat, actor_tau, old_cond_flat, old=True
                ).float()

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

            if not stored_l_old:
                v_old_r_f = v_old.reshape(n, num_mc, *flow_shape)
                l_old_bin = compute_cfm_loss(v_old_r_f, v_star_f, cfg, "bin")
                l_old = compute_cfm_loss(v_old_r_f, v_star_f, cfg, "sample")
            adv_mc = advantages.float().unsqueeze(1)
            l_new_bin = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "bin")
            rho_bin = compute_policy_ratio(l_old_bin, l_new_bin, cfg)

            l_new = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "sample")
            rho = compute_policy_ratio(l_old, l_new, cfg, advantage=adv_mc)
            self.cfm_above_clamp_frac = (l_new > cfg.cfm_clamp).float().mean().detach()

            psi = compute_aspo_surrogate(rho, adv_mc, cfg, cfg.epsilon_clip)
            policy_per_chunk = -psi.mean(dim=1)

        with autocast_context(device):
            v_pred = self.critic(states).float()
        returns_d = returns.reshape(-1)
        if v_pred.shape != returns_d.shape:
            raise ValueError(
                f"value/return shape mismatch: v_pred={tuple(v_pred.shape)}, "
                f"returns_d={tuple(returns_d.shape)}"
            )
        se = torch.square(v_pred - returns_d)
        if cfg.use_clipped_value_loss and "values" in batch:
            old_values = to_tensor(
                batch["values"], dtype=storage_dtype, device=device
            )
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.epsilon_clip, cfg.epsilon_clip
            )
            clipped_se = torch.square(v_clipped - returns_d)
            value_per_chunk = torch.maximum(se, clipped_se)
        else:
            value_per_chunk = se

        if reduction == "mean":
            if sample_mask is None:
                policy_loss = policy_per_chunk.mean()
                value_loss = value_per_chunk.mean()
            else:
                weight = sample_mask.to(policy_per_chunk.dtype)
                count = weight.sum().clamp_min(1)
                policy_loss = (policy_per_chunk * weight).sum() / count
                value_loss = (value_per_chunk * weight).sum() / count
            loss = policy_loss + cfg.vf_coef * value_loss
        elif reduction == "none":
            if sample_mask is None:
                loss = policy_per_chunk + cfg.vf_coef * value_per_chunk
            else:
                weight = sample_mask.to(policy_per_chunk.dtype)
                loss = weight * (
                    policy_per_chunk + cfg.vf_coef * value_per_chunk
                )
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        if check_finite and not bool(torch.isfinite(loss).all()):
            raise SJFPOAborted(
                f"Non-finite loss: "
                f"policy_loss={float(policy_per_chunk.mean().detach())}, "
                f"value_loss={float(value_per_chunk.mean().detach())}"
            )

        if not compute_info:
            return loss, None

        info = self._compute_info(
            l_old=l_old,
            l_new=l_new,
            rho=rho,
            rho_bin=rho_bin,
            v_star_f=v_star_f,
            v_pred=v_pred,
            returns_d=returns_d,
            policy_per_chunk=policy_per_chunk,
            value_per_chunk=value_per_chunk,
            advantages=advantages,
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
        rho,
        rho_bin,
        v_star_f,
        v_pred,
        returns_d,
        policy_per_chunk,
        value_per_chunk,
        advantages,
        cond_stats,
        cfg,
    ) -> dict[str, float]:
        """Compute every diagnostic reduction, batched into a single sync.

        The log ratio is taken on the raw delta rather than the clamped one, so
        the reported figure is never an artefact of the clamp. The reported
        clamp-saturation fraction matters because samples above ``cfm_clamp``
        contribute no gradient at all, so a run where it stays high is training
        the critic only. Every scalar is packed into one tensor and transferred
        in a single sync.

        Only called when ``forward(compute_info=True)``; the training loop can
        skip it entirely when nothing will consume the result.

        Args:
            l_old: CFM loss under the frozen behavior policy.
            l_new: CFM loss under the updated policy.
            rho: Elementwise policy ratio.
            rho_bin: Per-frequency-bin policy ratio.
            v_star_f: Flattened target velocities.
            v_pred: Flattened predicted velocities.
            returns_d: Value targets.
            policy_per_chunk: Per-chunk policy loss.
            value_per_chunk: Per-chunk value loss.
            advantages: Per-chunk advantages.
            cond_stats: Conditioner modulation statistics.
            cfg: Active configuration, supplying ``cfm_clamp`` and friends.

        Returns:
            Dict of diagnostic scalars for the wandb log.
        """
        delta = l_old - l_new
        log_rho_abs = delta.abs()
        log_rho_max = log_rho_abs.max()
        log_rho_clip_frac = (log_rho_abs > cfg.log_ratio_cap).float().mean()
        cfm_above_clamp_frac = (l_new > cfg.cfm_clamp).float().mean()

        rho_mean = rho.mean()
        rho_std = rho.std(correction=0)

        rho_bin_mean = rho_bin.mean(dim=(0, 1))
        n_bins = rho_bin_mean.numel()
        split = max(1, n_bins // 2)
        rho_bin_range = rho_bin_mean.max() - rho_bin_mean.min()
        rho_bin_low = rho_bin_mean[:split].mean()
        rho_bin_high = (
            rho_bin_mean[split:].mean() if n_bins > 1 else rho_bin_mean.mean()
        )
        rho_bin_outside = (
            ((rho_bin < 1.0 - cfg.epsilon_clip) | (rho_bin > 1.0 + cfg.epsilon_clip))
            .float()
            .mean()
        )
        eps_low = cfg.epsilon_clip_lowfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        eps_high = (
            cfg.epsilon_clip_highfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        )
        low_slice = rho_bin[..., :split]
        rho_bin_low_outside = (
            ((low_slice < 1.0 - eps_low) | (low_slice > 1.0 + eps_low)).float().mean()
        )
        if n_bins > 1:
            high_slice = rho_bin[..., split:]
            rho_bin_high_outside = (
                ((high_slice < 1.0 - eps_high) | (high_slice > 1.0 + eps_high))
                .float()
                .mean()
            )
        else:
            rho_bin_high_outside = rho_bin_low_outside

        returns_var = returns_d.var(correction=0).clamp_min(1e-8)
        explained_var = 1.0 - (returns_d - v_pred).var(correction=0) / returns_var
        value_error_max = torch.nn.functional.smooth_l1_loss(
            v_pred, returns_d, reduction="none", beta=1.0
        ).max()
        value_pred_max = v_pred.abs().max()
        return_max = returns_d.abs().max()
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
            value_per_chunk.mean(),
            explained_var,
            delta.mean(),
            log_rho_max,
            log_rho_clip_frac,
            value_error_max,
            value_pred_max,
            return_max,
            rho_mean,
            rho_std,
            rho_bin_range,
            rho_bin_low,
            rho_bin_high,
            rho_bin_outside,
            rho_bin_low_outside,
            rho_bin_high_outside,
            v_star_rms,
            l_old.mean(),
            l_new.mean(),
            l_old_std,
            l_new_std,
            cfm_above_clamp_frac,
            l_old_new_corr,
            advantages.abs().mean(),
            returns_d.std(correction=0),
            v_pred.std(correction=0),
        ]
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
            "log_rho_max",
            "log_rho_clip_frac",
            "value_error_max",
            "value_pred_max",
            "return_max",
            "flow_rho_mean",
            "flow_rho_std",
            "flow_rho_bin_range",
            "flow_rho_bin_lowfreq",
            "flow_rho_bin_highfreq",
            "flow_rho_bin_outside_frac",
            "flow_rho_bin_low_outside_frac",
            "flow_rho_bin_high_outside_frac",
            "flow_v_star_rms",
            "l_old_mean",
            "l_new_mean",
            "l_old_std",
            "l_new_std",
            "cfm_above_clamp_frac",
            "l_old_new_corr",
            "adv_mean_abs",
            "value_target_std",
            "value_pred_std",
        ]
        info = dict(zip(keys, vals))
        for i, k in enumerate(cond_keys):
            info[f"cond_{k}"] = vals[len(keys) + i]
        return info