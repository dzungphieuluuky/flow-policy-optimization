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
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.collector import collect_chunks, collect_chunks_vec
from sjfpo.dct_utils import (
    decode_coeffs_to_velocity,
    extract_dct_coeffs,
    sample_freq_weighted_noise,
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


# ---------------------------------------------------------------------------
# Small helpers (avoid repeating shape arithmetic)
# ---------------------------------------------------------------------------


def _num_spectral_tokens_kept(cfg: SJFPOConfig) -> int:
    """Number of spectral tokens kept per action chunk."""
    return max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon)) if cfg.use_dct_transform else cfg.action_horizon


def _sample_spectral_eps(
    cfg: SJFPOConfig,
    n: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw MC noise matching the token layout (DCT or raw), on ``device``."""
    if cfg.use_dct_transform:
        Ka = _num_spectral_tokens_kept(cfg)
        M_modes = max(1, Ka - 1) if Ka > 1 else 0
        eps = sample_freq_weighted_noise(
            M=M_modes,
            d=action_dim,
            T=1.0,
            B=n * cfg.num_mc_samples,
            sigma=1.0,
            device=device,
        )
        return eps.reshape(n, cfg.num_mc_samples, Ka, action_dim)
    return sample_noise(
        (n, cfg.num_mc_samples, cfg.action_horizon, action_dim),
        cfg,
        device,
        None,
    )


def _actor_call(actor, x_t, tau, cond, *, use_ckpt: bool):
    """Actor forward, optionally under gradient checkpointing."""
    if use_ckpt:
        return cp.checkpoint(actor, x_t, tau, cond, use_reentrant=False)
    return actor(x_t, tau, cond)


def _unwrap_compiled(m: torch.nn.Module) -> torch.nn.Module:
    """Return the eager module behind a ``torch.compile`` wrapper, if any."""
    return getattr(m, "_orig_mod", m)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


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
            config.dit_time_dim,
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

        # Frozen behavior snapshot (PPO π_old); refreshed per collection round.
        self.old_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.old_conditioner = (
            copy.deepcopy(self.conditioner).requires_grad_(False).eval()
        )

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=config.n_obs_steps)
        self._action_buffer: deque[torch.Tensor] = deque()
        self.step_count = 0
        self.preprocessor, self.postprocessor = make_sjfpo_pre_post_processors(self)
        del dataset_stats  # kept for base-class signature compatibility

    # -- module state --------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        self.old_actor.eval()
        self.old_conditioner.eval()
        return self

    def reset(self):
        self._frame_buffer.clear()
        self._action_buffer.clear()

    @torch.no_grad()
    def update_observation_stats(self, observations: torch.Tensor) -> None:
        """Update the live running statistics from raw post-step observations."""
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

    def get_actor_optim_params(self) -> dict:
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
        return {
            "params": [
                {
                    "params": list(self.critic.parameters()),
                    "lr": self.config.critic_lr,
                }
            ]
        }

    def get_optim_params(self) -> dict:
        """Compatibility view containing both optimizer ownership groups."""
        return {
            "params": [
                *self.get_actor_optim_params()["params"],
                *self.get_critic_optim_params()["params"],
            ]
        }

    # -- velocity helpers ----------------------------------------------

    @staticmethod
    def compute_target_velocity(a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """v* = a − ε broadcast over MC draws: [B, *flow] x [B, m, *flow] → [B, m, *flow]."""
        return a.unsqueeze(1) - eps

    def compute_predicted_velocity(self, x_t, tau, states, old: bool = False):
        conditioner = self.old_conditioner if old else self.conditioner
        net = self.old_actor if old else self.actor
        return net(x_t, tau, conditioner(states))

    # -- observation coercion ------------------------------------------

    def standardize_obs_tensor(self, states: torch.Tensor) -> torch.Tensor:
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

    # -- inference -----------------------------------------------------

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Full action chunk ``[B, H, action_dim]``; ``noise=True`` samples, else zeros."""
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

        return self.postprocessor({ACTION: scaled})[ACTION]

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
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

    # -- rollout -------------------------------------------------------

    def _initial_cfm_losses(
        self,
        states_n: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        token_count = eps.shape[2]
        if cfg.use_dct_transform:
            spectral_action = extract_dct_coeffs(actions, M=token_count - 1)
        else:
            spectral_action = actions
        tau_bn, eps_flat, action_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action, eps, tau
        )
        x_t = (
            (1.0 - tau_bn[:, None]) * eps_flat
            + tau_bn[:, None] * action_rep
        ).reshape(-1, *flow_shape)
        actor = _unwrap_compiled(self.actor)
        conditioner = _unwrap_compiled(self.conditioner)
        with torch.inference_mode(), autocast_context(actions.device):
            condition = conditioner(states_n)
            condition_flat = einx.id(
                "n d -> n m d", condition, m=num_mc
            ).reshape(num_mc * condition.shape[0], condition.shape[-1])
            v_initial = actor(x_t, tau_bn, condition_flat).float()
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
        """Roll out open-loop chunks; return a forward-ready batch."""
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

        if int(getattr(env, "num_envs", 1)) > 1:
            chunks = collect_chunks_vec(
                env,
                propose_batch,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                chunk_size=cfg.action_horizon,
            )
        else:
            chunks = collect_chunks(
                env,
                propose,
                total_steps,
                action_low,
                action_high,
                n_obs_steps=cfg.n_obs_steps,
                n_action_steps=cfg.n_action_steps,
                chunk_size=cfg.action_horizon,
            )

        n_chunks = len(chunks)
        action_dim = int(self.config.action_feature.shape[0])
        num_spectral_tokens_kept = _num_spectral_tokens_kept(cfg)

        # Stack chunk data and retain the post-step windows for step-level GAE.
        lengths_cpu = [int(c["executed_n_action_steps"]) for c in chunks]
        lengths = torch.as_tensor(lengths_cpu, dtype=torch.long, device=device)
        states = to_device_async(torch.stack([c[OBS_STATE] for c in chunks]), device)
        actions = to_device_async(torch.stack([c["action"] for c in chunks]), device)
        all_rewards = torch.cat(
            [
                c[REWARD].reshape(-1).to(device=device, dtype=torch.float32)
                for c in chunks
            ]
        )

        step_next_chunks = []
        for chunk, n_executed in zip(chunks, lengths_cpu, strict=True):
            step_next = getattr(chunk, "step_next_states", None)
            if step_next is None:
                raise ValueError(
                    "collector transition is missing step_next_states; "
                    "step-level GAE requires every executed post-step window"
                )
            expected_shape = (
                n_executed,
                *tuple(chunk["next.observation.state"].shape),
            )
            if tuple(step_next.shape) != expected_shape:
                raise ValueError(
                    f"step_next_states shape {tuple(step_next.shape)} != "
                    f"{expected_shape}"
                )
            step_next_chunks.append(step_next)
        step_next_states = to_device_async(
            torch.cat(step_next_chunks, dim=0), device
        )

        # Each executed step's current state is either a chunk start or the
        # previous executed step's post-state. This preserves the exact window
        # observed by the environment without changing the public Transition
        # key contract.
        step_states = []
        chunk_starts_cpu = []
        step_terminated = torch.zeros(
            int(sum(lengths_cpu)), dtype=torch.bool, device=device
        )
        step_truncated = torch.zeros_like(step_terminated)
        offset = 0
        for chunk_index, (chunk, n_executed) in enumerate(
            zip(chunks, lengths_cpu, strict=True)
        ):
            if n_executed <= 0:
                raise ValueError("every chunk must execute at least one step")
            chunk_starts_cpu.append(offset)
            step_states.append(states[chunk_index : chunk_index + 1])
            if n_executed > 1:
                step_states.append(
                    step_next_states[offset : offset + n_executed - 1]
                )
            if bool(chunk[DONE]):
                step_terminated[offset + n_executed - 1] = True
            if bool(chunk[TRUNCATED]):
                step_truncated[offset + n_executed - 1] = True
            offset += n_executed
        step_states = to_device_async(torch.cat(step_states, dim=0), device)
        chunk_starts = torch.tensor(
            chunk_starts_cpu, dtype=torch.long, device=device
        )

        # Sanity: chunk shape.
        if actions.dim() != 3 or actions.shape[1] != cfg.action_horizon:
            raise ValueError(
                f"chunk actions shape {tuple(actions.shape)} != expected "
                f"[B, H={cfg.action_horizon}, D={action_dim}]"
            )

        with torch.inference_mode():
            # One batched critic pass for both sides of every executed step.
            all_step_states_n = self.preprocessor(
                {OBS_STATE: torch.cat([step_states, step_next_states], dim=0)}
            )[OBS_STATE]
            states_n = all_step_states_n.index_select(0, chunk_starts)
            with autocast_context(device):
                all_step_values = self.critic(all_step_states_n).float()
            total_executed = int(sum(lengths_cpu))
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
        # The policy predicts one action chunk, so expose the first executed
        # step's step-GAE target as the chunk-level policy target.
        advantages = step_advantages.index_select(0, chunk_starts)
        returns = step_returns.index_select(0, chunk_starts)
        if advantages.numel() > 0:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            if cfg.advantage_clamp is not None:
                advantages = advantages.clamp(
                    -cfg.advantage_clamp, cfg.advantage_clamp
                )
        advantages = advantages.to(storage_dtype)
        returns = returns.to(storage_dtype)

        is_pad = torch.arange(cfg.action_horizon, device=device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)

        tau = sample_timesteps(
            (n_chunks, cfg.num_mc_samples), cfg, device, None
        ).unsqueeze(-1)
        eps = _sample_spectral_eps(cfg, n_chunks, action_dim, device)

        # Single shape check against the layout that will actually be used.
        expected_eps_shape = (n_chunks, cfg.num_mc_samples, num_spectral_tokens_kept, action_dim)
        if tuple(eps.shape) != expected_eps_shape:
            raise ValueError(
                f"MC noise shape {tuple(eps.shape)} != expected "
                f"{expected_eps_shape}"
            )

        initial_cfm_loss, initial_cfm_loss_bin = self._initial_cfm_losses(
            states_n, actions, tau, eps
        )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": is_pad,
            "advantages": advantages,
            "returns": returns,
            "values": values.detach(),
            # One raw post-step frame per executed environment step. The
            # training loop folds these into the live normalizer after the
            # update; keeping them separate avoids counting warmup/history
            # frames from the overlapping observation windows.
            "normalizer_observations": step_next_states[:, -1, :].detach(),
            # Independent per-chunk/MC CFM draws, reused for old/new losses.
            "mc_tau": tau,
            "mc_eps": eps,
            "initial_cfm_loss": initial_cfm_loss,
            "initial_cfm_loss_bin": initial_cfm_loss_bin,
        }

    # -- training forward ---------------------------------------------

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
        compute_info: bool = True,
    ) -> tuple[torch.Tensor, dict | None]:
        """Combined ASPO policy + value loss from a ``generate_rollout`` batch.

        When ``compute_info=False`` the diagnostic reductions are skipped
        entirely; the returned info is ``None``. Training runs without wandb
        should pass ``compute_info=False`` — the info pass is ~25 small
        reductions per minibatch that would otherwise be discarded.
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

        # Treat each selected action chunk as one macro-action for the policy
        # loss. The unexecuted suffix is still part of the chunk selected by
        # the behavior policy, so it must contribute to the FPO ratio when
        # receding-horizon execution is enabled. ``action_is_pad`` records
        # execution bookkeeping; only chunks with no executed action are
        # invalid for the policy update.
        policy_valid = ~action_is_pad.all(dim=1)

        states = self.preprocessor({OBS_STATE: states})[OBS_STATE]

        num_spectral_tokens_kept = _num_spectral_tokens_kept(cfg)
        M_modes = num_spectral_tokens_kept - 1
        # Spectral encoding: first M+1 DCT bins (low frequency) kept; token count = M+1 (derived from top_p * H).
        n = states.shape[0]
        action_dim = actions.shape[-1]

        spectral_action_tokens = (
            extract_dct_coeffs(actions, M=M_modes) if cfg.use_dct_transform else actions
        )

        # Reuse the per-chunk/MC CFM draws stored during rollout.
        has_mc_tau = "mc_tau" in batch
        has_mc_eps = "mc_eps" in batch
        if has_mc_tau != has_mc_eps:
            raise ValueError("mc_tau and mc_eps must be provided together")
        if has_mc_tau:
            tau = to_tensor(batch["mc_tau"], dtype=storage_dtype, device=device)
            expected_tau = (n, cfg.num_mc_samples, 1)
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
                (n, cfg.num_mc_samples), cfg, device, None
            ).unsqueeze(-1)
            eps = _sample_spectral_eps(cfg, n, action_dim, device)

        # Noisy anchors.
        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
        )
        x_t_flat = (
            (1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep
        ).reshape(-1, *flow_shape)

        # Condition fusion: [n, D] -> [n*m, D].
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
                v_old = self.old_actor(x_t_flat, tau_bn, old_cond_flat).float()

        with autocast_context(device):
            if cfg.use_gradient_checkpointing:
                v_new = _actor_call(
                    self.actor, x_t_flat, tau_bn, cond_flat, use_ckpt=True
                )
            else:
                v_new = self.actor(x_t_flat, tau_bn, cond_flat)
            v_new = v_new.float()

        cond_stats = None
        if compute_info:
            # Re-run just the modulation MLPs on one row to grab stats —
            # cheap relative to attention, and only on info-requesting steps.
            with torch.inference_mode(), autocast_context(device):
                _, cond_stats = self.actor(
                    x_t_flat[:1], tau_bn[:1], cond_flat[:1], return_stats=True
                )

        # CFM losses under target velocities (fp32 for stability).
        with torch.autocast(device_type=device.type, enabled=False):
            v_star_f = self.compute_target_velocity(spectral_action_tokens, eps).float()
            v_new_r_f = v_new.reshape(n, num_mc, *flow_shape)

            if not stored_l_old:
                v_old_r_f = v_old.reshape(n, num_mc, *flow_shape)
                l_old_bin = compute_cfm_loss(v_old_r_f, v_star_f, cfg, "bin")
                l_old = compute_cfm_loss(v_old_r_f, v_star_f, cfg, "sample")
            l_new_bin = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "bin")
            rho_bin = compute_policy_ratio(l_old_bin, l_new_bin, cfg)  # computed once

            l_new = compute_cfm_loss(v_new_r_f, v_star_f, cfg, "sample")
            rho = compute_policy_ratio(l_old, l_new, cfg)

            adv_mc = advantages.float().unsqueeze(1)

            if cfg.use_per_bin_clip:
                n_bins = rho_bin.shape[-1]
                split = max(1, n_bins // 2)
                eps_bin = torch.full(
                    (n_bins,), cfg.epsilon_clip_highfreq, device=device
                )
                eps_bin[:split] = cfg.epsilon_clip_lowfreq
                psi = compute_aspo_surrogate(
                    rho_bin, adv_mc.unsqueeze(-1), cfg, eps_bin
                )
                policy_per_chunk = -psi.mean(dim=(1, 2))
            else:
                psi = compute_aspo_surrogate(rho, adv_mc, cfg, cfg.epsilon_clip)
                policy_per_chunk = -psi.mean(dim=1)

        # Value loss (Huber).
        with autocast_context(device):
            v_pred = self.critic(states).float()
        returns_d = returns.reshape(-1)
        if v_pred.shape != returns_d.shape:
            raise ValueError(
                f"value/return shape mismatch: v_pred={tuple(v_pred.shape)}, "
                f"returns_d={tuple(returns_d.shape)}"
            )
        se = torch.nn.functional.smooth_l1_loss(
            v_pred, returns_d, reduction="none", beta=1.0
        )
        if cfg.use_clipped_value_loss and "values" in batch:
            old_values = to_tensor(
                batch["values"], dtype=storage_dtype, device=device
            )
            v_clipped = old_values + (v_pred - old_values).clamp(
                -cfg.epsilon_clip, cfg.epsilon_clip
            )
            clipped_se = torch.nn.functional.smooth_l1_loss(
                v_clipped, returns_d, reduction="none", beta=1.0
            )
            value_per_chunk = torch.maximum(se, clipped_se)
        else:
            value_per_chunk = se

        policy_valid = policy_valid.to(policy_per_chunk.dtype)
        masked_policy_per_chunk = policy_per_chunk * policy_valid
        valid_policy_count = policy_valid.sum().clamp_min(1)

        if reduction == "mean":
            policy_loss = masked_policy_per_chunk.sum() / valid_policy_count
            value_loss = value_per_chunk.mean()
            loss = policy_loss + cfg.vf_coef * value_loss
        elif reduction == "none":
            loss = masked_policy_per_chunk + cfg.vf_coef * value_per_chunk
        else:
            raise ValueError(f"reduction={reduction!r} must be 'mean' or 'none'")

        if not bool(torch.isfinite(loss).all()):
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

    # -- diagnostics (single device->host transfer) --------------------

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
        """All diagnostic reductions in one go, batched into a single sync.

        Only called when ``forward(compute_info=True)``; the training loop
        can skip it entirely when nothing will consume the result.
        """
        # Log-ratio on raw delta (never clamped-then-checked).
        delta = l_old - l_new
        log_rho_abs = delta.abs()
        log_rho_max = log_rho_abs.max()
        log_rho_clip_frac = (log_rho_abs > cfg.log_ratio_cap).float().mean()

        # rho stats
        rho_mean = rho.mean()
        rho_std = rho.std(correction=0)

        # per-bin rho
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

        # value-health stats
        returns_var = returns_d.var(correction=0).clamp_min(1e-8)
        explained_var = 1.0 - (returns_d - v_pred).var(correction=0) / returns_var
        value_error_max = torch.nn.functional.smooth_l1_loss(
            v_pred, returns_d, reduction="none", beta=1.0
        ).max()
        value_pred_max = v_pred.abs().max()
        return_max = returns_d.abs().max()
        v_star_rms = v_star_f.pow(2).mean().sqrt()

        # Concrete diagnostics (Step 3)
        l_old_std = l_old.std(correction=0)
        l_new_std = l_new.std(correction=0)
        corr_denom = l_old_std * l_new_std + 1e-8
        l_old_new_corr = (
            (l_old - l_old.mean()) * (l_new - l_new.mean())
        ).mean() / corr_denom

        # Pack into one tensor and transfer once.
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
            "l_old_new_corr",
            "adv_mean_abs",
            "value_target_std",
            "value_pred_std",
        ]
        info = dict(zip(keys, vals))
        for i, k in enumerate(cond_keys):
            info[f"cond_{k}"] = vals[len(keys) + i]
        return info