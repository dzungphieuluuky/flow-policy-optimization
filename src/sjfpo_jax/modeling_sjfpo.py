"""SJFPO policy (JAX): flow head + state conditioner + critic.

Parameters are stored as a plain pytree; the policy is a "stateful function"
object rather than a Flax module because training/inference need to swap the
actor snapshot and carry RNG explicitly.
"""

from __future__ import annotations

import copy
import dataclasses
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD, TRUNCATED

from sjfpo_jax.collector import collect_chunks, collect_chunks_vec
from sjfpo_jax.configuration_sjfpo import SJFPOConfig
from sjfpo_jax.dct_utils import (
    decode_coeffs_to_velocity,
    extract_dct_coeffs,
    sample_freq_weighted_noise,
)
from sjfpo_jax.fpo_core import (
    compute_aspo_surrogate,
    compute_cfm_loss,
    compute_policy_ratio,
    run_mc_simulation,
    sample_action,
    sample_noise,
    sample_timesteps,
)
from sjfpo_jax.gae import compute_advantages
from sjfpo_jax.networks import (
    Critic,
    SJFPOFlowModel,
    StateConditioner,
    normalize_with_stats,
    running_mean_std_update,
)
from sjfpo_jax.utils import resolve_storage_dtype

STORAGE_DTYPE = resolve_storage_dtype()


class SJFPOAborted(RuntimeError):
    """Non-finite training loss (message names the offending term)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

def _trainable_subtree(params: PolicyParams) -> dict:
    """Actor / conditioner / critic — the only float-only leaves we optimize."""
    return {
        "actor": params.actor,
        "conditioner": params.conditioner,
        "critic": params.critic,
    }


def _merge_subtree(params: PolicyParams, subtree: dict) -> PolicyParams:
    """Reassemble a full ``PolicyParams`` from a trainable subtree."""
    return dataclasses.replace(
        params,
        actor=subtree["actor"],
        conditioner=subtree["conditioner"],
        critic=subtree["critic"],
    )


def _num_spectral_tokens_kept(cfg: SJFPOConfig) -> int:
    if cfg.use_dct_transform:
        return max(1, round(cfg.top_p_spectral_tokens * cfg.action_horizon))
    return cfg.action_horizon


def _raise_if_nonfinite(loss, policy_loss, value_loss):
    if not np.isfinite(loss):
        raise SJFPOAborted(
            f"Non-finite loss: policy={float(policy_loss)}, value={float(value_loss)}"
        )


def _sample_spectral_eps(cfg, n, action_dim, key):
    if cfg.use_dct_transform:
        Ka = _num_spectral_tokens_kept(cfg)
        M_modes = max(1, Ka - 1) if Ka > 1 else 0
        eps = sample_freq_weighted_noise(
            M=M_modes,
            d=action_dim,
            T=1.0,
            B=n * cfg.num_mc_samples,
            sigma=1.0,
            key=key,
        )
        return eps.reshape(n, cfg.num_mc_samples, Ka, action_dim)
    return sample_noise(
        (n, cfg.num_mc_samples, cfg.action_horizon, action_dim),
        key=key,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PolicyParams:
    """Parameter pytree for the SJFPO policy.

    Registered as a JAX pytree node so ``jax.tree.map`` (and Optax, and
    ``jax.grad``, and ``jax.jit``) recurse into the six array subtrees
    instead of treating the whole dataclass as a single leaf. Without
    ``register_dataclass``, Optax's ``zeros_like`` would see an opaque
    ``PolicyParams`` object and raise ``TypeError``.
    """

    actor: Any
    conditioner: Any
    critic: Any
    old_actor: Any
    old_conditioner: Any
    obs_stats: dict

@dataclasses.dataclass
class SJFPOPolicy:
    """Stateless-in-module, params-carrying policy.

    Module instances are cheap; parameters are held in ``self.params``.
    """

    config: SJFPOConfig
    actor: SJFPOFlowModel
    conditioner: StateConditioner
    critic: Critic
    params: PolicyParams

    # Runtime state (mutable, not part of the pytree).
    action_low: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([-1.0], dtype=np.float32)
    )
    action_high: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0], dtype=np.float32)
    )
    step_count: int = 0

    def __post_init__(self):
        self._frame_buffer: deque = deque(maxlen=self.config.n_obs_steps)
        self._action_buffer: deque = deque()


    @classmethod
    def create(cls, config: SJFPOConfig, obs_dim: int, action_dim: int, key):
        """Build a fresh policy with initialized parameters."""
        actor = SJFPOFlowModel(
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            dit_depth=config.dit_depth,
            dit_heads=config.dit_heads,
            dit_mlp_ratio=config.dit_mlp_ratio,
            time_embed_dim=config.time_embed_dim,
            timestep_scale=config.timestep_scale,
        )
        conditioner = StateConditioner(obs_dim=obs_dim, hidden_dim=config.hidden_dim)
        critic = Critic(obs_dim=obs_dim, hidden_dim=config.hidden_dim)

        keys = jax.random.split(key, 4)
        # Dummy inputs for shape inference.
        dummy_states = jnp.zeros((1, config.n_obs_steps, obs_dim), dtype=jnp.float32)
        cond = conditioner.apply(
            conditioner.init(keys[0], dummy_states), dummy_states
        )
        actor_params = actor.init(
            keys[1],
            jnp.zeros(
                (1, _num_spectral_tokens_kept(config), action_dim), dtype=jnp.float32
            ),
            jnp.zeros((1,), dtype=jnp.float32),
            cond,
        )
        critic_params = critic.init(keys[2], dummy_states)

        params = PolicyParams(
            actor=actor_params,
            conditioner=conditioner.init(keys[0], dummy_states),
            critic=critic_params,
            old_actor=actor_params,
            old_conditioner=conditioner.init(keys[0], dummy_states),
            obs_stats={
                "mean": jnp.zeros((obs_dim,), dtype=jnp.float32),
                "var": jnp.ones((obs_dim,), dtype=jnp.float32),
                # JAX defaults to x64-off; fp32 is plenty for a frame count.
                # Saturation only matters past ~2^24 frames, far beyond the
                # episodic training budget in this port.
                "count": jnp.asarray(0.0, dtype=jnp.float32),
                "uninitialized": jnp.asarray(True),
            },
        )
        return cls(
            config=config,
            actor=actor,
            conditioner=conditioner,
            critic=critic,
            params=params,
        )


    def reset(self) -> None:
        self._frame_buffer.clear()
        self._action_buffer.clear()

    def set_action_bounds(self, low, high) -> None:
        self.action_low = np.asarray(low, dtype=np.float32)
        self.action_high = np.asarray(high, dtype=np.float32)

    def snapshot_behavior_policy(self) -> None:
        """Refresh frozen π_old weights (pure pytree copy)."""
        self.params = dataclasses.replace(
            self.params,
            old_actor=self.params.actor,
            old_conditioner=self.params.conditioner,
        )

    def _standardize_obs(self, states):
        """[B, T, obs] / [T, obs] / [obs] -> [B, T, obs]."""
        if states.ndim == 3:
            return states
        if states.ndim == 2:
            if states.shape[0] == self.config.n_obs_steps:
                return states[None]
            return jnp.broadcast_to(
                states[None], (1, self.config.n_obs_steps, states.shape[-1])
            )
        raise ValueError(f"observation.state bad shape: {states.shape}")


    def predict_action_chunk(self, batch: dict, *, noise: bool = False, key=None):
        cfg = self.config
        states = self._standardize_obs(batch[OBS_STATE])

        condition = self.conditioner.apply(self.params.conditioner, states)
        Ka = _num_spectral_tokens_kept(cfg)
        action_dim = int(cfg.action_feature.shape[0])
        shape = (states.shape[0], Ka, action_dim)

        eps = None
        if not noise:
            eps = jnp.zeros(shape, dtype=jnp.float32)
        elif key is None:
            key = jax.random.PRNGKey(0)

        def predict_velocity(x, t, cond):
            return self.actor.apply(
                self.params.actor, x, t, cond
            )

        num_steps = cfg.num_euler_steps if noise else cfg.num_inference_steps
        x0 = sample_action(
            predict_velocity,
            shape,
            condition,
            num_inference_steps=num_steps,
            eps=eps,
            key=key,
        )

        if cfg.use_dct_transform:
            tau_grid = jnp.linspace(0.0, 1.0, cfg.action_horizon, dtype=jnp.float32)
            scaled = decode_coeffs_to_velocity(
                x0, tau=tau_grid, T=1.0, K_original=cfg.action_horizon
            )[:, : cfg.action_horizon, :]
        else:
            scaled = x0[:, : cfg.action_horizon, :]
        scaled = jnp.clip(scaled, -1.0, 1.0)

        # Unscale to env action bounds.
        low = jnp.asarray(self.action_low, dtype=scaled.dtype)
        high = jnp.asarray(self.action_high, dtype=scaled.dtype)
        return 0.5 * (high - low) * scaled + 0.5 * (high + low)

    def select_action(self, batch: dict, *, key=None) -> jnp.ndarray:
        obs = jnp.asarray(batch[OBS_STATE])
        if obs.ndim == 1:
            if not self._frame_buffer:
                for _ in range(self.config.n_obs_steps):
                    self._frame_buffer.append(obs)
            else:
                self._frame_buffer.append(obs)
            window = jnp.stack(list(self._frame_buffer), axis=0)
        elif obs.ndim == 2:
            window = obs
        else:
            raise ValueError(f"select_action expects [obs] or [T, obs]; got {obs.shape}")

        if not self._action_buffer:
            chunk = self.predict_action_chunk(
                {OBS_STATE: window}, noise=True, key=key
            )
            n_exec = max(1, min(self.config.n_action_steps, chunk.shape[1]))
            for i in range(n_exec - 1, -1, -1):
                self._action_buffer.append(chunk[0, i])
        return self._action_buffer.popleft()


    def generate_rollout(
        self,
        env,
        total_steps: int,
        *,
        action_low=-1.0,
        action_high=1.0,
        noisy: bool = True,
        key=None,
    ) -> dict:
        if total_steps <= 0:
            raise ValueError(f"total_steps={total_steps} must be positive")
        cfg = self.config

        def propose(state):
            chunk = self.predict_action_chunk(
                {OBS_STATE: state}, noise=noisy, key=key
            )
            # [1, H, action_dim] -> [H, action_dim]; the single-env collector
            # indexes step ``i`` on the leading axis, so the batch dim must go.
            return np.asarray(chunk[0])

        def propose_batch(states):
            chunk = self.predict_action_chunk(
                {OBS_STATE: states}, noise=noisy, key=key
            )
            # [B, H, action_dim]; the vector collector indexes ``proposed[k]``.
            return np.asarray(chunk)
        
        if int(getattr(env, "num_envs", 1)) > 1:
            chunks = collect_chunks_vec(
                env, propose_batch, total_steps, action_low, action_high,
                n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.action_horizon,
                key=key,
            )
        else:
            chunks = collect_chunks(
                env, propose, total_steps, action_low, action_high,
                n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.action_horizon,
                key=key,
            )

        n_chunks = len(chunks)
        action_dim = int(self.config.action_feature.shape[0])
        Ka = _num_spectral_tokens_kept(cfg)

        lengths = jnp.asarray(
            [int(c["executed_n_action_steps"]) for c in chunks], dtype=jnp.int32
        )
        states = jnp.stack([c[OBS_STATE] for c in chunks])
        next_states = jnp.stack([c["next.observation.state"] for c in chunks])
        actions = jnp.stack([c["action"] for c in chunks])
        terminated = jnp.asarray([bool(c[DONE]) for c in chunks], dtype=bool)
        truncated = jnp.asarray([bool(c[TRUNCATED]) for c in chunks], dtype=bool)
        all_rewards = jnp.concatenate(
            [c[REWARD].reshape(-1).astype(jnp.float32) for c in chunks]
        )

        if actions.ndim != 3 or actions.shape[1] != cfg.action_horizon:
            raise ValueError(
                f"chunk actions shape {actions.shape} != expected "
                f"[B, H={cfg.action_horizon}, D={action_dim}]"
            )

        # Normalize states via running stats (identity until first update).
        stats = self.params.obs_stats
        normalized_states = normalize_with_stats(states, stats)
        next_states_n = normalize_with_stats(next_states, stats)
        values = self.critic.apply(self.params.critic, normalized_states)
        v_next_all = self.critic.apply(self.params.critic, next_states_n)
        next_values = jnp.where(terminated, 0.0, v_next_all)

        advantages, returns = compute_advantages(
            all_rewards, values, next_values, lengths, terminated,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
            normalize=True, truncated=truncated, clamp=cfg.advantage_clamp,
        )
        advantages = advantages.astype(STORAGE_DTYPE)
        returns = returns.astype(STORAGE_DTYPE)

        is_pad = (
            jnp.arange(cfg.action_horizon)[None, :] >= lengths[:, None]
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        k_tau, k_eps = jax.random.split(key)
        tau = sample_timesteps(
            cfg.num_mc_samples, (cfg.min_timestep, cfg.max_timestep), key=k_tau
        )
        eps = _sample_spectral_eps(cfg, n_chunks, action_dim, k_eps)

        expected_eps = (n_chunks, cfg.num_mc_samples, Ka, action_dim)
        if tuple(eps.shape) != expected_eps:
            raise ValueError(
                f"CRN eps shape {eps.shape} != expected {expected_eps}"
            )

        return {
            OBS_STATE: states,
            ACTION: actions,
            "action_is_pad": is_pad,
            "advantages": advantages,
            "returns": returns,
            "values": values,
            "tau": tau,
            "eps": eps,
        }


    def forward(self, batch: dict, *, compute_info: bool = True):
        cfg = self.config
        states = jnp.asarray(batch[OBS_STATE], dtype=STORAGE_DTYPE)
        actions = jnp.asarray(batch[ACTION], dtype=STORAGE_DTYPE)
        advantages = jnp.asarray(batch["advantages"], dtype=STORAGE_DTYPE)
        returns = jnp.asarray(batch["returns"], dtype=STORAGE_DTYPE)

        if states.shape[-2] != cfg.n_obs_steps:
            raise ValueError(
                f"state length {states.shape[-2]} != n_obs_steps {cfg.n_obs_steps}"
            )
        if actions.shape[-2] != cfg.action_horizon:
            raise ValueError(
                f"action horizon {actions.shape[-2]} != action_horizon {cfg.action_horizon}"
            )
        # Normalize states before conditioning / critic.
        states = normalize_with_stats(states, self.params.obs_stats)

        Ka = _num_spectral_tokens_kept(cfg)
        M_modes = Ka - 1
        n = states.shape[0]
        action_dim = actions.shape[-1]

        spectral_action_tokens = (
            extract_dct_coeffs(actions, M=M_modes)
            if cfg.use_dct_transform
            else actions
        )

        if "tau" in batch and "eps" in batch:
            tau = jnp.asarray(batch["tau"], dtype=STORAGE_DTYPE)
            if tau.ndim == 2:
                tau = tau[0]
            eps = jnp.asarray(batch["eps"], dtype=STORAGE_DTYPE)
            expected = (n, cfg.num_mc_samples, Ka, action_dim)
            if tuple(eps.shape) != expected:
                raise ValueError(f"stored eps shape {eps.shape} != {expected}")
        else:
            key = jax.random.PRNGKey(0)
            k_tau, k_eps = jax.random.split(key)
            tau = sample_timesteps(
                cfg.num_mc_samples, (cfg.min_timestep, cfg.max_timestep), key=k_tau
            )
            eps = _sample_spectral_eps(cfg, n, action_dim, k_eps)

        tau_bn, eps_flat, a_rep, flow_shape, num_mc = run_mc_simulation(
            spectral_action_tokens, eps, tau
        )
        x_t_flat = (
            (1.0 - tau_bn[:, None]) * eps_flat + tau_bn[:, None] * a_rep
        ).reshape(-1, *flow_shape)

        cond = self.conditioner.apply(self.params.conditioner, states)
        cond_flat = jnp.broadcast_to(
            cond[:, None, :], (n, num_mc, cond.shape[-1])
        ).reshape(n * num_mc, cond.shape[-1])

        old_cond_flat = jnp.broadcast_to(
            self.conditioner.apply(self.params.old_conditioner, states)[:, None, :],
            (n, num_mc, cond.shape[-1]),
        ).reshape(n * num_mc, cond.shape[-1])
        v_old = self.actor.apply(
            self.params.old_actor, x_t_flat, tau_bn, old_cond_flat
        )

        v_new = self.actor.apply(self.params.actor, x_t_flat, tau_bn, cond_flat)

        # CFM losses under target velocities.
        v_star = (spectral_action_tokens[:, None] - eps).astype(jnp.float32)
        v_old_r = v_old.reshape(n, num_mc, *flow_shape).astype(jnp.float32)
        v_new_r = v_new.reshape(n, num_mc, *flow_shape).astype(jnp.float32)

        l_old_bin = compute_cfm_loss(v_old_r, v_star, reduce_mode="bin")
        l_new_bin = compute_cfm_loss(v_new_r, v_star, reduce_mode="bin")
        rho_bin = compute_policy_ratio(l_old_bin, l_new_bin)

        l_old = compute_cfm_loss(v_old_r, v_star, reduce_mode="sample")
        l_new = compute_cfm_loss(v_new_r, v_star, reduce_mode="sample")
        rho = compute_policy_ratio(l_old, l_new)

        adv_mc = advantages.astype(jnp.float32)[:, None]

        if cfg.use_per_bin_clip:
            n_bins = rho_bin.shape[-1]
            split = max(1, n_bins // 2)
            eps_bin = jnp.full(
                (n_bins,), cfg.epsilon_clip_highfreq, dtype=jnp.float32
            )
            eps_bin = eps_bin.at[:split].set(cfg.epsilon_clip_lowfreq)
            psi = compute_aspo_surrogate(rho_bin, adv_mc[:, :, None], eps_bin)
            policy_per_chunk = -psi.mean(axis=(1, 2))
        else:
            psi = compute_aspo_surrogate(rho, adv_mc, cfg.epsilon_clip)
            policy_per_chunk = -psi.mean(axis=1)

        # Value loss (Huber).
        v_pred = self.critic.apply(self.params.critic, states)
        returns_d = returns.reshape(-1)
        se = _huber(v_pred, returns_d, beta=1.0)
        if cfg.use_clipped_value_loss and "values" in batch:
            old_values = jnp.asarray(batch["values"], dtype=STORAGE_DTYPE)
            v_clipped = old_values + jnp.clip(
                v_pred - old_values, -cfg.epsilon_clip, cfg.epsilon_clip
            )
            clipped_se = _huber(v_clipped, returns_d, beta=1.0)
            value_per_chunk = jnp.maximum(se, clipped_se)
        else:
            value_per_chunk = se

        policy_loss = policy_per_chunk.mean()
        value_loss = value_per_chunk.mean()
        loss = policy_loss + cfg.vf_coef * value_loss

        jax.debug.callback(_raise_if_nonfinite, loss, policy_loss, value_loss)

        if not compute_info:
            return loss, None
        return loss, self._compute_info(
            l_old=l_old, l_new=l_new, rho=rho, rho_bin=rho_bin,
            v_star_f=v_star, v_pred=v_pred, returns_d=returns_d,
            policy_per_chunk=policy_per_chunk, value_per_chunk=value_per_chunk,
            advantages=advantages, cfg=cfg,
        )


    def _compute_info(
        self, *, l_old, l_new, rho, rho_bin, v_star_f, v_pred, returns_d,
        policy_per_chunk, value_per_chunk, advantages, cfg,
    ) -> dict[str, float]:
        delta = l_old - l_new
        log_rho_abs = jnp.abs(delta)
        log_rho_max = log_rho_abs.max()
        log_rho_clip_frac = (log_rho_abs > cfg.log_ratio_cap).astype(jnp.float32).mean()

        rho_mean = rho.mean()
        rho_std = rho.std()
        rho_bin_mean = rho_bin.mean(axis=(0, 1))
        n_bins = rho_bin_mean.shape[0]
        split = max(1, n_bins // 2)
        rho_bin_range = rho_bin_mean.max() - rho_bin_mean.min()
        rho_bin_low = rho_bin_mean[:split].mean()
        rho_bin_high = rho_bin_mean[split:].mean() if n_bins > 1 else rho_bin_mean.mean()
        rho_bin_outside = (
            ((rho_bin < 1.0 - cfg.epsilon_clip) | (rho_bin > 1.0 + cfg.epsilon_clip))
            .astype(jnp.float32)
            .mean()
        )
        eps_low = cfg.epsilon_clip_lowfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        eps_high = cfg.epsilon_clip_highfreq if cfg.use_per_bin_clip else cfg.epsilon_clip
        low_slice = rho_bin[..., :split]
        rho_bin_low_outside = (
            ((low_slice < 1.0 - eps_low) | (low_slice > 1.0 + eps_low))
            .astype(jnp.float32)
            .mean()
        )
        if n_bins > 1:
            high_slice = rho_bin[..., split:]
            rho_bin_high_outside = (
                ((high_slice < 1.0 - eps_high) | (high_slice > 1.0 + eps_high))
                .astype(jnp.float32)
                .mean()
            )
        else:
            rho_bin_high_outside = rho_bin_low_outside

        returns_var = jnp.maximum(returns_d.var(), 1e-8)
        explained_var = 1.0 - (returns_d - v_pred).var() / returns_var
        value_error_max = _huber(v_pred, returns_d, beta=1.0).max()
        value_pred_max = jnp.abs(v_pred).max()
        return_max = jnp.abs(returns_d).max()
        v_star_rms = jnp.sqrt((v_star_f ** 2).mean())

        l_old_std = l_old.std()
        l_new_std = l_new.std()
        corr_denom = l_old_std * l_new_std + 1e-8
        l_old_new_corr = (
            (l_old - l_old.mean()) * (l_new - l_new.mean())
        ).mean() / corr_denom

        vals = [
            policy_per_chunk.mean(), value_per_chunk.mean(), explained_var,
            delta.mean(), log_rho_max, log_rho_clip_frac, value_error_max,
            value_pred_max, return_max, rho_mean, rho_std, rho_bin_range,
            rho_bin_low, rho_bin_high, rho_bin_outside, rho_bin_low_outside,
            rho_bin_high_outside, v_star_rms, l_old.mean(), l_new.mean(),
            l_old_std, l_new_std, l_old_new_corr,
            jnp.abs(advantages).mean(), returns_d.std(), v_pred.std(),
        ]
        keys = [
            "policy_loss", "value_loss", "explained_var", "mean_log_ratio",
            "log_rho_max", "log_rho_clip_frac", "value_error_max",
            "value_pred_max", "return_max", "flow_rho_mean", "flow_rho_std",
            "flow_rho_bin_range", "flow_rho_bin_lowfreq", "flow_rho_bin_highfreq",
            "flow_rho_bin_outside_frac", "flow_rho_bin_low_outside_frac",
            "flow_rho_bin_high_outside_frac", "flow_v_star_rms", "l_old_mean",
            "l_new_mean", "l_old_std", "l_new_std", "l_old_new_corr",
            "adv_mean_abs", "value_target_std", "value_pred_std",
        ]
        return {k: float(v) for k, v in zip(keys, vals)}


def _huber(pred, target, beta: float = 1.0):
    """Elementwise Huber loss (reduction='none')."""
    diff = pred - target
    abs_diff = jnp.abs(diff)
    return jnp.where(
        abs_diff < beta,
        0.5 * diff ** 2 / beta,
        abs_diff - 0.5 * beta,
    )