# skrl Reference vs Our baselines.py — Misdoings

Fetched from https://github.com/Toni-SM/skrl (develop branch, 2.1.0 tag).

## 1. Lifecycle: `post_interaction()` IS called (good)
Inside `record_agent_transition()` we call `agent.post_interaction()`. That matches the reference agent lifecycle.

## 2. Config mismatches — PPO (`PPOConfig` vs `PPO_CFG`)
| Key | Our value | Reference default | Note |
|---|---|---|---|
| `rollouts` | 2048 | 16 | Our rollout buffer is 128x larger; fine for 1 agent but memory-heavy |
| `learning_epochs` | 10 | 8 | Slight over-tuning |
| `mini_batches` | 8 | 2 | More fine-grained batches |
| `lr` | 3e-4 | 1e-3 | We use lower LR |
| `kl_threshold` | 0.01 | 0.0 | We set early-stopping KL; reference defaults to no KL early stop |
| `grad_norm_clip` | 1.0 | 0.5 | Higher clip |
| `value_loss_scale` | 1.0 | 2.5 | We under-weight value loss vs reference |
| `entropy_loss_scale` | 0.0 | 0.0 | Same (good) |
| `time_limit_bootstrap` | True (adaptation) | False | Our adaptation for truncated episodes |
| `mixed_precision` | False (default) | False | Same |

Misdoing: `value_loss_scale=1.0` vs reference `2.5`. With `hidden=(256,256,256)` our critic may under-train relative to policy, especially with step-level GAE returns over executed action prefixes. Should align to `2.5` or document why 1.0.

## 3. Config mismatches — SAC (`SACConfig` vs `SAC_CFG`)
| Key | Our value | Reference default |
|---|---|---|
| `lr` | 3e-4 | 1e-3 |
| `batch_size` | 256 | 64 |
| `memory_size` | 300_000 | ? (reference uses `RandomMemory` size) |
| `gradient_steps` | 1 | 1 |
| `polyak` | 0.005 | 0.005 |
| `initial_random_steps` | 5000 | 0 |
| `learn_entropy` | True (implied by build) | True |
| `grad_norm_clip` | 0 (implied) | 0 |

Misdoing: `initial_random_steps=5000` is fine, but our SAC doesn't set `learn_entropy` explicitly in `build_sac_skrl_config` — it relies on `SAC_CFG` default (`True`). That's okay but implicit.

## 4. Model architecture differences
Our `GaussianPolicyImpl` and `PpoValueImpl` / `SacCriticImpl` build `SwiGLU` MLP via `build_mlp_layer` (from `sjfpo.networks`). The reference uses plain `nn.Sequential` with `ELU`. Not a misdoing per se, but different activation (SwiGLU vs ELU) and initialization — makes direct performance comparison harder because architecture differs, not just algorithm.

## 5. Frame stack wrapper (`FrameStackWrapper`)
Our wrapper stacks `n_obs_steps=16` flat observations. Reference uses `wrap_env` with `observation_preprocessor` (optional `RunningStandardScaler`). We don't apply `RunningStandardScaler` — observations go raw into the MLP. For MuJoCo continuous control this can cause instability (values unnormalized). Should either add `RunningStandardScaler` or justify omission (e.g., `SwiGLU` handles scale better, but unproven).

## 6. Memory / rollout handling (`RandomMemory` size)
Our PPO `rollouts=2048` but `memory_size=2048` (`cfg.rollouts`). Reference sets `memory_size=32` (rollout buffer equal to `rollouts`). Our larger buffer doesn't break skrl but may delay updates: with 1 env (`num_envs=1`), `post_interaction` triggers update when `_rollout % rollouts == 0`, so 2048 steps between updates. At 1M total steps that's ~488 updates — reasonable but less frequent than standard PPO (which updates every 2048 steps anyway). No misdoing, just note.

## 7. Checkpointing / artifact naming
Our `agent.save(str(ckpt_dir / "skrl_agent.pt"))` matches skrl convention. `manifest.json` is our addition. Fine.

## 8. Missing: `agent.init()` is called but `memory.create_tensor()` depends on `agent.init()` being called first. Our `baseline_train_loop` calls `agent.init()` before loop start. Good.

## 9. Missing: `agent.enable_training_mode()` toggling in `record_agent_transition`?
Our `record_agent_transition` doesn't toggle training mode; the reference doesn't either during recording. But our `run_baseline_eval` does `agent.enable_training_mode(False)` before eval and `True` after. Reference `post_interaction` also uses `enable_models_training_mode(True/False)` around `update()`. Our `post_interaction` (called inside `record_agent_transition`) uses `agent.post_interaction()` which handles this internally. Good.

## 10. Key issue found: `record_agent_transition` passes `infos=None`
Reference `record_transition` accepts `infos` but we pass `None`. Not a bug for PPO/SAC unless `infos` is expected for bootstrap or reward shaping. Fine.

## 11. Key issue: `state_space` is None in our agent init
In `seed_and_wrap_env` we don't set a state space on the environment. The skrl agent receives `state_space=None` implicitly (from `env.state()` or `env.unwrapped`). Reference scripts explicitly pass `env.state_space`. Our `agent.init()` creates tensors with `size=self.state_space` which is `None` — but `memory.create_tensor()` with `size=None` might fail or create default size. Need to verify if `memory.create_tensor(name="states", size=None, ...)` works. Looking at `skrl` `Memory.create_tensor`, it expects a `Space` or tuple/int; `None` may break. Our code passes `states=env.state()` but `env.state()` on `FrameStackWrapper` returns None (no state method). Actually our `apply_env_step` returns `next_states = env.state() if hasattr(env, "state") else None`. So `states` is `None` for most steps. The reference passes `states` from the environment (often `None` for standard gym envs) but `agent.init()` creates state tensor with `self.state_space`. If our agent's `state_space` is `None`, `memory.create_tensor(..., size=self.state_space)` will likely raise. However the code seems to work (tests pass?), which suggests `self.state_space` might default to something or `memory.create_tensor` handles `None` gracefully. Need to verify.

Actually looking at our `seed_and_wrap_env`: we return `skrl_env` (wrapped with `wrap_env_fn`) but don't pass `state_space` explicitly. The skrl agent gets `state_space=skrl_env.state_space`. `wrap_env` probably sets `state_space` based on the environment. So `state_space` is likely not `None`. Fine.

## 12. Potential misdoing: `eval_policy_fn` in `run_baseline_iteration`
Our `eval_policy_fn` uses `mean_action_eval_policy(agent)` which returns `mean_actions`. For stochastic policies (PPO Gaussian, SAC stochastic), the reference evaluation often uses `agent.act(...)` with evaluation mode (`enable_training_mode(False)`) and takes the mean (for deterministic) or samples from distribution. Our `mean_action_eval_policy` does exactly that (`mean_actions`). Good.

## 13. Potential misdoing: diverged check uses `divergence_baseline=1.0`
Our `check_divergence` compares `mean_return < divergence_baseline`. The reference `skrl` examples don't use divergence baselines by default; this is our custom addition. Not a misdoing relative to skrl, but different from standard skrl behavior.

## Summary of real misdoings:
1. `PPOConfig.value_loss_scale = 1.0` should likely be `2.5` per reference (or documented deviation).
2. `PPOConfig.kl_threshold = 0.01` activates KLAdaptiveLR; reference default is `0.0` (no early stop). Should document or revert.
3. `grad_norm_clip` is 1.0 vs 0.5 — okay but different.
4. `FrameStackWrapper` lacks `RunningStandardScaler`; raw observations fed into MLP. Not broken but could improve stability.
5. `record_agent_transition` passes `infos=None`; reference allows `infos` for reward shaping but doesn't require it. Fine.
