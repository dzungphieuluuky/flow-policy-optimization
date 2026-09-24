# SJFPO Training Pipeline

This document describes the active Torch training path used by:

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --method sjfpo \
  --env Hopper-v5 \
  --seed 0 \
  --total_steps 1000000
```

The JAX method is dispatched separately through `scripts/train.py --method sjfpo_jax`; its differences are noted at the end.

## 1. CLI dispatch

`scripts/train.py` parses the run configuration and dispatches the method:

```text
sjfpo     -> sjfpo.train_sjfpo.train_sjfpo_run
ppo/sac   -> sjfpo.baselines
sjfpo_jax -> sjfpo_jax.train_sjfpo.train_sjfpo_run
```

The Torch SJFPO run creates or loads the policy, builds optimizer state, creates the training environment, evaluator, W&B logger, and checkpoint directory, then enters the rollout/update loop.

## 2. Policy construction

`build_policy()` constructs an `SJFPOPolicy` with:

- `StateConditioner`: observation window to a fused condition vector.
- `SJFPOFlowModel`: conditional flow/DiT actor.
- `Critic`: state-value predictor.
- `RunningMeanStd`: live observation normalization statistics.
- Frozen `old_actor` and `old_conditioner` snapshots for the behavior policy.

The policy is moved to the selected device and the actor, old actor, and critic are compiled when CUDA compilation is enabled. Set `SJFPO_COMPILE=0` to disable compilation.

## 3. Action representation and normalization

The simulator uses its real action bounds, for example `[-0.4, 0.4]`.

The policy predicts normalized actions in approximately `[-1,1]`. The postprocessor maps them back to the real environment bounds:

```text
real_action = 0.5 * (high - low) * normalized_action
              + 0.5 * (high + low)
```

The collector executes the real action, then converts it back to normalized space before storing the transition:

```text
stored_action = 2 * (real_action - low) / (high - low) - 1
```

Therefore, `Transition["action"]` stores normalized actions, not raw environment-range actions.

## 4. Rollout collection

Each training iteration collects:

```text
collect_steps = min(chunks_per_update * action_horizon, steps_remaining)
```

For each chunk:

1. The observation window is prepared and normalized.
2. The conditioner produces a condition vector.
3. The flow policy predicts a full action chunk.
4. The policy decodes the flow output into real-space actions.
5. Actions are scaled to the real environment bounds.
6. The chunk is executed open-loop in the environment.
7. Rewards, next observations, done/truncated flags, executed length, and clipping information are stored.

`action_is_pad` marks unexecuted tail steps when an episode ends before the full chunk is executed.

The default training path uses the full predicted chunk for collection. Evaluation may execute only a configured prefix and replan afterward.

## 5. Snapshot timing

At the beginning of each new rollout, the live actor and conditioner are copied into the frozen behavior snapshot:

```text
policy.snapshot_behavior_policy()
```

The snapshot is held fixed for all minibatches and epochs belonging to that rollout. It is refreshed only at the next rollout boundary.

## 6. GAE and value targets

The collector returns raw observations and rewards. The critic evaluates the initial and next observation windows. Chunk-level semi-MDP GAE then computes:

```text
advantages  [N]
returns     [N]
```

GAE discounts rewards within a chunk, handles termination/truncation, and optionally standardizes and clamps advantages. Rewards and observations remain FP32 in the collector; storage dtype is used later for training tensors.

The batch returned by `generate_rollout()` contains:

```text
observation.state   [N, T, obs_dim]
action              [N, H, action_dim]
action_is_pad       [N, H]
advantages          [N]
returns             [N]
values              [N]
tau                 [M]
eps                 [N, M, Ka, action_dim]
initial_cfm_loss   [N, M]
initial_cfm_loss_bin [N, M, Ka]
```

`N` is the number of chunks, `M` is `num_mc_samples`, and `Ka` is the number of retained spectral tokens when DCT is enabled.

## 7. CRN: Common Random Numbers

CRN means **Common Random Numbers**.

For each rollout, the sampler stores:

```text
tau: [M]
eps: [N, M, Ka, action_dim]
```

The exact same values are reused for:

- the rollout-time initial CFM loss;
- every minibatch;
- every epoch;
- the old/current policy comparison.

This makes the policy ratio meaningful because both policies are evaluated at exactly the same flow interpolation points and with exactly the same noise.

`tau` remains one-dimensional and is deliberately not sliced in `sample_minibatch`; all chunk-indexed tensors are sliced.

## 8. DCT action representation

When `use_dct_transform=True`, normalized real action chunks are transformed along the temporal horizon:

```text
A_real       [N, H, action_dim]
      DCT
A_spectral   [N, Ka, action_dim]
```

The number of retained coefficients is:

```text
Ka = round(top_p_spectral_tokens * action_horizon)
```

For `H=16` and `top_p_spectral_tokens=0.5`, `Ka=8`. Only the first eight low-frequency coefficients are retained.

Training currently uses the reduced spectral representation rather than a full `[H]` tensor plus a loss mask. The model is explicitly reduced to `[Ka]`, and every retained output participates in the loss.

## 9. Flow-matching sample construction

For every chunk and MC sample:

```text
x_t = (1 - tau) * eps + tau * A_spectral
```

The shapes are:

```text
x_t: [N*M, Ka, action_dim]
```

The target flow velocity is:

```text
v* = A_spectral - eps
```

The flow direction is `tau=0 -> 1`:

```text
tau=0: noise
 tau=1: clean spectral action
```

The actor predicts the velocity in spectral space.

## 10. Initial behavior CFM loss

The rollout stores the behavior CFM loss once, using the exact CRN draws:

```text
initial_cfm_loss     [N, M]
initial_cfm_loss_bin [N, M, Ka]
```

This is the CFM loss of the policy that generated the rollout. The update path does not recompute it for every minibatch.

The stored losses are sliced with the batch and used as the `l_old` terms in the policy ratio.

For compatibility, hand-built batches without these keys fall back to evaluating the frozen old actor during `forward`.

## 11. Current-policy forward

For a minibatch:

1. Normalize observations.
2. Apply the current conditioner.
3. Build the current-policy flow input from the stored `tau` and `eps`.
4. Run the current actor.
5. Compute `l_new` and optional `l_new_bin` CFM losses.
6. Combine stored `l_old` with `l_new` into the policy ratio.
7. Run the ASPO/PPO-style surrogate using advantages.
8. Run the critic and compute the value loss.

The normal policy ratio is:

```text
rho = exp(l_old - l_new)
```

If `use_per_bin_clip=True`, the bin-level ratio is used with separate low-frequency and high-frequency clipping thresholds.

## 12. Total loss

The policy loss is combined with the critic loss:

```text
total_loss = policy_loss + vf_coef * value_loss
```

The value loss is a Huber/smooth-L1 loss against GAE returns. If clipped value loss is enabled, the value prediction is also clipped relative to the rollout's old value estimate.

The loss rejects non-finite values with `SJFPOAborted`.

## 13. Epochs and minibatches

`mini_batch_indices()` produces the order of minibatches across all configured epochs.

Within one rollout:

- CRN `tau` and `eps` do not change.
- Stored initial CFM losses do not change.
- The behavior snapshot does not change.
- The live actor, conditioner, and critic update after each minibatch.
- Multiple epochs reuse the same rollout data and CRN samples.

This is the main difference from collecting a new behavior snapshot after every individual epoch.

## 14. Separate actor and critic optimizers

The live parameters are split into two ownership groups:

### Actor optimizer

Owns:

```text
actor parameters
conditioner parameters
```

It uses `policy_lr`.

### Critic optimizer

Owns:

```text
critic parameters
```

It uses `critic_lr`.

Frozen snapshot parameters are never included in either optimizer.

For every minibatch:

```text
actor_optimizer.zero_grad()
critic_optimizer.zero_grad()
total_loss.backward()
clip actor gradients separately
clip critic gradients separately
actor_optimizer.step()
critic_optimizer.step()
```

The actor and critic can therefore use different learning rates and independent optimizer states, while the combined loss still supplies both sets of gradients in one backward pass.

## 15. W&B logging

When enabled, training logs include:

```text
train/policy_loss
train/value_loss
train/explained_var
train/mean_log_ratio
train/flow_rho_*
train/grad_norm_preclip
train/grad_norm_after_clip
train/rollout_jerk
train/action_magnitude
train/total_loss
gae/advantages_mean
gae/advantages_std
gae/returns_mean
gae/values_mean
timing/fps
timing/elapsed_s
```

Evaluation logs are grouped by metric family:

```text
eval/mean_return
eval/min_return
eval/max_return
eval/std_return
eval/mean_length
eval/max_length
eval/std_length
eval/success_rate
eval/mean_jerk
eval/max_jerk
```

## 16. Evaluation and divergence

At each evaluation interval, the evaluator runs deterministic episodes with zero-noise action sampling where applicable. It writes the five-column CSV:

```text
episode,return,length,success,jerk
```

The random baseline is evaluated over 30 episodes, while regular policy evaluation remains 12 episodes by default.

The divergence check compares the current evaluation mean return against a fraction of the random baseline return for multiple consecutive evaluations.

## 17. Checkpointing and resume

New checkpoints store:

```text
config.json
model.safetensors
optimizer.pt
counters.json
rng.pt
```

The split optimizer payload is versioned and contains separate actor and critic states. Legacy single-optimizer checkpoints remain readable, but resumed legacy runs rebuild the split optimizers rather than attempting an unsafe state migration.

Resume starts a fresh environment rollout; it restores policy weights, optimizer state, counters, and optional RNG state, but not arbitrary environment state.

## 18. JAX path

The active JAX CLI path is:

```text
scripts/train.py
  -> sjfpo_jax.train_sjfpo.train_sjfpo_run
```

It uses the same overall rollout/CRN/flow/loss concepts, but:

- the loss and gradient update are wrapped with JAX JIT;
- Optax handles parameter updates;
- the policy is represented as JAX/Flax-style parameter trees;
- the current CLI path uses the Gymnasium vector collector;
- the separate Phase-5 MJX path is not the same live entrypoint.

## 19. Validation commands

Focused low-frequency DCT smoke:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/train.py \
  --method sjfpo \
  --env Hopper-v5 \
  --seed 0 \
  --total_steps 32 \
  --eval_steps 32 \
  --save_steps 32 \
  --wandb_enabled false \
  --sjfpo.use_dct_transform true \
  --sjfpo.top_p_spectral_tokens 0.5 \
  --sjfpo.num_envs 1 \
  --sjfpo.hidden_dim 8 \
  --sjfpo.dit_depth 1 \
  --sjfpo.dit_heads 2 \
  --sjfpo.dit_mlp_ratio 2.0 \
  --sjfpo.dit_time_dim 8 \
  --sjfpo.num_mc_samples 2 \
  --sjfpo.num_euler_steps 2 \
  --sjfpo.num_inference_steps 2 \
  --sjfpo.chunks_per_update 1 \
  --sjfpo.num_epochs 1 \
  --sjfpo.num_minibatches 1
```

Expected result:

```text
method: sjfpo
global_step: 32
```

with evaluation CSV and checkpoints under `.omo/evidence/...`.
