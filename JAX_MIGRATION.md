# JAX_MIGRATION.md — port SJFPO from PyTorch to JAX

Target: tens of millions of env steps per run. Learner in JAX/Flax NNX, MuJoCo
envs stay on CPU, one compiled program per rollout update.

Decisions taken (2026-09-30):

| # | Decision | Choice |
|---|---|---|
| 1 | Env strategy | JAX learner + CPU gymnasium/MuJoCo envs |
| 2 | Network style | Flax NNX modules |
| 3 | Layout | new `src/sjfpo_jax/`, torch kept as oracle |
| 4 | Checkpointing | Orbax `CheckpointManager`, new format |

---

## 0. Blockers before any code

| # | Blocker | Check | If it fails |
|---|---|---|---|
| 1 | `jax[cuda12]` not installed. `.venv` has jax 0.5.3, `jax.devices()` → `[CpuDevice(id=0)]` | `uv pip install -e ".[jax]"` or `uv pip install "jax[cuda12]"` | port is pointless, stop |
| 2 | Blackwell profile not taken. PROFILER.md CPU split says backward+opt = 61.4% | run PROFILER.md Step 2 on the GPU first | if env+rollout > 60% on GPU, the learner is not the bottleneck; fix the collector instead |
| 3 | CPU env throughput floor unknown | measure raw `AsyncVectorEnv.step` at `num_envs ∈ {64, 256, 1024}` | if the floor is < 2000 fps, 30M steps is unreachable regardless of framework |

`PROFILER.md` explicitly says "Do not carry the CPU conclusion over." Milestone 0
is that measurement, not the port.

## 1. The numbers the port has to beat

Current CPU reference (`PROFILER.md` Step 2, `num_envs=64`, 4096 steps/update):
13.71 s/update, **299 fps**.

| Budget | At 299 fps | Target | Required speedup |
|---|---|---|---|
| 30M steps | 27.9 h | 4 h | ≥ 7.0× |
| 100M steps | 92.9 h | 12 h | ≥ 7.7× |

Where the 299 fps goes on CPU (must be re-measured on Blackwell):

| segment | % of wall |
|---|---|
| `collect_chunks_vec` | 6.4 |
| `generate_rollout` minus collect | 9.5 |
| `update_forward` | 22.7 |
| **`backward` + optimizer** | **61.4** |
| drift diagnostics | 1.1 |

### What JAX actually removes

| win | mechanism | est. share of 299 fps |
|---|---|---|
| 8 separate fwd/bwd per update (4 epochs × 2 minibatches) become one `lax.scan` | 8 Python traversals + 8 launch batches → 1 | large part of the 84.1% |
| 2 critic forwards per optimizer step for drift metrics become tree reductions in the same program | deletes 1.1% and part of the 22.7% | ~2–3% |
| batch H2D once per rollout instead of once per minibatch | one memcpy, not 8 | small |
| donated `params` / `opt_state` | no alloc per step | wall-clock only, not VRAM |
| no bf16 autocast bookkeeping (see §7) | removes 6 dtype casts and their graph nodes | small but compounds in compile time |

### What JAX does not remove

- CPU MuJoCo `env.step`. This is the hard ceiling and it is unchanged by the port.
- The 16-step Euler rollout per chunk (still a `lax.scan`, not free).
- `num_mc_samples=16` × `batch_size=128` = 2048 flow evaluations per minibatch.

**Consequence to accept up front:** after the port, the env is the bottleneck. The
learner becomes cheap. That is the goal, but it means the next scaling lever is
MJX (deferred — decision 1), not more JAX learner work.

## 2. Target layout

```
src/sjfpo_jax/
  __init__.py
  config.py            # plain dataclass, mirrors SJFPOConfig field-for-field
  networks.py          # NNX: SwiGLU, CondMLP, SinusoidalEncoding, SelfAttention,
                       #       DiTBlock, SJFPOFlowModel, StateConditioner, Critic,
                       #       RunningMeanStd
  fpo_core.py          # sample_timesteps, sample_noise, run_monte_carlo_sampling,
                       # compute_cfm_loss, compute_aspo_surrogate, compute_policy_ratio,
                       # sample_action
  dct_utils.py         # extract_dct_coeffs, velocity_to_action_steps,
                       # decode_coeffs_to_velocity
  gae.py               # compute_chunk_advantages
  state.py             # TrainState (struct.PyTreeNode), build_optimizer
  policy.py            # forward (loss), predict_action_chunk, EMA, target critic
  collector.py         # gymnasium loop + jitted tensor bookkeeping
  checkpointing.py     # Orbax save/restore
  tracking.py          # wandb/tensorboard, unchanged keys
  train.py             # train_sjfpo_run
tests/
  test_*_jax.py        # mirrors of the 8 existing test modules
  test_parity.py       # torch vs jax, same inputs
```

Torch `src/sjfpo/` stays untouched. `src/sjfpo_jax/` must not import `lerobot` or
`torch` — `lerobot` is a torch library, and importing it would drag the whole
torch runtime into the JAX process. That forces `config.py` to be a plain
dataclass rather than a `PreTrainedConfig`.

`configs/sjfpo.yaml` is read by `scripts/train.py` via draccus. Reuse the same
file, same field names, so one YAML drives both frameworks.

## 3. torch → JAX API map

| torch | jax / flax | notes |
|---|---|---|
| `nn.Linear` | `nnx.Linear(in, out)` | NNX kernel is `[in, out]`; a checkpoint converter must transpose torch's `[out, in]` |
| `nn.Linear` + `nn.init.normal_(std=0.02)` | `nnx.Param(0.02 * jax.random.normal(k, shape))` | `token_pos` |
| zero-init (`adaLN.fc2`, `adaLN_out`, `out`) | `nnx.Param(jnp.zeros(...))` | keeps adaLN-Zero identity at step 0 |
| `F.silu` | `jax.nn.silu` | verified |
| `F.rms_norm` | `jax.nn.rms_norm(x, (D,), weight, eps)` | **verify** signature in the installed flax |
| `F.scaled_dot_product_attention` | `jax.nn.dot_product_attention(q, k, v, bias=...)` | **verify**; SDPA-style bool mask polarity is the opposite of JAX's additive bias. Port as: `keep` `[B, L]` → `bias = where(keep[:, None, None, :], 0.0, -inf)`, then force `keep[:, 0] = True` to kill the all-masked NaN row |
| `torch.chunk(2, -1)` on a fused `gate_up` | `jnp.split(..., 2, axis=-1)` | keep the fused GEMM |
| `torch.Tensor.unbind(dim=)` | `jnp.split(..., 6, axis=-2)` | adaLN 6-way |
| `nn.ModuleList` of 2 `StateConditioner` | stacked `nnx.Param` with leading member axis + `jax.vmap` | **required**, not optional: a `ModuleList` inside `lax.scan` reintroduces the per-member Python loop. See §6 |
| `torch.no_grad` / `inference_mode` | `jax.lax.stop_gradient` + `nnx.jit(..., static_argnames=...)` | there is no inference mode; put the no-grad pass behind `jax.lax.stop_gradient` on params, or run it in its own jitted fn |
| `.detach()` | `jax.lax.stop_gradient` | |
| `.float()` / `autocast(enabled=False)` | `.astype(jnp.float32)` | JAX does **not** auto-promote int→float. Every torch `.float()` needs an explicit cast or you get an int matmul |
| `torch.randperm` | `jax.random.permutation` | |
| `torch.randn` | `jax.random.normal` | |
| `torch.Generator` | `jax.random.key` (typed keys) + `fold_in`, threaded in `TrainState.key` | typed keys, not `PRNGKey`. Verified in the JEP 9263 docs |
| `torch.argsort(stable=True)` | `jnp.argsort(env_id, stable=True)` | GAE env grouping |
| `torch.linalg.vector_norm` | `jnp.linalg.norm` | drift metrics |
| `torch.clamp(x, max=c)` | `jnp.clip(x, max=c)` | **must be `clip`, not `minimum`** — see §7 |
| `torch.where` / `einsum` / `rsqrt` / `square` | `jnp.where` / `jnp.einsum` / `jnp.rsqrt` / `jnp.square` | 1:1 |
| `copy.deepcopy` for `old_actor` / `target_critic` | a second pytree key in `TrainState` | `old_actor`, `old_conditioner`, `target_critic`, `ema_shadow` are 4 extra sub-trees of `TrainState`, all donation-eligible |
| `optimizer.step()` (one AdamW, 2 param groups) | see §5 | per-role clip is the constraint, not per-role lr |
| `clip_grad_norm_` (returns pre-clip norm) | custom 6-line optax transform, see §5 | `optax.clip_by_global_norm` does not surface the norm |
| `orbax`-less hand-rolled save | `ocp.CheckpointManager` + `ocp.args.StandardSave(nnx.state(model))` | verified pattern |
| `safetensors` | keep a converter both ways | see §8 |
| `lru_cache` device freq tables | module-level `jnp` constants built once | no device keying needed; one device |
| `torch.autocast(bf16)` | **do not port** | see §7 |

## 4. The chunk-GAE port (`gae.py`)

Current code is 1 gather + 1 sort + a **Python loop** over `n_chunks`
(`gae.py:112-114`). JAX has no Python loop, so:

1. `chunk_rewards`: unchanged, still a `[B, chunk_size]` gather + `gamma**k` weights.
2. `order = jnp.argsort(env_id, stable=True)`; gather all per-chunk fields by `order`. Unchanged semantics.
3. `same_env_sorted`, `gamma_h`, `delta`, `continuation_factor` — all vectorized already.
4. Replace the loop:

```python
# c[i] is the coefficient on A_{i+1}; c[-1] = 0 kills the tail, exactly like
# continuation_factor[-1] staying zero in the torch version.
_, advantages = lax.scan(
    lambda a_next, dl_c: (dl_c[0] + dl_c[1] * a_next, dl_c[0] + dl_c[1] * a_next),
    jnp.float32(0.0),
    (delta[::-1], continuation_factor[::-1]),
    reverse=True,
)
advantages = advantages[::-1]
```

`lax.scan(..., reverse=True)` is verified in the JAX control-flow docs and is
reverse-mode differentiable, so the value-loss path is unaffected.

`lax.associative_scan` would turn the O(n_chunks) recursion into a work-efficient
parallel scan. n_chunks ≈ 256 at default settings, so the win is small and the
API (custom associative operator) is unverified. **Do not use it.** Revisit only
if `chunks_per_update` grows past ~4096.

## 5. Optimizer: keeping the per-role clip

`build_optimizer` (modeling_sjfpo.py:128) uses **one** AdamW with two param
groups, because the critic's gradient norm is 45–170× the actor's and a shared
clip would let the critic own the actor's step. `optax.adamw` is one
`GradientTransformation` over one pytree, so per-role clip norms need an
explicit split:

```python
def global_norm_and_clip(max_norm):
    """optax.clip_by_global_norm, but stashes the pre-clip norm in the state.

    The training loop logs actor/critic grad norms separately, and
    optax.clip_by_global_norm discards them.
    """
```

then

```python
adamw = optax.adamw(lr=policy_lr, weight_decay=optimizer_weight_decay)
updates, opt_state = adamw.update(grads, opt_state, params)
```

with `grads` shaped `{"actor": clip_actor(actor_grads), "critic": clip_critic(critic_grads)}`
before the AdamW. The pre-clip norms come back from the two clip states. This
reproduces the torch contract exactly and is ~10 lines.

`grad_accum_steps`: `optax.MultiSteps(adamw, every_k_schedule=...)` gives the same
"flush the final partial group" behaviour the loop hand-rolls today
(train_sjfpo.py:426-479) — except `MultiSteps` needs a schedule, and the torch
loop's real contract is "average within a group of `grad_accum_steps`, clip and
step at the boundary." At the default `grad_accum_steps=1` this is a no-op.
**Recommendation: drop `grad_accum_steps` from the JAX config and require the
batch to fit.** If it is ever needed, add `MultiSteps` then. Do not port a
feature the default config disables.

## 6. Networks: the two structural changes

**(a) Critic becomes a `vmap`, not a `ModuleList`.** `Critic.forward`
(networks.py:711) stacks `num_critics` independent `StateConditioner`s. Keeping
`ModuleList` means a Python loop inside the jitted loss. Stack the params on a
leading member axis and `vmap` the member function. This is the single most
important structural change: it is the "struct-of-arrays, no Python loops inside
`jit`" rule that JAX forum advice converges on (jax discussion #23200 — a Python
loop over a list of train states caused compile times from minutes to 12 hours).

Consequence: the checkpoint key layout changes from
`critic.conditioners.0.frame_proj.kernel` to
`critic.frame_proj.kernel[0]`. The converter in §8 handles it.

**(b) `nnx.Module` or functional pytrees at the train-step boundary.** Build the
networks as `nnx.Module` (readable, 1:1 with `networks.py`) but drive them from
the jitted step through the functional API:

```python
graphdef, state = nnx.split(policy)
# inside the jitted step:
model = nnx.merge(graphdef, state)
```

Do **not** use `@nnx.jit` / `nnx.Optimizer`. Two known issues: `nnx.Variable`
implements `__jax_array__`, which makes JAX transforms treat it as an Array and
fail intermittently (flax #4722, resolved by `.value`); and the flax maintainers
recommend the plain `jax.jit` + split/merge path for anything that crosses a
`lax.scan` boundary. Also: passing an `nnx.Module` as a *closure* into a
`lax.scan` is explicitly unsupported.

## 7. Precision: do not port bf16

`utils.py:54-88` selects bf16 storage and bf16 autocast on the RTX PRO 1000
Blackwell. Porting that into JAX is the highest-risk item in the plan:

- `jnp.sum` on bf16 silently upcasts to fp32, but **reductions that XLA
  auto-generates in the backward pass** (from broadcasts, i.e. every
  Linear-with-bias) run in bf16 and are numerically unstable — jax issue #30310.
- There is no autocast. The equivalent is
  `jax.config.update("jax_default_matmul_precision", ...)`, which is a *global*
  switch, not a scoped context. You cannot express "fp32 storage, bf16 matmul,
  fp32 reductions" per-region.
- The existing code already hard-codes fp32 regions: `compute_cfm_loss` under
  `autocast(enabled=False)`, `_RMSNorm` reducing in fp32, the DCT basis forced to
  fp32. Those are the numerically load-bearing parts and they want fp32 in JAX
  too.

**Plan: the JAX port is fp32 throughout. No `SJFPO_DTYPE`, no
`jax_default_matmul_precision`.** Milestone 5 revisits this as a separate
experiment with the §1 harness as the arbiter.

Two precision traps that *will* bite during the port:

1. `compute_policy_ratio` (fpo_core.py:281-285) clamps `delta` at
   `log_ratio_cap` via a one-sided straight-through estimator. The
   `torch.clamp(..., max=cfm_clamp)` calls above it must become `jnp.clip`, not
   `jnp.minimum` — the whole `ABOVE_CLAMP_ABORT_UPDATES` machinery depends on the
   gradient being *exactly zero* above `cfm_clamp` (train_sjfpo.py:538-548).
2. The GAE comment (gae.py:78-84) is explicit: a cumsum-diff discount needs
   `gamma**-s_i`, which loses all fp32 precision past a few thousand steps. Keep
   the gather + `gamma**k` form. Do not "optimize" it to a `jnp.cumsum`.

## 8. Checkpointing

```python
manager = ocp.CheckpointManager(
    out_dir / "checkpoints",
    options=ocp.CheckpointManagerOptions(
        preservation_policy=ocp.preservation_policy.LatestN(2),
        temporary_path_class=atomicity.CommitFileTemporaryPath,
    ),
)
manager.save(step, args=ocp.args.StandardSave(checkpoint_pytree))  # async
```

`checkpoint_pytree` holds `nnx.state(policy)`, the optax state, the PRNG key,
`global_step`, and the `RunningMeanStd` `count` (fp64 — keep it fp64, torch dies
at 2^24 frames, per networks.py:4).

**Donation hazard:** if `params` is passed with `donate_argnums`, the donated
buffer is invalid after the call, and Orbax's async save still holds a reference
to it. With ~1–2M params (~8 MB) the fix is trivial: pass
`out_shardings` and hand Orbax a `jax.device_get` copy, or keep the params copy
outside the donated argument. Do not skip this — it is a silent
wrong-checkpoint class of bug.

**Converter:** one script, `scripts/convert_torch_to_jax.py`, reading the existing
safetensors + torch optim state and emitting the Orbax tree. It needs the §6(a)
transpose and the `[out,in] → [in,out]` linear transpose. Without it, every
existing run is a dead end. **Build the converter before the trainer.**

## 9. Collector: hybrid Python env, JAX bookkeeping

The env stays `gym.vector.AsyncVectorEnv`. The Python loop stays, because
`env.step` is a Python call. Everything else moves to JAX.

Current cost (collector.py:689-810): per wavefront, a Python loop over 64 envs
doing `torch.zeros` / `copy_` / `float()` per env, plus 16 window shifts and
reward writes. That is thousands of tiny ops per wavefront.

Port:

| now | jax |
|---|---|
| `windows` `[n, T, obs]` list of deques | one `jnp` array, shifted with `.at[:, :-1].set(...)` |
| `reward_windows` / `step_frames` per wavefront | jnp arrays, `.at[:, pos].set(...)` |
| `RolloutBuffer` `torch.zeros` + `copy_` + `ensure_chunk_capacity` | preallocated jnp arrays at max capacity; the dynamic chunk count becomes a Python int that slices the prefix at use time — no growth path needed |
| `buffer.copy_to()` host→device staging | gone; the buffer *is* on device. Deletes the pinned-memory + `cuda.Event` machinery in modeling_sjfpo.py:210-212 |
| 16 per-step window shifts per wavefront | one jitted `advance_windows(windows, obs, rewards, term, trunc)` over the `chunk_size` inner steps, called once per wavefront |
| `actions_t.cpu().numpy()` | one `jax.device_get` per wavefront, batched over envs (not per env) |
| `chunk_env_id` / wavefront bookkeeping | stays Python ints; the env loop cannot be vmapped |

One D2H per wavefront is the floor and is already what the code does. Do not
chase the per-env copy.

`collect_chunks_vec` is 6.4% of wall on CPU and unknown on Blackwell. Port it in
milestone 3, not milestone 1. If the Blackwell profile says collection is small,
a direct `RolloutBuffer`-shaped jnp port with no jitted inner loop is enough.

## 10. The update step

One `@jax.jit` per rollout, structured as brax PPO does it:

```
update_step(state, batch, key) -> (state, metrics)
├── lax.scan over epochs            (num_learning_epochs)
│   └── lax.scan over minibatches   (num_mini_batches, from a jnp permutation)
│       ├── behavior CFM loss   (stop_gradient on old_actor / old_conditioner)
│       ├── v_new forward, v_star, CFM losses, ratio, ASPO surrogate
│       ├── critic forward, value loss
│       ├── value_and_grad over the merged {actor, critic} param tree
│       ├── per-role clip + AdamW
│       ├── target_critic EMA, actor EMA
│       └── drift metrics as tree reductions
└── metrics dict
```

Everything currently spread over train_sjfpo.py:426-531 becomes one program. The
per-epoch `jax.random.permutation` reshuffles for free inside the scan.

**Over-jitting warning:** jax discussion #33912 reports the *outer* jit being 5×
*slower* than inner jits with `donate_argnums` on a large buffer, because XLA
stops proving safe in-place reuse and materialises a full copy. There is no large
donated buffer inside `update_step` (the batch comes in as an argument), so the
risk is low — but **measure**. Fallback if it regresses: jit the minibatch step
only, scan outside. Report which one won in the milestone.

`minibatch_indices` (train_sjfpo.py:36) keeps its proportional-split semantics
(`(i*n)//n_mb`), computed once on the host and passed in as a jnp array.

## 11. Milestones

Each ends with a number, not a feeling.

| # | Milestone | Done when |
|---|---|---|
| 0 | Install `jax[cuda12]`; run PROFILER.md Steps 2+3 on Blackwell | wall-clock split on the real GPU recorded in PROFILER.md; the §0 blockers resolved |
| 1 | `fpo_core.py`, `dct_utils.py`, `gae.py` ports + `test_parity.py` | JAX and torch agree to `rtol=1e-5, atol=1e-6` on fixed-seed inputs for `compute_cfm_loss` (all 3 reductions, masked and unmasked), `compute_aspo_surrogate` (ppo/spo/aspo), `compute_policy_ratio` (incl. the zero-grad-above-clamp case), `sample_noise`/`sample_timesteps` statistics, both DCT functions, `compute_chunk_advantages` (incl. timeout-keeps-bootstrap, truncation split, budget-truncated chaining) |
| 2 | `networks.py` + `policy.py` ports | forward parity: same params → same velocity/value/`compute_info` dict to `rtol=1e-4` (fp32 GEMM ordering differs); `test_rollout_storage` and `test_critic_window` mirrored |
| 3 | `collector.py` port | rollout parity: given a fixed policy and fixed `envs.reset(seed=...)`, the JAX collector and the torch collector produce identical `chunk_state` / `chunk_actions` / `action_is_pad` / `step_rewards` on `Reacher-v5` with `num_envs=4` |
| 4 | `state.py` + `train.py` | 1000-step run on `Reacher-v5`; eval return within noise of the torch run at the same seed; `fps` recorded |
| 5 | Throughput | **≥ 7.0× vs the 299 fps CPU reference, or ≥ 2000 fps, on a 4 h run.** If the env floor from §0.3 is the limit, that number is the finding — report it and stop |
| 6 | `checkpointing.py` + converter | resume from a converted torch checkpoint reproduces the torch eval return; resume from a JAX checkpoint is bit-identical across 2 runs |
| 7 | (optional) bf16 experiment | only if milestone 5 is env-bound. Compare `jax_default_matmul_precision` against fp32 with §1's harness. Expect the §7 backward-reduction instability |

**Tolerances are not negotiable.** If a parity test fails at 1e-6, the JAX code is
wrong. Do not widen the tolerance to make it pass.

## 12. Traps, with sources

| trap | source | mitigation |
|---|---|---|
| static arg in `jit` → recompile per value | jax discussion #23200 (minutes → 12 h) | one batched train state; no static agent/env index |
| async dispatch: enqueue queue ~32 deep, then hang; `print()` on a device value lies | jax #21946 | `jax.block_until_ready(metrics)` once per update, exactly once |
| donation needs `out_shardings` at `jit` time, else `with_sharding_constraint` is not a substitute and you get "donated buffers were not usable" | jax #18263 | pass `out_shardings`; single device here, so `donate_argnames` on `params`/`opt_state` only |
| bf16 backward-pass reductions run in bf16; TPU already upcasts so GPU/TPU differ silently | jax #30310 | §7: fp32 only |
| `vmap` materialises large intermediates → OOM at 128 GB host RAM | jax #32201 | `lax.scan` over the epoch/minibatch axes (already the design); `jax.lax.map` only if a specific batch OOMs |
| `vmap` over envs: `key_envs` must be reshaped to the training batch size | brax #591 | shape keys `(num_envs,)` at the call site; add a test |
| slow compile regression in jax 0.4.36–0.4.38 (XLA inliner + partitioner) | jax #26162, brax #569, mujoco_playground #11 | jax 0.5.3 is past it. If a hang appears, check the version first |
| `jit` inside `pmap` pathologies; brax's `train.py` is the known-good scaffold partly because it avoids them | jax #14602 | no `pmap` in this plan (single device) |
| device→device copy blocks the other stream; 384 small memcpys → 1 large memcpy gained 10% | jax discussion #14693 | one batched `device_get` per wavefront (§9) |
| extracting concrete values from tracers (`int(x)`, `.item()`) | nboyd porting notes | all shapes static; chunk count stays a host int |
| Python `for` loops unroll at trace time, bloating compile time and memory | nboyd porting notes | `lax.scan` for epochs/minibatches/GaE/Euler |
| `lax.cond` runs *both* branches under `vmap` | nboyd porting notes | prefer `jnp.where`; the trust-region ASPO selection already is one |

## 13. Do not port

Scope discipline. Each of these is a torch-ism or a LeRobot integration with no
place in a JAX learner:

- `processor_sjfpo.py` — LeRobot pre/post-processor plumbing. In JAX,
  `RunningMeanStd.forward` is 4 lines and the action rescale is 2.
- `checkpointing.py`'s `ResumeError`, atomic-write helpers, and RNG save.
- `tracking/` — copy as-is, it is framework-agnostic (tensorboard/wandb take
  floats).
- `evaluator.py` — numpy/venv code, no torch.
- `torch.compile` concerns from PROFILER.md Steps 6/7. No analogue.
- `grad_accum_steps` (§5).
- `SJFPO_DTYPE` / `autocast_context` (§7).
- `resolve_device` / `is_rtx_1000_pro_blackwell` — device selection is
  `jax.default_device()` and an env var read.

## 14. Kill criteria

Stop and report if any of these hold:

1. Milestone 0 shows the Blackwell wall-clock split has collect+rollout > 60%.
   The learner was never the problem.
2. Milestone 0.3 shows the CPU `AsyncVectorEnv` floor < 2000 fps. 30M steps is
   out of reach with gymnasium/MuJoCo, and the correct next move is MJX
   (MuJoCo Playground, decision 1 revisited), not more learner work.
3. Milestone 1 parity cannot reach 1e-6 on `gae.py` or `compute_cfm_loss`. That
   means a semantic difference, not a precision one, and a wrong GAE would
   silently ruin every long run.
4. Milestone 5 reaches < 3× and the remaining time is still in the fused update.
   At that point XLA is the floor and the answer is a smaller/faster kernel
   (Pallas), not more porting.

## 15. Names (settled 2026-09-30)

Confirmed with the user. No open naming questions remain.

| item | decision | note |
|---|---|---|
| package | `sjfpo_jax` | §2 |
| module split | `networks.py`, `fpo_core.py`, `dct_utils.py`, `gae.py`, `state.py`, `policy.py`, `collector.py`, `checkpointing.py`, `config.py`, `train.py` | as §2 |
| state container | `SJFPOState`, a `struct.PyTreeNode` in `state.py` | avoids shadowing `flax.training.train_state.TrainState`; the `nnx.Optimizer` path is rejected, §6 |
| NNX classes | identical to torch: `SJFPOFlowModel`, `StateConditioner`, `Critic`, `DiTBlock`, `SelfAttention`, `SwiGLU`, `CondMLP`, `SinusoidalEncoding`, `RunningMeanStd`, `modulate` | makes the port auditable against `networks.py` line by line. `Critic` internals still change per §6(a) |
| math functions | identical to torch: `compute_cfm_loss`, `compute_aspo_surrogate`, `compute_policy_ratio`, `sample_timesteps`, `sample_noise`, `run_monte_carlo_sampling`, `sample_action`, `compute_chunk_advantages`, `extract_dct_coeffs`, `velocity_to_action_steps`, `decode_coeffs_to_velocity` | parity tests read as direct calls on both sides |
| collector buffer | `RolloutBuffer` | jnp arrays, preallocated, no growth path (§9) |
| new fn 1 | `global_norm_and_clip` | optax clip transform that stashes the pre-clip norm, §5 |
| new fn 2 | `update_step` | the jitted rollout→update program, §10 |
| new fn 3 | `advance_windows` | jitted per-wavefront collector bookkeeping, §9 |
| config | `SJFPOConfig` in `src/sjfpo_jax/config.py`, plain dataclass, field-for-field mirror of the torch one | the two classes coexist in the parity test without aliasing |
| dropped config fields | `grad_accum_steps` (must be 1, raises otherwise), `SJFPO_DTYPE` | §5, §7. `grad_accum_steps` stays in `configs/sjfpo.yaml` for compatibility |
| W&B / TB keys | unchanged from the torch loop (`train/policy_loss`, `train/critic_grad_norm_preclip`, `train/boot_value_delta`, `train/value_target_drift`, …) | no new metric names |
| script | `scripts/convert_torch_to_jax.py`, both directions (`--to-jax`, `--to-torch`) | §8. Needed so `play.py` keeps reading JAX checkpoints |
