# Optimization plan

The seven items from the profile, plus one harness bug, ordered so each step
is independently verifiable and the risky one is last.

**Nothing here is implemented.** This is the plan.

## Profile that sets the ranking

Actor forward, 2048 rows, CPU 4 threads: **1165 ms**.

| Component | ×6 blocks | share |
|---|---|---|
SwiGLU MLP | 566 ms | **49%** |
attention | 177 ms | 15% |
unaccounted (proj, pos, Python) | 204 ms | 17% |
`time_mlp` + timestep | 68 ms | 6% |
adaLN `CondMLP` | 67 ms | 6% |
norms | 5 ms | <1% |

Critic + encoder + losses: ~155 ms of a 2257 ms forward (**7%**). The SwiGLU
is 77.3 GFLOP, measured at 136 GFLOP/s. On a T4 that is 9.5 ms in fp32 and
**1.2 ms in fp16** — which is why step 7 is the only large lever and the rest
are one-liners.

## Blocker: no live test coverage

Every symbol this plan touches is referenced by **zero** live test files:

```
configure_torch_cuda         NONE
generate_mini_batch_indices  NONE
_EvalPolicy                 NONE
get_analysis_basis           NONE      (test_parity.py only, torch-vs-jax)
resolve_compute_dtype        NONE
autocast_context             NONE
```

`7204a6b "delete old tests"` removed `test_device.py`, `test_fpo_core.py`,
`test_config_yaml_parity.py` and `test_gae.py`. Those were the modules that
would have covered this work. **Each step below therefore carries its own
test, written before the change.**

Also live in `tests/__pycache__/`: 11 orphaned `.pyc` from deleted test
modules, including `test_train_run_parity`, `test_update_step_parity`,
`test_select_action_parity`, `test_train_cli`. Harmless, but they are what is
left of the coverage that was removed. Worth one `find` cleanup.

---

## Step 0 — harness bug (blocks the measurement for every later step)

`tmp/compile_sweep.py:131` calls `pol.critic(states)`. `Critic.forward` needs
`(states, actions)`, so `--target critic` fails with `TypeError` and that
sweep row cannot be produced at all. Anyone following `T4-KAGGLE.md` §7 on a
T4 hits this before measuring anything.

- Pass `batch[ACTION]` through alongside `states`.
- Re-run `--target critic --modes eager` to confirm the row appears.

No test: temp file. **Do this first** — it is the instrument for steps 4–7.

---

## Step 1 — `cache_size_limit` (free, unblocks step 6)

`torch._dynamo.config.cache_size_limit` defaults to **8**. Past 8 shape
variants dynamo **silently falls back to eager**: you pay full compile cost
and receive none of the benefit. Nothing in `src/` or `scripts/` sets it.

- Set `torch._dynamo.config.cache_size_limit = 32` inside
  `configure_torch_cuda()` in `scripts/train.py`, next to the existing CUDA
  tuning. One line.
- Needs no new name, no config field, no test.

**Why first:** it is the precondition for believing any measurement in step 6.
Without it, a compile sweep silently measures eager after 8 variants.

---

## Step 2 — resolve the matmul-precision conflict

`scripts/train.py:59` and `:65` set the **same knob twice with opposite
intent**:

```python
torch.backends.cuda.matmul.fp32_precision = "ieee"   # no TF32
...
torch.set_float32_matmul_precision("high")             # TF32 for fp32 GEMMs
```

Last call wins, so the `ieee` line is dead code and the effective setting is
`high`. Inductor reads the same setting.

- Decide which one is wanted and delete the other. **Open question for you:**
  `ieee` is the numerically strict choice, `high` is the ~8× faster one, and
  the comment ("TF32 for leftover fp32 GEMMs; autocast still handles bf16")
  says `high` was deliberate. My read is that `high` was intended and `ieee`
  is the leftover, but this changes fp32 GEMM numerics on any sm_80+ GPU, so
  it is your call and not mine to make silently.
- Test: assert the effective value after `configure_torch_cuda()`, so the
  surviving line and the intended setting cannot drift apart again.

---

## Step 3 — drop the `tau` range guard (unblocks compile)

`src/spectral_joint_policy/dct_utils.py:145` and `:159`:

```python
if not bool(torch.all((tau_q >= 0) & (tau_q <= T))):
    raise ValueError(...)
```

`torch.all(...)` → `bool(...)` is a device→host sync inside the CFM-loss path,
and the guard is unreachable: the only caller, `velocity_to_action_steps`,
builds the grid itself and lands in `[0.0, 0.9375]` with `T = 1.0`.

- Delete both guards.
- State the guarantee as a comment on `velocity_to_action_steps`: it owns the
  grid, so `tau` is in range by construction.
- Do **not** replace it with a compile guard — that reintroduces the sync.

Measured on CPU, deleting only this: **13 graphs → 4, 12 breaks → 3.**

- Test: extend the existing DCT tests to assert `decode_coeffs_to_velocity`
  raises on an out-of-range `tau` only where it is genuinely reachable, and
  that `velocity_to_action_steps` output is unchanged. `test_parity.py` already
  exercises the transform, so this is an addition, not a new file.

---

## Step 4 — `_EvalPolicy` per-step `eval()`/`train()` toggle

`src/spectral_joint_policy/train_sjfpo.py:80-86`:

```python
was_training = self.policy.training
self.policy.eval()
try:
    action = self.policy.select_action(...)
finally:
    self.policy.train(was_training)
```

Called once per env step during eval. Measured in `BLACKWELL.md`: **+0.835 ms
on a 1.375 ms policy call, +61%**, and it is pure Python.

- Hoist the mode switch to the episode boundary: set `eval()` in
  `_EvalPolicy.reset()` and restore in a `finish()`, so the per-step path is
  just `select_action`.
- The `finally` currently guarantees restoration on an exception; moving the
  switch changes where a mid-episode exception leaves the mode. Needs a
  decision: restore in `Evaluator` after `evaluate()` instead, or keep a
  try/finally at that level.

- Test: assert `_EvalPolicy` leaves `policy.training` unchanged across a full
  episode, and across an episode that raises.

---

## Step 5 — `get_analysis_basis` `@lru_cache`

`src/spectral_joint_policy/dct_utils.py:30`. Dynamo **ignores the cache
wrapper**, so inside a compiled region the basis is rebuilt per trace and a
warning is emitted. The work is trivial (`cos` on a `[4,16]` table) — this is
about the warning and trace cleanliness, not speed.

- Hoist to a module-level dict keyed by `(K, num_modes, device)`, which dynamo
  can reason about, or register the basis as a buffer on the flow model.
- Lowest priority in the plan. **Do it only if the warning shows up in a real
  compiled run**; do not do it speculatively.

---

## Step 6 — constant-shape minibatches (the real work)

`generate_mini_batch_indices` (`train_sjfpo.py:42`) splits `(i*n)//n_mb`, and
`n = n_chunks` is data-dependent: an episode ending mid-chunk produces a short
chunk, so the row count changes every rollout.

Measured compile cost of that churn on this box:

```
rows 2048   29.467 s  (compile)
rows 2032    7.981 s  (recompile)
rows 2016    1.029 s  (dynamic)
rows  700    0.312 s
```

That is minutes burned per run at Hopper's eval cadence.

- Pad `n` up to `num_mini_batches * ceil(n / num_mini_batches)` with repeated
  indices — repeat rather than drop, so no chunk is silently unlearned.
- **The same problem exists for the critic pass**: `generate_rollout` runs it
  over `cat([states, next_states])` = `2C` rows with `C` varying.
  `RolloutBuffer` already has a fixed `chunk_capacity` (`collector.py:80`), so
  pad to `2 * chunk_capacity` and carry a validity mask so the `amin` masks
  instead of slicing.
- **Verify padding waste stays under ~10%** using the wavefront histogram
  (`BLACKWELL.md` step 5). If padding exceeds that, the rows are being wasted
  on compute and the trade flips.

This step is what unlocks `dynamic=False`, which is the highest-value switch
in `TORCH-COMPILE.md` — dynamic kernels give up autotuned tile sizes and
epilogue fusion.

- Test: a padded minibatch must produce the same gradient sum as the unpadded
  one on the same chunks, i.e. padding must not change the update.
- Test: assert the compiled graph count does not grow across two rollouts with
  different chunk counts.

---

## Step 7 — fp16 autocast (the 8× lever; needs naming sign-off)

The repo already degrades correctly on a T4, and that is the problem:
`torch.cuda.is_bf16_supported()` is `False` on sm_75, so
`resolve_compute_dtype` returns **fp32** and `autocast_context` becomes a
`nullcontext`. The SwiGLU — 49% of the actor, 77.3 GFLOP — then runs on the
8.1 TFLOPS fp32 path instead of the 65 TFLOPS fp16 tensor path.

Two hard-coded spots:

| File | Current |
|---|---|
`utils.py:73` | `if want not in {"auto", "fp32", "bf16"}` |
`utils.py:97` | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` |

`autocast_context` hardcodes bf16, so this is not an env-var flip.

**Naming sign-off required before implementing** (`AGENTS.md`). Proposal:

- add `"fp16"` to the accepted `SJFPO_DTYPE` set, keeping `auto` on the
  current arch-probe behaviour so nothing changes by default
- branch in `autocast_context` on the resolved compute dtype

**Risks, in order of how likely they are to bite:**

1. **Range.** fp16 max is 65504. The critic regresses on **raw returns**, and
   Hopper best in this repo is **997**. That fits, but the margin is 65×, not
   1000×, and `test_value_target_precision.py` exists precisely because the
   return magnitude matters.
2. **No loss scaling in the current code.** `GradScaler` is not wired anywhere.
   fp16 training without it can underflow gradients. This is the real reason
   to think twice.
3. `_RMSNorm` already round-trips through fp32, which helps.

**Recommended sequence:** steps 1–6 are free or one-liners and must land first.
Step 7 is a separate commit with its own numerics gate — an eager fp32 vs fp16
twin on the same seed, diffing `explained_var`, `gae_linked_frac`,
`advantage_std_fraction`, and `actor_grad_norm_preclip` vs
`critic_grad_norm_preclip`, plus a `torch.isfinite` check on the raw returns
path. Do not fold it into a commit that also touches the trust region.

---

## Ordering

| Step | Risk | Needs a name? | Test |
|---|---|---|---|
0 harness bug | none | no | no |
1 `cache_size_limit` | none | no | no |
2 matmul precision | fp32 numerics | **decision** | yes |
3 `tau` guard | none | no | add to existing |
4 `_EvalPolicy` toggle | mode restoration | no | yes |
5 `lru_cache` | none | no | only if warning appears |
6 constant shapes | padding correctness | no | yes |
7 fp16 autocast | **precision** | **sign-off** | yes, twin run |

Steps 0, 1, 3 and 5 are zero-risk and can land as one commit.
Step 2 needs your decision. Steps 4 and 6 can share a commit.
Step 7 is alone.

## Not doing

Recorded so these are not re-litigated. All measured, all in `BLACKWELL.md`
or this session:

| Candidate | Measured | Verdict |
|---|---|---|
Critic members → `vmap` / `baddbmm` | 0.239 → 0.169 ms | critic is 0.007% of the update |
Nested `autocast(enabled=False)` in `_RMSNorm` | 1 graph, 0 breaks | traces fine |
`num_euler_steps` 4 vs 16 | 211.7 vs 212.0 return | saturated |
Batching 50 eval episodes | 1.375 ms/env at batch 1 vs 1.747 at batch 50 | compute-bound |
Gated residual → `addcmul` | 2.18× portable, changes residual numerics | declined this session |
`qkv` reshape → `permute+unbind` | ~3 µs of a 1165 ms forward | not worth the readability |
`_RMSNorm` fp32 round-trip | `BLACKWELL.md` Part C | leave it |
`SinusoidalEncoding` → `diffusers.get_timestep_embedding` | max abs diff 1.99 | rejected, also diverges from the JAX copy |