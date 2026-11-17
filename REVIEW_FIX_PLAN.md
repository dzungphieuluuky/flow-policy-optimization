# REVIEW FIX PLAN — `src/myfpo/myfpo.py`

> **Status 2026-11-18 (archival): phases 0–4 are complete and committed.**
> This file is the historical record of the fix plan written against HEAD
> `12a38cb` (red test suite). See `FINDINGS.md` for what landed and how the
> benchmark turned out. Phase 5 (affine ablations, target-critic option B)
> was not run.

Scope: the code-review feedback on `src/myfpo/myfpo.py` (single file, 1823 lines,
HEAD `12a38cb`). This file checks every claim against the code, then gives the fix
plan. Claims that do not hold, and items this plan drops, are in section 8 with the
reason.

This plan follows the repo's flat style rules (root `AGENTS.md`): no new
abstractions without a caller, no wrappers for one implementation, no speculative
generality.

---

## 0. Blocker that the review did not name

**The test suite is red today.** It cannot gate any change below.

```
pytest -q
  7 collection errors  (stale imports of names the single-file rewrite deleted)
  11 failed            (error-message wording drift, no behaviour difference)
  104 passed
```

`ruff check src scripts tests` is also red: 12 errors at HEAD.

```
src/myfpo/myfpo.py:1305:5  E731  Do not assign a `lambda` expression, use a `def`
src/myfpo/myfpo.py:1502:39 ARG002 Unused method argument: `seed`
src/myfpo/ppo.py:512,514,515,530,531,557,558,559  F821 Undefined name
```

### The PPO baseline crashes at the first eval

This is outside the review's scope, but it blocks the comparison method, so it
is a Phase 0 item.

`src/myfpo/ppo.py` still calls the backend abstraction that the single-file
rewrite deleted everywhere else:

```
NameError: name 'backends' is not defined
  File "src/myfpo/ppo.py", line 512, in run_ppo_training
    if backends:
```

`backends`, `log_metrics` and `finish_backends` are never defined or imported.
The file already has the replacement: `wandb_run = _init_wandb(...)` at line 333
and `wandb_run.log(...)` at line 477. The `if backends:` blocks at lines 512 and
557 are leftovers and must go.

Every `--method ppo` run dies at the first eval with exit code 2. The PPO
baseline has produced no numbers since the single-file rewrite. Any comparison
between myfpo and PPO made since then is missing its baseline.

Collection errors (the names no longer exist):

| test module | missing name | old home |
|---|---|---|
| `test_gae.py` | `gae_trace` | inlined into `compute_chunk_advantages` |
| `test_grad_clip.py` | `_param_groups_for_clip` | inlined into `update` |
| `test_shared_env_carry.py` | `_shared_env` | inlined into the training loop |
| `test_cfm.py` | `sample_noise` | inlined into `sample_chunk` / `run_rollout` |
| `test_cfm_loss_target.py` | `cfm_target` | inlined into `token_cfm_loss` |
| `test_bootstrap_values.py` | `estimate_state_values` | renamed `_estimate_state_values` |
| `test_checkpoint_eval.py` | `sample_random_baseline` | replaced by `evaluator.random_policy` |

The 11 failures are all message wording. The guard still raises `ValueError`, but
with a different string:

| test expects | code raises |
|---|---|
| `states must have shape` | `expected [B, 16, 39], got (4, 16, 40)` |
| `max_obs_steps must be >= 1` | `invalid StateEncoder(hidden_dim=128, max_obs_steps=0)` |
| `hidden_dim must be at least 2` | `invalid StateEncoder(hidden_dim=1, max_obs_steps=16)` |
| `chunk capacity exceeded` | `RolloutBuffer capacity exceeded` |
| `dim must be >= 2` | `invalid SinusoidalEncoding(dim=1, max_period=10000.0)` |
| `max_period must be > 0` | `invalid SinusoidalEncoding(dim=16, max_period=0.0)` |
| `done rows` | `SAME_STEP env reset but supplied no final observation for rows [0]` |

**Phase 0 repairs the gates before any other phase starts.** Rule 2 of
`SIMPLIFICATION_PLAN.md` says a change must not add a failure. A red suite cannot
show that.

---

## 1. Verdict summary

Severity: **S1** changes the learning result. **S2** changes cost or reproducibility.
**S3** is style.

| # | claim | verdict | evidence | phase |
|---|---|---|---|---|
| 1a | target critic barely moves | **confirmed (S1)** | probe: after 6 rollouts `\|Q_target\|/\|Q_online\| = 0.011` | P1 |
| 1b | rollout size uses `action_horizon` | **confirmed (S1)** | probe: `num_action_steps=2` gives 128 chunks for a 64-chunk target | P1 |
| 1c | first rollout trains on raw obs | **confirmed (S1)** | `obs_normalizer.uninitialized` is still `True` after collection | P1 |
| 1d | `"sqrt"` reduction is a mean under the mask | **confirmed (S2)** | probe: masked loss `0.75` equals the mean, the sqrt value is `1.5` | P1 |
| 1e | idle envs step with zero actions | **confirmed (S2)** | probe: 3 of 4 envs idle-stepped with zeros | P1 |
| 1f | `reward_mean_last10` averages every eval | **confirmed (S3)** | one line in the return dict | P1 |
| 1g | `FlowMLP.forward` unpacks before the dim check | **confirmed (S3)** | line 368 | P1 |
| 1g | single-env `Evaluator` consumes the global torch RNG | **confirmed (S2)** | `Evaluator.evaluate` has no `fork_rng`; `VectorEvaluator` has one | P1 |
| 1g | `trust_region_mode` and `cfm_loss_reduction` unvalidated | **confirmed (S2)** | a typo falls through to `"aspo"` and `"mean"` | P1 |
| 2 | replace the ring buffer with a per-step shift | **confirmed (S3)** | the ring is correct, so this is a readability win | P2 |
| 3a | GPU envs round-trip through the CPU each step | **confirmed (S2)** | `obs_r.detach().cpu().float()` per step | P3 |
| 3b | per-minibatch `.item()` calls | **confirmed (S2)** | 12 syncs per minibatch, 64 minibatches per update | P3 |
| 3c | vectorise the chunk GAE | **confirmed, low value (S3)** | measured 10 ms per rollout, 1% of the rollout | skip |
| 3d | the baseline is expensive | **confirmed, and worse than stated (S2)** | measured: the baseline costs 4.9x the collection | P3 |
| 3e | `np.random.shuffle` plus a host copy | **confirmed (S3)** | 64 host copies per update | P3 |
| 3f | `critic.eval()` toggling is a no-op | **confirmed (S3)** | no dropout, no BatchNorm | P3 |
| 4 | optimizer grouping by string match | **partly (S3)** | `"out."` does match `norm_out.`, but both rules land in the same group | P4 |
| 4 | config validation and `Literal` types | **confirmed (S3)** | no `__post_init__` checks | P4 |
| 4 | `StateAffine` and `ActionAffine` are redundant | **confirmed as an observation** | an ablation, not a cleanup | P5 (optional) |
| 4 | `RunningMeanStd.update` uses `inference_mode` | **confirmed (S3)** | `no_grad` is the conventional choice | P4 |
| 4 | dead code list | **partly** | 2 of 7 names are dead. See section 6 | P4 |
| 4 | magic constants | **confirmed (S3)** | `TIME_SCALE`, RoPE base | P4 |
| 5 | split the file into 7 modules | **not recommended** | commit `03b68f6` collapsed the repo to one file on purpose | section 8 |
| new | the PPO baseline crashes at the first eval | **confirmed (S1, outside the review)** | `NameError: name 'backends' is not defined`, `ppo.py:512` | P0 |
| new | ruff is red at HEAD | **confirmed (S2)** | 12 errors: 2 in `myfpo.py`, 10 in `ppo.py` | P0 |

---

## 2. Phase 0 — repair the gates

No behaviour change. The suite goes from red to green.

- Rewrite the 7 stale test modules against the current API. Keep every contract
  they pin. The missing names were renames or inlining, so the contracts still
  hold.
- `test_gae.py` needs two source changes, not one:
  - Extract `gae_trace(delta, trace, successor)` back out of
    `compute_chunk_advantages`. It was inlined by the single-file rewrite. The
    function is the unit-testable part of the backup.
  - Add the length check the test expects: `chunk tensors must have the same
    length`. The current code reshapes `v_current` and `v_next` and never
    compares their lengths to the chunk count.
- `test_shared_env_carry.py` needs `_shared_env(train_env, env)`. It is 3 lines in
  the training loop today. Extract it so the test can pin it.
- `test_grad_clip.py` needs the actor and critic parameter split that `update`
  builds today with `id()` sets. Extract it as `_param_groups_for_clip(policy)`.
  Phase 4 reuses it.
- The other four modules need import and call updates only.
- Restore the 7 specific error messages in the source. The old wording names the
  field and the limit, so the guard says what to fix. Do not weaken the tests.
- Fix `src/myfpo/ppo.py`: replace the two `if backends:` blocks with
  `if wandb_run is not None:` and `wandb_run.log(...)`, and finish the run with
  `wandb_run.finish(exit_code=0)`. The pattern is already in the file at line
  477.
- Fix the 2 ruff errors in `myfpo.py`: replace the `stat = lambda` at line 1305
  with a `def`, and mark the unused `seed` argument of
  `VectorEvaluator.random_policy` with a reason. The signature must stay,
  because the training loop calls it the same way for both evaluators.

Acceptance: `pytest -q` gives 0 errors and 0 failures. `ruff check` is clean.
A `--method ppo` smoke run exits 0.

---

## 3. Phase 1 — the correctness fixes (S1)

Items 1a, 1b and 1c change the learning result. Each needs a smoke run and a
written reason, per rule 1 and rule 4 of `SIMPLIFICATION_PLAN.md`.

### 1a. The target critic

The Polyak update runs once per rollout, with `tau=0.005` hard-coded at
`myfpo.py:1752`. A rollout is `chunks_per_update * action_horizon = 8192` env
steps, so a 200k-step run gets 24 target updates. The target holds
`1 - 0.995^24 = 11%` of the online critic.

`v_cur` comes from the online critic. `v_nxt` comes from the target. The TD
error therefore mixes two value functions that differ by about 9x, and the
bootstrap term is nearly gone.

Fix:

- Add `target_tau: float = 0.005` to `MyfpoConfig`.
- Move the Polyak step from the training loop into `update()`, right after each
  `optimizer.step()`. One rollout is 64 gradient steps, so the target moves
  `1 - 0.995^64 = 27%` per update. The lag then closes over a few rollouts.
- Log `train/critic_target_lag`, the mean `|Q_online - Q_target|` on the rollout
  states. The lag must be visible in the run, not inferred.

Option B, listed in the review: drop `critic_target` and read both ends from
the online critic. That removes a module and the `deepcopy`, and it changes the
algorithm. Recommend option A for now. Keep option B as an ablation.

Test: after 200 gradient steps the target tracks the online critic to within 5%.

### 1b. The rollout size

`rollout_steps = chunks_per_update * action_horizon` counts env steps, but a
chunk runs `num_action_steps` env steps, not `action_horizon`. With the default
`num_action_steps == action_horizon` the two agree. With
`num_action_steps < action_horizon` the rollout holds more chunks than
`chunks_per_update`, so the minibatch count and the number of gradient steps per
rollout both grow. Probe: `num_action_steps=2` gives 128 chunks for a 64-chunk
target.

Fix: budget the rollout in chunks, not in steps.

- `chunks_per_env = max(1, chunks_per_update // num_envs)`.
- The collector takes a chunk count, not a step count.
- The buffer capacity becomes `chunks_per_update` chunks and
  `chunks_per_update * num_action_steps` steps.
- Keep a step cap for the final rollout of a run, so the last rollout is shorter.

Test: the chunk count per rollout equals `chunks_per_update` for
`num_action_steps` in `{1, 2, 4}`.

### 1c. The normalizer order

`RunningMeanStd` passes raw obs through until the first `update()`. The first
update happens in the training loop, after `update(policy, optimizer, batch, cfg)`.
So rollout 1 collects with raw obs, computes its behaviour loss with raw obs, and
takes its gradient steps with raw obs. Rollout 2 switches to normalised obs. The
encoder sees two different input distributions.

Fix:

- Move `policy.obs_normalizer.update(...)` from the training loop into
  `run_rollout`, right after the collection and before `norm_states` and the
  behaviour loss.
- The behaviour loss and the update loss then use the same statistics, so the
  ratio compares two velocities computed under one normalizer.
- The collection itself still uses the statistics from the previous rollout. That
  is unavoidable and it is fine, because both ends of the ratio are recomputed
  under the current statistics.

Test: after the first `run_rollout`, `obs_normalizer.uninitialized` is `False`
and `norm_states` differs from the raw states.

### 1d. The `"sqrt"` reduction

`action_is_pad` is always passed, so the `"sqrt"` branch divides by
`n_valid = n_valid_tokens * action_dim`. The result is the mean over the valid
elements, not the sum scaled by `sqrt(n)`. Probe: the masked loss is `0.75`, the
mean; the true sqrt-scaled value is `1.5`. The logged name `cfm_loss_new` does
not match the math.

Fix, in the order of risk:

- Rename the mode to `"mean"`, which is what the code does, and keep the default
  numbers. The mode set becomes `{"sum", "mean"}`.
- Add the true sqrt variant as `"sqrt"` with `flat.sum(-1) / n_valid.sqrt()` and
  leave it off by default. Turning it on changes the loss scale, so it changes
  the ratio and the trust region width. That is an experiment, not a cleanup.

Test: the masked loss equals the mean over the valid elements.

### 1e. Idle envs

`next_actions.zero_()` then only the live rows are filled. An env that reached
its step target keeps taking zero-action steps while the other envs finish.
Those transitions are dropped, but they advance the real episode with
off-policy actions. Probe: with 4 envs and targets `[9, 8, 8, 8]`, the last vec
step sends zero actions to 3 envs.

Fix: the chunk budget from 1b makes every env close its chunks at the same
global step, so the idle path disappears.

- Delete the idle window branch and the `next_actions.zero_()`.
- The final partial chunk of a run is handled by a per-env length cap, not by
  idling.

Test: no env receives a zero action in a rollout where the policy output is
non-zero.

### 1f. `reward_mean_last10`

Use `all_eval_returns[-10:]` for the mean and the median.

### 1g. The small items

- `FlowMLP.forward`: move the `x_t.dim() != 3` check above the unpack.
- `Evaluator.evaluate`: wrap the run in `torch.random.fork_rng` and
  `torch.manual_seed(self.eval_seed)`, the way `VectorEvaluator` does. The eval
  then does not move the training RNG stream, and the docstring becomes true.
- `Evaluator.random_policy`: seed `env.action_space` when no seed is given. The
  random baseline is currently not reproducible, because `gym.make` seeds the
  action space from the OS entropy.
- `MyfpoConfig.__post_init__`: validate `trust_region_mode` in
  `{"ppo", "spo", "aspo"}` and `cfm_loss_reduction` in `{"sum", "mean"}`.
  Validate `1 <= num_action_steps <= action_horizon`,
  `num_obs_steps >= 1`, `chunks_per_minibatch <= chunks_per_update`,
  `0.0 <= target_tau <= 1.0`, and `0.0 <= gamma, gae_lambda <= 1.0`.

Acceptance for the phase: `pytest -q` green, `ruff check` clean, the smoke run
completes, and the smoke summary states which numbers moved and why.

---

## 4. Phase 2 — the collector window (S3)

The ring buffer in `collect_chunks_vec` is the hardest part of the file. It is
also correct. I checked the invariant by hand: the window holds the last
`num_obs_steps` frames, the chunk overwrites the `L` oldest slots, and the
rotation by `L` moves the new frames to the end. The result matches the true
last-n-frames for any chunk length, including a chunk cut short by a terminal.

So this phase is a simplification, not a bug fix. It also removes the
`num_action_steps <= num_obs_steps` limit, which only the ring needs.

Fix, per the review:

```python
# every step, for every env:
windows = torch.cat([windows[:, 1:], obs_t[:, None]], dim=1)

# at chunk start:  start_window[rows] = windows[rows]
# at chunk close:  next_window = windows[rows].clone()
# on done, after the chunk is recorded: windows[rows] = reset_frame repeated
```

- Delete `roll_table`, the `torch.gather` call, and the idle-env window branch.
- Delete the `num_action_steps > num_obs_steps` guard.
- Keep `chunk_state` and `chunk_next_state` semantics exactly as they are now.
  `test_collector_batch_equivalence.py` pins them and must pass unchanged.

Test changes:

- Delete `test_a_chunk_longer_than_the_obs_window_is_rejected`. The guard is
  gone.
- Add a test that a chunk longer than the window works and lands in order.
- The window invariants in `test_obs_windows_slide_by_exactly_one_frame_per_step`
  and `test_a_chunk_as_deep_as_the_window_still_lands_chronologically` must pass
  unchanged. They are the equivalence check for this phase.

---

## 5. Phase 3 — performance (S2)

### 3a. Keep GPU envs on the device

Every step moves the observation to the CPU, runs the window bookkeeping on the
CPU, and moves the action back to the GPU. The `.cpu()` call also forces a
device sync every step, which is worse than the byte count.

- `RolloutBuffer` takes a `device` argument. The chunk and step rows live there.
- The collector keeps `windows`, `raw_plan`, `exec_plan` and `next_actions` on
  `env.device` for a torch env, and on the CPU for a Gymnasium env.
- The numpy path stays for Gymnasium envs.
- `take_final_observations` already handles a torch `final_obs`.
- If the policy device and the env device differ, move the action to the env
  device and the observation to the policy device. `ManiSkillTorchEnv.step_torch`
  already moves the action.

Risk: medium. This path has no CPU test. It needs one ManiSkill GPU run to
confirm the numbers match the CPU-side behaviour.

### 3b. Per-minibatch `.item()` calls

`compute_loss` makes 12 host syncs per minibatch. A rollout has 64 minibatches,
so one update costs about 770 syncs. On the CPU this is close to free. On CUDA
each `.item()` waits for the queue, so it costs the full pipeline depth.

- `compute_loss` returns detached tensors, not floats.
- `update` accumulates them with `torch.stack` and calls `.item()` once per
  metric at the end.
- The metric keys stay the same. They are a logged contract.

The update is the largest stage of the cycle (10.4 s against 1.25 s for the
rollout), so this is worth doing before the run grows.

### 3c. The chunk GAE (skip)

The Python loop runs 2048 times, once per rollout. Measured, that is 10 ms,
which is 1% of the rollout and under 1% of the cycle. The vectorised version in
the review also drops `chunk_prev` and `_last_chunk`, which removes an
explicit, tested data structure. The gain does not pay for the risk. Skip
unless the GAE is rewritten for another reason.

### 3d. The baseline (promoted to a main item)

The review counts 80 actor passes per rollout and calls the baseline cheap.
That count is in passes, not rows. Each baseline pass runs over all 2048 chunks
at once. Each collection pass runs over about 16 envs. The rows differ by 128x.

Measured on this machine (CPU, 2048 chunks, 8192 steps, `hidden_dim=128`):

| stage | time | share of the rollout |
|---|---|---|
| collection | 211 ms | 17% |
| baseline (`2 x num_baseline_samples x num_steps_noise_start`) | 1030 ms | 82% |
| chunk GAE loop | 10 ms | 1% |
| update (16 epochs x 4 minibatches) | 10424 ms | separate, 8.3x the rollout |

So the baseline costs 4.9x the whole collection. It is the second largest item
in the cycle, and the review's "80 actor passes" framing hides it.

- The cheap version: batch the `num_baseline_samples` samples into one forward.
  This cuts the call count, not the rows, so the gain is small.
- The real version: add a V-head trained on the same lambda returns. One critic
  pass per state replaces 40 actor passes plus 4 critic passes per state. The
  row count drops about 40x. It also removes the variance of a 4-sample
  estimate of `V(s)`, which today goes straight into the advantage.
- The stopgap: lower `num_baseline_samples` from 4 to 2. It halves the cost and
  raises the variance. Use it only if the V-head is deferred.

Recommend the V-head, in its own phase, with its own smoke run. The critic keeps
its Q head for the value loss.

### 3e. The shuffle

Use `torch.randperm(n, device=device)` instead of `np.random.shuffle` plus a
host-to-device copy per minibatch.

### 3f. The no-op eval toggling

`policy.critic.eval()` and `policy.critic.train()` in `run_rollout` do nothing.
The critic has no dropout and no BatchNorm, only LayerNorm. Delete the
toggle and the `try/finally`.

---

## 6. Phase 4 — cleanups (S3)

### The optimizer groups

The review says the string match is fragile, and `"out."` does match
`norm_out.`. But both rules put `norm_out.weight` in the no-decay group, so the
outcome is the same. The real difference is intent: the `ndim` rule would add
weight decay to the actor output layer, which the current code excludes on
purpose.

Fix, without a math change:

- Group by `ndim`: 2-D weights get decay, 1-D parameters do not.
- Exclude the actor output layer (`policy.actor.out`) from decay explicitly, to
  keep today's numbers.
- Read the actor and critic parameter lists from the optimizer param groups in
  `update`, instead of rebuilding them with `id()` sets on every update.
- Keep `torch.optim.Adam`. The review proposes `AdamW`. That is a different
  update rule, so it needs an amendment and a smoke run, like any other
  algorithm change.

### The config

- Use `Literal` for `trust_region_mode` and `cfm_loss_reduction`.
- Validate in `__post_init__` (see 1g).
- `asdict` becomes `dataclasses.asdict(self)`.
- Delete the dead `critic_hidden_dim is not None` branch in `MyfpoPolicy`. The
  field is an `int`, so it is never `None`.
- Delete `env_squash_terms` and the `num_minibatches` property. Both are dead.

### The dead-code list from the review

Checked each name against `src/`, `scripts/` and `tests/`:

| name | verdict |
|---|---|
| `env_squash_terms` | dead. delete |
| `num_minibatches` | dead. delete |
| `is_maniskill_id` | used by `scripts/train.py` and `scripts/play.py`. keep |
| `register_gym_envs` | used by `scripts/play.py`. keep |
| `bound_for_storage` | used by the collector and pinned by a test. keep |
| `to_critic_action` | used by the collector and by `compute_loss`. keep |
| `RolloutState` | used. keep |
| `chunk_env_id` | written by the buffer, read only by tests. keep, and Phase 3c would use it if that phase is ever revived |

### The rest

- `RunningMeanStd.update`: use `@torch.no_grad()` instead of
  `@torch.inference_mode()`. Inference tensors cannot be used in autograd, and
  `no_grad` is the conventional choice for an in-place buffer update.
- Comment the sampler: the step size comes from the unclamped grid, and the
  velocity is evaluated at the clamped `t`. That is on purpose.
- Name the magic numbers: `TIME_SCALE = 1000.0` for the CFM time embedding, and
  the RoPE base `10000.0`.
- `__all__` lists 6 names out of about 60. Either widen it to the real public
  surface, or drop it and let `myfpo/__init__.py` be the surface. Low priority.

---

## 7. Phase 5 — optional experiments (each needs its own smoke run)

- Ablate `StateAffine` and `ActionAffine`. The critic applies `StateAffine`
  (RMS over the whole flattened window) on top of `RunningMeanStd`, and
  `ActionAffine` scales actions that are already in `[-1, 1]`. Both are
  learnable, so this is an ablation, not dead code.
- Drop the target critic and read both ends from the online critic (see 1a).

---

## 8. Items this plan does not do

- **Split the file into 7 modules.** Commit `03b68f6` ("refactor: single-file
  myfpo, add ppo baseline") collapsed the repo to one file on 2026-11-15, on
  purpose. The review's module map reverses that decision. If the file grows
  past about 2000 lines after Phase 4, revisit it.
- **Vectorise the chunk GAE.** About 1 ms per rollout (see 3c).
- **Change `Adam` to `AdamW`.** A different update rule (see Phase 4).
- **Change the default `cfm_loss_reduction` to a true sqrt.** It moves the loss
  scale, the ratio and the trust region width (see 1d).

---

## 9. Baseline and gates

Test baseline at HEAD `12a38cb`:

```
pytest -q
  7 collection errors, 11 failed, 104 passed

ruff check src scripts tests
  12 errors (2 in myfpo.py, 10 in ppo.py)
```

Smoke baseline (Hopper-v5, seed 0, 2000 steps), recorded at HEAD `12a38cb`:

| run | exit | fps | eval return mean | episodes |
|---|---|---|---|---|
| `smoke_before` (this file, section 10) | 0 | 371.9 | 24.33 | 50 |
| `aspo` default (`SIMPLIFICATION_PLAN.md` Amendment 1) | 0 | 292.7 | 5.20 | 50 |
| `ppo` override (Amendment 1) | 0 | 300.0 | 4.64 | 50 |

The two rows are not comparable. The Amendment 1 runs and this run differ in
the eval setup, and the return sits at the random level either way. Use this
file's row as the "before" number and compare "before" and "after" with the
same command.

A 2000-step run is one rollout, so it checks for runtime errors, not learning.

Cost split of one full rollout, measured on this machine (CPU, 2048 chunks,
8192 env steps, `hidden_dim=128`, `num_mc_samples=16`, `num_baseline_samples=4`):

| stage | time | share |
|---|---|---|
| collection | 211 ms | 17% of the rollout |
| baseline value estimate | 1030 ms | 82% of the rollout |
| chunk GAE loop | 10 ms | 1% of the rollout |
| update (16 epochs x 4 minibatches) | 10424 ms | 8.3x the rollout |

The update dominates the cycle. The baseline is the second largest item, and
the review understates it (see 3d).

Gate commands:

```
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src scripts tests
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000 \
    --out_dir /tmp/opencode/smoke_after --myfpo.report_to none
```

Phase order: P0 gates, P1 correctness, P2 collector, P3 performance, P4
cleanup, P5 experiments. P1 and P2 need a smoke run each. P3 needs a ManiSkill
GPU run for 3a.

---

## 10. The smoke run recorded here

Command:

```
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 --total_steps 2000 \
    --out_dir /tmp/opencode/smoke_before --myfpo.report_to none
```

Result at HEAD `12a38cb`:

| field | value |
|---|---|
| exit code | 0 |
| fps | 371.9 |
| best return | 24.33 |
| eval episodes | 50 |
| eval return min / median / max | 5.84 / 15.58 / 92.30 |
| global step | 2000 |

The run logs this warning exactly once:

```
UserWarning: RunningMeanStd: forward before the first update(); returning raw input
```

That is the normalizer bug of item 1c, seen in a real run. Rollout 1 collects,
computes its loss and takes its gradient steps on raw observations.
