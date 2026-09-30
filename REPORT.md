# Training pipeline review

5 passes, one pipeline stage each. Findings accumulate below; each pass records
what it checked, what it found, and what it could not settle.

Severity: **S1** breaks training, **S2** biases it, **S3** cosmetic or latent.

---

## Current state (read this first)

The pass-by-pass body below is a **historical audit**. Much of the code it
describes is gone. Do not read the body as a description of the current tree.

### Critic redesign — done, and NOT validated

`Critic` was a flatten-plus-position-code MLP over the 16-frame window, with
`num_critics` members batched through two `baddbmm` calls. It is now one
`StateConditioner` plus one linear head per member, in a `ModuleList`.

Why: the actor's `StateConditioner` already does a full orthonormal DCT of the
window. The critic did not. The function that sets the advantage scale was
estimated by a strictly weaker approximator than the policy it criticizes. The
old critic also had no recency structure at all — measured per-frame
`|dV/dframe|` was 0.0507 for the first 8 frames and 0.0500 for the last 8, equal
to three significant figures. The new one is non-uniform (0.0113 vs 0.0049).

Cost: **22,785 -> 467,736 params** at `num_critics=2`, about 20x. 85% of that is
the conditioner's `SwiGLU(128, 512, 128)`. Output shape is unchanged
(`[num_critics, B]`), so no call site changed.

**The motivating evidence did not reproduce.** The old health run reported
`Reacher-v5` DCT-off diverging at `cfm_above_clamp_frac=0.98`. A clean A/B on
that exact arm, seed 3407, 24576 steps:

| | old critic | new critic |
|---|---|---|
| `diverged` | false | false |
| `reward_mean_last10` | -10.95 | -11.87 |

Neither diverged, and the old one scored marginally better. Treat the redesign as
architecturally motivated and empirically **unvalidated**. A single seed and a
single eval number settle nothing.

### Audit findings: what still holds

| # | finding | status |
|---|---|---|
| 1 | Actor and critic clipped separately, so `max_grad_norm` is 2x loose | **resolved** — `critic_max_grad_norm` is a separate field; both are 1.0 (was 200.0, which never bound) |
| 2 | Critic target non-stationary via `return_normalizer` | **resolved** — no `return_normalizer` in the tree; GAE and the value loss are both on raw returns |
| 3 | `forward` docstring contradicts `cfm_mask_unexecuted` | **resolved** — docstring rewritten |
| 4 | Missing `return_normalizer.*` key resumes silently inert | **resolved** — the key does not exist |
| 5 | Stale `# <- move inside` marker | **resolved** — deleted from `train_sjfpo.py` |
| 6 | `collect_steps` vs `global_step` disagree | **resolved** — `collect_steps` is gone |
| 7 | `compute_policy_loss` dead | **resolved** — deleted |
| 8 | `reduction="none"` unreachable | **live** — `modeling_sjfpo.py:1208`. `reduction` is a separate param from `cfm_loss_reduction`, which is validated to `mean\|sum\|sqrt` and never reaches it |
| 9 | Value-loss docstring says MSE, code was Huber | **resolved** — the code is now `torch.square`, so the docstring was right and is kept |
| 10 | `top_p` in docstring | **resolved** — no `top_p` in the tree |
| 11 | Conditioner spectral branch has no frequency identity | **live** — `freq_encoding` runs on the feature axis, then mean-pools over frequency. `frame_norm` covers the frame path, not this |
| 12 | `step_truncated` written but never read | **resolved** — both `step_terminated` and `step_truncated` deleted from the collector |
| 13 | Critic trunk had no normalization | **resolved** — `frame_norm` added between `frame_proj` and `frame_mlp` |
| 14 | `max_grad_norm` 200.0 gave no clipping at all | **resolved** — both bounds are 1.0; the actor clip now binds on 44% of updates |
| 15 | `configs/sjfpo.yaml` did not parse and had drifted from the dataclass | **resolved** — regenerated from `SJFPOConfig`; 67 keys, zero mismatches |
| 16 | Rollout collection did O(n_obs) work per env-step | **resolved** — ring-buffered obs window, batched writes, one gather for actions. 378 ms -> 114 ms at 64 envs |

### Still-open, worth doing

- The **ACPPO chunk-level ratio audit**. Hahn et al. argue the PPO importance
  ratio must be taken over the whole action chunk. Check whether
  `compute_policy_ratio` in `fpo_core.py` is already evaluated at chunk
  granularity, and whether it is consistent with the `gamma ** h` backup in
  `gae.py`. Unstarted.
- `np.float_` was removed in NumPy 2.0 and `wandb` still calls it. The default
  `report_to="wandb"` path **fails outright**; every run needs an explicit
  `--sjfpo.report_to tensorboard`. Pre-existing, breaks `tests/test_tracking.py`.
- `configs/sjfpo.yaml` has orphaned indented `report_to` and `num_critics` lines
  near line 56.

---

## Pass 1 of 5 — rollout and data collection

**Area:** `src/sjfpo/collector.py` (908 lines), the rollout half of
`modeling_sjfpo.SJFPOPolicy.generate_rollout`.

**Method:** read the env-stepping loops, the buffer writes, the autoreset state
machine, and the budget accounting. Then instrumented `RolloutBuffer.add_chunk`
and `mark_incomplete` on a 2048-step Hopper run to replace reading with counting.

### S2 — `chunk_truncated` conflated two causes (fixed in `e6ec875`)

The rollout budget cutting a chunk short wrote the same flag as the env's
time-limit truncation. GAE reads that flag to decide whether a chunk hands
`gamma * lam * advantage` to the next chunk of the same episode, so budget cuts
severed the chain on live episodes. Split into `chunk_budget_truncated`, which
the GAE never sees. `collect_chunks` was the worse instance: it marked
`done_kind == "continue"`, i.e. every chunk of a live episode.

Measured budget-cut rate after the fix:

| path | budget cuts | GAE chains |
|---|---|---|
| `collect_chunks_vec` (`num_envs=8`) | 0.0229 | 0.3954 |
| `collect_chunks` (`num_envs=1`) | 0.5646 | 0.3825 |

### Resolved — a suspicion of mine that was wrong

I read the vec path's budget marking (line 787) as running per chunk and
therefore marking every normally-completed chunk, the same defect just fixed in
the single-env path. **The marking loop is outside the `while`, not inside it.**
It runs once after the rollout and marks only the last chunk per env. Counting
confirmed 7 marks against 8 envs, one of whose final chunks ended on a done.

The vec path was always correct here. Only `collect_chunks` was broken.

### Resolved — the dct-true `truncated_chunk_frac = 1` mystery

`truncated_chunk_frac` is derived from `action_is_pad`, not from the env's
`truncated` flag. On this Hopper run the env reported `truncated` **zero** times
(100 terminations, 76 continues, 0 truncations out of 176 chunks).

So dct-true's `truncated_chunk_frac = 1.0` means every one of its chunks was cut
short by a fall, with `executed_action_step_frac = 0.5625` giving roughly 9 of 16
steps. That is a **policy that falls mid-chunk every time**, not an env
reporting spurious time limits. The env-termination hypothesis I raised is dead.

### S3 — the autoreset handshake spends a real env step and discards it

`collect_chunks_vec` implements `AutoresetMode.NEXT_STEP` as a two-step
handshake. An env that returned `done` is flagged `awaiting`; on the next
`step()` the reset observation is picked up and the loop `continue`s past it.
Two consequences, both unverified:

- The env is stepped with `actions_t[i] == 0` during that handshake, so under
  `NEXT_STEP` the observation retained as the new episode's initial state may be
  one zero-action step *past* the reset state rather than the reset state itself.
  Needs a check against the installed gymnasium's `step_wait`.
- That step's reward is discarded and its transition is never recorded. If the
  zero action causes an immediate fall, the fall is attributed to the next chunk
  rather than to the reset, off by one.

### S3 — `executed_total` counts recorded steps, not env steps

`execution_limit = min(n_action_steps, targets[i] - executed_total[i])` and
`executed_total[i] += n_exec` only on chunk finalisation, while the vec env is
stepped on every while-iteration regardless. Envs that are idle or `awaiting`
are still stepped with zero actions. The true env step count therefore exceeds
`total_steps` by an amount that grows with imbalance between envs. The recorded
batch is consistent; the budget is not a true env-step budget.

### Not settled

- The gymnasium autoreset semantics behind the S3 above.

---

## Late-training regression (300k steps, `regressed=True`)

Several runs peak near 300k steps, then deteriorate until the regression
detector fires. Not classical catastrophic forgetting: this is single-task
on-policy, so there is no earlier task to forget.

Likely causes, ranked by the evidence available at 8k steps:

| candidate | 8k-step evidence | verdict |
|---|---|---|
| trust region broke (PFO) | `clipfrac` 0.07-0.09, healthy | not at 8k; unread at 300k |
| CFM loss saturated | `cfm_above_clamp_frac` 0.0000 | ruled out |
| critic cannot track the target | `returns_rms` 9.7 -> 28.7 in 8k steps, 3x | strongest candidate |
| normalization loss (Lyle 2024) | critic trunk had no norm; `explained_var` negative at 1M | partly addressed by `frame_norm` |

`explained_var` flipped sign between two 8k-step runs (+0.106, -0.027) at one
seed. It does not discriminate at that horizon. Read `clipfrac` and
`explained_var` at 200k/300k/350k before choosing a fix.

**Reward normalization: keep raw returns.** With `normalize_advantage=True`
the whitened advantage is invariant to reward scale (max deviation 2.4e-7 over
scales 0.01 to 100), so normalizing rewards cannot change the policy gradient.
It can only affect the critic, and naive standardization measurably hurts it:
at equal head scale, `explained_var` fell 0.60 -> 0.01, because the head must
traverse the output scale that Adam cannot cover in the available steps.

PopArt would fix the target scale while preserving predictions, but it is not a
drop-in: `Critic.forward` returns the head output directly with no affine
wrapper, so `V = y_norm * sigma + mu` is a restructuring, and it would break
every checkpoint again. Deferred.

Applied: gradient bounds 200.0 -> 1.0 (with the actor/critic split), clipped
value loss on, `num_learning_epochs` 4 -> 2 (PFO shows more epochs per batch
accelerate feature-rank loss), `frame_norm` on the critic trunk. None of these
is verified against the 300k peak.

---

## Pass 2 of 5 — GAE, advantage, and the return normalizer

**Area:** `src/sjfpo/gae.py` (124 lines), the backup in
`modeling_sjfpo.generate_rollout`, the advantage normalisation, and the
`return_normalizer` update order.

**Method:** traced the Bellman backup against the module docstring's stated
formula, checked the sort/gather/unsort ordering, and checked where the
normalizer is updated relative to the values that consume it.

### S2 — the critic's regression target is non-stationary

**Resolved.** The `return_normalizer` no longer exists; the critic regresses on
raw returns. Kept because the *question* is still open: `returns_rms` still
grows ~3x in the first 8k steps, so the target scale is still moving, just
without a normalizer in the way. See "Late-training regression" above.

The critic is trained to predict `return_normalizer(raw_returns)`, and
`return_normalizer.update` is called **every rollout**, on that rollout's
returns. So the regression target's scale moves while the critic is being fit
to it. The GAE is handed the *previous* statistics via
`return_normalizer.unnormalize(chunk_value)`, so the backup and the target are
deliberately on different generations of the same normalizer.

This is consistent as written and the one-step lag is documented, but it is a
real difference from rsl-rl, which has no return normalizer and therefore a fixed
target scale. It matches the observed critic underfit: `value_pred_std` 0.44
against `value_target_std` 1.00 at 16k steps, i.e. the prediction is shrunk
toward a mean that has already moved. Every measurement of the critic across
this session is consistent with "chasing a moving target" rather than "not enough
capacity".

Options, none implemented: slow the normalizer with a momentum term, freeze it
for a fixed number of rollouts, or normalise the critic's **output layer**
instead of its target so the target stays in raw units and the loss stays
O(1) by construction.

### S3 — stale marker left in `mini_batch_indices`

`train_sjfpo.py:74` carries the comment `perm = torch.randperm(n, device=device)
# <- move inside`. The `randperm` is already inside the epoch loop; the marker is
a leftover. Harmless, but it reads as an outstanding task.

### S3 — `collect_steps` and `global_step` count different things

`collect_steps = chunks_per_update * action_horizon` is a nominal step budget,
while `global_step += (~action_is_pad).sum()` advances by steps actually
executed. With `executed_action_step_frac` at 0.56 in the dct-true run, one
million global steps corresponds to roughly 1.78M nominal requested. The batch
is self-consistent and `total_steps` is honestly measured in executed steps, so
this is not a bug — but the data collected per update is not what the config
reads like, and it shifts over training as the policy's survival changes.

### Verified correct

- `V(s_{s_i + h_i})` is taken from `step_next_states[last_step]`, which is the
  same state as `step_states[chunk_starts[i+1]]`, so the bootstrap and the next
  chunk's own `V` do not disagree.
- `R_i = sum_{k<h_i} gamma^k r_{s_i+k}` with the intra-chunk `valid` mask, and
  `gamma_h = gamma ** lengths` across the boundary. Matches the docstring.
- The `valid` mask is built before the by-env sort and `chunk_rewards` is sorted
  after, so the two line up.
- The advantage is whitened once per rollout over all chunks, then minibatched.
  Same as rsl-rl, and correctly not per-minibatch.
- `torch.std` is the unbiased estimator in both the whitening and the
  diagnostic, matching rsl-rl.
- One rollout per update with `snapshot_behavior_policy()` before collection, so
  `l_old` is genuinely frozen.

### Not settled

- Whether `gae_lambda` having no intra-chunk smoothing is intended. The whole
  chunk's reward is one lump, so `lam` only shapes inter-chunk credit and the
  effective GAE horizon is chunk-level. `lam = 0.95` does not mean here what it
  means in per-step GAE.

---

## Pass 3 of 5 — loss functions and the update step

**Area:** `src/sjfpo/fpo_core.py` (343 lines) plus the update block in
`train_sjfpo.py` (`mini_batch_indices`, gradient accumulation, clipping,
`optimizer.step`).

**Method:** read each reduction and surrogate, then checked the update loop's
grouping, the non-finite handling, and whether anything the config exposes is
actually reachable.

### S2 — actor and critic are gradient-clipped separately, so `max_grad_norm` is 2x loose

```python
pre_clip_actor  = torch.nn.utils.clip_grad_norm_(actor_params,  cfg.max_grad_norm)
pre_clip_critic = torch.nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
```

Two independent bounds, so the joint norm can reach `2 * max_grad_norm` and the
actor-to-critic ratio is never constrained. rsl-rl clips the two parameter sets
together, which bounds the total step. The dct-true run logs
`critic_grad_norm_preclip 1.018` against `actor_grad_norm_preclip 0.086`, a 12x
gap, so the two paths are far from equivalent in general even though they happen
to differ by under 2% in that particular instance.

One `clip_grad_norm_` over the concatenated parameter list is the fix, but that
changes what the two logged metrics mean, so the diagnostic names would need
revisiting too.

### S3 — ~~`compute_policy_loss` is dead and duplicates the inline path~~ RESOLVED

`fpo_core.compute_policy_loss` was defined and never called. `forward` inlined the
identical computation. Two copies of the surrogate assembly that could drift apart
silently.

**Resolved:** `compute_policy_loss` deleted from `src/sjfpo/fpo_core.py`. The single
inline path in `SJFPOPolicy.forward` (`src/sjfpo/modeling_sjfpo.py:1115-1153`) is now
the only copy of the surrogate assembly.

### S3 — `reduction="none"` is unreachable

`forward` supports a per-chunk `reduction="none"`, but the training loop never
passes it and guards against it explicitly:

```python
if loss.dim() != 0:
    raise ValueError(f"Loss must be scalar before backward; got {tuple(loss.shape)}.")
```

So the branch exists, cannot be used from the trainer, and would be rejected if
it were.

### Verified correct

- `(loss / group_size).backward()` with `group_size` the number of microbatches in
  the accumulation group is the right scaling, and minibatches are equal-sized
  (`start = (i*n)//n_mb`), so the mean of per-minibatch means is the overall
  mean.
- `check_finite=False` plus an accumulated `finite_error` flag is deliberate: a
  non-finite microbatch is caught at the accumulation boundary *before*
  `optimizer.step()`, so no bad update lands. Better than raising mid-accumulation.
- `clip_grad_norm_` returns the pre-clip norm, and `*_grad_norm_preclip` records
  exactly that, so the metric names are accurate.
- The STE cap on the log-ratio is one-sided by construction:
  `safe_delta.detach() + delta - delta.detach()` gives the capped forward value
  with the unclipped gradient, which is the intent.
- PPO's `min(ratio*A, clip(ratio)*A)` is the pessimistic bound for negative `A`,
  since it selects the more negative term.
- `cfm_clamp=20` never binds because the masked `"sample"` reduction is a
  per-element mean and therefore O(1). Consistent with `cfm_above_clamp_frac = 0`
  in every run logged.

### Not settled

- The SPO term uses `advantage.abs()` with a single `clip_param` on both sides.
  Standard SPO uses `A^2` with one `eps`. Using `|A|` is defensible for an
  asymmetric variant but it is not something I verified against the FPO++ paper,
  and it is the default path for every negative-advantage sample.
- `num_learning_epochs` is now just a multiplier on a flat minibatch stream, so
  accumulation groups cut across epoch boundaries. Mathematically fine, but the
  name suggests per-epoch structure that no longer exists.

---

## Pass 4 of 5 — networks, conditioner, and the policy forward

**Area:** `StateConditioner`, `Critic`, `SJFPOFlowModel` in
`src/sjfpo/networks.py`, and `SJFPOPolicy.forward` plus the eval/inference path
in `modeling_sjfpo.py`.

> `Critic` has since been rewritten — see "Current state" at the top. The
> "Verified correct" list in this pass covers the old flatten-and-`baddbmm` critic
> and no longer applies to it.

**Method:** traced tensor ranks and normalisations from `forward` down through the
conditioner and the DiT, and cross-read every docstring claim against the code it
describes.

### S2 — `forward`'s docstring now asserts the opposite of the code

```
The unexecuted suffix is still part of the chunk the behavior policy selected,
so it must contribute to the FPO ratio when receding-horizon execution is
enabled; ``action_is_pad`` records execution bookkeeping only
```

With `cfm_mask_unexecuted=True`, which is the default since `7cc479b`, the
unexecuted suffix is **masked out of the CFM loss entirely** and contributes
nothing. This paragraph is the pre-mask rationale, kept verbatim, and it is the
docstring of the main forward. A reader checking whether the mask is safe would
read this and conclude the opposite.

There is also no `receding_horizon` config field in the repo; only
`select_action`'s "receding through a cached chunk" docstring at line 530 is
real. So the sentence cites a setting that does not exist and a behaviour that
was deliberately reversed.

### S3 — two more stale claims in the same docstring

- "the value loss is a mean squared error, as the official FPO++ does" — it is
  `smooth_l1_loss` with `beta=1.0` since `0c9a87f`. Someone reasoning about the
  critic's loss scale from this would be reasoning about a loss that no longer
  exists.
- "a token count of `M+1` derived from `top_p * H`" — `top_p` was replaced by
  `num_kept_tokens` in `37ab3e0`. No `top_p` remains anywhere in the package.

### S3 — the conditioner's spectral branch has no frequency identity

```python
coeffs = extract_dct_coeffs(states, M=states.shape[1] - 1)     # [B, T, obs_dim]
spec = self.key_proj(self.freq_encoding(self.freq_norm(coeffs))).mean(dim=1)
```

`SinusoidalEncoding(obs_dim)` encodes along the **observation-feature** axis, and
the result is then mean-pooled over the frequency axis. So the network is never
told which DCT bin a coefficient came from, and the mean-pool discards the
ordering. The branch reduces to "a fixed linear mix of all window frequencies",
which the raw per-frame path already provides. Not a bug, but the "DCT-frequency
features" framing overstates what the branch computes.

Note this conditioner DCTs the **observation** window, which is orthogonal to the
DCT applied to the action chunk. No interaction.

### Verified correct

- `forward` re-applies `self.preprocessor` to the raw `batch[OBS_STATE]`, and
  `generate_rollout` stores the raw `states` in the batch while using
  `normalised_states` internally. So the critic and the actor see the same
  preprocessing in the rollout and in the update. No double-normalisation, no
  skew between the two paths.
- The condition vector is expanded `[n, D] -> [n*m, D]` so one conditioning
  vector covers every Monte-Carlo draw, and the DiT consumes it through adaLN.
  Shapes line up.
- `Critic.forward` takes `[B, T, obs]` and returns `[B]`, matching the
  `returns.reshape(-1)` it is compared against.
- `compute_info=False` skips the diagnostic reductions and returns `None`, and
  the trainer honours that when wandb is off.

### Not settled

- Whether the conditioner's frequency branch earns its parameters at all, given
  the mean-pool above. Ablating it would answer that; I have not run it.

---

## Pass 5 of 5 — config validation, checkpointing, and the eval path

**Area:** `SJFPOConfig` validation, `checkpointing.py`, `RunningMeanStd`
persistence, `select_action`'s cached chunk, and both evaluators.

**Method:** enumerated every config field against every validation branch, then
dumped `policy.state_dict()` to see what actually persists.

### S2 — a missing normalizer key resumes silently inert

`RunningMeanStd._load_from_state_dict` re-derives the latch from the count:

```python
self._uninitialised.copy_(self.count.eq(0))
```

So a checkpoint **missing** `return_normalizer.*` loads cleanly with
`count = 0`, `_uninitialised = True`, and the module then passes its input
through unchanged in both directions. `unnormalize` becomes the identity, so the
critic's regression target silently becomes raw returns and the GAE baseline
silently loses its scale. No error, no warning.

This is exactly the failure class that bit the codebase three times already: the
mask/layout split, the separate clip, and the budget/truncation split. All were
"correct-looking numbers, wrong semantics."

It is live right now for one concrete case: a checkpoint written between
`10c4919` and `0c9a87f` has no `return_normalizer.*` keys, and loading it into
`main` today produces a dead return normalizer rather than an error. That window
is local and unpushed, so the practical risk is nil, but the mechanism is
general.

### Verified correct

- Both normalizers are `nn.Module` buffers and **do** persist: `state_dict()`
  carries `obs_normalizer.{mean,var,count}` and `return_normalizer.{mean,var,count}`,
  112 keys total.
- `save_checkpoint` strips `old_actor.*` and `old_conditioner.*` from the saved
  dict, so the frozen snapshot is not persisted and cannot go stale.
- The EMA shadow is only substituted into `state_dict` when
  `policy.ema_ready(global_step)`, so eval and train see the same weights at the
  same step.
- `select_action` accepts both `[obs]` and `[T, obs]`, warms the frame buffer on
  a rank-1 input, and buffers the chunk in reverse so `popleft` yields actions in
  order.
- **Both evaluators call `policy.reset()` per episode** (`evaluator.py:135` and
  `:196`), so the cached action chunk and the frame buffer cannot leak across
  episodes. This is the correct behaviour and the dct-true `std_length = 0.276`
  is a real degenerate policy, not a buffer-leak artefact.
- Only `cfm_mask_unexecuted`, `normalize_advantage` and `use_clipped_value_loss`
  are unvalidated, and all three are booleans needing no range check.

---

# Summary

**Historical.** Superseded by the "Current state" table at the top of this file.
The findings below were true when written; several now describe deleted code.
See the audit table near the top for current status.

Five passes, one stage each. Nothing implemented; this pass list is findings
only. 51 tests, no new failures.

## S2 — biases training, worth fixing

| # | finding | pass | now |
|---|---|---|---|
| 1 | Actor and critic gradient-clipped **separately**, so `max_grad_norm` bounds each, not the joint. Logged gap is 12x, so the bound is 2x loose. rsl-rl clips jointly. | 3 | **resolved** — separate `critic_max_grad_norm`, both 1.0 |
| 2 | The critic's regression target is **non-stationary**: `return_normalizer.update` runs every rollout, so the target's scale moves while the critic is fit to it. Consistent with `value_pred_std 0.44` vs `value_target_std 1.00`. | 2 | **resolved** — normalizer deleted |
| 3 | `forward`'s docstring asserts the unexecuted suffix **must** contribute to the ratio. With `cfm_mask_unexecuted=True` (default) it is masked out. The docstring states the reverse of the code. | 4 | **resolved** — docstring rewritten |
| 4 | A missing `return_normalizer.*` key in a checkpoint resumes **silently inert** rather than erroring. | 5 | **resolved** — the key does not exist |

## S3 — latent, cosmetic, or dead

| # | finding | pass | now |
|---|---|---|---|
| 5 | Stale `# <- move inside` marker in `mini_batch_indices` | 2 | **resolved** — deleted |
| 6 | `collect_steps` and `global_step` count nominal vs executed steps | 2 | **resolved** — `collect_steps` gone |
| 7 | ~~`compute_policy_loss` is dead and duplicates the inline surrogate~~ **RESOLVED** (deleted) | 3 | — |
| 8 | `reduction="none"` is unreachable and guarded against | 3 | **live** — `modeling_sjfpo.py:1208` |
| 9 | `forward` docstring: "value loss is a mean squared error" — it is Huber | 4 | **resolved** — the code uses `torch.square`, the docstring was right |
| 10 | `forward` docstring: token count "derived from `top_p * H`" — no `top_p` exists | 4 | **resolved** — no `top_p` in the tree |
| 11 | The conditioner's spectral branch mean-pools over frequency and encodes along the feature axis, so it has no frequency identity | 4 | **live** — `frame_norm` covers the frame path, not this |
| 12 | `step_truncated` is written in two places, read by nothing | 1 | **resolved** — both terminated/truncated deleted |

## Suspicions raised and then refuted

Recording these because each looked like a real bug on inspection:

- **The vec path marked every live chunk budget-truncated.** The marking loop is
  *outside* the `while`, not inside. Counting confirmed 7 marks for 8 envs. The
  vec path was always correct; only `collect_chunks` was broken.
- **`update_ema` / `update_observation_stats` ran per epoch.** The epoch loop
  lives inside the `mini_batch_indices` generator; both run once per rollout.
- **The env reported spurious time limits on dct-true.** `truncated` fired
  **zero** times on a 176-chunk Hopper run. `truncated_chunk_frac` is derived
  from `action_is_pad`, so dct-true's value of 1.0 means its policy falls
  mid-chunk on every chunk.
- **Eval leaked a cached action chunk across episodes.** Both evaluators reset
  per episode.

## Biggest open question, unchanged by all five passes

`explained_var` was **negative** in both 1M-step runs (−0.185 and −0.064) and
`advantage_std_fraction` was **above 1.0** (1.037), which is impossible for a
working baseline. A critic worse than a constant is upstream of everything else
in this list: no loss-mask or advantage work shows up through it. Item 2 above is
the most likely mechanism, and it is the one thing here with a concrete fix.

**Update:** item 2's proposed mechanism (`return_normalizer`) is gone from the
tree, so this diagnosis is no longer current and the negative `explained_var` is
**unexplained**. It is the highest-value open question in the repo. Re-measure
`explained_var` on the new critic before drawing any conclusion.
