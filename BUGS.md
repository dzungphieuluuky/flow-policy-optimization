# SJFPO Bug Audit

Audit scope:

```text
spectral-joint-policy/src/sjfpo/
```

The nested `spectral-joint-policy/` checkout was excluded. This document records the bugs and suspicious implementations found in the active Torch training pipeline. The active Torch package no longer contains an MJX rollout path; MJX findings are retained below as historical audit records.
## Severity

- **P0:** Core training correctness or environment-control bug.
- **P1:** Likely serious correctness, resume, precision, or configuration bug.
- **P2:** Meaningful reliability, performance, or reproducibility bug.
- **P3:** Stale, dead, or misleading code/documentation.

---

# P0 — Core correctness bugs

## 1. Truncated GAE drops the bootstrap value

**Status:** Resolved in the active Torch path
**File:** `src/sjfpo/gae.py:91-120`

The GAE implementation uses:

```python
done = terminated | truncated
```

and then applies zero next value and zero recursion continuation whenever `done` is true. This treats time-limit truncation exactly like true termination.

A truncated transition must:

- bootstrap its TD error from the terminal next-state value;
- stop propagation only because the episode boundary ends the sequence.

The current code discards the next value for both cases.

### Reproduction

```text
reward = 1
V(s) = 0
V(s') = 10
gamma = 0.9
lambda = 1
truncated = True
terminated = False
```

Observed:

```text
advantage = 1
return = 1
```

Expected bootstrapped result:

```text
advantage = 10
return = 10
```

### Impact

Humanoid and other time-limited environments will systematically lose terminal-state value at truncation boundaries. This biases advantages and returns.

### Implemented fix

The active Torch path now uses separate masks in the single step-level GAE helper:

```python
cut_recursion = terminated | truncated
bootstrap = terminated
```

A terminated step masks the next-state bootstrap and cuts recursion. A truncated step keeps the next-state bootstrap but cuts recursion. The active Torch path evaluates step-level GAE over stored post-step observation windows and selects the first executed step's target for each action chunk.

---

## 2. Policy action bounds do not update the actual postprocessor

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/modeling_sjfpo.py:145-186`
- `src/sjfpo/processor_sjfpo.py:64-77,140-152`
- `src/sjfpo/train_sjfpo.py:81-95`

`SJFPOPolicy` creates its postprocessor during initialization. `set_action_bounds()` later updates the policy's `action_low` and `action_high` buffers, but does not update the postprocessor's cloned bounds or cached affine mapping.

### Reproduction

```text
set_action_bounds([-2], [2])

policy.action_low     = -2
postprocessor.low     = -1
```

The postprocessor therefore continues converting normalized actions using the `[-1, 1]` mapping.

### Impact

This produces incorrectly scaled actions for any environment whose action range is not exactly `[-1, 1]`, especially positive-only action spaces.

### Implemented fix

`SJFPOPolicy.set_action_bounds()` now validates the new bounds, updates the policy buffers, and calls the postprocessor's atomic setter. The postprocessor recomputes its affine scale/bias and clears both device/dtype caches, so repeated bound changes cannot reuse stale coefficients.

---

## 3. Unexecuted action suffixes still contribute to training loss

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/collector.py:245-250,375-384`
- `src/sjfpo/modeling_sjfpo.py:499-522,576-685`
- `src/sjfpo/train_sjfpo.py:328-331`

The collector marks unexecuted action positions with `action_is_pad`, but `SJFPOPolicy.forward()` never reads that mask. The full predicted chunk is passed through DCT, flow loss, ASPO, and policy-ratio loss.

### Reproduction scenario

If an episode ends after 3 executed actions from a 16-action chunk, the remaining 13 unexecuted actions still receive gradients.

### Impact

Training learns from action coordinates that were never sent to the simulator. This is especially problematic for terminated and truncated chunks.

DCT also mixes the time axis, so a simple post-DCT mask is not sufficient: the valid-prefix boundary must be handled before or during spectral transformation.

### Implemented fix

`SJFPOPolicy.forward()` now treats `action_is_pad` as a chunk-level policy-loss eligibility mask. Full chunks contribute to the policy loss; chunks with an unexecuted suffix contribute only to the critic/value loss. This is necessary because the DCT mixes the time axis, so a post-DCT suffix mask would not correspond to the executed action prefix.

---

# P1 — Serious configuration, rollout, and resume bugs

## 4. Vector collector can execute beyond the requested step budget

**Status:** Resolved in the active Torch path
**File:** `src/sjfpo/collector.py:290-316,347-390`

The collector computes a per-environment target, but when executing an open-loop chunk it does not cap the chunk to the remaining budget.

### Reproduction

Four environments, `total_steps=4`, `n_action_steps=8`:

```text
requested steps: 4
executed steps:  32
```

### Impact

Runs with non-divisible step budgets execute too many simulator steps, distorting intended sample counts, evaluation schedules, and throughput measurements.

### Implemented fix

The vector collector now distributes the requested global budget exactly across environments using quotient/remainder targets, and stores a per-environment execution limit:

```python
targets = [base_target + int(i < remainder) for i in range(n)]
execution_limit = min(n_action_steps, targets[i] - executed_total[i])
```

It closes the transition at that limit, so final transition lengths, rewards, stored post-step windows, and `executed_total` all use the capped execution length.

---

## 5. Custom actor architecture configuration is ignored

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/modeling_sjfpo.py:138-151`
- `src/sjfpo/networks.py:398-439`
- `src/sjfpo/train_sjfpo.py:81-93`

Before the fix, `SJFPOPolicy` passed only `hidden_dim` to `SJFPOFlowModel`. Other actor settings were resolved through a module-global default `SJFPOConfig`, not the policy's actual configuration.

Ignored settings include:

- `dit_depth`
- `dit_heads`
- `dit_mlp_ratio`
- `dit_time_dim`
- `timestep_scale`

### Reproduction

Requested:

```text
dit_depth=1
dit_heads=2
hidden_dim=32
timestep_scale=10
```

Actual actor:

```text
blocks=4
heads=4
timestep_scale=1000
```

### Impact

Debug runs, small-resource runs, and benchmark configurations can silently instantiate a different architecture from the one requested.

### Implemented fix

`SJFPOPolicy` now passes `hidden_dim`, `dit_depth`, `dit_heads`, `dit_mlp_ratio`, `dit_time_dim`, and `timestep_scale` explicitly to `SJFPOFlowModel`. The Torch network constructors no longer resolve actor settings from a module-global configuration, and `build_policy()` no longer derives an overriding `n_action_steps` value. The canonical values are defined by `SJFPOConfig` and mirrored in `configs/sjfpo.yaml`.

---

## 6. FPO math uses global defaults instead of the policy configuration

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/fpo_core.py:22-224`
- `src/sjfpo/modeling_sjfpo.py:104-109,395-399,565-775`

Before the fix, `fpo_core.py` had its own global default configuration. Policy-specific values could therefore be ignored for:

- `cfm_clamp`
- `log_ratio_cap`
- `use_aspo`
- other omitted CFM defaults

### Impact

Changing policy configuration can produce policy ratios and surrogate losses that do not match the requested algorithm.

### Implemented fix

The active Torch FPO helpers now require an explicit `SJFPOConfig` (and explicit reduction/device/generator arguments where applicable). `SJFPOPolicy` passes its live configuration through timestep sampling, noise sampling, CFM loss, policy-ratio, and surrogate-loss calls. The global lazy fallback and per-call actor/FPO defaults were removed; sampling modes and Beta parameters are dataclass/YAML-backed.

---

## 7. Running observation statistics are never updated

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/modeling_sjfpo.py:186-196,584-604`
- `src/sjfpo/train_sjfpo.py:375-379`
- `src/sjfpo/networks.py:236-309`

`RunningMeanStd.update()` exists, but no active training-loop call updated it. Its `_uninitialized` latch therefore remained true, so normalization was effectively identity throughout training.

### Reproduction

After a rollout:

```text
normalizer_count = 0
uninitialized = True
```

### Impact

The implementation claims running observation normalization but does not perform it. This changes the intended input distribution and invalidates reproducibility expectations.

### Implemented fix

The training loop now updates the live normalizer once after each completed rollout and optimizer update. The policy exposes the exact raw post-step frame for every executed environment step; using `step_next_states[:, -1, :]` avoids counting overlapping warmup/history frames. Evaluation and rollout inference remain read-only, and the next rollout uses the updated statistics.

---

## 8. Checkpoint resume loses action bounds

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/modeling_sjfpo.py:145-150,199-223`
- `src/sjfpo/train_sjfpo.py:231-248`
- `src/sjfpo/checkpointing.py:271-307`

Action bounds are runtime environment state. Resume now accepts the current environment bounds, reapplies them through `set_action_bounds()`, and synchronizes the action postprocessor immediately after loading.

### Impact

A resumed policy can revert to `[-1, 1]` postprocessing even when the original environment used different bounds. Resumed and uninterrupted runs are not equivalent.

### Implemented fix

`load_policy_checkpoint()` now accepts optional `action_low` and `action_high` values, validates that they are supplied together, and reapplies them before returning the policy. The Torch training resume path passes the current environment bounds into the loader, ensuring the policy buffers and postprocessor use the resumed environment's mapping.

---

## 9. Checkpoint resume loses normalizer initialization state

**Status:** Resolved in the active Torch path
**File:** `src/sjfpo/networks.py:236-309`

`RunningMeanStd._uninitialized` remains non-persistent for checkpoint backward compatibility, but it is reconstructed from the persisted `count` during `load_state_dict`. A resumed policy with nonzero statistics therefore no longer ignores those statistics.

### Implemented fix

The policy checkpoint roundtrip now verifies that nonzero running statistics resume with `_uninitialized=False` and produce the same normalized outputs as the pre-checkpoint policy. Existing standalone `RunningMeanStd` state-dict coverage remains in place.

---

## 10. Blackwell CPU execution can use BF16 inputs with FP32 parameters

**Status:** Confirmed hardware-conditional runtime bug
**Files:**

- `src/sjfpo/utils.py:64-67`
- `src/sjfpo/modeling_sjfpo.py:552-568,688-690`
- `src/sjfpo/networks.py:66-79`

`STORAGE_DTYPE` is selected from GPU hardware presence, not the actual policy device. On Blackwell, selecting CPU leaves parameters FP32 while rollout tensors can be BF16. CPU autocast is intentionally a no-op.

### Impact

CPU runs on a Blackwell machine can fail with BF16-input/FP32-weight `Linear` errors or silently incur an unintended precision path.

### Minimal fix

Resolve storage dtype using the actual execution device. Force FP32 storage when the execution device is CPU.

---

## 11. `SJFPO_DTYPE=fp32` is ignored on Blackwell

**Status:** Confirmed configuration contradiction
**File:** `src/sjfpo/utils.py:70-88`

The Blackwell early return occurs before `SJFPO_DTYPE` is read.

### Impact

The environment variable does not provide the expected FP32 override on Blackwell.

### Minimal fix

Read and validate `SJFPO_DTYPE` before applying the Blackwell default, or explicitly document Blackwell as a forced-BF16 exception.

---

## 12. MJX truncation finding — removed from the active Torch path

**Status:** Historical finding; active code removed
**Original file:** `src/sjfpo/mjx_env.py` (deleted)

The former MJX collector autoreset completed environments and returned the reset observation as the transition's next state. That could have made truncation bootstrapping incorrect.

This is no longer an active Torch bug: `src/sjfpo/mjx_env.py` and the `SJFPO_MJX` branch in `SJFPOPolicy.generate_rollout()` were removed. No repair is required for the current Torch package.

## 13. 8-bit optimizer configuration is not preserved on resume

**Status:** Confirmed optimizer/resume inconsistency
**Files:**

- `src/sjfpo/train_sjfpo.py:99-124`
- `src/sjfpo/checkpointing.py:221-253`

Fresh training can construct `bitsandbytes.AdamW8bit`, but checkpoint loading always uses the normal optimizer preset builder.

### Impact

A resumed 8-bit run may use a different optimizer class and state semantics than the original run. Fused AdamW behavior is also not preserved consistently.

### Minimal fix

Centralize optimizer construction and use the same helper for fresh runs and resume, including 8-bit and fused settings.

---

# P2 — Reliability, logging, and precision issues

## 14. MJX reward precision finding — removed from the active Torch path

**Status:** Historical finding; active code removed
**Original file:** `src/sjfpo/mjx_env.py` (deleted)

The former MJX rollout path used `STORAGE_DTYPE` for rollout storage, which could quantize rewards to BF16 on Blackwell before GAE converted them back to FP32.

This is no longer an active Torch bug: the MJX implementation and its dedicated test were removed. The separate `src/sjfpo_jax/` backend is outside the active Torch scope of this audit.

## 15. `grad_accum_steps` has no effect

**Status:** Confirmed dead configuration
**Files:**

- `src/sjfpo/configuration_sjfpo.py:83-84,174-175`
- `src/sjfpo/train_sjfpo.py:347-373`

The option is validated, but every minibatch performs:

```python
loss.backward()
clip_grad_norm_()
optimizer.step()
```

No accumulation counter, loss scaling, or delayed step exists.

### Minimal fix

Implement accumulation or remove the option. A real implementation must scale losses, delay optimizer steps, and clip only at the accumulation boundary.

---

## 16. `n_action_steps` is contradictory between training and evaluation

**Status:** Confirmed configuration/behavior mismatch
**Files:**

- `src/sjfpo/train_sjfpo.py:81-93`
- `src/sjfpo/modeling_sjfpo.py:331-335,415-428`
- `src/sjfpo/configuration_sjfpo.py:20-30`

`build_policy()` overwrites the configured value with half the action horizon. Training rollouts ignore it and use the full horizon, while evaluation uses it.

### Impact

The same policy collects and evaluates with different action execution lengths, making evaluation behavior unlike training behavior and making the setting misleading.

### Minimal fix

Use one explicit execution-length contract for collection and evaluation, or add a clearly named separate training/evaluation setting.

---

## 17. Periodic evaluation overwrites `eval_history.csv`

**Status:** Confirmed logging bug
**Files:**

- `src/sjfpo/train_sjfpo.py:412-414`
- `src/sjfpo/evaluator.py:299-320`
- `src/sjfpo/evaluator.py:31-52`
- `src/sjfpo/baselines.py:389-394`

The evaluator defaults to `append_csv=False`, and the live training loop does not enable appending. Every evaluation rewrites the file.

### Impact

The file named `eval_history.csv` contains only the most recent evaluation batch, making training curves and final reporting incorrect.

### Minimal fix

Append after the first evaluation, or accumulate all evaluation rows and rewrite the complete history atomically.

---

## 18. Actor positional embedding is identical for every action token

**Status:** Confirmed architecture defect
**File:** `src/sjfpo/networks.py:457-462,495`

The token positional parameter has shape:

```python
(1, 1, hidden_dim)
```

and is expanded over the token axis. Every action token receives the same embedding.

### Impact

The actor cannot distinguish token positions through this parameter. The comment describes a learned positional embedding, but the implementation is only a shared learned vector.

### Minimal fix

Use a token-length dimension and slice/interpolate it to the current token count, including the reduced DCT token count.

---

## 19. Checkpoints include frozen behavior-policy duplicates

**Status:** Confirmed inefficient/dead checkpoint state
**Files:**

- `src/sjfpo/modeling_sjfpo.py:152-156`
- `src/sjfpo/checkpointing.py:147-149`

`old_actor` and `old_conditioner` are registered child modules and are likely included in `state_dict()`, although they are refreshed at the next rollout and are not needed for live optimizer state.

### Minimal fix

Exclude frozen snapshot modules from persisted state where compatible, or document that they are intentionally persisted.

---

# P3 — Dead or misleading code/documentation

## 20. Legacy chunk storage removed

**Status:** Resolved
**File:** `src/sjfpo/rollout_storage.py`

The inactive chunk-level storage and GAE implementation were removed. The active pipeline now uses `sjfpo.gae.compute_step_advantages_torch()` over stored post-step windows; `rollout_storage.py` retains only the active minibatch-index helper.

### Verification

The legacy storage class is no longer present:

```text
hasattr(sjfpo.rollout_storage, "RolloutStorage") == False
```

---

## 21. Self-attention key-mask documentation contradicts implementation

**File:** `src/sjfpo/networks.py:180-217`

The docstring says `True` means ignore, while the implementation comment and `keep = ~key_mask` use `True` as keep. There are currently no active callers passing the mask.

### Recommended action

Correct the documentation and add a direct unit test before enabling the API.

---

## 22. Environment matrix comments have stale run counts

**File:** `src/sjfpo/config.py:83-102`

The current four core environments with three methods and five seeds produce 60 runs, but comments still describe 45.

### Recommended action

Update comments or derive the count from the registry.

---

## 23. Package `AGENTS.md` contains stale runtime claims

**File:** `src/sjfpo/AGENTS.md`

Examples include:

- three-column evaluation CSV, while the code writes five columns;
- stale spectral-token defaults;
- references to removed modules such as `runner.py`;
- FP32-only claims despite current BF16/autocast paths.

### Recommended action

Synchronize documentation with executable source.

---

## 24. Checkpoint atomicity documentation is too strong

**File:** `src/sjfpo/checkpointing.py:12-14`

The module says all writes are atomic, but the policy model is written directly through `save_pretrained(d)`, while optimizer, counters, and RNG use temporary-file helpers.

### Recommended action

Implement a complete temporary-directory/replacement flow or narrow the claim to the files actually protected by the helpers.

---

# Verification summary

The highest-impact findings were reproduced in the active code:

```text
truncated GAE:                 observed advantage=1, expected=10
vector budget:                 requested=4, executed=32
postprocessor bounds:          policy=-2, postprocessor=-1
normalizer after rollout:      count=0, uninitialized=True
custom actor config:           requested depth=1, actual depth=4
```

No changes are required for the historical MJX findings in the current Torch package. The remaining recommended repair order is:

```text
1. Truncated GAE
2. Action-bound postprocessor synchronization
3. Padded action loss handling
4. Vector collector budget capping
5. Policy configuration propagation
6. Normalizer update and checkpoint state
7. Blackwell/device dtype handling
8. Logging/configuration cleanup
```
