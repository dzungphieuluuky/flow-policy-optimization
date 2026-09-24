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

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/utils.py:64-93`
- `src/sjfpo/modeling_sjfpo.py:196-223,293-304,416-430,633-650`
- `src/sjfpo/networks.py:47-61`
- `src/sjfpo/fpo_core.py:82-89`

Storage dtype is now resolved from the actual execution device. CPU always uses FP32, including on Blackwell hosts; CUDA Blackwell uses BF16 storage when enabled. Model parameters and RMS normalization remain FP32-safe while runtime data follows the device policy.

---

## 11. `SJFPO_DTYPE=fp32` is ignored on Blackwell

**Status:** Resolved in the active Torch path
**File:** `src/sjfpo/utils.py:70-93`

`SJFPO_DTYPE` is now parsed and validated before the Blackwell default is applied. `SJFPO_DTYPE=fp32` therefore overrides Blackwell BF16, while `bf16` is selected only when CUDA supports it.

---

## 12. MJX truncation finding — removed from the active Torch path

**Status:** Removed by policy decision
**Original file:** `src/sjfpo/mjx_env.py` (deleted)

The former MJX collector autoreset completed environments and returned the reset observation as the transition's next state. The stale `SJFPO_MJX` test and Torch integration were removed; training always uses the Gymnasium Torch path. The separate `src/sjfpo_jax/` backend remains outside this Torch-specific removal.

# P2 — Reliability, logging, and precision issues

## 14. MJX reward precision finding — removed from the active Torch path

**Status:** Removed by policy decision
**Original file:** `src/sjfpo/mjx_env.py` (deleted)

The former MJX rollout path and its stale test were removed. The active Torch training path uses Gymnasium only; the separate `src/sjfpo_jax/` backend remains outside this Torch-specific removal.

## 15. `grad_accum_steps` has no effect

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/configuration_sjfpo.py:83-87,174-186`
- `src/sjfpo/train_sjfpo.py:341-399`

The training loop now follows Accelerate's accumulation contract: every microbatch performs a scaled backward pass, gradients and diagnostics are clipped only at the accumulation boundary, optimizer steps occur only at boundaries, and a final partial group is flushed. This is the single-process equivalent of `Accelerator.accumulate`; distributed `no_sync` behavior is not needed by the current non-distributed Torch trainer.

---

## 16. `n_action_steps` is contradictory between training and evaluation

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/configuration_sjfpo.py:26-30,101-115`
- `src/sjfpo/modeling_sjfpo.py:416-464`
- `src/sjfpo/collector.py:293-318,448-478`
- `scripts/train.py:141-178,343-372`

`n_action_steps` is loaded from `SJFPOConfig`, overlaid by `--config` YAML, and preserved when an explicit `--sjfpo.n_action_steps` CLI field is present. Both single-env and vector collection now pass the configured execution length, while evaluation already uses the same policy configuration. The collector also receives the configured action horizon explicitly instead of resolving it from a module-global config.

---

## 17. Periodic evaluation overwrites `eval_history.csv`

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/train_sjfpo.py:411-416`
- `src/sjfpo/baselines.py:389-395`
- `src/sjfpo/evaluator.py:31-52,299-320`

Periodic SJFPO and baseline evaluations now pass `append_csv=True`. The evaluator writes the header only for a new file and preserves all prior evaluation rows, so `eval_history.csv` remains a complete run history.

---

## 18. Actor positional embedding is identical for every action token

**Status:** Resolved in the active Torch path
**File:** `src/sjfpo/networks.py:399-473`

`SJFPOFlowModel` now stores a learned positional table with shape:

```python
(1, action_horizon, hidden_dim)
```

The forward pass slices the table to the current token count, so reduced DCT token counts retain distinct positions. Inputs exceeding `action_horizon` fail explicitly. Checkpoints are intentionally not backward-compatible with the old shared-position shape because training will restart from scratch.

---

## 19. Checkpoints include frozen behavior-policy duplicates

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/modeling_sjfpo.py:162-169`
- `src/sjfpo/checkpointing.py:147-158,294-306`
- `tests/test_policy_checkpoint.py:122-163`

Frozen `old_actor` and `old_conditioner` snapshots remain available in memory for behavior-policy ratios, but are filtered out of `model.safetensors`. On resume they are reconstructed from the loaded live modules before training continues, so no optimizer or rollout behavior is lost.

---

# P3 — Dead or misleading code/documentation

## 20. Legacy chunk storage removed

**Status:** Resolved
**Files:**

- `src/sjfpo/rollout_storage.py` (deleted)
- `src/sjfpo/train_sjfpo.py`

The inactive chunk-level storage and GAE implementation were removed. The active pipeline uses `sjfpo.gae.compute_step_advantages_torch()` over stored post-step windows; the remaining minibatch-index helper now lives directly in the Torch training entrypoint.

### Verification

The legacy storage module is no longer present:

```text
Path("src/sjfpo/rollout_storage.py").exists() == False
```

---

## 21. Self-attention key-mask documentation contradicts implementation

**Status:** Resolved in the active Torch path
**Files:**

- `src/sjfpo/networks.py:156-240`
- `tests/test_networks.py:59-71`

`SelfAttention` and `WindowEncoder` now use the Hugging Face convention: `1`/`True` marks a valid key and `0`/`False` marks an ignored key. The mask is passed directly to SDPA, with the first key restored as a valid fallback for fully masked rows. The unit test now expresses the same valid/ignored convention.

---

## 22. Environment matrix comments have stale run counts

**Status:** Removed by policy decision
**File:** `src/sjfpo/config.py`

The automatic `RUN_MATRIX`, `TINY_MATRIX`, method/seed expansion, and matrix validation code were removed. Training configurations are now selected manually through the CLI/config path; `config.py` only retains environment lookup, baseline dataclasses, and `RunSpec` for callers that explicitly manage their own runs.

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

**Status:** Resolved by documentation correction
**File:** `src/sjfpo/checkpointing.py:12-15`

The module now documents the actual guarantee: optimizer, counters, and RNG files use temporary-file replacement, while model/config serialization follows the LeRobot/Hugging Face `save_pretrained` path. It no longer claims that the complete multi-file checkpoint is atomic.

---

# Verification summary

The highest-impact findings were reproduced before their fixes:

```text
truncated GAE:                 observed advantage=1, expected=10
vector budget:                 requested=4, executed=32
postprocessor bounds:          policy=-2, postprocessor=-1
normalizer after rollout:      count=0, uninitialized=True
custom actor config:           requested depth=1, actual depth=4
```

The active Torch path now uses Gymnasium only, has device-aware BF16/FP32 resolution, and no longer exposes the removed 8-bit optimizer configuration. The remaining repair order is:

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
