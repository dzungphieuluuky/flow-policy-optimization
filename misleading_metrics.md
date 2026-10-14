# Misleading Metric Implementations (Detected from Code Audit)
Source: src/, tests/test_metric_faithfulness.py, docs/DIAGNOSE_DEGRADATION.md, tests/test_ema_and_huber_defects.py

---

## 1. `budget_truncated_chunk_frac` Reports NaN as "Unmeasured" (Not Zero)

**Location**: `src/spectral_joint_policy/modeling_sjfpo.py` line 119; `collector.py` line 269; `tests/test_metric_faithfulness.py` line 14.

**What it claims**: A fraction of chunks truncated by the step budget (`chunk_frac`).

**What it actually does**:
- When `budget_truncated` is `None` (unsupplied): reports `float("nan")`.
- When supplied (`torch.Tensor` of bool): reports `.mean()` as a real fraction.

**Why misleading**:
The name (`chunk_frac`) implies a measurable fraction. A reader sees NaN and interprets it as "data missing" or "error," not as "no budget truncation occurred." The docstring in `collector.py` admits: `"budget_truncated_chunk_frac reports NaN rather than 0.0"`. The test explains the rationale: `"0.0 would assert 'no chunk was cut short by the budget', which is a claim about the data."` — i.e., zero claims measurement; NaN claims absence.

**Impact**: Any dashboard or automated analysis that expects 0.0 = no truncation will misread NaN. The metric behaves differently from standard fraction metrics (e.g., `success_rate`) which report 0.0 when no successes exist.

---

## 2. `_GAE_DIAG_LOG_KEYS`: Name Mapping Mismatch (`return_rms` → `returns_rms`)

**Location**: `src/spectral_joint_policy/train_sjfpo.py` lines 42-50.

**Code**:
```python
_GAE_DIAG_LOG_KEYS = {
    "linked_frac": "gae_linked_frac",
    "advantage_std": "gae_advantage_std",
    "return_rms": "returns_rms",          # singular -> plural
    "advantage_std_fraction": "advantage_std_fraction",
    "budget_truncated_chunk_frac": "budget_truncated_chunk_frac",
}
```

**Comment**: `# GAE diagnostic name -> logged name. Deliberately not a uniform prefix: return_rms is logged as returns_rms.`

**Why misleading**: The source key (`return_rms`) and logged key (`returns_rms`) don't match. A researcher tracing `returns_rms` back to the computation must know to search for `return_rms` (singular). The comment admits it's deliberate (`"deliberately not a uniform prefix"`), but doesn't explain why the singular/plural mismatch is necessary — it just breaks traceability.

**Impact**: Metric tracing, debugging, and reproducibility suffer. The `tests/test_metric_faithfulness.py` asserts `not math.isnan(out["return_rms"])` (singular), confirming the source name is singular, but the tracker emits `train/returns_rms` (plural).

---

## 3. `action_magnitude` → `action_abs_mean`: Previous Name Was an L2 Norm Claim

**Location**: `tests/test_metric_faithfulness.py` line 9; `tests/test_metric_audit.py` line 63.

**Historical defect**: The metric was previously named `action_magnitude` (`tests/test_metric_faithfulness.py`: `"It was named action_magnitude, which reads as a norm."`). The actual computation is `mean|a|` (mean absolute value), not `||a||` (L2 norm).

**Fix applied**: Renamed to `action_abs_mean`; docstring now clarifies: `"action_abs_mean is mean|a|, not an L2 norm -- it is named to pair with action_abs_max."`

**Why it was misleading**: The old name (`magnitude`) implies Euclidean magnitude (`sqrt(sum(a²))`). A user comparing `action_abs_mean` to physics-based magnitude thresholds would apply the wrong conversion.

**Current state**: The metric is now correctly named, but the test file (`test_metric_audit.py`) preserves the regression guard: `test_stored_actions_never_exceed_unit_abs_so_a_magnitude_guard_is_useless` explains why a magnitude guard (`>1.05`) was dead — because actions are clipped to `[-1, 1]` before storage, so `abs().max()` is always `1.0`.

---

## 4. `compute_spectral_energy_retained`: Dead Weight / Ghost Metric

**Location**: `tests/test_metric_faithfulness.py` lines 114-121; `tests/test_metric_audit.py` line 273.

**Evidence**:
- The test searches for `"compute_spectral_energy_retained"` in `train_sjfpo.py` source and asserts it's NOT present (`assert "compute_spectral_energy_retained" not in src`).
- The test comment: `"d0f97c2 removed its source; a passthrough slot for it is dead weight."`
- The audit test (`test_trainer_emits_no_key_it_cannot_compute`) scans AST for keys read from `batch` that are never produced upstream — `compute_spectral_energy_retained` was the only example.

**Why misleading**: There was a metric key reference (`compute_spectral_energy_retained`) that had no corresponding computation. A tracker or logger might reference it (as a passthrough), producing a missing-key error or silent omission. The metric name (`spectral_energy_retained`) implies a spectral analysis of retained energy — a concrete measurement — but no code produces it.

---

## 5. Huber Knee Scaled by `returns_rms` (Not Residual RMS) — Defect in Metric Usage

**Location**: `tests/test_ema_and_huber_defects.py` lines 1-3, 108; `src/spectral_joint_policy/configuration_sjfpo.py` line 109.

**Defect description** (from test file):
> "The Huber knee was scaled by returns_rms, putting it at ~2x the residual spread, so 97% of residuals took the quadratic branch and the loss was MSE in disguise."

**Why misleading**:
The metric `returns_rms` (return root-mean-square) was being used to scale the Huber `delta` parameter. But `delta` should scale with the **residual spread** (the spread of `pred - target`), not the return scale. If returns are large (e.g., humanoid control with rewards in hundreds), the Huber knee (`delta`) becomes huge, making the loss quadratic (MSE-like) for almost all points. The metric `returns_rms` is a property of the return distribution, not the residual distribution — using it for `delta` confuses two different statistical quantities.

**Fix verified by test**: `test_huber_knee_is_not_built_from_the_return_scale` asserts that `delta` does NOT scale with `returns_rms`.

---

## 6. Gradient Norm Collapse (Single Metric, Not Split)

**Location**: `docs/DIAGNOSE_DEGRADATION.md`; `src/spectral_joint_policy/train_sjfpo.py` lines 617-624.

**What the doc says**:
> "`train/actor_grad_norm_preclip` and `train/critic_grad_norm_preclip` were collapsed into `train/grad_norm_preclip` in commit `fc8724b`, which is the one metric that could answer it."

**Why misleading**: The single metric `grad_norm_preclip` (joint over actor + critic) reports a combined gradient norm. When it shows `2291.5` (vs budget `1`), it hides whether the actor, the critic, or both are saturated. A user sees the number and assumes joint saturation, but without the split, attribution is impossible. The diagnostic file explicitly states this prevents answering the key question: "How does the 2291x split between actor and critic?"

---

## 7. `eval/reward_improvement_pct`: Only Computed When Last Mean Matches Previous

**Location**: `src/spectral_joint_policy/tracking/base.py` lines 75-82.

**Code**:
```python
if len(all_eval_returns) >= 2 and all_eval_returns[-1] == mean_return:
    prev = all_eval_returns[-2]
    data["eval/reward_improvement"] = float(mean_return - prev)
    data["eval/reward_improvement_pct"] = float((mean_return - prev) / max(abs(prev), 1e-8))
```

**Why misleading**: The metric `reward_improvement_pct` is only emitted when `all_eval_returns[-1] == mean_return` (the last entry equals the current mean). This is a guard against gaps, but the metric's absence in other cases could be misread as "no improvement" rather than "not computed due to gap." A reader comparing runs might miss that one run has the metric and another doesn't — not because performance differed, but because of series continuity.

---

## Summary of Misleading Behaviors

| Metric / Pattern | What It Claims | What It Actually Does / Was | Risk |
|---|---|---|---|
| `budget_truncated_chunk_frac` | Fraction of budget-cut chunks | NaN when unmeasured (not 0.0) | False "missing data" reading |
| `_GAE_DIAG_LOG_KEYS` mapping | Consistent naming | `return_rms` → `returns_rms` (singular/plural) | Traceability broken |
| `action_magnitude` (legacy) | Magnitude (L2) | Mean absolute value (`mean\|a\|`) | Wrong physical interpretation |
| `compute_spectral_energy_retained` | Spectral energy measurement | Not implemented (dead weight) | Phantom metric |
| Huber `delta` scaled by `returns_rms` | Residual-sensitive clipping | Return-scale clipping (MSE in disguise) | Loss behaves incorrectly |
| `grad_norm_preclip` (joint only) | Gradient health indicator | Combined actor+critic norm (split collapsed) | Cannot diagnose which module fails |
| `eval/reward_improvement_pct` | Improvement over previous eval | Only computed when series continuous | False absence = false conclusion |

---

*Sources: `tests/test_metric_faithfulness.py`, `tests/test_metric_audit.py`, `tests/test_ema_and_huber_defects.py`, `docs/DIAGNOSE_DEGRADATION.md`, `src/`, `README.md`.*
