# CUDA-only bugs and verification gaps

Everything here is either a defect that **cannot reproduce without a GPU**, or a
correctness risk that sits in a code path which **does not execute on CPU**. None
of it can be validated on the machine this was written on:

```
torch 2.11.0+cu130 | cuda available: False | device count: 0
```

Do not treat any item as confirmed-fixed without a CUDA run. Items 1 and 2 are
read-confirmed defects; the rest are unverified risks that only a GPU can settle.

## Status

Still accurate as written. All four checklist metrics (`explained_var`,
`value_target_std`, `advantage_std_fraction`, `critic_grad_norm_preclip`) are
still logged by `train_sjfpo.py`, and `resolve_storage_dtype` is still at
`utils.py:55`.

Two things changed since this file was written:

- **The critic was rewritten.** `Critic` is now one `StateConditioner` plus one
  linear head per member. E1 below is unaffected, since it concerns the buffer
  storage dtype and not the network. But any CUDA run of stage 1 measures a
  different critic than the one these numbers were first taken on, so
  before/after comparisons across the redesign are not apples-to-apples.
- **numpy is 2.2.6, and `wandb` calls `np.float_`,** removed in numpy 2.0. The
  default `report_to="wandb"` path raises `AttributeError` at import. Pass
  `--sjfpo.report_to tensorboard`. This also fails
  `tests/test_tracking.py::test_all_fans_out_to_every_backend` on CPU.

---

---

# Run checklist for a Blackwell (or any CUDA) box

10 items, ordered. Stages 1 and 2 need no code change. Stage 3 needs a decision
from me first, so do not start it without asking.

`scripts/cuda_probe.py` is the tool. It prints the diagnostics that only reach
W&B, plus a preflight header with the device name, the sm version, whether
`resolve_storage_dtype` is bf16, and peak VRAM.

```
ENV=Hopper-v5 SEED=3407 DCT=true STEPS=32768 .venv/bin/python scripts/cuda_probe.py
```

If the preflight header says `resolve_storage_dtype torch.float32`, this is not a
bf16-storage device, and **1.1 and 1.2 cannot be answered here**. The rest still
can.

CPU fp32 reference numbers, for comparison:

| run | explained_var | advantage_std_fraction | value_pred_std | returns_rms |
|---|---|---|---|---|
| InvertedPendulum, dct on, 32768 | -0.003 to -0.026 | 1.000-1.011 | 0.05-0.27 | ~2.9 |
| Hopper, dct on, 16384 | 0.22-0.24 | 0.88-0.90 | 0.44 | ~15 |

## Stage 1 — measurements, no code change

- [ ] **1.1 Verify the E1 fp32 fix holds on real bf16 storage**
  `ENV=Hopper-v5 SEED=3407 DCT=true STEPS=32768 .venv/bin/python scripts/cuda_probe.py`
  Pass: `value_target_std` is O(10), not sitting on a coarse quantisation ladder.
  Fail: it steps in visible jumps, meaning `raw_returns` is still being cast.
  Note this only confirms the dtype path, not that the fix helped. That is 1.2.

- [ ] **1.2 Is the negative explained_var a bf16 artefact?** — the single most
  important question here. Same run, or a shorter one.
  Pass: `explained_var` goes **positive** and `advantage_std_fraction` drops below
  1.0. Then the negative values were a bf16 storage artefact, E1 was the cause,
  and the critic was never actually broken.
  Fail: `explained_var` stays at or below 0, matching CPU. Then bf16 is not the
  cause and the critic genuinely is not fitting — the likelier outcome, since every
  CPU run also showed it, and it sends us back to the critic architecture.
  Record both numbers unrounded either way.

- [ ] **1.3 Does the pinned-memory H2D path race after the reuse fix?**
  `cd51f9d` made the host buffer genuinely reused instead of reallocated, so a
  missing sync would now corrupt data rather than being masked by a fresh buffer.
  Run the identical command twice and compare metric for metric.
  Pass: identical, or differing only in the last floating-point digit.
  Fail: any metric differs between identical runs, especially `returns_rms` or
  `explained_var`. Then the async copy races the host buffer and the reuse is
  unsafe at that batch size. Try a larger `--sjfpo.chunks_per_update` to see
  whether it scales with the copy.

- [ ] **1.4 Is peak VRAM flat across a num_kept_tokens sweep?**
  The DCT basis is cached in an `lru_cache` keyed on `(K, M, device)`, so the
  cached CUDA tensors are only freed on eviction.
  `for k in 2 4 8 16; do KEPT=$k STEPS=16384 OUT_DIR=/tmp/sweep_$k .venv/bin/python scripts/cuda_probe.py 2>&1 | grep -E "peak VRAM|value_target_std"; done`
  Pass: peak VRAM does not grow monotonically with `K`, or grows by less than the
  basis size. Fail: monotonic growth, so the cache needs a smaller `maxsize` or an
  explicit eviction.

- [ ] **1.5 Sanity: does a bf16 run produce the same reward as CPU?**
  `ENV=InvertedPendulum-v5 SEED=3407 DCT=false STEPS=32768 .venv/bin/python scripts/cuda_probe.py`
  Pass: final `eval/mean_return` inside the CPU seed spread, roughly 17-24 across
  the three CPU seeds. Fail: collapse, divergence, or `diverged: true` in the
  summary JSON. Then something in the bf16 path is wrong beyond precision, and
  that outranks everything in stage 3.

## Stage 2 — tests to write and run

Ask me and I will write these before you run them.

- [ ] **2.1 B4 — is a non-finite gradient detected?** The only item here that can
  silently ruin a long run. Nothing checks gradient finiteness, only the loss.
  Under bf16 an overflowing gradient leaves the loss finite, and
  `clip_grad_norm_` either returns `inf` as the total norm without raising, or
  raises, depending on torch version.
  Decide first: guard the `clip_grad_norm_` return value, or run `isfinite` over
  the gradients. Then inject a `nan` gradient and assert the run aborts rather
  than stepping with zero-scaled parameters.
  Also record what `critic_grad_norm_preclip` does on this torch version when a
  grad is `nan`, since that tells us which of the two behaviours we have.

- [ ] **2.2 B2 and B3 — dtype asserts inside the bf16 autocast region.** Both are
  already correct by reading; these are regression guards.
  B2: under CUDA bf16 autocast, assert the DCT output dtype is fp32. The guards
  use `autocast(device_type=vel.device.type, enabled=False)`, and a wrong device
  type would silently run the einsum in bf16 despite the explicit `.to(float32)`.
  B3: assert `compute_policy_ratio` returns fp32 under CUDA bf16 autocast. `exp`
  overflows near 11 in bf16 versus 88 in fp32, so a future edit moving the ratio
  inside autocast would saturate the surrogate silently.

## Stage 3 — fixes that need a decision from me first

Do not start these without answering. Each has an open design question.

- [ ] **3.1 A1 — the EMA shadow is left on CPU after a CUDA resume.** Now
  **reachable**, because `cd51f9d` made `ema_ready()` return true. Before that the
  shadow was never used after a resume.
  Test: save, move to CUDA, resume, run one `update_ema`, assert no exception and
  `shadow.device == param.device`. Expected failure today: `RuntimeError` from
  `shadow.mul_(decay).add_(cuda)`.
  The fix is agreed in principle, take the device from the live actor and raise on
  unknown shadow names, but it is not written.
  Open: whether checkpoints should persist live or EMA actor weights at all.
  `cd51f9d` made that question live and it is still unanswered — a resume currently
  starts from the EMA'd actor.

- [ ] **3.2 A2 — `_load_rng` leaves the CUDA setters unguarded.** Test: write an
  `rng.pt` holding a CUDA state sized for 4 devices, resume on a 1-GPU box, assert
  the resume completes. Fix: move `set_rng_state_all` inside the existing `try`.
  Low risk, but it blocks a resume, which is the worst time to find out.

- [ ] **3.3 B1 — are the variance-ratio diagnostics safe in bf16?** Run only if
  1.2 shows the numbers differ between CPU and Blackwell. `explained_var` is
  `1 - Var(R-V)/Var(R)` and `advantage_std_fraction` is a ratio of two standard
  deviations; both subtract or divide nearly-equal quantities, the worst case for
  an 8-bit mantissa. If they disagree with CPU by more than the seed spread, the
  diagnostics need an fp32 accumulation step, and every conclusion drawn from them
  across this project needs revisiting.

## Result table

Keep the raw numbers, not the rounded ones.

| item | run | key metric | value | verdict |
|---|---|---|---|---|
| 1.1 | | `value_target_std` | | |
| 1.2 | | `explained_var` | | |
| 1.2 | | `advantage_std_fraction` | | |
| 1.3 | | run-to-run delta | | |
| 1.4 | | peak VRAM | | |
| 1.5 | | `eval/mean_return` | | |
| 2.1 | | `critic_grad_norm_preclip` on nan | | |
| 3.1 | | resume + `update_ema` | | |
| 3.2 | | resume on fewer devices | | |
| 3.3 | | CPU vs Blackwell delta | | |


---

## A. Confirmed defects, only observable on CUDA

### A1. EMA shadow is left on CPU after a CUDA resume

`checkpointing.py:248` loads with `map_location="cpu"` — correct for the optimizer
and counters. `modeling_sjfpo.py:356-364`:

```python
self._ema_shadow = {
    name: value.detach().clone()
    for name, value in state["shadow_params"].items()
}
```

`_ema_shadow` is a plain dict of tensors, not registered buffers, so
`policy.to(device)` does not move it and neither does this method. The next
`update_ema` runs:

```python
shadow.mul_(decay).add_(current[name].detach(), alpha=1.0 - decay)
```

with `shadow` on CPU and `current[name]` on CUDA, which raises. On CPU-only runs
both tensors are CPU and the bug is invisible.

**Fix** — take the device from the live actor so the method is self-sufficient
rather than dependent on call order:

```python
live = dict(self.actor.named_parameters())
missing = [n for n in state["shadow_params"] if n not in live]
if missing:
    raise ValueError(f"EMA shadow names absent from the actor: {missing[:3]}")
self._ema_shadow = {
    name: value.detach().clone().to(live[name].device)
    for name, value in state["shadow_params"].items()
}
```

The `missing` guard matters: today a checkpoint from a different architecture is
accepted silently and then dies with a `KeyError` inside `update_ema`, several
updates later.

**Test (GPU only)** — save, move to CUDA, resume, run one `update_ema`, assert
no exception and `shadow.device == param.device`. A CPU-only assertion of
`shadow.device == param.device` passes vacuously and proves nothing.

**Interaction:** this only bites now that `cd51f9d` makes `ema_ready()` return
true. Before that, the shadow was never used after a resume.

### A2. `_load_rng` leaves the CUDA setters unguarded

`checkpointing.py:185-200`. The `try` covers only `torch.load`:

```python
try:
    rng = torch.load(path, map_location="cpu", weights_only=True)
except Exception:
    return                      # corrupt RNG file shouldn't block resume
if "torch" in rng:
    torch.set_rng_state(rng["torch"])
if "cuda" in rng and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(rng["cuda"])   # not guarded
```

`set_rng_state_all` raises on a device-count mismatch — a realistic case when
resuming a multi-GPU run on a different node, or a single-GPU run on a machine
that now exposes several. The docstring promises a corrupt RNG file "shouldn't
block a resume", and this path does.

**Fix** — move the setters inside the `try`, or wrap them in their own.

**Test (GPU only)** — write an `rng.pt` with a CUDA state sized for 4 devices,
resume on a 1-GPU box, assert resume completes.

---

## B. The bf16 path — large, and entirely absent on CPU

`utils.py:100-109`:

```python
def autocast_context(device=None):
    if not cuda_available():
        return contextlib.nullcontext()      # CPU: never autocast
    if resolve_compute_dtype() != torch.bfloat16:
        return contextlib.nullcontext()
    ...
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

Every autocast region in the actor, critic, and loss is CUDA-gated. On CPU the
whole low-precision surface is dead code. This is where I would look first for a
CUDA-only numerical regression.

### B1. Variance-ratio diagnostics under bf16

`explained_var` is `1 - Var(R - V) / Var(R)`, and `advantage_std_fraction` is a
ratio of two standard deviations. Both subtract or divide two nearly-equal
quantities, which is the worst case for bf16's 8-bit mantissa. `explained_var` has
already been observed at or below zero on CPU, where the arithmetic is fp32 —
so the bf16 result is not obviously safe.

**Test** — run the identical seed on CPU and CUDA, bf16 on CUDA, and compare
`explained_var` and `advantage_std_fraction`. A divergence larger than the CPU
seed-to-seed spread means the diagnostics need fp32 accumulation.

### B2. The DCT's autocast opt-out depends on `device.type`

`extract_dct_coeffs`, `decode_coeffs_to_velocity` and `velocity_to_action_steps`
all do:

```python
with torch.autocast(device_type=vel.device.type, enabled=False):
```

This is correct — `autocast(device_type="cpu", enabled=False)` would **not**
disable an enclosing CUDA autocast region, so if the device type is ever wrong the
`einsum` silently runs in bf16 despite the explicit `.to(torch.float32)`. The
guards are right today; there is no test that would catch a regression.

**Test** — under CUDA bf16 autocast, assert the DCT output dtype is fp32.

### B3. `exp` in the policy ratio

`compute_policy_ratio` returns `exp(old - new)` with a one-sided STE cap at
`log_ratio_cap`. The cap bounds the forward value, and the CFM losses are cast to
fp32 before this point, so the current path is fp32. If a future edit moves the
ratio inside an autocast region, `exp` overflows near 11 in bf16 versus 88 in
fp32, and the surrogate silently saturates.

**Test** — assert `compute_policy_ratio` returns fp32 when called under CUDA
bf16 autocast.

### B4. Gradient non-finiteness is never checked, only the loss

`train_sjfpo.py:390` accumulates `finite_error |= ~torch.isfinite(loss)` and
`modeling_sjfpo.py:1113` checks the loss. Nothing checks the **gradients**. Under
bf16 an overflowing gradient does not make the loss non-finite, and
`clip_grad_norm_` either returns `inf`/`nan` as the total norm (older torch, no
error) or raises (newer torch, inconsistent across versions).

So on CUDA the run can degrade silently: `actor_grad_norm_preclip` and
`critic_grad_norm_preclip` go to `inf`, the clip divides by `inf`, every parameter
is scaled to zero, and training stops without any error. The preclip norms are
already logged, so `inf` there is the only signal.

**Test (GPU only)** — inject a `nan` gradient and assert the run aborts rather
than stepping with zero-scaled parameters. Decide first whether to guard
`clip_grad_norm_`'s return value or to run `isfinite` on the grads.

---

## C. Untestable on CPU, but I read them as correct

Recording these so nobody re-raises them as bugs without looking.

### C1. Pinned-memory H2D overlap is correctly ordered

`modeling_sjfpo.py:707-709` synchronizes the previous rollout's event at the top
of `generate_rollout`, before the host buffer is rewritten. `782-784` records the
event after `copy_to(..., non_blocking=True)` is enqueued, and
`active` is a view of the destination on the same stream. The ordering is right.

**This got riskier with `cd51f9d`.** The host buffer is now genuinely reused
across rollouts rather than reallocated, so a missing sync would now corrupt data
instead of being masked by a fresh allocation. It is correct by reading, and worth
one GPU test with `CUDA_LAUNCH_BLOCKING=0` and a large batch, comparing against a
`non_blocking=False` reference.

### C2. `_get_analysis_basis` caches device tensors in an `lru_cache`

`dct_utils.py:28`, `maxsize=64`, keyed on `(K, M, device)`. The cached CUDA
tensors are never freed until eviction, so sweeping `num_kept_tokens` or
alternating devices retains up to 64 bases. Small tensors, so this is VRAM
hygiene rather than a leak, and the key includes the device so the tensors are
never handed to the wrong device.

**Test (GPU only)** — assert peak VRAM is flat across a `num_kept_tokens` sweep.

### C3. Normalizer buffers do follow the device

`train_sjfpo.py:123` calls `policy_cls(policy_cfg).to(device)`, and
`RunningMeanStd`'s `mean`/`var`/`count` are registered buffers, so they move.
`RunningMeanStd.update` accumulates on the buffer's own device and `forward`
does `.to(x.device)` on the statistics. No cross-device bug found.

---

## D. Recommended order

1. **A1** — a hard crash on any CUDA resume, and now reachable because the EMA is
   live. Fix is three lines. Needs one GPU test to confirm.
2. **B4** — the only item on this list that can silently ruin a long run without
   raising.
3. **B1** — measure the diagnostics under bf16 before trusting any of the numbers
   from a GPU run. Everything downstream is read against those diagnostics.
4. **A2** — narrow, only bites on a changed GPU topology.
5. **C1** — a regression guard for the reuse fix, not a known bug.
6. **B2, B3, C2, C3** — cheap assertions, low risk, no known defect.

---

## E. Added after the bf16 investigation

### E1. The critic's regression target was quantised to bf16

`resolve_storage_dtype` (`utils.py:63-73`) returns `bfloat16` on RTX PRO 1000
Blackwell and fp32 everywhere else. `generate_rollout` cast `raw_returns` and
`raw_values` to it on the way into the buffer, and `forward` cast them back on the
way out, so the target arrived at the critic already quantised.

bf16 has 8 mantissa bits. `1234.5` round-trips to `1232.0`. `returns_rms` runs
to 35 and the values reach the hundreds, so the per-element error was on the order
of 0.1-0.5 — imposed on the regression target before any learning happened. It
was invisible on CPU, where the storage dtype is always fp32.

**Fixed** — both tensors now stay fp32 on write and are read back as fp32. The
value loss was already computed under `autocast(enabled=False)` with a `.float()`,
so only the storage cast was losing precision.

Why the fix is free: these two tensors are 256 elements each. The buffers holding
`step_states` and `chunk_actions` are already hardcoded fp32 via
`collector._OBS_DTYPE`, so `storage_dtype` was never saving meaningful memory. It
was paying a precision cost for nothing.

What legitimately still rides `storage_dtype`, and why each is safe:

| tensor | magnitude | why bf16 is acceptable |
|---|---|---|
| `normalized_advantages` | mean 0, std 1 | 0.4% relative becomes a 0.4% shift in `exp`, against `clip_param = 0.2` |
| `behavior_cfm_loss[_token]` | O(1) | absolute error ~5e-4 on the ratio's numerator |
| `mc_eps`, `mc_tau` | O(1) | inside the flow interpolation, not a target in its own right |
| `raw_advantages` | O(1) after the cast is irrelevant | only `.shape[0]` and `.device` are ever read |

**Not yet verified.** `explained_var` was at or below zero in every run so far, on
CPU, in fp32. This fix removes one plausible contributor on Blackwell hardware but
does not establish that it was the cause. The test that matters is still: run the
same seed on a bf16-storage device and check whether `explained_var` goes
positive. Until that is done, treat the negative `explained_var` as unresolved.
