# CUDA-only bugs and verification gaps

Everything here is either a defect that **cannot reproduce without a GPU**, or a
correctness risk that sits in a code path which **does not execute on CPU**. None
of it can be validated on the machine this was written on:

```
torch 2.11.0+cu130 | cuda available: False | device count: 0
```

Do not treat any item as confirmed-fixed without a CUDA run. Items 1 and 2 are
read-confirmed defects; the rest are unverified risks that only a GPU can settle.

---

# Run checklist for a Blackwell (or any CUDA) box

Everything below is ordered. Stages 1 and 2 need no code change and can be run as
they are. Stage 3 needs a decision from me first, so do not start it without
asking.

`scripts/cuda_probe.py` is the tool. It prints the diagnostics that only reach
W&B, plus a preflight header with the device name, the sm version, whether
`resolve_storage_dtype` is bf16, and peak VRAM.

```
ENV=Hopper-v5 SEED=3407 DCT=true STEPS=32768 .venv/bin/python scripts/cuda_probe.py
```

If the preflight header says `resolve_storage_dtype torch.float32`, this is not a
bf16-storage device and **stage 1.1 and 1.2 do not apply** — the E1 and B1
questions cannot be answered here. The rest still can.

Reference numbers from CPU, fp32, for comparison:

| run | explained_var | advantage_std_fraction | value_pred_std | returns_rms |
|---|---|---|---|---|
| InvertedPendulum, dct on, 32768 | -0.003 to -0.026 | 1.000-1.011 | 0.05-0.27 | ~2.9 |
| Hopper, dct on, 16384 | 0.22-0.24 | 0.88-0.90 | 0.44 | ~15 |

---

## Stage 1 — measurements, no code change

### [ ] 1.1 Verify the E1 fp32 fix holds on real bf16 storage

```
ENV=Hopper-v5 SEED=3407 DCT=true STEPS=32768 .venv/bin/python scripts/cuda_probe.py
```

Check the preflight header first. Then confirm the batch dtype, which is the
thing `6aee3f1` changed:

- **Pass:** `value_target_std` is O(10), not collapsed toward a quantised grid,
  and the checkpoint at `OUT_DIR` reloads with the same value
- **Fail:** `value_target_std` sits on a coarse step ladder, which means
  `raw_returns` is still being cast
- **Note:** this only confirms the *dtype path*. It does **not** confirm the fix
  helped. That is 1.2.

### [ ] 1.2 Is the negative explained_var a bf16 artefact?

Same run, or a shorter one. This is the single question that matters most.

- **Pass:** `explained_var` goes **positive** and `advantage_std_fraction` drops
  below 1.0. Then the negative values were a bf16 storage artefact, E1 was the
  cause, and the critic was never actually broken
- **Fail:** `explained_var` stays at or below 0, matching CPU. Then bf16 is not
  the cause and the critic is genuinely not fitting. That is the more likely
  outcome given every CPU run also showed it, and it sends us back to the critic
  architecture rather than to precision
- Either way, record both numbers. Do not round

### [ ] 1.3 Does the pinned-memory H2D path race after the reuse fix?

`cd51f9d` made the host buffer genuinely reused instead of reallocated, so a
missing sync would now corrupt data rather than being masked by a fresh buffer.

```
ENV=Hopper-v5 SEED=3407 DCT=true STEPS=32768 \
  .venv/bin/python scripts/cuda_probe.py
```

Then rerun the identical command and compare metric-for-metric.

- **Pass:** two runs of the same seed are identical, or differ only in the last
  floating-point digit
- **Fail:** any metric differs between identical runs, especially
  `returns_rms` or `explained_var`. Then the async copy is racing the host buffer
  and the reuse is unsafe at that batch size. Try a larger
  `--sjfpo.chunks_per_update` to see whether it scales with the copy

### [ ] 1.4 Is peak VRAM flat across a num_kept_tokens sweep?

The DCT basis is cached in an `lru_cache` keyed on `(K, M, device)`, so the
cached CUDA tensors are only freed on eviction.

```
for k in 2 4 8 16; do
  KEPT=$k STEPS=16384 OUT_DIR=/tmp/sweep_$k .venv/bin/python scripts/cuda_probe.py 2>&1 \
    | grep -E "peak VRAM|value_target_std"
done
```

- **Pass:** peak VRAM does not grow monotonically with `K`, or grows by less than
  the basis size
- **Fail:** monotonic growth. Then the cache needs a smaller `maxsize` or an
  explicit eviction

### [ ] 1.5 Sanity: does a bf16 run produce the same reward as CPU?

```
ENV=InvertedPendulum-v5 SEED=3407 DCT=false STEPS=32768 .venv/bin/python scripts/cuda_probe.py
```

- **Pass:** final `eval/mean_return` within the CPU seed spread, which is roughly
  17-24 across the three CPU seeds
- **Fail:** collapse, divergence, or `diverged: true` in the summary JSON. Then
  something in the bf16 path is wrong beyond precision, and that outranks
  everything in stage 3

---

## Stage 2 — tests to write and run

These need test code. Ask me and I will write them before you run them.

### [ ] 2.1 B4 — is a non-finite gradient detected?

The only item here that can silently ruin a long run. Nothing checks gradient
finiteness; only the loss is checked. Under bf16 an overflowing gradient leaves
the loss finite, and `clip_grad_norm_` either returns `inf` as the total norm
without raising, or raises, depending on the torch version.

- **Decide first:** guard the `clip_grad_norm_` return value, or run `isfinite`
  over the gradients
- **Then:** inject a `nan` gradient and assert the run aborts rather than stepping
  with zero-scaled parameters
- **Also record:** what `critic_grad_norm_preclip` does on this torch version when
  a grad is `nan`. That tells us which of the two behaviours we have

### [ ] 2.2 B2 and B3 — dtype asserts inside the bf16 autocast region

Both are already correct by reading. These are regression guards.

- **B2:** under CUDA bf16 autocast, assert the DCT output dtype is fp32. The
  guards use `autocast(device_type=vel.device.type, enabled=False)`, and a wrong
  device type would silently run the einsum in bf16 despite the explicit
  `.to(float32)`
- **B3:** assert `compute_policy_ratio` returns fp32 when called under CUDA bf16
  autocast. `exp` overflows near 11 in bf16 versus 88 in fp32, so a future edit
  moving the ratio inside autocast would saturate the surrogate silently

---

## Stage 3 — fixes that need a decision from me first

Do not start these without answering. Each has an open design question.

### [ ] 3.1 A1 — the EMA shadow is left on CPU after a CUDA resume

This is now **reachable**, because `cd51f9d` made `ema_ready()` return true.
Before that, the shadow was never used after a resume.

- **Test:** save, move to CUDA, resume, run one `update_ema`, assert no exception
  and `shadow.device == param.device`
- **Expected failure today:** `RuntimeError` from `shadow.mul_(decay).add_(cuda)`
- **Fix is agreed in principle** (take the device from the live actor, and raise
  on unknown shadow names), but it has not been written
- **Open:** whether checkpoints should persist live or EMA actor weights at all.
  `cd51f9d` made that question live and it is still unanswered. A resume currently
  starts from the EMA'd actor

### [ ] 3.2 A2 — `_load_rng` leaves the CUDA setters unguarded

- **Test:** write an `rng.pt` holding a CUDA state sized for 4 devices, resume on
  a 1-GPU box, assert the resume completes
- **Fix:** move `set_rng_state_all` inside the existing `try`
- **Low risk**, but it blocks a resume, which is the worst time to find out

### [ ] 3.3 B1 — are the variance-ratio diagnostics safe in bf16?

Run only if 1.2 shows the numbers differ between CPU and Blackwell.

`explained_var` is `1 - Var(R-V)/Var(R)` and `advantage_std_fraction` is a ratio
of two standard deviations. Both subtract or divide nearly-equal quantities, the
worst case for an 8-bit mantissa. If they disagree with the CPU run by more than
the seed spread, the diagnostics need an fp32 accumulation step, and every
conclusion drawn from them across this project needs revisiting.

---

## Result table

Fill in as you go. Keep the raw numbers, not the rounded ones.

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

