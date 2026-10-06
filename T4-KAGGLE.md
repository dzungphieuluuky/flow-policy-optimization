# T4 / Kaggle — `torch.compile` setup

Target: NVIDIA Tesla T4 (Turing, **sm_75**) on Kaggle. Companion to
`TORCH-COMPILE.md`, which targets RTX PRO 1000 Blackwell and assumes things
that are **false** on a T4. Read this file; do not copy the mode table across.

Nothing here is measured. Every T4 number is a decision rule, and each rule
says what to measure to replace it.

---

## 0. Get the T4 at all, or nothing else matters

This is the first trap and it is silent.

`enable_gpu: true` alone gives you a **P100 (sm_60)**. Modern PyTorch wheels
require sm_70+, so a P100 session produces kernels that build and then do
nothing — no error, just a dead run. The CLI also does **not** persist a GPU
model chosen in the Kaggle UI.

`kernel-metadata.json` needs it on **every** push:

```json
{
  "id": "owner/kernel-slug",
  "title": "sjfpo",
  "code_file": "sjfpo.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": []
}
```

The `--accelerator NvidiaTeslaT4` push flag is reported **not** to work; the
metadata key is the path that was verified with real runs. Source:
`Kaggle/kaggle-cli#821`, corroborated Aug 2026.

**Confirm before spending a session:**

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
.venv/bin/python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.major, p.minor, round(p.total_memory/2**30,1),'GiB')"
```

Expect `T4`, `7`, `5`, `15.7`GiB. If `6` and `0` print, stop — you are on a P100.

## 1. What is different from the Blackwell setup

| | Blackwell (`TORCH-COMPILE.md`) | **T4** |
|---|---|---|
| compute capability | sm_120 | **sm_75** |
| bf16 tensor cores | yes | **no** — needs sm_80+ |
| repo dtype on this GPU | bf16 storage + bf16 autocast | **fp32 storage + fp32 autocast** |
| FP32 | — | 8.1 TFLOPS |
| FP16 tensor | — | **65 TFLOPS** |
| VRAM | large | **16 GB — the binding constraint** |
| TMA / Hopper features | n/a | none (sm_90+) |
| throttling | laptop, real | passive 70 W, not a factor |
| CPU cores | host | **4** |
| session cap | none | **12 h**, 30 h/week, 20 min idle |

### The dtype consequence is the biggest single number on this page

The repo already degrades correctly and **without any code change**:

- `resolve_storage_dtype` returns bf16 only when `is_rtx_1000_pro_blackwell()`,
  so on a T4 it returns **fp32**.
- `resolve_compute_dtype` falls back to `torch.cuda.is_bf16_supported()`, which is
  **False** on sm_75, so it returns **fp32**.
- `autocast_context` becomes a `nullcontext`.

So a T4 run executes in **fp32 at 8.1 TFLOPS**, on a card whose fp16 tensor path
is **65 TFLOPS**. That is an ~8× ceiling you are leaving on the table before
`torch.compile` contributes anything.

**Footgun:** `SJFPO_DTYPE=bf16` on a T4 does *not* error and does *not* give
bf16. It silently falls through to fp32. Do not use it as a probe.

Going to fp16 is a **code change**, not a config flip:
`resolve_compute_dtype` accepts only `{"auto", "fp32", "bf16"}`, and
`autocast_context` hardcodes `torch.bfloat16`. Adding `fp16` means new config
surface in `utils.py`, and `AGENTS.md` requires naming sign-off first. Until that
is done, treat every number below as *the fp32 number on the best tensor core T4
has*.

## 2. Kaggle budget — size the run to the box

| Limit | Value | Consequence here |
|---|---|---|
| session | **12 h** | compile time must amortise inside 12 h, including the sweep |
| weekly quota | 30 h, **2× T4 bills 2×** | a failed 6 h run costs 12 h of quota |
| idle timeout | **20 min** without browser input | an unattended run is killed |
| disk | 20 GB auto-saved at `/kaggle/working` | checkpoint + inductor cache |
| CPU | **4 cores** | `num_envs` and compile time both hurt |

The 2×T4 are **two separate 16 GB devices**, not one 32 GB pool. Without DDP and
NCCL you get one 16 GB card and a second idling device. Run on
`CUDA_VISIBLE_DEVICES=0` and do not plan around 32 GB.

Because idle timeout kills unattended runs, either keep the browser tab alive,
or run in **batch mode** (`Save & Run All`) which is not subject to it.

---

## 3. Step 1 — repo-side config, before any compile flag

```bash
# 4 cores: 64 vec envs would be 64 processes on 4 cores. This is the single
# biggest config change the T4 forces.
.venv/bin/python scripts/train.py \
  --env Hopper-v5 --seed 3407 --total_steps 65536 \
  --eval_steps 16384 --save_steps 16384 \
  --sjfpo.num_envs 8 --sjfpo.chunks_per_update 256 --sjfpo.batch_size 128 \
  --out_dir ./tmp/t4 --sjfpo.report_to none
```

Verify before continuing:

```python
from spectral_joint_policy.utils import (is_rtx_1000_pro_blackwell,
    resolve_storage_dtype, resolve_compute_dtype)
print("blackwell fast path:", is_rtx_1000_pro_blackwell())   # expect False
print("storage dtype:", resolve_storage_dtype("cuda"))       # expect float32
print("compute dtype:", resolve_compute_dtype())              # expect float32
```

`False / float32 / float32` on a T4 is correct, not a misconfiguration.

Keep inductor and Triton caches in `./tmp` — never the default `/tmp`, and on
Kaggle `/tmp` is not auto-saved anyway:

```bash
mkdir -p ./tmp/{inductor,triton}
export TORCHINDUCTOR_CACHE_DIR=$PWD/tmp/inductor
export TRITON_CACHE_DIR=$PWD/tmp/triton
```

## 4. Step 2 — apply the Blackwell Stage 1 fix first

Unchanged by architecture, and it is the highest-value change in either
document. From `TORCH-COMPILE.md` §1.1:

`src/spectral_joint_policy/dct_utils.py:145` and `:159` contain

```python
if not bool(torch.all((tau_q >= 0) & (tau_q <= T))):
    raise ValueError(...)
```

`torch.all(...)` → `bool(...)` is a device→host sync inside the CFM-loss path,
and it is unreachable: the only caller, `velocity_to_action_steps`, builds the
grid itself and always lands in `[0.0, 0.9375]`. Measured effect on CPU:

```
as-is                         graphs=13  breaks=12
without the tau range check   graphs= 4  breaks= 3
```

Delete both guards and state the guarantee as a comment on
`velocity_to_action_steps`. **This matters more on T4 than on Blackwell**,
because a `bool()` sync stalls the pipeline and a T4 has far less occupancy to
hide behind than a Blackwell.

## 5. Step 3 — gate the compile wrapper before measuring

Identical to `TORCH-COMPILE.md` Stage 0, and the reason is architectural, not
cosmetic: **never assign the wrapper back**.

```python
def compile_probe():
    """Gate: compiling must not change parameter identity or names."""
    import sys, dataclasses, torch; sys.path.insert(0, "src")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
    from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy
    cfg = dataclasses.replace(SJFPOConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (17,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))})
    pol = SJFPOPolicy(cfg)
    keys_before = set(pol.state_dict())
    store = {}   # plain dict: not registered by nn.Module
    store["actor"] = torch.compile(pol.actor, dynamic=True)
    assert set(pol.state_dict()) == keys_before, "compile leaked into state_dict"
    names = {n for n, _ in pol.actor.named_parameters()}
    assert "out.weight" in names, sorted(names)[:5]   # no _orig_mod. prefix
    pol.snapshot_behavior_policy()                    # must not raise
    print("gate 0 ok")
```

Both failure modes are silent on the Blackwell path and are equally silent here:

- `get_actor_optim_params()` splits no-decay on `name.startswith("out.")`. Under
  a wrapper the name is `_orig_mod.out.weight`, so **`out.*` silently joins the
  weight-decay group**. Different optimizer, no error.
- `snapshot_behavior_policy()` does `old_actor.load_state_dict(...)`. Mismatched
  prefixes raise — that one is loud, which is why it is the safe canary.

## 6. Step 4 — mode, in the order to try on T4

The ranking differs from Blackwell, and the reason is the box, not the GPU.

| Order | `mode` | `dynamic` | Why on T4 |
|---|---|---|---|
| 1 | *(eager)* | — | reference |
| 2 | `default` | `true` | lowest compile cost; always fits the 12 h budget |
| 3 | `default` | `false` | static shapes → autotuned tile sizes |
| 4 | `reduce-overhead` | `false` | **likely the best row.** cudagraphs remove launch overhead, and a T4 is launch-bound, not FLOP-bound |
| 5 | `max-autotune-no-cudagraphs` | best of 3/4 | GEMM autotuning; no cudagraphs |
| 6 | `max-autotune` | best of 5 | **probably skip on Kaggle.** Autotune benchmarks every candidate at compile time on 4 CPU cores |

Rows 4 and 5 are the ones worth the time on T4 specifically:

- **`reduce-overhead` first.** `sample_action` runs a 4-step Euler loop
  (`num_euler_steps=4`), each step launching the full 6-block DiT. That is
  launch-bound, which is exactly what cudagraphs fix, and it is why the
  Blackwell doc rates the rollout-propose row highly.
- **`max-autotune` last or never.** Its compile-time search is CPU-bound and
  Kaggle gives 4 cores for ≤12 h. Row 5 gets the GEMM tuning without paying
  twice.

**Stop rule.** After row 3, if `ms med` improves <15% over eager, compile is not
the lever on this box — **fp16 autocast is** (8× of available tensor throughput
vs the ~15% compile might buy). Prefer spending the session on §1.

**Noise floor: require ≥15%, medians of 3, `rm -rf ./tmp/{inductor,triton}`
between configurations.** Never compare a cold cache against a warm one — on
Kaggle that mistake is easy because each session starts cold by default.

## 7. Step 5 — sweep harness

`./tmp/compile_sweep.py` from `TORCH-COMPILE.md` §Stage 2 is architecture-agnostic
and runs unmodified on a T4. Add two T4-only flags to your invocation:

```bash
# update path, both dynamics, with the eager reference
CUDA_VISIBLE_DEVICES=0 .venv/bin/python ./tmp/compile_sweep.py \
  --target forward --modes eager default reduce-overhead \
  --dynamic true false --iters 20

# the rollout propose shape: batch = num_envs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python ./tmp/compile_sweep.py \
  --target forward --chunks 8 --modes eager reduce-overhead --dynamic false --iters 50
```

`peak GiB` matters more on T4 than on Blackwell — it is the column that tells you
whether row 4 fits. Watch it at `--chunks 128` first; if `reduce-overhead`
OOMs, drop to `--chunks 64` before blaming the mode.

## 8. Step 6 — cudagraph safety, if you reach row 4

`reduce-overhead` is only correct for graphs that "do not mutate inputs". Three
sites mutate parameters **in place**, all between forward calls:

| Site | Mutation |
|---|---|
`swap_in_ema` / `swap_out_ema` | `torch._foreach_copy_` into live actor storage, once per eval |
`snapshot_behavior_policy` | `load_state_dict` into `old_actor` / `old_encoder` |
`update_target_critic` | `_foreach_mul_` + `_foreach_add_` on `target_critic` |

Addresses survive, so replay should re-read the values. Do not assume it — run
the gate from `TORCH-COMPILE.md` §Stage 4 on the eager policy first, then on the
compiled one. It perturbs `actor.out.bias` specifically, because `out` is
zero-init and the adaLN-Zero stack would absorb any upstream perturbation and
make the gate pass vacuously.

## 9. Step 7 — persist the compile cache across sessions

The inductor cache is the one artifact that converts session 2+ into session 1
speed, and it is lost on every Kaggle restart.

1. `export TORCHINDUCTOR_CACHE_DIR=/kaggle/working/inductor`
   **and** `export TRITON_CACHE_DIR=/kaggle/working/triton` — not `./tmp`, since
   only `/kaggle/working` is auto-saved.
2. At the end of a session, save that directory as a Kaggle dataset.
3. At the start of the next, add it to `dataset_sources` and unpack it to the
   same path before importing torch.

**Never compare a run with a warm cache against one without** and report the
speedup — that is the whole measurement. Report compile seconds separately from
steady state.

## 10. Step 8 — end-to-end

`train.py`'s `fps` includes eval and checkpoint saving, so run the update-only
arm to get the update-path number:

```bash
# update-dominated: no eval, no periodic save
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python scripts/train.py \
  --env Walker2d-v5 --seed 3407 --total_steps 65536 \
  --eval_steps 99999999 --save_steps 99999999 \
  --sjfpo.num_envs 8 --out_dir ./tmp/t4_update --sjfpo.report_to none

git checkout -- wandb/
```

Run it against an eager twin on the same seed and diff the health metrics:
`gae_linked_frac`, `explained_var`, `advantage_std_fraction`,
`actor_grad_norm_preclip` vs `critic_grad_norm_preclip`, `cfm_above_clamp_frac`.
A compiled run whose `explained_var` diverges from its eager twin is a numerics
failure, not a performance result.

Use **Walker2d-v5**, not Reacher: Reacher episodes are 50 steps, so eval and the
fixed compile cost dominate and you measure the compile overhead instead of the
update.

## 11. Naming sign-off required before implementation

`AGENTS.md` requires a naming decision before adding config surface. Three
proposals, none implemented:

| Proposal | Where | Note |
|---|---|---|
| `compile: bool = False` on `SJFPOConfig` → `--sjfpo.compile=true` | `configuration_sjfpo.py` | must be mirrored into `src/sjfpo_jax/configuration_sjfpo.py` or `test_config_parity.py` gets worse; it already fails on `adam_betas` |
| `compile_mode: str = "default"` | same | one of `default` / `reduce-overhead` / `max-autotune-no-cudagraphs` / `max-autotune`; naming T4 preferences into a Blackwell-tuned default is a mistake |
| `fp16` added to `SJFPO_DTYPE` and to `resolve_compute_dtype` | `utils.py` | the 8× lever from §1; also needs an `autocast_context` change, since it hardcodes bf16 |

`torch._dynamo.config.cache_size_limit = 32` (default 8) goes next to
`configure_torch_cuda()` in `scripts/train.py` and needs no new name.

---

## Traps specific to T4 / Kaggle

| Trap | Consequence |
|---|---|
`enable_gpu: true` without `machine_shape` | silent **P100 sm_60**, dead kernels on modern torch |
`--accelerator` push flag | reported not to work; use `machine_shape` |
`SJFPO_DTYPE=bf16` | silently fp32 on sm_75; not a probe |
Assuming 32 GB | 2× T4 are two 16 GB devices, not a pool |
`num_envs=64` on 4 cores | 64 forkserver processes on 4 cores |
20-minute idle timeout | unattended runs are killed; use Save & Run All |
Warm vs cold inductor cache | invents a speedup that does not exist |
`max-autotune` on 4 cores | compile-time search may not amortise in 12 h |
`self.actor = torch.compile(self.actor)` | `_orig_mod.` prefix → wrong no-decay split, broken EMA keys |
Default `cache_size_limit=8` | silent eager fallback after 8 shape variants |
`bool(torch.all(...))` in the CFM path | device sync; 13 graphs instead of 4 |

## Do not bother on T4

| Candidate | Reason |
|---|---|
`mode="max-autotune"` | compile-time search on 4 cores, ≤12 h; `...-no-cudagraphs` gets the tuning |
DDP across the 2 T4 | the update is 8 ms-scale; comms dominate, and it is not where the time goes |
Hand-written cudagraph capture | run `reduce-overhead` first; if it cannot capture, yours will not either |
Raising `batch_size` for throughput | 16 GB is the binding constraint and 4 cores limit the step rate anyway |