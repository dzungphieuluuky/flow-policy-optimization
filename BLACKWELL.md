# BLACKWELL.md — everything that must be measured on a Blackwell GPU

**Nothing in this file can be run on the CPU box.** Every step needs a real
NVIDIA GPU, and the ones marked **RTX PRO 1000** additionally need that
specific SKU, because two code paths branch on the device *name string*
(`utils.py:is_rtx_1000_pro_blackwell`).

The numbers quoted as "measured" were all taken on **CPU only** (torch
2.11.0+cu130, no CUDA). They are the reference to check against, not the
result. The CPU profile has since been reworked twice — the collector is now
3.0–3.9x faster and the diagnostics are gated — so the balance has moved
again and **step 2 below is still the measurement that decides everything
else.**

Every script and output path is `./tmp/`, per the Temp Files rule in
`AGENTS.md`. `mkdir -p ./tmp` first.

---

## Preconditions

| # | Check | Why it matters |
|---|---|---|
| 0 | `nvidia-smi` shows a Blackwell part, `sm_100` or `sm_120` | Everything below assumes it. Two Blackwell SKUs behave very differently. |
| 1 | **Do not run `uv sync` or `uv run`.** | `uv run` re-syncs `.venv` from the lockfile. That lock used to pin the 49 KB `lerobot` 0.1.0 stub and `torch` 2.2.2, which has no `nn.RMSNorm`. It is fixed as of `b44e90d` (`lerobot>=0.6.1`, torch 2.11.0), but `uv lock` still resolves `torch` from PyPI's default build. Prefer `.venv/bin/python` directly; use `uvx ruff` for linting, which runs in an isolated env. |
| 2 | After any `uv sync`, **check for partial installs before trusting `import torch`** | A sync on the external NVMe mount silently produced three incomplete packages — `nvidia-nccl-cu13` missing its 218 MB `libnccl.so.2`, plus `nvidia-cudnn-cu13` and `opencv-python-headless`. Symptom: `ImportError: libnccl.so.2: cannot open shared object file`. Detect and repair with the integrity check in step A5. |
| 3 | `uv lock` warns `jax==0.5.3 does not have an extra named cuda13` | Harmless warning, but `pyproject.toml` still asks for `jax[cuda13]`. jax is only reachable via `mujoco-warp`, unused by the training path. |
| 4 | Smoke runs use `WANDB_MODE=offline` | Per `AGENTS.md`. The tensorboard and wandb backends write to `./tensorboard` and `./wandb` in the repo root **regardless of `--out_dir`**, and `wandb/debug.log`, `debug-internal.log`, `latest-run` are tracked in git — revert their churn before committing. |
| 5 | Confirm `resolve_storage_dtype('cuda')` and `resolve_compute_dtype()` | These fork on the device name. `resolve_compute_dtype` gates every `autocast_context`; if it is not bf16, all autocast sites are `nullcontext` and the forward runs fp32. See step A2. |

```bash
mkdir -p ./tmp
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
.venv/bin/python -c "
import torch, sys; sys.path.insert(0,'src')
print('torch      ', torch.__version__)
print('cuda       ', torch.version.cuda, '| available', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'device {i}   {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB  SMs={p.multi_processor_count}')
from spectral_joint_policy.utils import (is_rtx_1000_pro_blackwell,
    resolve_storage_dtype, resolve_compute_dtype, resolve_device)
print('is_rtx_1000_pro_blackwell', is_rtx_1000_pro_blackwell())
print('storage dtype (cuda)      ', resolve_storage_dtype('cuda'))
print('compute dtype (cuda)      ', resolve_compute_dtype())
"
```

---

# Part A — do these first, ranked by measured cost

Each item was found and quantified on CPU. None can be validated without the
GPU. A1 is the largest single number measured in this session.

## A1. CUDA-graph the 16-step Euler integration — **largest remaining item**

Measured on CPU at rollout scale (64 envs, obs 348, horizon 16):

```
one predict_action_chunk: 2006 dispatches (16 sequential Euler steps, 125/step)
a rollout makes ~4-9 such calls -> 8,024-18,054 dispatches, all serial
```

125 dispatches per Euler step, 16 deep, every one a round trip the GPU cannot
overlap with anything. That is comparable to the *entire* 8-microbatch update
(1,114 × 8 = 8,912 dispatches), and the steps are fixed-shape with no
data-dependent control flow — the exact shape CUDA graphs exist for.

`fpo_core.py:sample_action` is the loop. Measure it first:

```bash
cat > ./tmp/euler_probe.py <<'EOF'
import sys, time; sys.path.insert(0, "src")
import numpy as np, torch
from torch.utils._python_dispatch import TorchDispatchMode
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
from lerobot.utils.constants import OBS_STATE
class C(TorchDispatchMode):
    def __init__(self): self.n = 0
    def __torch_dispatch__(self, f, t, a=(), k=None):
        self.n += 1; return f(*a, **(k or {}))
cfg = SJFPOConfig()
pol = build_policy(11, 2, cfg, device="cuda",
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
x = torch.randn(cfg.num_envs, cfg.n_obs_steps, 11, device="cuda")
pol.eval()
def call():
    with torch.inference_mode():
        pol.predict_action_chunk({OBS_STATE: x}, zero_sampling=False)
call(); torch.cuda.synchronize()
c = C()
with c: call()
ts = []
for _ in range(20):
    torch.cuda.synchronize(); t0 = time.perf_counter(); call()
    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
print(f"predict_action_chunk: {c.n} dispatches, {min(ts)*1e3:.2f} ms/call, "
      f"{cfg.num_euler_steps} euler steps")
# graph-capture attempt
try:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): call()
    torch.cuda.synchronize()
    ts2 = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay()
        torch.cuda.synchronize(); ts2.append(time.perf_counter() - t0)
    print(f"  CUDAGraph replay: {min(ts2)*1e3:.2f} ms/call  "
          f"speedup {min(ts)/min(ts2):.2f}x")
except Exception as e:
    print(f"  CUDAGraph FAILED: {type(e).__name__}: {str(e)[:160]}")
EOF
.venv/bin/python ./tmp/euler_probe.py 2>&1 | tee ./tmp/euler_probe.txt
```

Also try `torch.compile(predict_action_chunk, mode="reduce-overhead")`, which
applies CUDA graphs automatically and will tell you if capture is viable
before you hand-write one.

**Read it as.** If the graph replay is 2x or better, this is worth more than
every other item in this file combined. If capture fails, the usual suspects
are the `.cpu()` in `propose_batch` (`modeling_sjfpo.py:762`) and the
`actions_t.cpu().numpy()` in the collector — both are host syncs inside the
region and both must be lifted out before capture can succeed.

## A2. bf16 storage dtype — 1.66x activation memory

Measured with `saved_tensors_hooks` on CPU, one microbatch at the defaults
(`batch_size=128`, `num_mc_samples=16`, hidden 128, depth 2):

| | saved for backward | tensors |
|---|---|---|
| fp32 | 1.560 GiB | 163 |
| bf16 autocast | **0.938 GiB** | 163 |

`resolve_compute_dtype` already returns bf16 for *compute* on any CUDA GPU, so
the matmuls are already bf16. But two things keep the storage path fp32:

1. `resolve_storage_dtype` returns bf16 **only** when the device name contains
   both `"rtx pro 1000"` and `"blackwell"`. On a B200 or GB200 it returns
   fp32 and the entire bf16-storage path is dead code.
2. `collector.py` hardcodes `_OBS_DTYPE = torch.float32` for every buffer
   tensor regardless, so even on the RTX PRO 1000 SKU the rollout buffer is
   fp32 and `forward` downcasts on read.

Measure the current peak, then the same run with `_OBS_DTYPE` and
`resolve_storage_dtype` forced to bf16:

```bash
.venv/bin/python -c "
import torch, sys; sys.path.insert(0,'src')
print('alloc backend:', torch.cuda.memory.get_allocator_backend())
print('peak reserved GiB', torch.cuda.max_memory_reserved()/2**30)
"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python ./tmp/split.py 64 4
```

**Read it as.** 0.938 vs 1.560 GiB is the activation saving alone; the rollout
buffer and its H2D copy halve on top. This is a **research decision, not a
free win** — obs and rewards are fp32 end to end on purpose, and `collector.py`
notes that bf16 rewards corrupt the GAE target. If you change the storage
dtype, keep the *reward and return* tensors fp32 and re-check
`gae/linked_frac`, `value_target_std` and `explained_var` from step 8 for
damage. A CPU fingerprint (`./tmp/fingerprint.py`, 54 values) exists to catch
numerics drift before you spend GPU hours on it.

## A3. The two `aten._local_scalar_dense` graph breaks

Measured on CPU: compiling the full `forward` is a **no-op**.

```
eager    full forward+backward:  62.6 ms, 1114 dispatches
compiled (mode=default)         72.3 ms, 1114 dispatches   <- 13% slower
dynamo.explain: graphs=3 breaks=2 ops=274
  2x "Data dependent operator: aten._local_scalar_dense has a non-Tensor
      output whose value is dependent on the data"
```

Compile captures 274 of 1,114 ops into 3 graphs and falls back for the rest, so
the dispatch count is unchanged. On CPU that is a net loss. **On GPU it may
not be** — capturing those 274 would fuse the RMSNorm / adaLN / `modulate`
chains, which is most of what commits `891e7f6` and `195fa4f` just
hand-optimised. Find the two sites:

```bash
TORCH_LOGS=+dynamo .venv/bin/python ./tmp/compile_probe.py 2>&1 | grep -A6 "Graph break"
```

The only data-dependent scalar extractions on the DCT path are the two
`bool(torch.all(...))` guards in `dct_utils.decode_coeffs_to_velocity`; check
those first. The actor's per-token `time_mlp` and `adaLN` are the fusion
payoff, so if capture succeeds, re-measure A1's Euler loop too — a captured
rollout propose path would fold the 16 Euler steps into the same graph.

**Do not** try to compile the CFM-loss region in isolation. Measured and
rejected: Dynamo fuses it only when the tensors are captured as *module-level
globals* (42 → 0 dispatches, 3.70x), and refuses when they arrive as function
arguments or closure cells (41 → 41, no Inductor kernels). As shipped at the
real call site it measured **79.6 ms vs 74.5 ms eager — a 7% regression**, and
re-creating the `torch.compile` wrapper per call costs ~2 ms of pure overhead.
The region is also only 42 of 1,114 dispatches, so the ceiling is a few
percent of the update.

## A4. Confirm the collector rewrite on GPU

The collector is host-side, so its CPU numbers should transfer, but the *shape*
of the win may not: the rewrite cut **dispatches**, and on GPU a dispatch is a
~3–5 µs launch rather than a ~0.4 µs Python call.

```bash
.venv/bin/python ./tmp/bench_collector.py 2>&1 | tee ./tmp/bench_blackwell.txt
```

| regime | CPU before | CPU after | CPU speedup |
|---|---|---|---|
| no falls (16-step chunks) | 378 ms | 114 ms | 3.32x |
| fall every 200 steps | 381 ms | 111 ms | 3.45x |
| fall every 50 steps | 394 ms | 132 ms | 2.99x |
| fall every 16 steps | 394 ms | 136 ms | 2.89x |
| Reacher dims (obs 11) | 348 ms | 90 ms | 3.87x |
| fall every step (worst) | 1216 ms | 1198 ms | 1.02x |

The repo's own operating point is in the fast band: `REPORT.md` records
`executed_action_step_frac = 0.5625`, i.e. ~9-step chunks, which lands between
fall/16 and fall/50. The pathological `fall every step` case is break-even by
design — the per-step savings are cancelled by the per-chunk window roll.

## A5. Confirm the Phase-A buffer changes on GPU

Two of the four allocation changes are invisible on CPU and only pay off with a
CUDA allocator:

- **Pinning.** Only 3 of 14 buffer tensors are now pinned
  (`_PIN_MIN_BYTES = 1 MiB`); the other 11 were each a `cudaHostAlloc` per
  rollout for sub-megabyte index and flag vectors. Count the calls before and
  after.
- **Capacity.** `chunk_capacity` now sizes from the previous rollout instead of
  `max_steps // action_horizon + 8`. A policy falling every 9 steps made 17x
  the predicted chunk count and forced 4 reallocations of 10 pinned tensors on
  the first rollout. Verify a falling policy no longer re-grows.

Integrity check to run after any `uv sync` (see precondition 2):

```bash
.venv/bin/python - <<'EOF'
import csv, pathlib
sp = pathlib.Path(".venv/lib/python3.12/site-packages")
bad = []
for rec in sorted(sp.glob("*.dist-info/RECORD")):
    missing = total = 0
    with rec.open() as f:
        for row in csv.reader(f):
            if not row: continue
            total += 1
            if not (sp / row[0]).exists(): missing += 1
    if missing: bad.append((rec.parent.name, missing, total))
print("no partial installs" if not bad else "")
for d, m, t in bad: print(f"  INCOMPLETE {d} {m}/{t} files missing")
EOF
```

---

# Part B — the original measurement plan

Unchanged in substance. **Step 2 is the one that decides where to spend
effort**; everything in Part A is a candidate, not a decision.

## Known blockers

| Blocker | Detail | Fix |
|---|---|---|
| `scripts/cuda_probe.py` will not run | Passes `--wandb_enabled`, which does not exist in `train.py`, and does `import wandb` | Ignore it; the steps below replace it. |
| `torch.compile` behaviour on this build | Broken for CPU (AOTAutograd crashes); a no-op for the full forward (A3) | Measure on Blackwell, do not assume. |

## Step 1 — confirm the CUDA path is actually taken

Several paths are gated on `.is_cuda` and are inert on CPU. Confirm each fires.

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import torch, numpy as np
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
cfg = SJFPOConfig()
pol = build_policy(10, 2, cfg, device='cuda',
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
from spectral_joint_policy.modeling_sjfpo import build_optimizer
opt = build_optimizer(pol)
print('optimizer impl     ', type(opt).__name__,
      'fused=', [g.get('fused') for g in opt.param_groups])
with torch.autocast('cuda', dtype=torch.bfloat16):
    print('  autocast active  ', torch.is_autocast_enabled('cuda'))
"
```

`fused=False` means the gate is not firing and the optimizer is on the slow path.

## Step 2 — wall-clock split

The single most important measurement. Scripts: `./tmp/split.py`,
`./tmp/prof_bwd.py`, `./tmp/prof_xfer.py`, `./tmp/wavefront.py`,
`./tmp/compile_probe.py` (all in `./tmp/`, regenerate from git history if the
directory was cleaned).

cat > ./tmp/split.py <<'EOF'
import sys, time
sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
import spectral_joint_policy.modeling_sjfpo as M
import spectral_joint_policy.train_sjfpo as T
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
from lerobot.utils.constants import OBS_STATE

NE   = int(sys.argv[1]) if len(sys.argv) > 1 else 64
UPD  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CFG  = SJFPOConfig()
cfg  = CFG
envs = gym.vector.SyncVectorEnv([lambda: gym.make("Reacher-v5") for _ in range(NE)])
D    = int(envs.single_observation_space.shape[0])
pol  = build_policy(D, 2, cfg, device="cuda",
                    action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
opt  = M.build_optimizer(pol)

acc = {}
def timed(name, fn):
    def wrap(*a, **k):
        t = time.perf_counter()
        try:    return fn(*a, **k)
        finally: acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
    return wrap

M.SJFPOPolicy.generate_rollout = timed("generate_rollout", M.SJFPOPolicy.generate_rollout)
M.SJFPOPolicy.forward          = timed("update_forward",  M.SJFPOPolicy.forward)
# one forward serves both drift metrics since 891e7f6
T.measure_value_drift          = timed("drift",           T.measure_value_drift)

TSTEP = cfg.chunks_per_update * cfg.action_horizon
def propose(s):
    with torch.inference_mode():
        return pol.predict_action_chunk({OBS_STATE: s}, zero_sampling=False)

t_all = time.perf_counter()
for _ in range(UPD):
    t = time.perf_counter()                      # generate_rollout owns collection
    batch = pol.generate_rollout(envs, TSTEP,
                                 action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
    acc["collect_plus_rollout"] = acc.get("collect_plus_rollout", 0.0) + time.perf_counter() - t
    n = batch["observation.state"].shape[0]
    for _ in range(cfg.num_learning_epochs):
        perm = torch.randperm(n)
        for m in range(n // cfg.batch_size):
            idx = perm[m*cfg.batch_size:(m+1)*cfg.batch_size]
            sl = {k: (v if k in ("normalizer_observations","gae_diagnostics") else v[idx])
                  for k, v in batch.items()}
            sl = {k: (v.to("cuda", non_blocking=True)
                      if isinstance(v, torch.Tensor) and v.device.type == "cpu" else v)
                  for k, v in sl.items()}
            loss, _ = pol.forward(sl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            opt.step()
            pol.step_count += 1
            pol.update_target_critic()
            T.measure_boot_value_drift(pol, sl, torch.device("cuda"))
            T.measure_value_target_drift(pol, sl, torch.device("cuda"))
envs.close()
total = time.perf_counter() - t_all
meas = acc.get("generate_rollout",0)+acc.get("update_forward",0)
print(f"num_envs={NE}  {UPD} updates x {TSTEP} steps  wall {total:.2f}s "
      f"({total/UPD:.2f}s/update, {UPD*TSTEP/total:.0f} fps)")
for k in ("collect_plus_rollout","generate_rollout","update_forward","drift"):
    if k in acc: print(f"  {k:24s} {acc[k]:7.2f}s {100*acc[k]/total:5.1f}%")
print(f"  {'backward + optimizer':24s} {total-meas:7.2f}s {100*(total-meas)/total:5.1f}%")
print(f"  peak VRAM reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB  "
      f"allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
EOF
.venv/bin/python ./tmp/split.py 64 4
```

CPU reference at `num_envs=64`, 4096-step rollouts, 4 updates, 13.71 s/update,
299 fps:

| segment | % |
|---|---|
| `collect_chunks_vec` | 6.4 |
| `generate_rollout` minus collect | 9.5 |
| `update_forward` | 22.7 |
| **`backward` + optimizer** | **61.4** |
| drift diagnostics | 1.1 |

**Stale.** The drift diagnostics are now gated on `tracker.enabled` and share
one critic forward (`891e7f6`), and the collector is 3.0–3.9x faster
(`100f66a`), so the collector share is well under 6.4% and the diagnostics
share is zero with `report_to=none`. Re-measure the whole table.

**Read it as:** if Blackwell inverts this — which it plausibly will, since the
rollout is launch-bound and the backward is compute-bound — the optimisation
target moves with it.

## Step 3 — kernel-level breakdown of the backward

Use `torch.profiler` with CUDA activities. For each row note
`self_cuda_time_total`, the op count, and whether it scales with
`num_mc_samples`.

```bash
cat > ./tmp/prof_bwd.py <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity, record_function
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
import spectral_joint_policy.train_sjfpo as T

cfg = SJFPOConfig()
pol = build_policy(10, 2, cfg, device="cuda",
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
opt = build_optimizer(pol)
n, H, D, A = cfg.chunks_per_update, cfg.n_obs_steps, 10, 2
tok = pol.get_token_count_for_cfm_loss(cfg)
g  = torch.Generator().manual_seed(0)
full = {
  "observation.state": torch.randn(n,H,D,generator=g).cuda(),
  "action": torch.randn(n,H,A,generator=g).cuda(),
  "action_is_pad": torch.zeros(n,H,dtype=torch.bool).cuda(),
  "raw_advantages": torch.randn(n,generator=g).cuda(),
  "normalised_advantages": torch.randn(n,generator=g).cuda(),
  "raw_returns": (torch.randn(n,generator=g)*10).cuda(),
  "raw_values": torch.randn(n,generator=g).cuda(),
  "normalizer_observations": torch.randn(n*H,D,generator=g).cuda(),
  "boot_observations": torch.randn(n,H,D,generator=g).cuda(),
  "boot_values_start": torch.randn(n,1,generator=g).cuda(),
  "mc_tau": torch.rand(n,cfg.num_mc_samples,cfg.num_flow_tokens,1,generator=g).cuda(),
  "mc_eps": torch.randn(n,cfg.num_mc_samples,cfg.num_flow_tokens,A,generator=g).cuda(),
  "chunk_clip_counts": torch.zeros(n,dtype=torch.int64).cuda(),
  "gae_diagnostics": {},
  "behavior_cfm_loss": torch.rand(n,cfg.num_mc_samples,generator=g).cuda(),
  "behavior_cfm_loss_token": torch.rand(n,cfg.num_mc_samples,tok,generator=g).cuda(),
}
idx = torch.randperm(n)[:cfg.batch_size].cuda()
sl  = {k:(v if k in ("normalizer_observations","gae_diagnostics") else v[idx])
       for k,v in full.items()}
for _ in range(3):                                   # warm up
    loss,_ = pol.forward(sl); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=False) as prof:
    for _ in range(2):
        with record_function("update"):
            loss,_ = pol.forward(sl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25))
EOF
.venv/bin/python ./tmp/prof_bwd.py 2>&1 | tee ./tmp/prof_bwd.txt
```

| Question | Where to look |
|---|---|
| Compute-bound or launch-bound? | Total kernel time ≪ region wall time means launch-bound, and the fix is CUDA graphs (A1) or `reduce-overhead`, not more FLOPs. |
| Does kernel time scale with `num_mc_samples`? | Re-run with 8 and 16. Linear means the `num_mc × num_euler` unroll is the cost. |
| Is the critic a large share? | 467,736 params at `num_critics=2` (obs 11), 640,954 at obs 348. It does **not** scale with `num_mc_samples`. |
| Any host-device sync in the loop? | `cudaStreamSynchronize`, `Memcpy DtoH`, `.item()`, `.cpu()`. The `.cpu()` in `propose_batch` and the drift diagnostics' one `.tolist()` are expected. |

## Step 4 — transfer accounting

Three transfers exist: `buffer.copy_to(..., non_blocking=True)` + a
`torch.cuda.Event` (overlapped), `propose_batch`'s `chunk.cpu()` once per chunk
(blocking, mid-loop), and `to_device_async` per wavefront (pinned staging).

```bash
cat > ./tmp/prof_xfer.py <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
from torch.profiler import profile, ProfilerActivity
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
cfg = SJFPOConfig()
NE = 64
envs = gym.vector.SyncVectorEnv([lambda: gym.make("Reacher-v5") for _ in range(NE)])
D = int(envs.single_observation_space.shape[0])
pol = build_policy(D, 2, cfg, device="cuda",
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
TSTEP = cfg.chunks_per_update * cfg.action_horizon
for _ in range(2):
    pol.generate_rollout(envs, TSTEP, action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    pol.generate_rollout(envs, TSTEP, action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
ka = prof.key_averages()
for e in ka:
    n = e.key.lower()
    if "memcpy" in n or "cuda" in n and "sync" in n:
        print(f"{e.key:52s} n={e.count:6d}  cuda={e.self_device_time_total/1e3:9.2f} ms")
print("\ntop CPU-side time (sync/setup):")
for e in sorted(ka, key=lambda x: -x.self_cpu_time_total)[:15]:
    print(f"  {e.key[:60]:60s} {e.self_cpu_time_total/1e3:9.2f} ms  n={e.count}")
EOF
.venv/bin/python ./tmp/prof_xfer.py 2>&1 | tee ./tmp/prof_xfer.txt
```

`n_chunks` small D2H calls of ~2 KB would mean the per-chunk `.cpu()` is
serialising the rollout. The chunk loop is data-dependent, so batching the D2H
is a real refactor, not a flag. Note that `to_device_async` calls
`x.pin_memory()` per transfer, which is a `cudaHostAlloc` each time; a reused
`torch.empty(..., pin_memory=True)` staging buffer is the standard fix and
**has never been measured** — it needs a driver, so CPU cannot test it.

## Step 5 — wavefront batch histogram

```bash
for E in Reacher-v5 Hopper-v5 InvertedPendulum-v5; do
  .venv/bin/python ./tmp/wavefront.py 64 $E 4096
done
```

```bash
cat > ./tmp/wavefront.py <<'EOF'
import sys, collections; sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
import spectral_joint_policy.collector as C
from spectral_joint_policy.collector import RolloutBuffer
NE, H = int(sys.argv[1]), 16
cfg   = C.get_default_config()
envs  = gym.vector.SyncVectorEnv([lambda: gym.make(sys.argv[2]) for _ in range(NE)])
D     = int(envs.single_observation_space.shape[0])
T     = int(sys.argv[3])
sizes = []
orig  = C.RolloutBuffer.add_chunk
def traced(self, **kw):
    return orig(self, **kw)
C.RolloutBuffer.add_chunk = traced
def policy(s):
    sizes.append(s.shape[0])
    return torch.zeros(s.shape[0], H, 2, device=s.device)
buf = RolloutBuffer(max_chunks=T, max_steps=T, obs_steps=H, obs_dim=D,
                    action_horizon=H, action_dim=2)
C.collect_chunks_vec(envs, policy, T, np.array([-2.,-2.]), np.array([2.,2.]),
                     n_obs_steps=H, n_action_steps=H, chunk_size=H, buffer=buf, seed=0)
envs.close()
hist = collections.Counter(sizes)
print(f"{sys.argv[2]} num_envs={NE} total={T}")
print(f"  policy calls {len(sizes)}  chunks {buf.n_chunks}  "
      f"distinct batch sizes {len(hist)}")
print(f"  mean batch {sum(sizes)/len(sizes):.1f} of {NE}  "
      f"efficiency {sum(sizes)/(len(sizes)*NE)*100:.0f}%")
for s, c in sorted(hist.items()): print(f"    batch {s:4d}: {c:5d} calls")
EOF
for E in Reacher-v5 Hopper-v5 InvertedPendulum-v5; do
  .venv/bin/python ./tmp/wavefront.py 64 $E 4096
done
```

CPU gave a perfectly uniform wavefront at `num_envs=64` (`efficiency` 1.00).
That is **not** guaranteed: envs awaiting autoreset shrink `len(need)`, and the
final wavefront is ragged. Below ~95% efficiency you are paying full-batch DiT
cost for a partial wavefront. More than 2–3 distinct batch sizes means
`torch.compile` will recompile per shape.

## Step 6 — does `torch.compile` work here?

Superseded by **A3**, which has the measured CPU verdict. Re-run this on
Blackwell for the propose path, and read it together with A3: the 16-step Euler
loop is the thing worth capturing.

```bash
cat > ./tmp/compile_probe.py <<'EOF'
import sys, time; sys.path.insert(0, "src")
import numpy as np, torch
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import build_optimizer
from spectral_joint_policy.train_sjfpo import build_policy
from lerobot.utils.constants import OBS_STATE
cfg = SJFPOConfig()
pol = build_policy(10, 2, cfg, device="cuda",
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
x = torch.randn(cfg.num_envs, cfg.n_obs_steps, 10, device="cuda")
fn = lambda t: pol.predict_action_chunk({OBS_STATE: t}, zero_sampling=False)
with torch.no_grad():
    for mode in (None, "default", "reduce-overhead", "max-autotune-no-cudagraphs"):
        kw = {} if mode is None else {"mode": mode}
        try:
            torch._dynamo.reset()
            t0 = time.perf_counter()
            c = torch.compile(fn, dynamic=False, **kw)
            for _ in range(3): c(x)
            torch.cuda.synchronize()
            print(f"  mode={str(mode):32s} OK    {(time.perf_counter()-t0)/3*1000:8.1f} ms/call", flush=True)
        except Exception as e:
            m = [l for l in str(e).splitlines() if "Error" in l or "ASSERT" in l]
            print(f"  mode={str(mode):32s} FAIL  {(m[0] if m else str(e).splitlines()[-1]).strip()[:56]}", flush=True)
        finally:
            try: torch._dynamo.reset()
            except Exception: pass
# graph-break census, independent of the backend
import torch._dynamo as dynamo
e = dynamo.explain(fn)(x)
print(f"\n  rollout propose: graphs={e.graph_count} breaks={e.graph_break_count} ops={e.op_count}")
for r, c in __import__("collections").Counter(r.reason for r in e.break_reasons).most_common(8):
    print(f"    {c:3d}x {r[:88]}")
EOF
.venv/bin/python ./tmp/compile_probe.py 2>&1 | tee ./tmp/compile_probe.txt
```

## Step 7 — dtype and matmul precision

Covered by **A2** for storage. For matmul precision, note that
`scripts/train.py:configure_torch_cuda` sets
`torch.backends.cuda.matmul.fp32_precision = "ieee"` and then
`torch.set_float32_matmul_precision("high")` — the same knob, opposite intent.
The later call wins, so the effective setting is `high` (TF32 allowed). Resolve
the contradiction rather than relying on call order.

CPU reference, 1024×1024 fp32 matmul: `highest` 6.76 ms / 6.485e-05 max err,
`high` 6.06 ms / 6.485e-05, `medium` 4.97 ms / 1.068e-04. `high` was
numerically free on CPU. On Blackwell the speedups will be far larger and the
error pattern may differ. **This is a research decision, not a free win.**

## Step 8 — per-update health metrics

```bash
WANDB_MODE=offline .venv/bin/python scripts/train.py --env Reacher-v5 --seed 3407 \
  --total_steps 24576 --eval_steps 8192 --save_steps 24576 \
  --out_dir ./tmp/tb 2>&1 | tail -3
tensorboard --logdir tensorboard --port 6006
```

| metric | healthy | meaning if wrong |
|---|---|---|
| `gae_linked_frac` | ~0.7 or higher | 0 means a dead trace. CPU measured 0.736 at `num_envs=64`. |
| `explained_var` | rising toward 1 | near 0 means the critic is inert. **Negative in both 1M-step runs** (−0.185, −0.064) and unexplained — the highest-value open question in the repo. |
| `advantage_std_fraction` | falling toward `1 - R²` | pinned near 1.0 means the critic does nothing |
| `actor_grad_norm_preclip` vs `critic_grad_norm_preclip` | critic much larger | logged 1.018 vs 0.086, a 12x gap |
| `cfm_above_clamp_frac` | low | near 1 means the policy gets no gradient |
| `boot_value_delta`, `value_target_drift` | stable | drift means the target critic lags |
| `flow_ratio_token_lowfreq` vs `highfreq` | — | per-token ratios; A3's `modulate` fusion targets this path |

## Step 9 — `num_envs` with the GPU in the loop

The CPU 3-seed A/B settled 64 (32 and 48 were worse on mean *and* variance).
But that was a GAE-quality result about sample decorrelation, and on a GPU the
throughput pressure points the other way: 64 `AsyncVectorEnv` workers competing
with the training process for host cores. If fps rises with `num_envs`, the
host is the bottleneck and you have a real trade against GAE quality.

## Step 10 — cleanups that need no measurement

| Site | Issue | Status |
|---|---|---|
| `networks.py` `RunningMeanStd.forward` | three `.to(x.device)` on buffers whose device never changes, plus a full-size `where` | **DONE** (`195fa4f`) — latch is a Python bool, 1.72x |
| `dct_utils.py` `_get_analysis_basis` | `lru_cache` bypassed by Dynamo, which retraces it | open; only matters under compile (A3) |
| `modeling_sjfpo.py` per-env `float()` reductions in `_gae_diagnostics` | six blocking D2H syncs per rollout | **DONE** (`891e7f6`) — one packed `.tolist()` |
| `train_sjfpo.py` drift diagnostics | a full critic pass per boundary, for numbers only the tracker reads; both metrics ran the identical forward | **DONE** (`891e7f6`) — gated on `tracker.enabled`, one forward, 3.0x less critic work |

## Step 11 — allocator and VRAM

```bash
.venv/bin/python -c "import torch; print('alloc conf:', torch.cuda.memory.get_allocator_backend())"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python ./tmp/split.py 64 4
```

Compare peak VRAM with and without `expandable_segments:True`.
`scripts/train.py` already sets it by default.

---

# Part C — measured dead ends, do not spend time here

Every one of these was implemented or benchmarked and **lost**. Recorded so
nobody re-litigates them.

| Candidate | Measurement | Verdict |
|---|---|---|
| `einsum` → `matmul` in `extract_dct_coeffs` | 51.8 → 54.0 µs, 17 → 10 dispatches | 0.96x, slower |
| `F.silu(g).mul_(u)` in `SwiGLU` | 69.1 → 79.9 ms | 0.87x, slower, and in-place breaks autograd |
| qkv `chunk` → `view+permute+unbind` in `SelfAttention` | 26.3 → 27.4 ms, 19 → 15 dispatches | 0.98x, slower |
| `lru_cache` on `resolve_compute_dtype` | 0.6 µs × ~8 calls/update | negligible |
| `torch.compile` on the CFM-loss region | 79.6 vs 74.5 ms in situ | 7% regression; see A3 |
| one GEMM for `SinusoidalEncoding` | impossible | `cos(t·f − φ)` is not linear in `t` |
| GAE reverse scan → closed-form cumprod/cumsum | 10 dispatches, wrong answer | underflows within a few hundred chunks; the same reason the code rejects the cumsum trick for the reward gather |
| `SinusoidalEncoding` via `torch.polar` | one fused kernel | interleaves the `(cos, sin)` layout, permuting what the learned `time_mlp` weights mean |
| `Critic` member loop → `vmap`/`baddbmm` | — | the old critic did this and was replaced on purpose; members must stay independent |
| `_RMSNorm` fp32 round-trip | +48 MiB of temporaries, bitwise identical to plain `F.rms_norm` in fp32 | the promotion only earns its keep for bf16 autocast activations, which is the intended path. Leave it. |

Two items were also **declined on measurement**, not on principle:

- **A5 / item A5 in the collector audit (full-batch fast path for `need`).**
  ~16,384 Python iterations per rollout, ceiling ~1% of the collector, and it
  adds a second code path through the most safety-critical loop in the file.
- **Dropping the redundant `chunk_next_state` rows.** Only chunks ending in
  `continue` chain, so the redundancy is bounded by the terminal rate, not a
  flat 75%: 4.08 MiB at fall/200, ~2.8 MiB at fall/16, ~0 at fall-every-step.
  Revisit only if `obs_dim` exceeds ~1000.

---

# Bring back

1. The precondition environment block, verbatim.
2. **Step 2's segment table.** This is the answer to "where does the time go".
3. **A1's `predict_action_chunk` numbers** — dispatches, ms/call, and the
   CUDA-graph replay speedup. This is the headline result.
4. Step 3's top-25 kernel table, plus: compute-bound or launch-bound?
5. A2's peak-VRAM and fps with storage dtype forced to bf16, plus the
   fingerprint diff showing what numerics moved.
6. A3's graph-break census with the two sites named.
7. Step 4's transfer counts, especially the small D2H per rollout.
8. Step 5's batch histogram and efficiency for all three envs.
9. Step 8's `gae_linked_frac` and `explained_var` trajectories.

## Do not carry these over from the CPU profile

| CPU finding | Status |
|---|---|
| backward + optimizer is 61.4% | re-measure; the balance inverts easily |
| collector is 6.4% of runtime | re-measure; now 3.0–3.9x faster on CPU, and host-side work competes with the GPU feed |
| drift diagnostics are 1.1% | **zero** with `report_to=none`, since `891e7f6` |
| GAE is 0.0% of runtime | still 97% of the GAE call's *dispatches* (1,586 of 1,640) but only ~3–5 ms of serial launches; not worth prototype CUDA-only code unless step 2 says the update is short |
| `torch.compile` gives 1.05x | superseded by A3's measurement |
| `AdamW(fused=True)` is slower | **irrelevant on GPU** — `modeling_sjfpo.py:138` enables it there |
| `F.rms_norm` and SDPA are already fused | unchanged, nothing to do |
| `einx` symbolic shapes cost 0.002% | unchanged, nothing to do |
| num_envs 64 beats 32 and 48 | a GAE-quality result, not a throughput one; re-run step 9 |
