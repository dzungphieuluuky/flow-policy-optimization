# PROFILER.md — measuring SJFPO on a Blackwell GPU

Everything here was measured on **CPU only** (torch 2.11.0+cu130, 8 threads, no
CUDA). The split below is the CPU reference. On Blackwell the balance moves and
several code paths that are dormant here become live, so re-measure before
optimising. Each step says what to run, what to expect, and what a bad number
means.

Every script and output path below is `./tmp/`, per the Temp Files rule in
`AGENTS.md`. That directory is gitignored; `mkdir -p ./tmp` before step 2.

## Known blockers before you start

| Blocker | Detail | Fix |
|---|---|---|
| `scripts/cuda_probe.py` will not run | It passes `--wandb_enabled`, which does not exist in `train.py`, and it does `import wandb`, which raises `AttributeError: np.float_ was removed in NumPy 2.0` | Ignore it, or fix `numpy`. The steps below replace it. |
| **Every `train.py` run needs `--sjfpo.report_to`** | `report_to` defaults to `"wandb"`, and `import wandb` is broken on this numpy. `tests/test_tracking.py::test_all_fans_out_to_every_backend` fails for the same reason. | Always pass `--sjfpo.report_to tensorboard` (keeps metrics) or `none` (fastest). |
| venv is out of sync with `requirements.txt` | venv has numpy 2.2.6; `scripts/AGENTS.md` records the repo's own pin as `numpy<2.1` | `uv sync` before measuring, otherwise you are profiling a broken build. |

`torch.compile` is **broken on this build for CPU** — see step 6. Do not assume
it works on Blackwell; measure it.

---

## Step 0 — record the environment

Do not skip. Two Blackwell SKUs behave very differently (RTX PRO 6000/1000
workstation vs B200/GB200 datacenter), and the dtype path forks on the exact
name (`is_rtx_1000_pro_blackwell` in `utils.py:22`).

```bash
.venv/bin/python -c "
import torch, numpy, platform
print('torch      ', torch.__version__)
print('numpy      ', numpy.__version__)
print('cuda       ', torch.version.cuda)
print('available  ', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'device {i}   {p.name}  sm_{p.major}{p.minor}  '
          f'{p.total_memory/2**30:.1f} GiB  SMs={p.multi_processor_count}')
import sys; sys.path.insert(0,'src')
from sjfpo.utils import (is_rtx_1000_pro_blackwell, resolve_storage_dtype,
                         resolve_compute_dtype, resolve_device)
print('is_rtx_1000_pro_blackwell', is_rtx_1000_pro_blackwell())
print('resolve_device            ', resolve_device())
print('storage dtype (cuda)      ', resolve_storage_dtype('cuda'))
print('compute dtype (cuda)      ', resolve_compute_dtype())
"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit \
  --format=csv
```

Record the output. Two things decide everything downstream:

- `resolve_storage_dtype('cuda')` → if **not** `bfloat16`, the fp32-stored-batch
  bug class in `BUGS.md` E1/B1 does not apply to this run and the buffer
  transfer story changes.
- `resolve_compute_dtype()` → gates `autocast_context` (`utils.py:90`). If it is
  not bf16, every `autocast_context` call site is a `nullcontext` and the
  forward runs fp32.

## Step 1 — confirm the CUDA path is actually taken

Several paths are gated on `.is_cuda` / `device.type == "cuda"` and are inert on
CPU. Confirm each one fires rather than assume it.

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import torch, numpy as np
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.train_sjfpo import build_policy
cfg = SJFPOConfig()
pol = build_policy(10, 2, cfg, device='cuda',
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
from sjfpo.modeling_sjfpo import build_optimizer
opt = build_optimizer(pol)
print('optimizer impl     ', type(opt).__name__,
      'fused=' , [g.get('fused') for g in opt.param_groups])
print('pinned staging     ', pol.obs_normalizer.mean.is_pinned() if hasattr(pol.obs_normalizer.mean,'is_pinned') else 'n/a')
x = torch.randn(64,16,10, device='cuda')
from sjfpo.utils import to_device_async, autocast_context
y = to_device_async(x, torch.device('cuda'))
print('to_device_async    pinned_src=', y.is_cuda)
print('autocast active    ', torch.is_autocast_enabled('cuda') if False else 'see below')
with autocast_context(torch.device('cuda')):
    print('  inside ctx, cuda autocast =', torch.is_autocast_enabled('cuda'))
"
```

Expected on Blackwell: `fused=True` (`modeling_sjfpo.py:138` gates it on
`.is_cuda`), autocast enabled. A `fused=False` here means the gate is not
firing and the optimizer is running the slow path.

## Step 2 — wall-clock split

This is the single most important measurement and the one that decides where to
optimise. It is the CPU reference reproduced on your box.

```bash
cat > ./tmp/split.py <<'EOF'
import sys, time
sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
import sjfpo.modeling_sjfpo as M
import sjfpo.train_sjfpo as T
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.train_sjfpo import build_policy
from lerobot.utils.constants import OBS_STATE

NE   = int(sys.argv[1]) if len(sys.argv) > 1 else 64
UPD  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CFG  = SJFPOConfig()
cfg  = CFG
envs = gym.vector.SyncVectorEnv([lambda: gym.make("Reacher-v5") for _ in range(NE)])
D    = int(envs.single_observation_space.shape[0])
pol  = build_policy(D, 2, cfg, device="cuda",
                    action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
opt  = T.build_optimizer(pol)

acc = {}
def timed(name, fn):
    def wrap(*a, **k):
        t = time.perf_counter()
        try:    return fn(*a, **k)
        finally: acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
    return wrap

M.SJFPOPolicy.generate_rollout = timed("generate_rollout", M.SJFPOPolicy.generate_rollout)
M.SJFPOPolicy.forward          = timed("update_forward",  M.SJFPOPolicy.forward)
T.measure_boot_value_drift     = timed("drift_boot",      T.measure_boot_value_drift)
T.measure_value_target_drift   = timed("drift_target",    T.measure_value_target_drift)

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
for k in ("collect_plus_rollout","generate_rollout","update_forward",
          "drift_boot","drift_target"):
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

**Read it as:** on CPU, the backward is the target and the rollout is not. If
Blackwell inverts this — which it plausibly will, since the rollout is launch-
bound and the backward is compute-bound — the optimisation target moves with it.
Do not carry the CPU conclusion over.

## Step 3 — kernel-level breakdown of the backward

**This is the measurement that was impossible on CPU and is the main reason to
re-profile.** The CPU `cProfile` said `run_backward` was 12.34 s of a ~15 s run
but could not say whether that was the DiT blocks, the `num_mc × num_euler`
unroll, or the critic. `cProfile` is Python-frame granularity. Use
`torch.profiler` with CUDA activities.

```bash
cat > ./tmp/prof_bwd.py <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity, record_function
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.train_sjfpo import build_policy
import sjfpo.train_sjfpo as T

cfg = SJFPOConfig()
pol = build_policy(10, 2, cfg, device="cuda",
                   action_low=np.array([-2.,-2.]), action_high=np.array([2.,2.]))
opt = T.build_optimizer(pol)
n, H, D, A = cfg.chunks_per_update, cfg.n_obs_steps, 10, 2
tok = pol.get_token_count_for_cfm_loss(cfg)
g  = torch.Generator().manual_seed(0)
full = {
  "observation.state": torch.randn(n,H,D,generator=g).cuda(),
  "action": torch.randn(n,H,A,generator=g).cuda(),
  "action_is_pad": torch.zeros(n,H,dtype=torch.bool).cuda(),
  "raw_advantages": torch.randn(n,generator=g).cuda(),
  "normalized_advantages": torch.randn(n,generator=g).cuda(),
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

**What to extract.** For each row note `self_cuda_time_total`, the op count, and
whether `num_repeats` scales with `num_mc_samples`:

| Question | Where to look |
|---|---|
| Is the backward compute-bound or launch-bound? | If total kernel time << wall time of the region, you are launch-bound and the fix is CUDA graphs or `torch.compile(mode="reduce-overhead")`, not more FLOPs. |
| Does kernel time scale with `num_mc_samples`? | Re-run with `--sjfpo.num_mc_samples 8` and `16`. Linear means the `num_mc × num_euler` unroll is the cost. |
| Is the critic a large share? | The critic is 467,736 params vs the actor's 926,850 and it does **not** scale with `num_mc_samples` (it runs on `2 * n_chunks` windows, `modeling_sjfpo.py:815-834`). |
| Any host-device sync inside the loop? | Any `cudaStreamSynchronize`, `Memcpy DtoH`, or `.item()` in the top rows. `to_device_async` and `predict_action_chunk`'s `.cpu()` per chunk are expected sync points (`modeling_sjfpo.py:718`). |

## Step 4 — transfer accounting

The rollout is host-generated and the training is device-side, so PCIe traffic is
a real cost. Three transfers exist:

1. `buffer.copy_to(self._device_rollout_buffer, non_blocking=True)` +
   `torch.cuda.Event()` (`modeling_sjfpo.py:790-791`) — intended to overlap.
2. `propose_batch` does `chunk.cpu()` **once per chunk** (`modeling_sjfpo.py:718`)
   — a blocking D2H in the middle of the rollout loop.
3. `to_device_async` per wavefront, pinned staging.

```bash
cat > ./tmp/prof_xfer.py <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
from torch.profiler import profile, ProfilerActivity
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.train_sjfpo import build_policy
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

**What to look for.** `n_chunks` `Memcpy DtoH` calls of ~2 KB each would mean
the per-chunk `.cpu()` in the propose loop is serialising the rollout. On
Reacher at `num_envs=64` a rollout is 5 policy calls and 320 chunks, so ~320
small D2H. If they show real time, batch the D2H — but note the chunk loop is
data-dependent, so that is a real refactor, not a flag.

## Step 5 — wavefront batch histogram

This is the shape-stability input. On CPU, Reacher at `num_envs=64` gave a
perfectly uniform wavefront:

```
policy calls: 5   chunks: 320   envs: 64   ideal calls at full batch: 5
batches: min=64 max=64 mean=64.0   env-slots proposed: 320 vs 320 chunks -> 1.00
```

That is *not* guaranteed. Envs awaiting autoreset shrink `len(need)`, and the
final wavefront is ragged. Measure the histogram before trusting `dynamic=False`.

```bash
cat > ./tmp/wavefront.py <<'EOF'
import sys, collections; sys.path.insert(0, "src")
import numpy as np, torch, gymnasium as gym
import sjfpo.collector as C
from sjfpo.collector import RolloutBuffer
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

**Read it as:** `efficiency` below ~95% means you are paying full-batch DiT cost
for a partial wavefront. Padding to `num_envs` costs FLOPs; the right answer
depends on step 3's launch-bound-vs-compute-bound verdict. `distinct batch
sizes` above 2–3 means `torch.compile` will recompile per shape.

## Step 6 — does `torch.compile` work here?

On this CPU build it does not, and the failure is **not** a graph break — the
propose path captures as a single clean 1659-op graph with **zero** breaks, then
AOTAutograd crashes on it in all four configurations tested:

| backend | result |
|---|---|
| `eager` | OK, but 1586 ms/call of pure capture overhead, no fusion |
| `aot_eager` | FAIL `RuntimeError: <weakref> to 'torch.storage.UntypedStorage'` |
| `aot_eager_decomp_partition` | FAIL, same |
| `inductor` | FAIL, same |

With DCT off it fails differently: `INTERNAL ASSERT FAILED at ATen/core/...`.

```bash
cat > ./tmp/compile_probe.py <<'EOF'
import sys, time; sys.path.insert(0, "src")
import numpy as np, torch
from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.train_sjfpo import build_policy
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

**Expectation from the CPU work: even where compile succeeded it returned 1.05x.**
Do not invest in compile until step 3 shows a real win available. If it works on
Blackwell, the two known graph-side cleanups are worth doing first — see
step 10.

## Step 7 — dtype and matmul precision

Blackwell tensor cores make reduced precision worth re-testing, and the CPU
numbers do not transfer. `resolve_compute_dtype` already selects bf16, so the
first question is whether autocast is actually engaged.

```bash
for AMP in true false; do
  for PREC in highest high medium; do
    OUT=$(SJFPO_PREC=$PREC .venv/bin/python -c "
import os, torch
torch.set_float32_matmul_precision(os.environ['SJFPO_PREC'])
print(torch.get_float32_matmul_precision())" 2>/dev/null)
    echo "--- precision=$PREC use_amp=$AMP (torch says $OUT)"
    .venv/bin/python scripts/train.py --env Reacher-v5 --seed 3407 \
      --total_steps 8192 --eval_steps 8192 --save_steps 8192 \
      --sjfpo.use_amp $AMP --sjfpo.chunks_per_update 16 --sjfpo.batch_size 16 \
      --sjfpo.report_to none --out_dir ./tmp/p_${PREC}_${AMP} 2>/dev/null \
      | python3 -c "import sys,json; d=json.load(sys.stdin); \
print(f\"      fps={d['fps']:.1f} reward={d['reward_mean_last10']:.3f} diverged={d['diverged']}\")"
  done
done
```

**Read it as.** CPU reference for the same knob, 1024×1024 fp32 matmul:

| level | ms | max err vs fp64 |
|---|---|---|
| `highest` | 6.76 | 6.485e-05 |
| `high` | 6.06 | 6.485e-05 |
| `medium` | 4.97 | 1.068e-04 |

`high` was numerically free on CPU. On Blackwell the matmul speedups will be far
larger and the error pattern may differ. **This is a research decision, not a free
win** — the repo is fp32-by-design on purpose, obs and rewards fp32 end to end
(`collector.py` header, and "bf16 rewards corrupt the GAE target"). If you change
it, re-check `gae/linked_frac` and `value_target_std` from step 8 for damage.

## Step 8 — per-update health metrics

Even with `report_to=none`, the diagnostics come from the info dict. Read them
from the tensorboard event file, or use `report_to=tensorboard` and
`tensorboard --logdir`.

```bash
SJFPO_DISABLE_WANDB=1 .venv/bin/python scripts/train.py --env Reacher-v5 --seed 3407 \
  --total_steps 24576 --eval_steps 8192 --save_steps 24576 \
  --sjfpo.report_to tensorboard --out_dir ./tmp/tb 2>&1 | tail -3
tensorboard --logdir ./tmp/tb --port 6006
```

Metrics that decide whether the GPU changed any conclusion:

| metric | healthy | meaning if wrong |
|---|---|---|
| `gae_linked_frac` | ~0.7 or higher | 0 means a dead trace: the GAE chain joined nothing and every advantage collapsed to a 1-step delta. On CPU at `num_envs=64` it measured 0.736, with the remainder exactly the episode ends. |
| `explained_var` | rising toward 1 | near 0 means the critic is inert |
| `advantage_std_fraction` | falling toward `1 - R²` | pinned near 1.0 means the critic is doing nothing |
| `actor_grad_norm_preclip` vs `critic_grad_norm_preclip` | critic much larger | a shared clip would be owned by the critic; that is why `critic_max_grad_norm` is separate |
| `cfm_above_clamp_frac` | low | near 1 means the policy gets no gradient (`clamp()` has zero gradient above its max) |
| `value_target_std`, `returns_rms` | stable | drift means the target critic is lagging |

## Step 9 — `num_envs` with the GPU in the loop

The CPU 3-seed A/B settled this and the answer was **64**: 32 and 48 were
statistically indistinguishable from each other and both worse on mean *and*
variance (std 2.85 / 2.78 vs 0.37). More parallel trajectories per GAE batch
means more decorrelated samples. **But that was a CPU result about GAE sample
quality, not a GPU result about throughput**, and on a GPU the throughput
pressure points the other way: 64 `AsyncVectorEnv` workers competing with the
training process for host cores.

```bash
for NE in 16 32 64 128; do
  .venv/bin/python scripts/train.py --env Reacher-v5 --seed 3407 \
    --total_steps 24576 --eval_steps 8192 --save_steps 24576 \
    --sjfpo.num_envs $NE --sjfpo.report_to none --out_dir ./tmp/ne$NE 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); \
print(f\"  num_envs=$NE fps={d['fps']:.1f} reward={d['reward_mean_last10']:.3f}\")"
done
```

**Read it as.** If fps rises with `num_envs` on GPU, the host is the bottleneck
and you have a real trade against the GAE quality result above. If fps falls, the
GPU was already saturated at 16 and 64 costs you sample quality for nothing —
keep 64 anyway, but note you are not buying throughput with it. Cross-check
against step 5's efficiency column.

## Step 10 — known cleanups, do these regardless

Independent of any measurement, these are free in eager and reduce the traced
graph. Listed with the survey that found them so you can re-check the premise.

| Site | Issue | Fix |
|---|---|---|
| `modeling_sjfpo.py:1129` | the **only** graph break in the codebase: `bool(torch.isfinite(loss).all())` | Hoist the finiteness guard out of the loss, or gate it on the tracker. Better than `capture_scalar_outputs=True`, which pulls a scalar sync into the graph. |
| `networks.py:307-311` | the obs normalizer calls `.to(x.device)` on `mean`, `var`, and `_uninitialized` on **every forward** | These are buffers whose device never changes. Three redundant casts per forward. |
| `dct_utils.py:36-38` | `_get_analysis_basis` rebuilds the DCT basis per call; it is `lru_cache`d but Dynamo **bypasses the cache and retraces it** (warns at `dct_utils.py:61` and `utils.py:93`, "silent incorrectness is a potential risk") | Precompute as a registered buffer. |

## Step 11 — allocator and VRAM

```bash
.venv/bin/python -c "
import torch
print('alloc conf:', torch.cuda.memory.get_allocator_backend())
print([a for a in dir(torch.cuda.memory) if 'expandable' in a.lower()])
" 2>/dev/null
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python ./tmp/split.py 64 4
```

Compare the peak VRAM line from step 2 with and without
`expandable_segments:True`. Fragmentation on a large-memory device with a
small, repeatedly-resized allocation set is a common cause of high reserved /
low allocated.

---

## Bring back

1. Step 0 environment block, verbatim.
2. Step 2 table — the segment percentages. **This is the answer to "where does
   the time go" and everything else is a footnote to it.**
3. Step 3 top-25 kernel table, plus the answer to: compute-bound or
   launch-bound?
4. Step 4 transfer counts, specifically how many small D2H per rollout.
5. Step 5 batch histogram and efficiency for all three envs.
6. Step 6 verdict on `torch.compile`.
7. Step 7 dtype sweep, with reward alongside fps.
8. Step 8 `gae_linked_frac` and `explained_var` trajectories.

## Do not carry these over from the CPU profile

Measured on CPU, may not hold on Blackwell:

| CPU finding | Status |
|---|---|
| backward + optimizer is 61.4% | re-measure; the balance inverts easily |
| rollout is 15.9%, so the collector is not a bottleneck | re-measure; the collector is host-side and the host is now also feeding the GPU |
| GAE is 0.0% of runtime | re-measure, but it is ~20 lines of Python, so it only matters if everything else gets 100x faster |
| `torch.compile` gives 1.05x | only ever measured on a build where AOTAutograd crashes |
| `AdamW(fused=True)` is slower | **irrelevant on GPU** — `modeling_sjfpo.py:138` enables it there, and on CPU the optimizer was only 0.35% of the update |
| `F.rms_norm` and SDPA are already fused | unchanged, nothing to do |
| `einx` symbolic shapes cost 0.002% | unchanged, nothing to do |
| num_envs 64 beats 32 and 48 | a GAE-quality result, not a throughput one; re-run step 9 |
