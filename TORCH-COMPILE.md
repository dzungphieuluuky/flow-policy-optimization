# TORCH-COMPILE.md — enabling `torch.compile` on RTX PRO 1000 Blackwell (ThinkPad P16v Gen 3)

Target: the highest-throughput `torch.compile` configuration for
`spectral-joint-policy`, chosen by measurement on the real GPU.

**Read `BLACKWELL.md` first.** It holds the CUDA profile, the wall-clock
split, the transfer accounting and the rollout-propose (Euler loop) numbers.
This file does not repeat them; it covers only what decides the compile
configuration. Every number quoted here as "measured" was taken on the **CPU
dev box** (torch 2.11.0+cu130, no CUDA, 4 threads). They are the reference to
check against, not the result.

Work in order. **Stage 0 and Stage 1 are gates**: do not benchmark timings
until both pass, because Stage 0's failures are silent and Stage 1's absence
makes every timing meaningless.

---

## Preconditions

```bash
mkdir -p ./tmp/{inductor,triton}
export TORCHINDUCTOR_CACHE_DIR=$PWD/tmp/inductor     # never the default /tmp
export TRITON_CACHE_DIR=$PWD/tmp/triton
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
.venv/bin/python -c "
import torch, sys; sys.path.insert(0,'src')
print('torch', torch.__version__, '| triton', __import__('triton').__version__)
print('cuda', torch.version.cuda, '| available', torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'device {i}: {p.name} sm_{p.major}{p.minor} {p.total_memory/2**30:.1f} GiB SMs={p.multi_processor_count}')
from spectral_joint_policy.utils import (is_rtx_1000_pro_blackwell,
    resolve_storage_dtype, resolve_compute_dtype)
print('rtx_pro_1000_blackwell', is_rtx_1000_pro_blackwell())
print('storage dtype (cuda)   ', resolve_storage_dtype('cuda'))
print('compute dtype (cuda)   ', resolve_compute_dtype())
"
```

| # | Check | Why it decides the sweep |
|---|---|---|
| 1 | **Do not run `uv sync` / `uv run`.** Use `.venv/bin/python` directly, `uvx ruff` for lint. | `uv lock` resolves torch from PyPI's default build; a past sync left `libnccl.so.2` missing. `BLACKWELL.md` precondition 1-2. |
| 2 | Record `sm_{major}{minor}` and the Triton version | The Triton backend, TMA and `max-autotune` GEMM choices all branch on arch. An sm number you assumed is worse than the printed one. |
| 3 | `resolve_storage_dtype('cuda')` is **bf16** on this SKU | Every graph is then specialized on a bf16 input signature. Expected, not a bug — but it doubles the graph count (bf16 storage vs the fp32 GAE path) and means numerics tolerance must be bf16-scale. |
| 4 | `resolve_compute_dtype()` is bf16 | If not, every `autocast_context` is a `nullcontext` and the compiled forward is fp32. A different sweep. |
| 5 | `torch._dynamo.config.cache_size_limit` (default **8**) | Raise it before anything else; see Stage 1.4. |
| 6 | `scripts/train.py:configure_torch_cuda` sets `torch.backends.cuda.matmul.fp32_precision="ieee"` **then** `set_float32_matmul_precision("high")` | Same knob, opposite intent, last call wins. Inductor reads it too, so resolve it rather than relying on call order (`BLACKWELL.md` step 7). |

---

## Stage 0 — correctness gates (no timings until these pass)

Measured on CPU, `torch.compile` returned an `OptimizedModule` whose
parameter names are prefixed:

```
type(torch.compile(m))        OptimizedModule
named_parameters()[0]         _orig_mod.token_pos      (was: token_pos)
state_dict keys renamed       96 of 219, e.g. actor._orig_mod.adaLN_out.bias
```

Assigning that wrapper back onto the module is what breaks things, so **the
wrapper must be called, never assigned**:

```python
def compile_probe():
    """Gate 0: compiling must not change parameter identity or names."""
    import sys, dataclasses, torch; sys.path.insert(0, "src")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
    from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy
    cfg = dataclasses.replace(SJFPOConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (17,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))})
    pol = SJFPOPolicy(cfg)
    before = dict(pol.named_parameters())
    keys_before = set(pol.state_dict())
    wrapper = torch.compile(pol.actor, dynamic=True)          # NOT pol.actor = ...
    assert next(iter(wrapper.named_parameters()))[0] != next(iter(before))[0]
    assert set(pol.state_dict()) == keys_before, "compile leaked into state_dict"
    # the correct shape: a plain dict attribute is not registered by nn.Module
    store = {}
    store["actor"] = torch.compile(pol.actor, dynamic=True)
    assert set(pol.state_dict()) == keys_before
    print("gate 0 ok: names, state_dict and param groups untouched")
compile_probe()
```

Then check the three things that break *silently* if the wrapper is assigned:

| Gate | Failure mode if violated | How to see it |
|---|---|---|
| **0a param names** | `get_actor_optim_params()` splits no-decay on `name.startswith("out.")`. Under a wrapper that is `_orig_mod.out.weight`, so `out.*` silently joins the **weight-decay group**. Different optimizer, no error. | `print(sorted({n for n,_ in pol.actor.named_parameters()}))` must contain `out.weight`, `adaLN.fc1.weight` |
| **0b EMA shadow** | `update_ema` keys `_ema_shadow` off `named_parameters()`; prefixed keys change the checkpoint layout and `ema_state_dict()` | `assert set(policy.ema_state_dict()["shadow_params"]) == {n for n,_ in policy.actor.named_parameters()}` |
| **0c frozen snapshot** | `snapshot_behavior_policy()` does `old_actor.load_state_dict(self.actor.state_dict())`. Prefixed keys on one side → **RuntimeError** (loud, good) | call `policy.snapshot_behavior_policy()` after any compile step |
| **0d numerics** | Inductor fuses differently; with bf16 storage the drift is bf16-scale, not fp32-scale | `./tmp/fingerprint.py` if it still exists (`BLACKWELL.md` A2), else compare `explained_var`, `gae_linked_frac`, `flow_ratio_token_lowfreq` from `BLACKWELL.md` step 8 over 2 rollouts |
| **0e optimizer still fused** | `build_optimizer` sets `fused=True` on CUDA; a recompile must not silently drop it | `print([g.get('fused') for g in build_optimizer(pol).param_groups])` |

**Naming sign-off required before implementation.** `AGENTS.md` requires a
naming decision before adding a config field. Proposal: a single
`compile: bool = False` on `SJFPOConfig`, exposed as `--sjfpo.compile=true`,
plus `torch._dynamo.config.cache_size_limit` raised next to
`configure_torch_cuda()` in `scripts/train.py`. Note the field must be added to
`src/sjfpo_jax/configuration_sjfpo.py` too or `test_config_parity.py::
test_field_set_is_torch_minus_lerobot_plus_the_two_dims` gets worse (it already
fails on `adam_betas`).

---

## Stage 1 — make the graph static (prerequisite for any real speedup)

Measured on CPU with `torch._dynamo.explain`, current tree (`95ca5b7`):

| Compiled target | graphs | breaks | ops |
|---|---|---|---|
| `SJFPOPolicy.actor.forward` | 1 | **0** | — |
| `StateEncoder.forward` | 1 | **0** | 3 |
| `Critic.forward` | 1 | **0** | 7 |
| `SJFPOPolicy.forward` | **13** | **12** | 367 |

**The modules are already static.** The fragmentation is entirely in
`SJFPOPolicy.forward`, and it is nearly all validation code reading scalars.
Note what this rules out: the 12 nested `autocast(enabled=False)` regions inside
`_RMSNorm` are traced fine by dynamo 2.11 (1 graph, 0 breaks) — do not spend
time there.

### 1.1 Remove the unreachable `tau` range guard — **the big one**

```
src/spectral_joint_policy/dct_utils.py:145 and :159
    if not bool(torch.all((tau_q >= 0) & (tau_q <= T))):
        raise ValueError(...)
```

`torch.all(...)` → `bool(...)` is a tensor→scalar read inside the CFM-loss
path. It is unreachable: the only caller, `velocity_to_action_steps`, builds the
grid itself.

```
velocity_to_action_steps -> tau = torch.arange(num_rows) / num_rows
tau min/max = 0.0 / 0.9375,  T = 1.0  ->  always in range: True
```

Measured effect of deleting just this guard:

```
as-is                       graphs=13  breaks=12  ops=367
without the tau range check graphs= 4  breaks= 3  ops=345
```

That is a caller contract enforced at runtime. Replace it with the guarantee,
stated as a comment on `velocity_to_action_steps`, not with a compile guard.

### 1.2 Constant minibatch width

`generate_mini_batch_indices` splits `(i*n)//n_mb`, and `n = n_chunks` is
data-dependent (an episode ending mid-chunk produces a short chunk), so the
row count varies every rollout. Measured compile-time cost of shape churn on
one compiled actor:

```
rows 2048  29.467 s (compile)   rows 1024  0.516 s   rows  64  0.023 s
rows 2032   7.981 s (recompile) rows  700  0.312 s   rows   1  7.993 s (recompile)
rows 2016   1.029 s (dynamic)   rows 2048  1.028 s (cache hit)
```

Fix: pad `n` up to `num_mini_batches * ceil(n / num_mini_batches)` with
repeated indices, or drop `< num_mini_batches` chunks. Then `B` is
`batch_size` forever.

### 1.3 Constant rollout width for the critic pass

`generate_rollout` runs the critic over `cat([states, next_states])` = `2C`
rows with `C` varying. `RolloutBuffer` already has a fixed `chunk_capacity`:
pad the batch to `2 * chunk_capacity` and carry a validity mask so the `amin`
masks instead of slicing. Verify with the wavefront histogram
(`BLACKWELL.md` step 5) that padding waste stays under ~10%.

### 1.4 `cache_size_limit`, and then `dynamic=False`

```python
torch._dynamo.config.cache_size_limit = 32   # default is 8
```

Past 8 variants dynamo **silently falls back to eager** — you pay compile cost
and get none of the benefit. Raise it before measuring anything.

Once 1.2 and 1.3 land, shapes are constant, and **`dynamic=False` becomes
viable**. That is the highest-value switch in this document: dynamic kernels
give up autotuned tile sizes and epilogue fusion. Sweep `dynamic=False` against
`dynamic=True` explicitly (Stage 3); do not assume `True` is the safe default.

### 1.5 Keep out of the compiled region

Already gated, keep it that way: `compute_info` (packed `.tolist()`),
`effective_rank` (`svdvals`), `check_finite` (`bool(torch.isfinite(...))`),
the collector (Python loop, pinned memory, `torch.cuda.Event`), and
`compute_gae_diagnostics`.

One known interaction: `get_analysis_basis` is `@lru_cache` and dynamo
**ignores the cache wrapper**, so basis caching stops applying inside a
compiled region and the basis is rebuilt per trace. Cheap (`cos` on a
`[4,16]` table) but it emits a warning; either hoist the basis to a module
global or accept it.

---

## Stage 2 — the sweep harness

`./tmp/compile_sweep.py` (validated end-to-end on CPU: eager and compiled
paths, compile time, graph census, medians, peak VRAM). Regenerate from this
doc if `./tmp` was cleaned:

```bash
cat > ./tmp/compile_sweep.py <<'EOF'
"""torch.compile sweep probe. Validated on CPU; the real numbers need a GPU.

    .venv/bin/python ./tmp/compile_sweep.py --target forward --iters 20
    .venv/bin/python ./tmp/compile_sweep.py --target actor --modes default max-autotune-no-cudagraphs

Prints one row per configuration: compile seconds, graphs/breaks/ops,
steady-state ms/update, and peak VRAM.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature
from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy


def make_batch(cfg, C, device):
    K, M, T, H, D = (
        cfg.num_flow_tokens, cfg.num_mc_samples, cfg.n_obs_steps,
        cfg.action_horizon, int(cfg.action_feature.shape[0]),
    )
    g = torch.Generator().manual_seed(0)
    r = lambda *s, dt=torch.float32: torch.randn(*s, generator=g, dtype=dt).to(device)
    return {
        "observation.state": r(C, T, int(cfg.robot_state_feature.shape[0])),
        "action": r(C, H, D),
        "action_is_pad": torch.zeros(C, H, dtype=torch.bool, device=device),
        "normalised_advantages": r(C),
        "raw_returns": r(C) * 10,
        "mc_tau": torch.rand(C, M, K, 1, generator=g).to(device),
        "mc_eps": r(C, M, K, D),
        "behavior_cfm_loss": torch.rand(C, M, generator=g).to(device),
        "behavior_cfm_loss_token": torch.rand(C, M, H, generator=g).to(device),
    }


def build(args):
    cfg = dataclasses.replace(
        SJFPOConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (args.obs_dim,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (args.act_dim,))},
        hidden_dim=args.hidden, dit_depth=args.depth,
    )
    pol = SJFPOPolicy(cfg)
    return cfg, pol.to(args.device)


def explain(fn, *a):
    try:
        e = torch._dynamo.explain(fn)(*a)
        return e.graph_count, e.graph_break_count, e.op_count
    except Exception as exc:  # explain is itself version-sensitive
        return -1, -1, type(exc).__name__


def timed(fn, iters, warmup, sync):
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("forward", "actor", "critic"), default="forward")
    ap.add_argument("--modes", nargs="+", default=["eager", "default", "max-autotune-no-cudagraphs"])
    ap.add_argument("--dynamic", nargs="+", default=["true"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunks", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--act-dim", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()

    sync = args.device.startswith("cuda") and torch.cuda.is_available()
    cfg, pol = build(args)
    opt = torch.optim.AdamW(
        [p for g in pol.get_actor_optim_params()["params"] for p in g["params"]]
        + [p for g in pol.get_critic_optim_params()["params"] for p in g["params"]],
        lr=1e-5,
    )
    batch = make_batch(cfg, args.chunks, args.device)

    if args.target == "forward":
        def step():
            loss, _ = pol(batch, compute_info=False, check_finite=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        base = lambda: step()
    elif args.target == "actor":
        K, M = cfg.num_flow_tokens, cfg.num_mc_samples
        rows = args.chunks * M
        x_t = torch.randn(rows, K, args.act_dim, device=args.device)
        tau = torch.rand(rows, K, 1, device=args.device)
        cond = torch.randn(rows, args.hidden, device=args.device)

        def step():
            out = pol.actor(x_t, tau, cond)
            opt.zero_grad(set_to_none=True)
            out.square().mean().backward()
            opt.step()
        base = step
    else:
        states = batch["observation.state"]

        def step():
            out = pol.critic(states)
            opt.zero_grad(set_to_none=True)
            out.square().mean().backward()
            opt.step()
        base = step

    print(f"target={args.target} chunks={args.chunks} hidden={args.hidden} "
          f"depth={args.depth} device={args.device} sync={sync}")
    print(f"{'config':38s} {'compile s':>9s} {'graphs':>7s} {'breaks':>7s} {'ops':>6s} "
          f"{'ms med':>9s} {'ms min':>9s} {'peak GiB':>9s}")

    for mode in args.modes:
        for dyn in (args.dynamic if mode != "eager" else ["n/a"]):
            torch._dynamo.reset()
            gc.collect()
            if sync:
                torch.cuda.reset_peak_memory_stats()
            tag = mode if dyn == "n/a" else f"{mode} dynamic={dyn}"
            kw = {} if mode == "eager" else {
                "mode": None if mode == "default" else mode,
                "dynamic": dyn == "true",
            }
            fn = base if mode == "eager" else torch.compile(base, **kw)
            t0 = time.perf_counter()
            try:
                fn()
                if sync:
                    torch.cuda.synchronize()
                compile_s = time.perf_counter() - t0
            except Exception as exc:
                print(f"{tag:38s}  FAIL  {type(exc).__name__}: {str(exc)[:60]}")
                continue
            graphs, breaks, ops = (("-", "-", "-") if mode == "eager"
                                   else explain(fn))
            med = timed(fn, args.iters, args.warmup, sync)
            best = timed(fn, args.iters, 0, sync)
            peak = (f"{torch.cuda.max_memory_reserved()/2**30:.2f}" if sync else "-")
            print(f"{tag:38s} {compile_s:9.1f} {str(graphs):>7s} {str(breaks):>7s} "
                  f"{str(ops):>6s} {med:9.1f} {best:9.1f} {peak:>9s}")


if __name__ == "__main__":
    main()
EOF
```

Invocation:

```bash
# update path, both dynamics, eager reference
.venv/bin/python ./tmp/compile_sweep.py --target forward \
    --modes eager default max-autotune-no-cudagraphs --dynamic true false

# the actor alone (what A1's fusion payoff actually lives in)
.venv/bin/python ./tmp/compile_sweep.py --target actor \
    --modes eager default --dynamic false

# rollout propose: batch = num_envs, the A1 shape
.venv/bin/python ./tmp/compile_sweep.py --target forward --chunks 64 --iters 50
```

Read the row, not the mode name:

| Column | Meaning |
|---|---|
| `compile s` | one-time. Must be amortised over the planned run length. |
| `graphs` / `breaks` / `ops` | **the only column that predicts fusion**. More graphs = less fused. |
| `ms med` | steady state, median of `--iters`. Median, not mean: this workload's eager spread was 3095–3509 ms (~12%) on 4 CPU threads. |
| `ms min` | best case; useful when the GPU is thermally limited |
| `peak GiB` | cudagraph modes reserve memory pools |

**Noise floor: require ≥15% to believe a result, and take medians of 3.** A
CPU reference from this session, for scale:

| Measurement | Result |
|---|---|
| actor fwd+bwd, eager vs compiled, 3 alternating rounds | **1.09×** (eager 3095/3509/3272, compiled 2996/3013/3031) |
| full training step fwd+bwd+opt | **0.97×** |
| actor forward, single run | 1.19× |

Within-noise on CPU. The GPU case is unmeasured — that is the point of this
document.

---

## Stage 3 — sweep matrix, in the order to run it

| # | Target | mode | dynamic | Isolates |
|---|---|---|---|---|
| 1 | `forward` | eager | — | reference |
| 2 | `forward` | default | `true` | automatic dynamic, current shapes |
| 3 | `forward` | default | `false` | **does static-shape specialization pay after Stage 1?** |
| 4 | `forward` | max-autotune-no-cudagraphs | best of 2/3 | GEMM/epilogue autotuning, no cudagraph risk |
| 5 | `forward` | max-autotune | best of 3/4 | adds cudagraphs — read Stage 4 first |
| 6 | `forward` | reduce-overhead | best of 3/4 | cudagraphs only; cheapest way to test capture viability |
| 7 | `actor` | best of the above | best | how much of the win is the 6 DiT blocks |
| 8 | rollout propose | reduce-overhead | `false` | **A1's Euler loop — probably bigger than rows 1-7 combined** |

**Stop rule.** Stop after row 4 if `ms med` improves <15% over eager: compile
is not the lever on this machine, and `BLACKWELL.md` A1 (CUDA-graph the
16-step Euler integration) is. Row 8 is the one to try first if you only have
time for one, because the rollout is launch-bound and 16 Euler steps deep.

`torch._inductor.list_mode_options()` prints what each mode actually sets —
check it rather than trusting the name. Do **not** hand-roll cudagraph capture
until row 6 has run: if `reduce-overhead` cannot capture, your hand-written
capture will not either, and the failures in A1 (`.cpu()` in `propose_batch`,
`actions_t.cpu().numpy()` in the collector) still have to be lifted first.

---

## Stage 4 — cudagraphs and in-place parameter mutation

`reduce-overhead` and `max-autotune` enable CUDA graphs, and the
`torch.compile` docstring is explicit: reduction of overhead "is only performed
for CUDA only graphs which do not mutate inputs". Three sites mutate
parameters **in place**, all between forward calls:

| Site | Mutation |
|---|---|
| `swap_in_ema` / `swap_out_ema` | `torch._foreach_copy_` into the live actor's parameter storage, once per eval |
| `snapshot_behavior_policy` | `load_state_dict` into `old_actor` / `old_encoder` |
| `update_target_critic` | `torch._foreach_mul_` + `_foreach_add_` on `target_critic` |

These are in-place, so graph *addresses* stay valid and cudagraph replay should
read the new values. Do not assume it — run this gate:

```python
def cudagraph_param_gate(policy):
    """A captured graph must observe in-place parameter updates.

    Perturbs ``actor.out.bias`` specifically: ``out`` is zero-init and the
    adaLN-Zero stack starts as an identity, so perturbing anything upstream
    (token_pos, adaLN) provably cannot move the output and the gate would
    pass vacuously. The output bias bypasses the whole stack.
    """
    import torch
    bias = policy.actor.out.bias
    with torch.inference_mode():
        x = torch.randn(8, policy.config.n_obs_steps,
                        int(policy.config.robot_state_feature.shape[0]))
        before = policy.predict_action_chunk({"observation.state": x}).clone()
        with torch.no_grad():
            bias.add_(1.0)
        after = policy.predict_action_chunk({"observation.state": x})
    assert not torch.allclose(before, after), "graph replay used stale parameters"
    print("cudagraph param gate ok")

# Run it on the eager policy first (must pass), then on the compiled one
# (row 6 of Stage 3). It only has teeth under cudagraph capture.
if __name__ == "__main__":
    import dataclasses, sys
    sys.path.insert(0, "src")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from spectral_joint_policy.configuration_sjfpo import SJFPOConfig
    from spectral_joint_policy.modeling_sjfpo import SJFPOPolicy

    cfg = dataclasses.replace(
        SJFPOConfig(),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (17,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    cudagraph_param_gate(SJFPOPolicy(cfg).eval())
```

Also count graphs per mode with the harness: `train fwd+bwd`, `inference_mode`
rollout, `compute_info=True` diagnostics and the bf16-vs-fp32 GAE path each
specialize separately. Expect ≥4 graphs per compiled module before Stage 1, and
check that count does not exceed `cache_size_limit`.

---

## Stage 5 — end-to-end, after the sweep picks a winner

`train.py`'s reported `fps` is `global_step / elapsed` and **includes eval and
checkpoint saving**, so it is not the update-path number. Two runs:

```bash
# update-dominated: no eval, no periodic save
WANDB_MODE=offline .venv/bin/python scripts/train.py --env Walker2d-v5 --seed 3407 \
  --total_steps 65536 --eval_steps 99999999 --save_steps 99999999 \
  --out_dir ./tmp/tc_update --sjfpo.compile=true

# eval-inclusive, with the EMA + stochastic pair
WANDB_MODE=offline .venv/bin/python scripts/train.py --env Walker2d-v5 --seed 3407 \
  --total_steps 65536 --eval_steps 16384 --save_steps 16384 \
  --out_dir ./tmp/tc_full --sjfpo.compile=true

git checkout -- wandb/
```

Run both against an eager twin with identical `--seed` and diff the health
metrics (`BLACKWELL.md` step 8): `gae_linked_frac`, `explained_var`,
`advantage_std_fraction`, `actor_grad_norm_preclip` vs
`critic_grad_norm_preclip`, `cfm_above_clamp_frac`. A compiled run whose
`explained_var` diverges from its eager twin is a numerics failure, not a
performance result.

Use `Walker2d-v5`, not `Reacher-v5`, for the timing runs: Reacher episodes are
50 steps, so eval and the fixed compile cost dominate a short run and you will
"measure" the compile overhead instead of the update.

---

## Stage 6 — benchmarking hygiene on a laptop

This is a thermally and power-limited part, and the limits are large enough to
invent speedups that do not exist.

| Risk | Control |
|---|---|
| **Sustained-clock throttle.** A short benchmark fits in the boost window; a long run does not. | `nvidia-smi -q -d PERFORMANCE` and read `clocks_throttle_reasons.active` before and after every run. Record SM clock alongside every timing. If throttling is active, fix clocks or report the throttled number as throttled. |
| **Power cap** | `nvidia-smi -q -d POWER` for the enforced limit. Laptop SKUs often enforce below the nominal TGP. Do not raise it; report it. |
| **Clock locking** | `sudo nvidia-smi -lgc <min>,<max>` and `-pl <watts>` if the driver accepts them. Many laptop parts reject `-lgc`; if rejected, that is the answer, not a problem to solve. |
| **CPU is a separate thermal domain.** Inductor tracing is 30–60 s of pure CPU, and the collector is host-side. | Measure `compile s` separately from steady state, and note CPU package temperature if the fan is audible. |
| **Compile cache makes run 2 look faster.** | `rm -rf ./tmp/inductor ./tmp/triton` between configurations, or keep the cache and say so. Never compare a cold run against a warm one. |
| **Concurrent jobs** | `nvidia-smi` shows other processes. One run at a time; `scripts/train.py` is one run per invocation, so do not fan out over envs while timing. |
| **Background throttling from the previous run** | Discard the first 3 iterations (`--warmup 3`), then take the median of `--iters`. |
| **Allocator** | `scripts/train.py` already exports `PYTORCH_CUDA_ALLOC_CONF` with `expandable_segments:True`. Keep it identical across arms. |

---

## Bring back

| # | Result | Where it goes |
|---|---|---|
| 1 | Stage 0 gate output (all five) | `AGENTS.md` audit trail; without it nothing else is trustworthy |
| 2 | Stage 1 `explain` census after the `dct_utils` fix | `BLACKWELL.md` A3 — the open "two sites" item, closed |
| 3 | Sweep rows 1–6 as a table, with `graphs`/`breaks`/`ops` | the choice of default mode |
| 4 | `dynamic=False` vs `true` at equal `graphs` | decides whether Stage 1's padding work was worth it |
| 5 | Recompile count during a real run (`torch._dynamo.utils.counters`) | whether `cache_size_limit=32` is enough |
| 6 | Row 8 rollout-propose number, or `reduce-overhead` capture failure | decides whether A1's hand-written cudagraph is worth it |
| 7 | Cudagraph param gate | whether `max-autotune` is safe here or only `...-no-cudagraphs` |
| 8 | End-to-end fps and peak VRAM, eager vs compiled, same seed | the actual number |
| 9 | Health-metric diff vs the eager twin | numerics verdict |
| 10 | Throttle reasons and SM clock per run | whether any of 8 is real |

## Do not spend time here

Recorded so this file is not re-litigated:

| Candidate | Measured | Verdict |
|---|---|---|
| Nested `autocast(enabled=False)` in `_RMSNorm` as a break source | 1 graph, 0 breaks | Not a problem; dynamo 2.11 traces it |
| `_RMSNorm` fp32 round-trip | `BLACKWELL.md` Part C | Leave it |
| `Critic` members → `vmap` / `baddbmm` | 0.239 → 0.169 ms on CPU, but the critic forward is **0.21 ms of a 3074 ms update** (0.007%) | 0.02% of the update; not worth the checkpoint key change |
| Batching the 50 eval episodes into one flow call | batch 1: 1.375 ms/env, batch 50: 1.747 ms/env | No win on CPU (compute-bound at batch 1). Re-test on GPU, where batch-1 is launch-bound |
| Vectorizing eval envs across workers | `env.step` 0.030 ms (Reacher) / 0.085 ms (Hopper) vs 1.375 ms policy call | Env is 2-6% of an eval step |
| Per-step `eval()`/`train()` toggle in `_EvalPolicy` | +0.835 ms on a 1.375 ms policy call (+61%) | Not parallelism, but it is pure Python overhead in the eval hot path and the cheapest remaining win in the repo |
| `SinusoidalEncoding` → `diffusers.get_timestep_embedding` | different denominator (`half_dim-1` vs `half_dim`), max abs diff 1.99 | Rejected; also diverges from the JAX copy |

## Traps specific to this repo

| Trap | Consequence |
|---|---|
| `self.actor = torch.compile(self.actor)` | 96 parameter names gain an `_orig_mod.` prefix; `get_actor_optim_params`'s `startswith("out.")` no-decay split silently changes; EMA shadow keys change; `snapshot_behavior_policy` raises |
| `torch.compile` called per-call | Measured ~2 ms of pure wrapper overhead per call (`BLACKWELL.md` A3). Memoize once |
| `mode="reduce-overhead"` with in-place param updates | See Stage 4's gate. Addresses survive, values must be re-read |
| Compiling the CFM-loss region in isolation | `BLACKWELL.md` A3: fuses only when tensors are module globals, 7% regression in situ |
| Default `cache_size_limit=8` | Silent eager fallback after 8 shape variants |
| `@lru_cache` on `get_analysis_basis` | Dynamo ignores the wrapper; basis rebuilt per trace, plus a warning |
| `configure_torch_cuda`'s two matmul-precision calls | Same knob, opposite intent; inductor reads it |
| bf16 storage on this SKU | Numerics tolerance is bf16-scale, not fp32; storage and compute dtypes each specialize their own graph |
| `TORCHINDUCTOR_CACHE_DIR` default | Writes to `/tmp`, which `AGENTS.md` forbids; point it at `./tmp` |