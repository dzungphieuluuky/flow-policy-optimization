# JAX_MIGRATION.md — port SJFPO from PyTorch to JAX

Target: tens of millions of env steps per run. Learner in JAX/Flax NNX, MuJoCo
envs stay on CPU, one compiled program per rollout update.

Status: **milestones 1 and 2 done, on CPU, plus the device layer.** `src/sjfpo_jax/`
now holds `fpo_core.py`, `dct_utils.py`, `gae.py`, `networks.py` and `runtime.py`,
with `tests/test_parity.py` (44), `tests/test_network_parity.py` (21) and
`tests/test_runtime.py` (22) holding all of it against the torch oracle and
against a CPU box. `policy.py` and upward are still unwritten, and no training
process imports jax yet. **No GPU exists on the development box** (no
`nvidia-smi`, `torch.cuda.is_available()` is False, jax is CPU-only), so nothing
here has been measured on Blackwell and the §0b blockers are suspended, not
cleared. Every file:line below was re-read against the current `jax` branch
(a7a65ef) while writing this revision.

Decisions taken (2026-09-30, unchanged):

| # | Decision | Choice |
|---|---|---|
| 1 | Env strategy | JAX learner + CPU gymnasium/MuJoCo envs |
| 2 | Network style | Flax NNX modules |
| 3 | Layout | new `src/sjfpo_jax/`, torch kept as oracle |
| 4 | Checkpointing | Orbax `CheckpointManager`, new format |

## 0. What moved under the plan since 2026-09-30

The decisions hold. The code they were written against did not. Every row below
is a live reference in the rest of this document that had to change.

| # | Was | Now | Where it bites |
|---|---|---|---|
| 1 | `PROFILER.md` | `BLACKWELL.md` (renamed 233c839) | §0, §1 |
| 2 | `chunks_per_update=256`, `num_learning_epochs=4`, `num_mini_batches=2` → 8 microbatches/update | `1024`, `2`, `8` → **16 microbatches/update** | §1, §4, §10, §11 |
| 3 | CPU reference 13.71 s/update = one default update | that update was 4096 steps; a default update is now `1024 × 16 = 16384` steps (`train_sjfpo.py:426`) | §1 |
| 4 | collector: per-env Python loop, per-step window slide, per-wavefront allocation | already rewritten: ring cursor `wpos` + `roll_table`, preallocated `plan`/`reward_windows`/`step_frames`, three batched scatters per vec step (`collector.py:697-745, 816-826`), measured 3.0–3.9× faster (`BLACKWELL.md` A4) | §9, §15 (`advance_windows` is dead) |
| 5 | drift diagnostics: a critic forward per optimizer step | one critic forward per **rollout**, both metrics from it, gated on `tracker.enabled` (`train_sjfpo.py:203-228, 540-542`, 891e7f6) | §1, §10 |
| 6 | `max_grad_norm = critic_max_grad_norm = 200.0` (inert) | both `1.0`, and `critic_max_grad_norm=None` is a supported log-only mode (8ab20b7) | §5 |
| 7 | `src/sjfpo/` | `src/spectral_joint_policy/`; hyperparameters in `configuration_sjfpo.py`, and `config.py` is the **env registry** | §2, §3 |
| 8 | `configs/sjfpo.yaml` read by `scripts/train.py` | nothing reads it at runtime; it is a `--config` flat overlay, opt-in, below the CLI (`scripts/train.py:126-146`, and the file's own header) | §2 |
| 9 | `use_clipped_value_loss=False` | `True`; value loss is Huber over the member axis with an rsl-rl clipped branch (`modeling_sjfpo.py:72-82, 1204-1226`) | §3, §10 |
| 10 | `use_dct_transform` off | `True`, `num_kept_tokens == action_horizon == 16`, so the DCT is the full orthonormal transform and a measured no-op on the loss (1.2e-07) | §3, §11 |
| 11 | 8 test modules | 11 (`tests/`) | §2, §11 |
| 12 | torch checkpoint holds every weight | `save_policy_checkpoint` **drops** `old_actor.*`, `old_conditioner.*`, `target_critic.*`; `ema.pt` is separate | §8 |
| 13 | milestone 1 is a plan | shipped: `src/sjfpo_jax/{fpo_core,dct_utils,gae}.py` + `tests/test_parity.py`, 44 tests green on CPU | §4, §11, §15 |
| 14 | milestone 2 is a plan | **networks** shipped: `src/sjfpo_jax/networks.py` + `tests/test_network_parity.py`, 21 tests green on CPU | §3, §6, §11, §15 |
| 15 | single device | a multi-GPU box is **data-parallel by env**: one process per GPU, each owning `num_envs / N` envs, gradients averaged across processes. The env is the unit of GAE coupling, so every GPU still holds whole trajectories, and no sharding enters `SJFPOState` | §1, §8, §10 |
| 16 | `jax.nn.rms_norm` is a port target | **it does not exist in jax 0.11.2.** `src/sjfpo_jax/networks.py` carries a three-line `rms_norm`, because flax ships RMSNorm only as a Module and a Module cannot hold the critic's stacked `[num_critics, hidden]` weight | §3 |
| 17 | x64 is off by default in JAX | left off, the `RunningMeanStd` count silently truncates to fp32 and dies at 2^24 frames. `sjfpo_jax/__init__.py` sets `jax_enable_x64`, and **every** other array pins `dtype=jnp.float32`, because x64 also changes the default float type | §3, §7 |
| 18 | the learner picks its own device | `src/sjfpo_jax/runtime.py` owns device detection, per-process device pinning, the fp32/TF32 policy, and the donation decision. One process per GPU, assigned by `SJFPO_GPU_INDEX` or `LOCAL_RANK` | §0c, §0d, §0e |
| 19 | `jax.struct.PyTreeNode` exists | **it does not.** `struct` is gone from jax 0.11.2; `SJFPOState` is a `flax.struct.dataclass`, with `global_step` marked `field(pytree_node=False)` so a Python int is not a traced leaf | §11b |
| 20 | milestone 2b is a plan | **steps 1-5 shipped**: `src/sjfpo_jax/{state,policy}.py` + `tests/test_policy_parity.py`, 26 tests green on CPU. Steps 6-9 not started | §11b |

Environment today: torch 2.11.0+cu130, jax **0.11.2**, flax 0.12.10, optax 0.2.8,
orbax-checkpoint 0.12.6. The last four are already declared in `pyproject.toml`
(`jax[cuda13]`, `flax>=0.10.6`, `optax>=0.2.8`, `orbax-checkpoint>=0.11.24`), so
§0.1 is a *resolution* problem, not a dependency problem.

---

## 0b. Blockers before any code

| # | Blocker | Check | If it fails |
|---|---|---|---|
| 1 | jax has no CUDA device. `.venv` has jax 0.11.2, `jax.devices()` → `[CpuDevice(id=0)]`, and the CUDA plugin fails with `cuInit(0) ... error 303` on this box | install the extra matching the torch build (`pyproject.toml` already asks for `jax[cuda13]`, matching torch's `cu130`; BLACKWELL.md precondition 3 notes `uv lock` warns the cuda13 extra is unresolvable and silently drops it) | port is pointless, stop |
| 2 | Blackwell profile not taken. The CPU split says backward+opt = 61.4%, and `BLACKWELL.md` marks that whole table **stale** (collector 3.0–3.9× faster, drift now gated → 0%) | run `BLACKWELL.md` Step 2 on the GPU first | if env+rollout > 60% on GPU, the learner is not the bottleneck; fix the collector instead |
| 3 | CPU env throughput floor unknown | measure raw `AsyncVectorEnv.step` at `num_envs ∈ {64, 256, 1024}` | if the floor is < 2000 fps, 30M steps is unreachable regardless of framework |

`BLACKWELL.md` explicitly says "Do not carry the CPU conclusion over." Milestone 0
is that measurement, not the port.

## 0c. The two GPU boxes, and what to run on them

The port targets **T4 (sm_75)** and **Blackwell** boxes. They disagree about
what fp32 even costs, so the parity recipe differs.

| | T4 (sm_75) | Blackwell (sm_100+/sm_120) |
|---|---|---|
| bf16 tensor cores | **no** | yes |
| TF32 | **no** | yes |
| VRAM | 16 GiB | varies, 96 GB on the RTX PRO 1000 |
| torch oracle storage | fp32 (`is_bf16_supported()` is False) | bf16, gated on the device name (`utils.py:55-66`) |
| torch oracle autocast | off | bf16 (`utils.py:92-101`) |
| JAX port | fp32 | fp32 |
| parity reachable at 1e-6 | **yes**, both sides are fp32 | **only with `SJFPO_DTYPE=fp32`** |
| default update | 16 microbatches, 1.560 GiB fp32 activations per `BLACKWELL.md` A2 -- fits 16 GiB | the same, with room to spare |

```bash
# T4 box: nothing to set, both sides are already fp32
.venv/bin/python -m pytest tests/test_parity.py tests/test_network_parity.py -q

# Blackwell box: force the torch oracle to fp32 or you are comparing bf16 to fp32
SJFPO_DTYPE=fp32 .venv/bin/python -m pytest \
    tests/test_parity.py tests/test_network_parity.py -q

# Blackwell box: confirm the two dtypes the torch code branches on
.venv/bin/python -c "
import torch, sys; sys.path.insert(0, 'src')
from spectral_joint_policy.utils import is_rtx_1000_pro_blackwell, resolve_storage_dtype, resolve_compute_dtype
print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())
print('blackwell', is_rtx_1000_pro_blackwell(),
      '| storage', resolve_storage_dtype('cuda'), '| compute', resolve_compute_dtype())
"
```

bf16 in JAX stays milestone 7. It is not a portability question: fp32 is the only
setting that is *identical* on both boxes, so a bf16 experiment has to be a
deliberate fork rather than a config value.

## 0d. Multi-GPU: data-parallel by env

One process per GPU. Each owns `num_envs / N` envs and runs its own rollout,
GAE and update; gradients are averaged across processes before one optimizer
step.

```
GPU0: envs 0-15    rollout -> GAE -> update_step -> grads ---\
GPU1: envs 16-31  rollout -> GAE -> update_step -> grads ----+-- all-reduce mean -> step
```

| why |
|---|
| The env is the unit of coupling. A chunk's GAE chains to the next chunk of the *same* env, so splitting by env keeps every trajectory whole. Splitting by chunk would cut traces in half. |
| `gae_linked_frac` is then a property of the config, not of the GPU count. It runs 0.938 at `num_envs=64`. |
| No sharding in `SJFPOState`, no `pmap`, no `in_axes` for params. The §12 `pmap` trap stays avoided for a reason rather than by luck. |
| `num_envs` is a **total**, not a per-GPU count. `chunks_per_update` stays global, so a 4-GPU box collects the same 1024 chunks per update as one GPU, just across four processes. |

The cost is real and should be measured, not assumed: each GPU runs its own
env workers, so a 4xT4 box wants 4x the host cores. If fps per GPU falls as
`N` grows, the host is the bottleneck and `num_envs / N` per process is the knob
(`BLACKWELL.md` Step 9).

## 0e. `runtime.py`: what "auto-detect the device" actually means

JAX optimises most of this for free -- XLA fuses, `lax.scan` lowers to a loop,
the update step compiles once. What is *not* free is **choosing a device** and
**deciding what fp32 means on it**. `src/sjfpo_jax/runtime.py` owns exactly
those, as pure functions of a device so the logic is testable on a CPU box.

| function | decision | T4 (7.5) | Blackwell (10.0/12.0) | CPU |
|---|---|---|---|---|
| `detect_backend()` | `'cuda'` or `'cpu'` | cuda | cuda | cpu |
| `device_index_from_env()` | which GPU this process owns | `SJFPO_GPU_INDEX`, else `LOCAL_RANK`, else 0 | same | 0 |
| `select_device()` | one device per process, or raise | -- | -- | -- |
| `matmul_precision_for()` | pin `'highest'` where fp32 could silently be TF32 | unset | **`'highest'`** | unset |
| `should_donate()` | donate buffers into the optimizer step | True | True | False |

```bash
# four processes, four GPUs, 64 envs in total
for i in 0 1 2 3; do
  SJFPO_GPU_INDEX=$i .venv/bin/python scripts/train.py --method sjfpo_jax \
      --env Hopper-v5 --sjfpo.num_envs 64 --out_dir ./tmp/gpu$i &
done
```

Three things it deliberately does *not* do:

- **No `.to(device)` calls.** JAX arrays live where XLA put them; the one real
  decision is which device the process commits to.
- **No fp32/bf16 switch.** That is milestone 7, and it is a fork with a measured
  comparison, not a config value (§7, §0c).
- **No env-count or thread budgeting.** `num_envs` belongs to the trainer, where
  the per-process split happens.

**Unverified on this box:** every CUDA branch. `compute_capability` is a
CUDA-only attribute (a `getattr` with a `None` default, or CPU raises), and
whether donation actually pays is milestone 5.

## 1. The numbers the port has to beat

CPU reference (`BLACKWELL.md` Step 2): **299 fps**, measured on torch 2.11.0 at
`num_envs=64` with 4096-step rollouts. The per-update figure (13.71 s/update)
belongs to a `chunks_per_update=256` config that no longer exists; the default
update collects 16384 steps, so at the same fps one update would cost 54.8 s.
The rate is the number that survives.

| Budget | At 299 fps | Target | Required speedup |
|---|---|---|---|
| 30M steps | 27.9 h | 4 h | ≥ 7.0× |
| 100M steps | 92.9 h | 12 h | ≥ 7.7× |

Where the 299 fps goes on CPU. **Stale, re-measure** (`BLACKWELL.md` Step 2):
the collector share is well under 6.4% after 100f66a, and the diagnostics share
is 0 with `report_to=none`.

| segment | % of wall | now |
|---|---|---|
| `collect_chunks_vec` | 6.4 | too high, collector rewritten |
| `generate_rollout` minus collect | 9.5 | |
| `update_forward` | 22.7 | |
| **`backward` + optimizer** | **61.4** | the target |
| drift diagnostics | 1.1 | 0, gated on `tracker.enabled` |

### What JAX actually removes

| win | mechanism | est. share of 299 fps |
|---|---|---|
| 16 separate fwd/bwd per update (2 epochs × 8 minibatches) become one `lax.scan` | 16 Python traversals + 16 launch batches → 1 | large part of the 84.1% |
| one rollout-level critic forward for drift (2 metrics) becomes tree reductions in the same program | deletes a full critic pass over `2·n_chunks` windows (`train_sjfpo.py:203-228`) | the 1.1%, now 0 |
| batch H2D once per rollout instead of once per minibatch: `sample_minibatch` slices ~12 tensors and `forward` re-issues `to_device_async` per key | one transfer, not 16 | small |
| the 16-step Euler integration per chunk (`fpo_core.py:sample_action`) becomes one compiled scan | `BLACKWELL.md` A1 counts 2006 dispatches per `predict_action_chunk` | **this is the largest single dispatch cost in the repo** |
| donated `params` / `opt_state` | no alloc per step | wall-clock only, not VRAM |
| no bf16 autocast bookkeeping (§7) | removes the dtype casts and their graph nodes | small, compounds in compile time |

### What JAX does not remove

- CPU MuJoCo `env.step`. This is the hard ceiling and it is unchanged by the port.
- `num_mc_samples=16` × `batch_size=128` = 2048 flow evaluations per minibatch,
  × 16 microbatches per update.
- The Python `for i in range(num_envs)` chunk-close loop and the awaiting-reset
  state machine in the collector (`collector.py:831-907`). `env.step` is a Python
  call; that loop cannot be vmapped.

**Consequence to accept up front:** after the port, the env is the bottleneck.
The learner becomes cheap. That is the goal, but it means the next scaling lever is
MJX (deferred — decision 1), not more JAX learner work.

## 2. Target layout

```
src/sjfpo_jax/
  __init__.py
  configuration_sjfpo.py  # plain dataclass, mirrors SJFPOConfig field-for-field
  networks.py             # NNX: SwiGLU, CondMLP, SinusoidalEncoding, SelfAttention,
                          #       DiTBlock, SJFPOFlowModel, StateConditioner, Critic,
                          #       RunningMeanStd, modulate
  fpo_core.py             # sample_timesteps, compute_freq_weighted_std, sample_noise,
                          # run_monte_carlo_sampling, compute_cfm_loss,
                          # compute_aspo_surrogate, compute_policy_ratio, sample_action
  dct_utils.py            # get_analysis_basis, extract_dct_coeffs,
                          # compute_spectral_energy_retained,
                          # velocity_to_action_steps, decode_coeffs_to_velocity
  gae.py                  # compute_chunk_advantages
  state.py                # SJFPOState (struct.PyTreeNode), build_optimizer
  policy.py               # forward (loss), predict_action_chunk, EMA, target critic
  collector.py            # gymnasium loop + jitted tensor bookkeeping
  runtime.py              # device detection, precision policy, device pinning
  checkpointing.py        # Orbax save/restore
  train.py                # train_sjfpo_run
tests/
  test_*_jax.py           # mirrors of the 11 existing test modules
  test_parity.py          # torch vs jax, same inputs
```

Torch `src/spectral_joint_policy/` stays untouched, and is the parity oracle.
`src/sjfpo_jax/` must not import `lerobot` or `torch`: `lerobot` is a torch
library, and importing it drags the whole torch runtime into the JAX process.

Two consequences of that constraint, both of which change the mirror:

1. `SJFPOConfig` is a LeRobot `PreTrainedConfig`, and the policy reads its dims
   off `input_features` / `output_features` as `PolicyFeature` objects
   (`modeling_sjfpo.py:196-215`). A plain dataclass cannot mirror that. Take
   `obs_dim: int` and `action_dim: int` as fields instead — `scripts/train.py`
   already resolves both before dispatch (`:246-258`). Drop the LeRobot-only
   fields (`repo_id`, `push_to_hub`, `private`, `tags`, `license`, `use_peft`,
   `hidden`, `input_features`, `output_features`). Everything that is real
   hyperparameter mirrors field-for-field, including the properties
   `num_mini_batches` and `num_flow_tokens`, which the trainer reads.
2. The env registry is reusable as-is: `spectral_joint_policy.config` reads
   `configs/envs.yaml` with PyYAML and imports nothing from torch.
   `__init__.py` is empty, so importing that submodule does not pull torch either.

`configs/sjfpo.yaml` is a documentation mirror of the defaults. `scripts/train.py`
reads it only when `--config` is passed, as a **flat overlay below the CLI**, and
its own header says "Nothing reads this file at runtime". Keep that file as-is
and keep it out of the JAX path; the JAX entry takes `--sjfpo_jax.*` flags and
dataclass defaults like the torch one does.

Train entry: branch in `scripts/train.py`, not a second script. `TrainConfig.method`
is `Literal["sjfpo"]` (`scripts/train.py:81`) and `run_single` imports
`train_sjfpo_run` directly (`:219-262`); add `"sjfpo_jax"` to the Literal and one
dispatch arm. The dim check against the live env space, the run-dir resolution
(`<method>_<env>_seed<seed>`), the resume handling and the exit codes are all
already there and should not be duplicated.

## 3. torch → JAX API map

Line cites are `src/spectral_joint_policy/`.

| torch | jax / flax | notes |
|---|---|---|
| `nn.Linear` | `nnx.Linear(in, out)` | NNX kernel is `[in, out]`; the checkpoint converter must transpose torch's `[out, in]` |
| `nn.init.normal_(std=0.02)` on `token_pos` | `nnx.Param(0.02 * jax.random.normal(k, shape))` | `networks.py:540-541` |
| zero-init (`adaLN.fc2`, `adaLN_out`, `out`) | `nnx.Param(jnp.zeros(...))` | `networks.py:358-359, 548-552`; keeps adaLN-Zero identity at step 0 |
| `F.silu` | `jax.nn.silu` | verified |
| `F.rms_norm` under `autocast(enabled=False)` (`_RMSNorm`, `networks.py:34-58`) | **none -- write it out** | **verified: `jax.nn.rms_norm` does not exist in jax 0.11.2.** flax ships RMSNorm only as a Module, and a Module cannot hold the critic's stacked `[num_critics, hidden]` weight, so `src/sjfpo_jax/networks.py` carries `rms_norm(x, scale)` in three lines: one mean, one reciprocal square root, one scale |
| `F.scaled_dot_product_attention` with a `[B,1,1,L]` bool mask | `jax.nn.dot_product_attention(q, k, v, bias=...)` | **verified, and the layout is the opposite**: jax takes `(batch, seq, heads, depth)` where torch takes `(batch, heads, seq, depth)`, so the port keeps the `(b, l, h, d)` reshape and drops the two transposes torch needs. Mask polarity is also opposite: `keep` `[B, L]` -> `bias = where(keep[:, None, None, :], 0.0, -inf)`, and `keep[:, 0]` is forced True to kill the all-masked NaN row (`networks.py:212-217` already forces it) |
| `gate_up(x).chunk(2, dim=-1)` | `jnp.split(..., 2, axis=-1)` | fused GEMM, keep it (`networks.py:83`) |
| `qkv(x).chunk(3, dim=-1)` | `jnp.split(..., 3, axis=-1)` | `networks.py:206` |
| `adaLN_out(c).chunk(2, dim=-1)` | `jnp.split(..., 2, axis=-1)` | `networks.py:647, 667` |
| `parts.unbind(dim=1/2/3)` on the 6-way adaLN | `jnp.split(..., 6, axis=...)` | three layout branches in `DiTBlock.forward` (`networks.py:388, 409, 417`): `[B,K,D_a,H]`, `[B,K,D]`, `[B,D]`. Static shapes, so all three trace |
| `torch.lerp(noise, action, t)` (`compute_mc_xt`, `modeling_sjfpo.py:193`) | `jnp.lerp` | 1:1 |
| `torch.addcmul(a, b, c)` (`modulate` `networks.py:332`, Euler step `fpo_core.py:337`, postprocessor `processor_sjfpo.py:127`) | `a + b * c` | no `addcmul` in `jnp`; the fusion is the reason the torch version exists, so this is a real (small) regression — note it |
| `einx.id("n d -> n m d", ...)` then reshape (`modeling_sjfpo.py:741-744, 1130-1133`) | `jnp.broadcast_to` + `reshape` | two sites only |
| `nn.ModuleList` of `num_critics` `StateConditioner`s + `heads` (`networks.py:710-715`) | stacked `nnx.Param` with a leading member axis + `jax.vmap` | **required**, see §6. Both `conditioners` and `heads` move |
| `nn.ModuleList` of `dit_depth` `DiTBlock`s (`networks.py:543`) | Python loop is fine | `dit_depth=2` is static, so this unrolls at trace time. Do not stack it; only the critic's per-member cost and key churn justify the churn |
| `torch.no_grad` / `inference_mode` (30+ sites) | `jax.lax.stop_gradient` + a separate jitted no-grad fn | there is no inference mode |
| `.detach()` | `jax.lax.stop_gradient` | |
| `.float()` / `autocast(enabled=False)` | `.astype(jnp.float32)` | JAX does **not** auto-promote int→float. Every torch `.float()` needs an explicit cast or you get an int matmul |
| `.to(dtype=torch.float32)` on `raw_returns` / `raw_values` | `.astype(jnp.float32)` | these stay fp32 on purpose, `modeling_sjfpo.py:951-958` |
| `.std(correction=0)` (advantage whitening `modeling_sjfpo.py:930-935`, `compute_info`) | `jnp.std(..., ddof=0)` | the unbiased default is NaN at one chunk |
| `torch.var_mean(x, dim=0, unbiased=False)` (`RunningMeanStd.update`) | `jnp.var(..., ddof=0)` + `jnp.mean` | the count is **fp64**; keep it fp64 |
| `F.huber_loss(..., reduction="none")` (`compute_value_huber_loss`, `modeling_sjfpo.py:72-82`) | `jnp.where`-form Huber | no `jnp` Huber; 3 lines |
| `torch.randperm` (`generate_mini_batch_indices`, `train_sjfpo.py:41-58`) | `jax.random.permutation` | the split bounds `(i*n)//n_mb` are static Python ints; only the permutation moves into the program |
| `torch.randn` / `torch.rand` | `jax.random.normal` / `jax.random.uniform` | `fpo_core.py:15-56, 70-98` |
| `torch.Generator` (threaded as `None` at every call site today) | `jax.random.key` (typed) + `fold_in`, threaded in `SJFPOState.key` | typed keys, not `PRNGKey`. Verified in the JEP 9263 docs |
| `torch.argsort(stable=True)` | `jnp.argsort(env_id, stable=True)` | `gae.py:89` and again in `compute_gae_diagnostics` (`modeling_sjfpo.py:107`) |
| `torch.linalg.vector_norm` (`utils.rms`) | `jnp.linalg.norm` | |
| `torch.clamp(x, max=c)` | `jnp.clip(x, max=c)` | **must be `clip`, not `minimum`** — see §7 |
| `torch.where` / `einsum` / `rsqrt` / `square` / `maximum` / `abs` | `jnp.where` / `einsum` / `rsqrt` / `square` / `maximum` / `abs` | 1:1 |
| `copy.deepcopy` for `old_actor` / `old_conditioner` / `target_critic` (`modeling_sjfpo.py:232-243`) | a second pytree key in `SJFPOState` | `snapshot_behavior_policy` then `load_state_dict`s into them (`:315-328`), so the JAX version copies trees |
| `torch._foreach_mul_` / `_foreach_add_` for the actor EMA and the Polyak target critic | plain `jax.tree.map` arithmetic | `modeling_sjfpo.py:330-370, 372-405`; the foreach was a torch launch-count optimisation with no JAX analogue and no reason to exist |
| EMA shadow keyed by parameter **name** (`modeling_sjfpo.py:241, 344-350`) | keyed by pytree path | `swap_in_ema` / `swap_out_ema` (`:412-433`) return the displaced weights so eval can run on the average |
| `optimizer.step()` (one AdamW, 4 param groups: actor decay, actor no-decay, conditioner, critic) | see §5 | per-role clip is the constraint, not per-role lr |
| `clip_grad_norm_` (returns the pre-clip norm) | custom 6-line optax transform, §5 | `optax.clip_by_global_norm` does not surface the norm |
| `critic_max_grad_norm is None` → log-only via `_foreach_norm` + `vector_norm` (`train_sjfpo.py:501-513`) | compute the norm, skip the clip | a supported config, not an edge case |
| Orbax-less HF/`safetensors` save | `ocp.CheckpointManager` + `ocp.args.StandardSave(...)` | see §8 |
| `lru_cache` device-keyed freq tables and DCT basis (`networks.py:114-127`, `dct_utils.py:28-42`) | module-level `jnp` constants built once | no device keying needed; one device |
| `torch.rsqrt` | `1.0 / jnp.sqrt(...)` | `jnp.rsqrt` does not exist in jax 0.11.2 |
| x64 off | `jax.config.update("jax_enable_x64", True)` in `sjfpo_jax/__init__.py` | the `RunningMeanStd` count is fp64 and JAX truncates a float64 request with only a warning. **x64 also changes the default float type**, so every other array must pin `dtype=jnp.float32` or the model silently runs in float64. Same trap in the math: `gamma ** int_offsets` takes the default float type, so `gae.py` casts the offsets to fp32 first |
| `torch.autocast(bf16)` | **do not port** | see §7 |
| `torch.compile`, CUDA graphs | no analogue | see §13 |

## 4. The chunk-GAE port (`gae.py`)

Current code is 1 gather + 1 stable sort + a **Python loop** over `n_chunks`
(`gae.py:110-115`). JAX has no Python loop, so:

1. `chunk_rewards`: unchanged, still a `[B, chunk_size]` gather + `gamma**k`
   weights (`gae.py:78-85`). Note `starts` is used here and is **not** part of
   the sort; `lengths` is.
2. `order = jnp.argsort(env_id, stable=True)`; gather `lengths`, `env_id`,
   `terminated`, `truncated`, `chunk_rewards`, `chunk_value`, `chunk_next_value`
   by `order` (`gae.py:89-96`). Unchanged semantics.
3. `same_env_sorted`, `gamma_h`, `delta`, `continuation_factor` — all
   vectorized already (`gae.py:95-108`).
4. Replace the loop (`gae.py:110-115`), and keep the unsort the torch version
   does at `gae.py:116`:

```python
# c[i] is the coefficient on A_{i+1}; c[-1] stays 0, which kills the tail
# exactly like continuation_factor[-1] in the torch version.
def step(next_advantage, delta_c):
    carry = delta_c[0] + delta_c[1] * next_advantage
    return carry, carry

_, advantages = lax.scan(
    step, jnp.float32(0.0), (delta, continuation_factor), reverse=True
)
advantages = advantages[unsort]
return advantages, advantages + sorted_values[unsort]
```

**Do not pre-reverse the inputs.** `reverse=True` already reverses `xs` and
flips `ys` back, so `(delta[::-1], continuation_factor[::-1])` reverses twice and
chains each env forward instead of backward. That bug produces plausible-looking
advantages that are wrong for exactly the chunks that start a trajectory, and
the parity test below is what catches it.

`lax.scan(..., reverse=True)` is verified in the JAX control-flow docs and is
reverse-mode differentiable, so the value-loss path is unaffected.

Also port, do not re-derive: the `[0, 1]` validation on `gamma`/`lam`, the
length-agreement `ValueError`s, and the empty-rollout early return
(`gae.py:44-76`) — all Python branches on static shapes, so they are fine outside
`jit`.

`n_chunks ≈ 1024` at the current default (`chunks_per_update`), 16384 executed
steps per rollout. `lax.associative_scan` would turn the O(n_chunks) recursion
into a work-efficient parallel scan, but the API (custom associative operator) is
unverified. **Do not use it.** Revisit only if `chunks_per_update` grows past
~4096.

`compute_gae_diagnostics` (`modeling_sjfpo.py:95-143`) sorts the same `env_id`
again and packs 5 floats into one `.tolist()` so a rollout pays one D2H, not
five. In JAX the sort comes free from `compute_chunk_advantages`; keep the
5-key packing contract, because the trainer indexes it by name
(`train_sjfpo.py:585-598`).

## 5. Optimizer: keeping the per-role clip

`build_optimizer` (`modeling_sjfpo.py:160-176`) uses **one** AdamW (`fused=True`
on CUDA) over four param groups, because the critic's gradient norm is 45–170× the
actor's and a shared clip would let the critic own the actor's step. `optax.adamw`
is one `GradientTransformation` over one pytree, so per-role clip norms need an
explicit split:

```python
def global_norm_and_clip(max_norm):
    """optax.clip_by_global_norm, but stashes the pre-clip norm in the state.

    The training loop logs actor/critic grad norms separately, and
    optax.clip_by_global_norm discards them.
    """
```

then

```python
adamw = optax.adamw(lr=policy_lr, weight_decay=optimizer_weight_decay)
updates, opt_state = adamw.update(grads, opt_state, params)
```

with `grads` shaped `{"actor": clip_actor(actor_grads), "critic": clip_critic(critic_grads)}`
before the AdamW. The pre-clip norms come back from the two clip states. This
reproduces `train_sjfpo.py:492-524` exactly and is ~10 lines.

Three contract details in that block that a naive port drops:

1. **Actor and conditioner share `policy_lr`; the critic has its own `critic_lr`**
   and `weight_decay=0.0` (`modeling_sjfpo.py:460-495`). Two lrs inside one
   optax chain is fine — chain the per-role transforms before `adamw`.
2. **Weight decay skips `adaLN*` and `out.*` in the actor** (`:465-472`). Port as
   a mask on the decay term, or as two chained `optax` transforms over disjoint
   subtrees.
3. **`critic_max_grad_norm=None` means "measure, do not clip"**
   (`train_sjfpo.py:501-513`). Keep it a supported config.

`grad_accum_steps`: `optax.MultiSteps(adamw, every_k_schedule=...)` gives the same
"flush the final partial group" behaviour the loop hand-rolls today
(`train_sjfpo.py:471-492`) — except `MultiSteps` needs a schedule, and the torch
loop's real contract is "average within a group of `grad_accum_steps`, clip and
step at the boundary". At the default `grad_accum_steps=1` this is a no-op.
**Recommendation: drop `grad_accum_steps` from the JAX config and require the
batch to fit.** If it is ever needed, add `MultiSteps` then. Do not port a
feature the default config disables.

## 6. Networks: the two structural changes

**(a) Critic becomes a `vmap`, not a `ModuleList`.** `Critic.forward`
(`networks.py:717-742`) stacks `num_critics` independent `StateConditioner`s, each
followed by its own linear head. Keeping `ModuleList` means a Python loop in the
traced loss graph. Stack the params on a leading member axis and `vmap` the member
function. This is the "struct-of-arrays, no Python loops inside `jit`" rule that
JAX forum advice converges on (jax discussion #23200 — a Python loop over a list
of train states caused compile times from minutes to 12 hours).

Shipped, and the implementation has three details the design above did not
anticipate:

* **Every module's forward is a module-level function of `(params, x)`**, and
  `__call__` extracts its own state with
  `nnx.to_pure_dict(nnx.state(self, nnx.Param))` and calls it. Without that, the
  actor path and the vmapped critic path need two implementations of the
  conditioner. It is also why the params are plain `nnx.Param` dicts rather than
  `nnx.Linear` submodules: vmap needs the members reachable as one stacked pytree.
* **`jax.vmap` maps the params pytree only; `states` is closed over.** Every
  critic member sees the same window batch, so the input has no member axis.
  Mapping both is not merely wasteful, it raises: two mapped axes of different
  lengths are an error.
* **Map a pytree, not a leaf list.** `nnx.vmap` on a `Module` class stacks the
  state correctly but does *not* vmap the call, so the forward runs a plain GEMM
  against a `[num, in, out]` kernel and dies in `dot_general`. The shape that
  works is `jax.vmap(lambda params: fn(params, x))(params_of(module))`.

Consequence: the checkpoint key layout changes from
`critic.conditioners.0.frame_proj.kernel` to `critic.conditioners.frame_proj.kernel`
with a leading member axis, and `critic.heads.0.weight [1, H]` to
`critic.head.kernel [num, H, 1]` -- a transpose as well as a stack, because torch
carries the singleton output axis. The converter in §8 handles it;
`tests/test_network_parity.py` carries the mapping in miniature.

`SJFPOFlowModel.blocks` (`networks.py:543-546`) is a `ModuleList` too, but
`dit_depth=2` is static and the loop unrolls at trace time. Leave it a Python
loop; stacking it buys nothing and costs readability.

**(b) `nnx.Module` or functional pytrees at the train-step boundary.** Build the
networks as `nnx.Module` (readable, 1:1 with `networks.py`) but drive them from
the jitted step through the functional API:

```python
graphdef, state = nnx.split(policy)
# inside the jitted step:
model = nnx.merge(graphdef, state)
```

Do **not** use `@nnx.jit` / `nnx.Optimizer`. Two known issues: `nnx.Variable`
implements `__jax_array__`, which makes JAX transforms treat it as an Array and
fail intermittently (flax #4722, resolved by `.value`); and the flax maintainers
recommend the plain `jax.jit` + split/merge path for anything that crosses a
`lax.scan` boundary. Also: passing an `nnx.Module` as a *closure* into a
`lax.scan` is explicitly unsupported.

## 7. Precision: do not port bf16

`utils.py:55-101` selects bf16 storage and bf16 autocast on the RTX PRO 1000
Blackwell. Porting that into JAX is the highest-risk item in the plan:

- `jnp.sum` on bf16 silently upcasts to fp32, but **reductions that XLA
  auto-generates in the backward pass** (from broadcasts, i.e. every
  Linear-with-bias) run in bf16 and are numerically unstable — jax issue #30310.
- There is no autocast. The equivalent is
  `jax.config.update("jax_default_matmul_precision", ...)`, which is a *global*
  switch, not a scoped context. You cannot express "fp32 storage, bf16 matmul,
  fp32 reductions" per-region. (`scripts/train.py:configure_torch_cuda` sets
  `set_float32_matmul_precision("high")`, i.e. TF32, for the torch baseline; the
  JAX port leaves the global unset, which is the fp32-everywhere choice this
  section assumes. Measure the parity gap before turning it on.)
- The existing code already hard-codes fp32 regions, and they are the
  numerically load-bearing ones: `compute_cfm_loss` and the ASPO surrogate under
  `autocast(enabled=False)` (`modeling_sjfpo.py:744-748, 1171-1194`), `_RMSNorm`
  reducing in fp32 (`networks.py:34-58`), the DCT basis forced to fp32
  (`dct_utils.py:69, 100, 158, 174`), the collector's obs and reward tensors fp32
  end-to-end (`collector.py:43-46`), and `raw_returns` / `raw_values` pinned to
  fp32 after collection (`modeling_sjfpo.py:951-958`).

**Plan: the JAX port is fp32 throughout. No `SJFPO_DTYPE`, no
`jax_default_matmul_precision`.** Milestone 7 revisits this as a separate
experiment with the §1 harness as the arbiter.

Three precision traps that *will* bite during the port:

1. `compute_policy_ratio` (`fpo_core.py:270-296`) clamps `l_old` and `l_new` at
   `cfm_clamp` (`:285-286`) and then caps `delta` at `log_ratio_cap` through a
   one-sided straight-through estimator (`:294-295`). Those `torch.clamp(..., max=)`
   calls must become `jnp.clip`, not `jnp.minimum` — the whole
   `ABOVE_CLAMP_ABORT_UPDATES` machinery (`train_sjfpo.py:38, 601-611`) depends on
   the gradient being *exactly zero* above `cfm_clamp`.
2. The GAE comment (`gae.py:78-80`) is explicit: a cumsum-diff discount needs
   `gamma**-s_i`, which loses all fp32 precision past a few thousand steps. Keep
   the gather + `gamma**k` form. Do not "optimize" it to a `jnp.cumsum`.
   `BLACKWELL.md` Part C already records the closed-form cumprod/cumsum version as
   a measured wrong answer.
3. `RunningMeanStd.count` is fp64 because fp32 dies at 2^24 frames
   (`networks.py:4, 231-247`). Keep it fp64 in JAX and in the checkpoint.

## 8. Checkpointing

```python
manager = ocp.CheckpointManager(
    out_dir / "checkpoints",
    options=ocp.CheckpointManagerOptions(
        preservation_policy=ocp.preservation_policy.LatestN(2),
        temporary_path_class=atomicity.CommitFileTemporaryPath,
    ),
)
manager.save(step, args=ocp.args.StandardSave(checkpoint_pytree))  # async
```

`checkpoint_pytree` holds `nnx.state(policy)`, the optax state, the PRNG key,
`global_step`, `step_count`, and the EMA shadow and its `initialised` flag.

**What the torch oracle actually stores** (`checkpointing.py:74-145`), and what
the converter therefore has to synthesise or read separately:

| torch artifact | contents | note |
|---|---|---|
| `model.safetensors` | live `actor.*`, `conditioner.*`, `critic.*`, `obs_normalizer.*` | `old_actor.*`, `old_conditioner.*` and `target_critic.*` are **dropped** (`checkpointing.py:101-113`); the target critic is a Polyak shadow, rebuilt from the critic on load (`:235`) |
| `ema.pt` | decay, `initialised`, shadow params by name | separate file, not in the safetensors |
| `optimizer.pt` | one AdamW `state_dict`, 4 param groups | `load_optimizer` refuses a `format == 2` state, i.e. the pre-merge twin-optimizer era |
| `counters.json` | `{global_step, step_count}` | |
| `rng.pt` | torch + CUDA RNG state | |
| `config.json` | LeRobot `PreTrainedConfig` | the JAX converter reads it for dims and hyperparameters |

**Donation hazard:** if `params` is passed with `donate_argnums`, the donated
buffer is invalid after the call, and Orbax's async save still holds a reference
to it. With ~0.5–1M params (critic alone is 467,736 at obs 11, 640,954 at obs
348, per `BLACKWELL.md` Step 3) the fix is trivial: pass `out_shardings` and hand
Orbax a `jax.device_get` copy, or keep the params copy outside the donated
argument. Do not skip this — it is a silent wrong-checkpoint class of bug.

**Converter:** one script, `scripts/convert_torch_to_jax.py`, reading the existing
safetensors + `ema.pt` + `optimizer.pt` + `config.json` and emitting the Orbax
tree. It needs the §6(a) member-axis reshape for both `conditioners` and `heads`,
and the `[out,in] → [in,out]` linear transpose. Without it, every existing run is
a dead end. **Build the converter before the trainer.**

## 9. Collector: hybrid Python env, JAX bookkeeping

The env stays `gym.vector.AsyncVectorEnv`. The Python loop stays, because
`env.step` is a Python call. Everything else moves to JAX.

**The collector has already been rewritten** (`100f66a`, `collector.py:646-924`),
so most of what this section used to propose is done in torch and is a *port*,
not a *redesign*:

| already in torch today | jax |
|---|---|
| `windows [n, T, obs]` ring cursor `wpos`, chronological restore via `roll_table` (`collector.py:697-745, 865-869`) | same, one `jnp` array, `.at[:, wpos].set(...)`; the ring is the port, not a shift |
| `reward_windows [n, chunk_size]`, `step_frames [n, chunk_size, obs]`, `plan [n+1, chunk_size, act]` preallocated | identical shapes in `jnp`; `plan`'s extra zero row is what makes a non-stepping env read zeros, keep it |
| three batched scatters per vec step (`collector.py:816-826`) | one jitted scatter call, same three writes |
| `RolloutBuffer` growth via `ensure_chunk_capacity` and a previous-rollout capacity guess (`collector.py:128-133, 186-218`) | **gone**: preallocate at max capacity and let the chunk count be a Python int that slices the prefix. The torch growth path exists because `n_chunks` is data-dependent; in JAX the buffer lives on device and re-allocation is free enough not to be worth a growth path |
| `buffer.copy_to()` pinned staging + `torch.cuda.Event` (`modeling_sjfpo.py:858-875`) | gone; the buffer *is* on device. Deletes the pinned-memory and event machinery at `modeling_sjfpo.py:244-245, 798-800` |
| `actions_t.copy_(plan[row, pos_t])` then one `.cpu().numpy()` per vec step (`collector.py:802-807`) | one `jax.device_get` per vec step, batched over envs. This is the floor; do not chase the per-env copy |
| `propose_batch`'s `chunk.cpu()` once per chunk (`modeling_sjfpo.py:792-796`) | one `device_get` per chunk, unavoidable: the actions have to reach `envs.step` |
| `chunk_env_id` / wavefront bookkeeping, `awaiting` reset state machine, `validate_autoreset_mode` (`collector.py:626-643, 831-907`) | stays Python ints and Python control flow; the env loop cannot be vmapped. The `NEXT_STEP` handshake is a correctness contract, not an optimisation — port it literally |

What is left to win in JAX, honestly: the vec-step scatters become one kernel
instead of three small indexed writes, and the buffer never round-trips through
host. `BLACKWELL.md` A4 puts the collector at 114 ms/rollout on CPU after the
rewrite, down from 378 ms; the JAX port will not move that much again.

`collect_chunks_vec` was 6.4% of wall before the rewrite and is well under that
now. Port it in milestone 3, not milestone 1, and only if the Blackwell profile
still shows it above noise.

## 10. The update step

One `@jax.jit` per rollout, structured as brax PPO does it:

```
update_step(state, batch, key) -> (state, metrics)
├── lax.scan over epochs            (num_learning_epochs = 2)
│   └── lax.scan over minibatches   (num_mini_batches = chunks_per_update // batch_size = 8)
│       ├── behaviour CFM loss   (stop_gradient on old_actor / old_conditioner)
│       ├── v_new forward, v_star, CFM losses, ratio, ASPO surrogate
│       ├── critic forward, Huber value loss + clipped branch
│       ├── value_and_grad over the merged {actor, critic} param tree
│       ├── per-role clip + AdamW
│       ├── target_critic Polyak, actor EMA
│       └── diagnostics as tree reductions
└── metrics dict
```

16 microbatches per update at the defaults. Everything currently spread over
`train_sjfpo.py:426-527` becomes one program. The per-epoch
`jax.random.permutation` reshuffles for free inside the scan, so only the
proportional split bounds `(i*n)//n_mb` come from the host, and those are static
Python ints at trace time (`generate_mini_batch_indices`, `train_sjfpo.py:41-62`).

Faithfully port these, they are the algorithm and not decoration:

- `l_old` / `l_old_token` come from `generate_rollout`
  (`modeling_sjfpo.py:1010-1011`), computed once against the frozen snapshot.
  `forward` raises `KeyError` rather than recomputing them (`:1134-1138`), and the
  JAX rollout must produce the same two tensors or the ratio compares different
  things.
- The CFM loss mask and the velocity layout are returned together by
  `prepare_cfm_loss_inputs` (`:670-710`); they are only consistent with each
  other and the mismatch fails silently.
- The value loss is Huber over the member axis with `delta` scaled by the batch
  return RMS, then `max()` against the clipped branch (`:1204-1226`).
- `advantage_std_fraction`, `cfm_above_clamp_frac` and the per-token ratio
  diagnostics (`compute_info`, `modeling_sjfpo.py:1261-1436`) are 28 scalars
  packed into **one** `.tolist()`. Keep the packing; each `float()` is a D2H.
- `check_finite` becomes `jnp.all(jnp.isfinite(loss))` on device; the loop's
  `finite_error` OR across microbatches becomes one reduction in the scan.

**Over-jitting warning:** jax discussion #33912 reports the *outer* jit being 5×
*slower* than inner jits with `donate_argnums` on a large buffer, because XLA
stops proving safe in-place reuse and materialises a full copy. There is no large
donated buffer inside `update_step` (the batch comes in as an argument), so the
risk is low — but **measure**. Fallback if it regresses: jit the minibatch step
only, scan outside. Report which one won in the milestone.

## 11. Milestones

Each ends with a number, not a feeling.

| # | Milestone | Done when |
|---|---|---|
| 0 | Get jax onto the GPU; run `BLACKWELL.md` Steps 2+3 on Blackwell | wall-clock split on the real GPU recorded in `BLACKWELL.md`; the §0b blockers resolved. **Open: this box has no GPU, so everything so far is CPU-only** |
| 1 | `fpo_core.py`, `dct_utils.py`, `gae.py` ports + `test_parity.py` | **done**, CPU only: `src/sjfpo_jax/{fpo_core,dct_utils,gae}.py` and `tests/test_parity.py`, 44 tests, all at `rtol=1e-5, atol=1e-6`. Covers `compute_cfm_loss` (4 reductions × 3 divisors × masked and unmasked), `compute_aspo_surrogate` (ppo/spo/aspo), `compute_policy_ratio` (value *and* gradient, incl. exactly-zero gradient above `cfm_clamp` and the one-sided STE), `run_monte_carlo_sampling`, `sample_action` against a numpy Euler recursion, `sample_timesteps`/`sample_noise` statistics across all four timestep modes and both noise modes, all four DCT functions plus Parseval round-trip, and `compute_chunk_advantages` (wavefront interleave, termination vs truncation, variable lengths, permutation invariance, empty rollout, no fp32 loss at 4800 steps in). **Nothing here has run on a GPU** |
| 2 | `networks.py` port | **done**, CPU only: `src/sjfpo_jax/networks.py` and `tests/test_network_parity.py`, 21 tests at `rtol=1e-4, atol=1e-5`. Same weights in, same velocity / conditioning / critic value / `return_stats` dict out, across the three time layouts and `num_critics` in {1, 2, 3}; critic parameter count equals torch's; attention-mask parity including the all-masked row; the adaLN-Zero identity at init; gradients reach every member. `policy.py` is **not** started |
| 3 | `collector.py` port | rollout parity: given a fixed policy and fixed `envs.reset(seed=...)`, the JAX collector and the torch collector produce identical `chunk_state` / `chunk_actions` / `action_is_pad` / `step_rewards` on `Reacher-v5` with `num_envs=4`, across a terminal and an autoreset |
| 4 | `state.py` + `train.py` + the `scripts/train.py` branch | 1000-step run on `Reacher-v5`; eval return within noise of the torch run at the same seed; `fps` recorded |
| 5 | Throughput | **≥ 7.0× vs the 299 fps CPU reference, or ≥ 2000 fps, on a 4 h run.** If the env floor from §0b.3 is the limit, that number is the finding — report it and stop |
| 6 | `checkpointing.py` + converter | resume from a converted torch checkpoint reproduces the torch eval return; resume from a JAX checkpoint is bit-identical across 2 runs |
| 7 | (optional) bf16 experiment | only if milestone 5 is env-bound. Compare `jax_default_matmul_precision` against fp32 with §1's harness. Expect the §7 backward-reduction instability |

**Tolerances are not negotiable.** If a parity test fails at 1e-6, the JAX code is
wrong. Do not widen the tolerance to make it pass. The one documented exception is
milestone 2's forward parity at 1e-4, and only because fp32 GEMM accumulation order
differs.

Note the default config is a weak parity target in one place: `num_kept_tokens ==
action_horizon` makes the DCT a full orthonormal rotation and a no-op on the loss
(README, measured 1.2e-07). Run the DCT parity cases with `num_kept_tokens=8`, or
they prove nothing.

## 11b. Implementation plan: `policy.py` (milestone 2b)

Decisions settled for this step (2026-10-01): the loss entry is a **free
function** over a state pytree, the batch is a **dict with torch's keys**, the
live obs normalizer and action bounds live on **`SJFPOState`**, `cfg` is an
**explicit argument**, the metrics leave as a **dict of 0-dim arrays**, and the
loss **always** computes them.

Why a free function and not a module: `SJFPOPolicy` is a LeRobot
`PreTrainedPolicy`, and nothing in the JAX learner needs that inheritance. A
function is directly `jax.jit`-able and directly `jax.grad`-able, which is the
whole point of §10, and the parity test hands it the same dict it hands torch's
method. The torch class stays the oracle and is never imported by the trainer.

Why `cfg` is an argument rather than state: config is a constant, and putting it
in the state tree makes it look trainable to the optimizer and checkpointable to
Orbax. The trainer holds one and threads it.

Why the metrics are always computed: under `jit` `compute_info` is a **static**
choice, so torch's runtime flag would need two traced variants to keep. Under
`jit` the reductions are tree ops XLA fuses into the surrounding program, and one
traced graph is far easier to trust than two.

Work in the order below. Each step is independently testable and leaves the tree
green; do not start step *n+1* until step *n*'s gate passes.

| # | step | gate |
|---|---|---|
| 1 | `state.py`: `SJFPOState` only, no optimizer | builds, splits, round-trips |
| 2 | `policy.py` pure helpers | value parity, all four reductions |
| 3 | behaviour CFM loss (`l_old`) | parity with torch |
| 4 | `forward_loss`: the policy half | per-chunk parity at `rtol=1e-4` |
| 5 | `forward_loss`: the value half | loss parity **and** gradient parity |
| 6 | `compute_info` | 28 scalars + `cond_*`, parity |
| 7 | `predict_action_chunk` | parity, both noise modes, DCT on and off |
| 8 | EMA, Polyak target, behaviour snapshot | N-step parity |
| 9 | validation parity | every `ValueError`, both sides |

Steps 1-5 are this pass; 6-9 follow once the loss path is green.

**Steps 1-5 shipped** (2026-10-01): `src/sjfpo_jax/state.py`, `src/sjfpo_jax/policy.py`,
`tests/test_policy_parity.py` — 26 tests, all green on CPU. Four things the port got
wrong that the gates caught, all in `compute_value_huber_loss` or its inputs:

| what | why it is dangerous |
|---|---|
| the linear branch written as `\|x\| - 0.5*delta` | this build's `F.huber_loss` computes `delta * (\|x\| - 0.5*delta)`. The two agree **only when delta == 1**, so a probe at delta=1 cannot tell them apart. At the running config delta is the batch return RMS (~9), so the whole value loss was off by ~9x and read as a critic that cannot fit |
| the target broadcast as `raw_returns[:, None]` | `[n, 1]` against `[num_critics, n]`; the member axis has to be the broadcast one |
| `jnp.lerp` | does not exist; and the blend needs the 3-D `[B*M, K, D]` layout, not the flattened rows `run_monte_carlo_sampling` returns |
| the PRNG key on `SJFPOState` | a typed key is not a differentiable leaf, so `jax.grad(state)` refuses to run. The key is threaded as its own argument, which is what §10 already specified |

### Step 1 — `SJFPOState`, without the optimizer

A `struct.PyTreeNode` holding everything `forward_loss` reads. **No
`build_optimizer`, no `global_norm_and_clip`** — those are §5 and need the
per-role clip design, which is its own step.

```python
@dataclass  # flax.struct.dataclass: jax.struct is gone in 0.11.2
class SJFPOState:
    actor: tuple                 # NNX (graphdef, state), one per network
    conditioner: tuple
    critic: tuple
    old_actor: tuple             # frozen behaviour snapshot, runtime-only
    old_conditioner: tuple
    target_critic: tuple         # Polyak teacher, runtime-only
    obs_normalizer: RunningMeanStd
    action_low: jax.Array        # [action_dim], drives the unscale
    action_high: jax.Array
    global_step: int = field(pytree_node=False)   # static: not a traced leaf
```

**No `key` field.** A typed PRNG key is not a differentiable leaf, so keeping it
on the state makes ``jax.grad(state)`` refuse to run. It is threaded as its own
argument, which is what §10 already specified, and nothing in the loss needs it:
the MC draws arrive in the batch.

- **`tuple` for the networks, not the modules.** A module captured in a closure is
  the flax pitfall in §6(b); a `(graphdef, state)` pair is an ordinary pytree and
  `nnx.merge(*state.actor)` rebuilds the module at trace time. That is what makes
  `jax.grad(forward_loss)(state, batch)` work with no plumbing.
- **`obs_normalizer` and the bounds are state, not code.** Both are mutated once
  per rollout by the trainer and read inside the loss. They live and die with the
  step, not with the process.
- **`global_step` is `field(pytree_node=False)`.** A Python int traced as a leaf
  would make the whole graph a function of a counter.

### Step 2 — the pure helpers

Four functions with no parameters and no network. This is where the DCT mask
contract lives, and the easiest parity win in the port.

| function | mirrors | the trap |
|---|---|---|
| `get_token_count_for_cfm_loss(cfg)` | `modeling_sjfpo.py:645` | the token axis is the action-step grid when the mask moved the loss onto rows, and the frequencies otherwise. Not always `flow_shape[0]` |
| `prepare_cfm_loss_inputs(cfg, action_is_pad, *velocities)` | `:670` | mask and layout are returned **together** because they are only consistent with each other, and a mismatch fails silently |
| `compute_target_velocity(a, eps)` | `:507` | `a[:, None] - eps`, broadcast over the MC draws |
| `compute_value_huber_loss(pred, target, delta)` | `:72` | no `jnp` Huber: three lines, `delta` is 0-dim and scaled by the batch return RMS |

Gate: same value and same shape as the torch counterpart, including the
"expand the target over the member axis" behaviour.

### Step 3 — the behaviour CFM loss (`l_old`)

`compute_behavior_cfm_losses` (`modeling_sjfpo.py:712`) becomes
`compute_behavior_cfm_loss(state, states, actions, tau, eps, action_is_pad)`.

This is `l_old`, the numerator of the FPO++ ratio, computed **once per rollout**
against the frozen snapshot. `forward_loss` raises `KeyError` rather than
recompute it (`:1135`), and the port must keep that: recomputing per minibatch
silently compares two different policy snapshots.

Gate: parity given the same `tau` and `eps`. The same MC draws must feed both
sides — that is the entire point of the ratio, and fresh noise per side makes the
test meaningless.

### Step 4 — `forward_loss`, the policy half

The torch order, which is the order that matters:

1. validate shapes (`states.shape[-2] == n_obs_steps`, `actions.shape[-2] ==
   action_horizon`, `action_is_pad` is `[n, action_horizon]`)
2. normalise obs through `state.obs_normalizer`
3. `extract_dct_coeffs(actions, num_modes=num_flow_tokens)` when the DCT is on
4. `run_monte_carlo_sampling` -> `x_t_flat`, `actor_tau`
5. conditioner, broadcast over MC: `jnp.broadcast_to(cond[:, None, :], (n, M, D))`
6. actor forward -> `v_new`
7. `compute_target_velocity`, then `prepare_cfm_loss_inputs`
8. `compute_cfm_loss(..., "both")` -> `l_new`, `l_new_token`
9. `compute_policy_ratio` on the per-sample **and** the per-token losses
10. `compute_aspo_surrogate` -> `policy_per_chunk = -surrogate.mean(axis=1)`

Gate: `policy_per_chunk` matches to `rtol=1e-4`. Compare the **per-chunk vector**,
not the scalar mean — the scalar hides a chunk that went the wrong way.

Traps:

- `adv_mc = normalised_advantages[:, None]`: the advantage is per chunk, the ratio
  is per chunk per MC draw.
- `compute_info` reads the **raw** log ratio, not the clamped one, on purpose.
- `cfm_above_clamp_frac` is `(l_new > cfm_clamp).mean()`, detached.

### Step 5 — `forward_loss`, the value half

1. critic forward on the same normalised states -> `[num_critics, n]`
2. `returns_rms = rms(raw_returns)`, `huber_delta = value_huber_delta * max(returns_rms, 1e-6)`
3. Huber per member, `.mean(axis=0)` over the member axis -> per-chunk value loss
4. if `use_clipped_value_loss`: the clipped branch, then `jnp.maximum`
5. `loss = policy_loss + value_loss_coef * value_loss`

Gate: two things, both required.

- **Value parity** at `rtol=1e-4` for `value_loss_per_chunk`.
- **Gradient parity**: `jax.grad(forward_loss)` against `torch.autograd.grad`, on
  a batch where some chunk is truncated so the mask and the clip are both live. A
  loss that matches while its gradients do not is the failure this port is most
  exposed to.

Traps:

- Broadcasting against the member axis trains **every** member. A `min` inside a
  value loss would only train the lowest one.
- `raw_values` stays fp32; it is the rollout snapshot, not a trainable tensor.
- Do not aggregate the ensemble here. Backup uses `min`, the loss uses `mean`, and
  that split is the design.

### Step 6 — `compute_info`

The 28 scalars plus `cond_*`, all from one packed transfer in torch
(`:1261-1436`). Port as a dict of **0-dim arrays that stay on device**; the
trainer calls `jax.device_get(info)` once per update, which is JAX's version of
torch's single `.tolist()`. The keys are an interface, not a convenience: the
trainer logs them by name (`train_sjfpo.py:570-600`).

Gate: same keys, same values, `rtol=1e-4`. Check `explained_var` and
`advantage_std_fraction` specifically — both are ratios of quantities that nearly
cancel, so they are where a silent fp32-vs-TF32 difference surfaces first.

The intermediates `compute_info` reads already come out of `forward_loss`, which
returns `(loss, terms)`; step 6 is a pure reduction over `terms` with no second
forward pass.

### Step 7 — `predict_action_chunk`

`predict_action_chunk(state, states, *, zero_sampling, cfg)` (`:567`):

1. normalise obs
2. `eps = zeros` when `zero_sampling`, else `sample_noise(key, ...)`
3. `sample_action(actor, eps, cond, num_inference_steps=cfg.num_euler_steps)` —
   the ported `lax.scan` Euler sampler
4. DCT on: `velocity_to_action_steps(x0, cfg.action_horizon)`; off: slice
5. `jnp.clip(-1, 1)`
6. `+ action_perturb_std * noise` when training
7. unscale from `action_low` / `action_high`

Gate: parity against torch with the same weights, **both** noise modes, DCT on and
off.

Traps:

- The perturbation goes in **before** the unscale, in normalised action space.
- `velocity_to_action_steps` reads coeffs as time samples on the half-open grid
  `t_j = j*H/H`. A `linspace(0, 1, K)` time-warps every executed action.
- Clip to `[-1, 1]`, *then* rescale. Reversed, the trainer's
  `action_abs_max > 1.05` guard trips on a perfectly healthy policy.

### Step 8 — EMA, Polyak target, behaviour snapshot

Four small functions over pytrees. No `torch._foreach_*`: that was a launch-count
optimisation with no JAX analogue and no reason to exist here.

| function | mirrors | semantics to preserve |
|---|---|---|
| `snapshot_behavior_policy(state)` | `:315` | copies actor -> `old_actor`, conditioner -> `old_conditioner` |
| `update_ema(state, global_step)` | `:330` | seeds the shadow on the first call, **hard-copies at warmup end**, decays only after |
| `update_target_critic(state)` | `:372` | hard-copies on the first call, then Polyak per optimizer step |
| `swap_in_ema` / `swap_out_ema` | `:412, :434` | eval runs on the average; the live weights come back and are restored |

Gate: run **N = 5** updates on both sides and compare after each one. The failure
this guards is the warmup gate — a one-shot latch behind `>=`, where an `==` is
never hit and the average is re-hard-copied forever. A single-call test cannot see
that; an N-step test can.

### Step 9 — validation parity

Torch raises `ValueError` in eight places inside `forward` (`modeling_sjfpo.py`
1067, 1071, 1076, 1103, 1109, 1147, 1154, 1193) plus one on `reduction`. Port all
of them: an invalid batch that silently reshapes is worse than one that raises.

Gate: one test per `ValueError`, both sides, same exception type.

### Explicitly not in this step

- `build_optimizer`, `global_norm_and_clip` (§5) — the per-role clip is a design
  decision with its own parity gate.
- `generate_rollout`, the collector, `sample_minibatch` — milestones 3 and 4.
- The `scripts/train.py` branch and the fused `update_step` — milestone 4.
- bf16 anywhere — milestone 7.

### Order of risk: what each step can still break

| step | if it is wrong |
|---|---|
| 2 | mask and layout disagree, and the CFM loss is fitted on rows the chunk never ran |
| 3 | the ratio compares two snapshots, so the trust region bounds nothing |
| 4 | the policy gradient is right-shaped and wrong-signed |
| 5 | the critic trains but explained variance never moves |
| 8 | eval reports a stale average, which reads as "the run plateaued" |

## 12. Traps, with sources

| trap | source | mitigation |
|---|---|---|
| static arg in `jit` → recompile per value | jax discussion #23200 (minutes → 12 h) | one batched train state; no static agent/env index |
| async dispatch: enqueue queue ~32 deep, then hang; `print()` on a device value lies | jax #21946 | `jax.block_until_ready(metrics)` once per update, exactly once |
| donation needs `out_shardings` at `jit` time, else `with_sharding_constraint` is not a substitute and you get "donated buffers were not usable" | jax #18263 | pass `out_shardings`; single device here, so `donate_argnames` on `params`/`opt_state` only |
| bf16 backward-pass reductions run in bf16; TPU already upcasts so GPU/TPU differ silently | jax #30310 | §7: fp32 only |
| `vmap` materialises large intermediates → OOM at 128 GB host RAM | jax #32201 | `lax.scan` over the epoch/minibatch axes (already the design); `jax.lax.map` only if a specific batch OOMs |
| `vmap` over envs: `key_envs` must be reshaped to the training batch size | brax #591 | shape keys `(num_envs,)` at the call site; add a test |
| slow compile regression in jax 0.4.36–0.4.38 (XLA inliner + partitioner) | jax #26162, brax #569, mujoco_playground #11 | jax 0.11.2 is long past it. If a hang appears, check the version first |
| `jit` inside `pmap` pathologies; brax's `train.py` is the known-good scaffold partly because it avoids them | jax #14602 | no `pmap` in this plan (single device) |
| device→device copy blocks the other stream | jax discussion #14693 | already banked in torch (`actions_t` gather, one D2H per vec step); do not regress it in the port |
| extracting concrete values from tracers (`int(x)`, `.item()`) | nboyd porting notes | all shapes static; chunk count stays a host int |
| Python `for` loops unroll at trace time, bloating compile time and memory | nboyd porting notes | `lax.scan` for epochs/minibatches/GaE/Euler; leave the collector's env loop alone, it cannot be traced |
| `lax.cond` runs *both* branches under `vmap` | nboyd porting notes | prefer `jnp.where`; the trust-region ASPO selection already is one (`fpo_core.py:267`) |
| `nnx.Variable` implements `__jax_array__`, so JAX transforms treat it as an Array | flax #4722 | split/merge + `.value`; §6(b) |

## 13. Do not port

Scope discipline. Each of these is a torch-ism or a LeRobot integration with no
place in a JAX learner:

- `processor_sjfpo.py` — LeRobot pre/post-processor plumbing
  (`ProcessorStepRegistry`). In JAX, `RunningMeanStd.forward` is 4 lines and the
  action unscale is one affine.
- `checkpointing.py`'s `ResumeError`, `write_bytes_atomically`,
  `save_torch_atomically`, and RNG save. Orbax owns atomicity.
- `tracking/` — copy as-is, it is framework-agnostic (tensorboard/wandb take
  floats). `build_tracker` reads `cfg` through `dataclasses.asdict`, so the JAX
  config must stay a dataclass.
- `evaluator.py` — numpy/venv code, no torch. Needs no port at all, only an
  import.
- `torch.compile` and CUDA-graph concerns from `BLACKWELL.md` A1/A3. No analogue.
- `grad_accum_steps` (§5).
- `SJFPO_DTYPE` / `autocast_context` / `resolve_storage_dtype` (§7).
- `resolve_device` / `is_rtx_1000_pro_blackwell` — device selection is
  `jax.default_device()` and `jax.devices()`.
- `configure_torch_cuda` in `scripts/train.py` — it has to become a no-op or a
  `try: import torch` guard when the method is `sjfpo_jax`, since the JAX process
  must not initialise a CUDA context it does not use.

## 14. Kill criteria

Stop and report if any of these hold:

1. Milestone 0 shows the Blackwell wall-clock split has collect+rollout > 60%.
   The learner was never the problem.
2. §0b.3 shows the CPU `AsyncVectorEnv` floor < 2000 fps. 30M steps is out of
   reach with gymnasium/MuJoCo, and the correct next move is MJX (MuJoCo
   Playground, decision 1 revisited), not more learner work.
3. Milestone 1 parity cannot reach 1e-6 on `gae.py` or `compute_cfm_loss`. That
   means a semantic difference, not a precision one, and a wrong GAE would
   silently ruin every long run. `gae_linked_frac` runs 0.938 at `num_envs=64`
   with the current rollout shape; a JAX run that reports materially less has a
   broken collector, not a slow one.
4. Milestone 5 reaches < 3× and the remaining time is still in the fused update.
   At that point XLA is the floor and the answer is a smaller/faster kernel
   (Pallas), not more porting.

## 15. Names (settled 2026-09-30, revised 2026-10-01 against the current tree)

Revised rows are marked. Everything else was confirmed with the user and stands.

| item | decision | note |
|---|---|---|
| package | `sjfpo_jax` | §2 |
| module split | `configuration_sjfpo.py`, `networks.py`, `fpo_core.py`, `dct_utils.py`, `gae.py`, `state.py`, `policy.py`, `collector.py`, `checkpointing.py`, `train.py` | as §2 |
| **revised** config module | `configuration_sjfpo.py`, not `config.py` | `config.py` is the env registry in the torch package; mirroring the file name keeps the §3 audit file-to-file. The registry itself is reused as-is (`spectral_joint_policy.config` imports no torch) |
| **revised** config fields | mirror `SJFPOConfig` minus the LeRobot-only ones, plus `obs_dim: int` and `action_dim: int` | dims come off `PolicyFeature` shapes in torch (`modeling_sjfpo.py:196-215`) and a plain dataclass cannot mirror that |
| **revised** train entry | `--method sjfpo_jax` branch in `scripts/train.py` | `TrainConfig.method` is `Literal["sjfpo"]` today (`:81`); one CLI, shared env-registry dim check, run-dir resolution and exit codes |
| **revised** dropped name | `advance_windows` is **removed** | the window ring and the batched per-vec-step scatters already exist in torch (`collector.py:697-745, 816-826`); there is nothing left for it to do |
| **revised** `run_monte_carlo_sampling` | three return values, not five | `flow_shape` comes off `a.shape` and `num_mc` off `eps.shape`; a jitted function cannot return Python ints |
| **revised** `run_monte_carlo_sampling` tau | tokenwise `[B, M, K, 1]` only | the scalar, `[M]` and `[B, M]` legacy forms have no caller in the tree, and shapes are static in JAX |
| **revised** `sample_action` | `(predict_velocity, eps, condition, num_inference_steps)` | the `shape` argument is unused in the torch version and `device` does not exist in JAX |
| **revised** `sample_timesteps` / `sample_noise` | take a `jax.random.key`, no device or generator | the torch versions take a shape *tuple* under a parameter named `num_samples`, and every call site passes `None` for the generator |
| **revised** network constructor | every ported class takes `rngs: nnx.Rngs` last | unavoidable: NNX draws its initialisation from a key, and the parity tests copy torch's weights in afterwards, so the initialisation itself is never under test |
| **revised** network parameters | plain `nnx.Param` dicts, kernels `[in, out]` | vmap needs the critic's members as one stacked pytree, which `nnx.Linear` submodules do not give; torch's `[out, in]` is the converter's transpose |
| **revised** `rms_norm` | a module-level helper in `networks.py`, not `jax.nn.rms_norm` | verified absent in jax 0.11.2; flax's Module form cannot hold a stacked weight. §3 |
| **revised** `SelfAttention` | no transpose; `(batch, seq, heads, depth)` throughout | that is jax's layout, not torch's. §3 |
| **revised** `DiTBlock` | the 4-D coordinate-axis branch is not ported | no caller in the tree: `SJFPOFlowModel` always passes `[B, K, H]`. Six lines to add if a caller appears |
| **new** device layer | `src/sjfpo_jax/runtime.py`, with `detect_backend`, `local_device_count`, `compute_capability`, `has_tf32`, `matmul_precision_for`, `device_index_from_env`, `select_device`, `configure_device`, `should_donate`, `describe` | §0e. Every one is a pure function of a device or an env dict, so the CUDA logic is testable on the CPU box |
| **new** launcher env vars | `SJFPO_GPU_INDEX`, falling back to `LOCAL_RANK` | one process per GPU, §0d. `LOCAL_RANK` is what torchrun already sets |
| state container | `SJFPOState`, a `struct.PyTreeNode` in `state.py` | avoids shadowing `flax.training.train_state.TrainState`; the `nnx.Optimizer` path is rejected, §6 |
| NNX classes | identical to torch: `SJFPOFlowModel`, `StateConditioner`, `Critic`, `DiTBlock`, `SelfAttention`, `SwiGLU`, `CondMLP`, `SinusoidalEncoding`, `RunningMeanStd`; plus the `modulate` **function** | makes the port auditable against `networks.py` line by line. `Critic` internals still change per §6(a) |
| math functions | identical to torch: `compute_cfm_loss`, `compute_aspo_surrogate`, `compute_policy_ratio`, `sample_timesteps`, `compute_freq_weighted_std`, `sample_noise`, `run_monte_carlo_sampling`, `sample_action`, `compute_chunk_advantages`, `get_analysis_basis`, `extract_dct_coeffs`, `compute_spectral_energy_retained`, `velocity_to_action_steps`, `decode_coeffs_to_velocity` | parity tests read as direct calls on both sides |
| collector buffer | `RolloutBuffer` | jnp arrays, preallocated, no growth path (§9) |
| new fn 1 | `global_norm_and_clip` | optax clip transform that stashes the pre-clip norm, §5 |
| new fn 2 | `update_step` | the jitted rollout→update program, §10 |
| dropped config fields | `grad_accum_steps` (must be 1, raises otherwise), `SJFPO_DTYPE` | §5, §7. `grad_accum_steps` stays in `configs/sjfpo.yaml` for compatibility |
| W&B / TB keys | unchanged from the torch loop (`train/policy_loss`, `train/critic_grad_norm_preclip`, `train/boot_value_delta`, `train/value_target_drift`, `train/gae_linked_frac`, …) | no new metric names; the trainer reads `compute_info` and `gae_diagnostics` by key (`modeling_sjfpo.py:1403-1432`, `train_sjfpo.py:570-600`) |
| script | `scripts/convert_torch_to_jax.py`, both directions (`--to-jax`, `--to-torch`) | §8. Needed so `scripts/play.py` keeps reading JAX checkpoints |
