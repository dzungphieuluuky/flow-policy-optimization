# PLAN.md — execution plan for the SJFPO JAX port

The design lives in `JAX_MIGRATION.md` at the repo root. This file is the
**ordered, checkable** version: what to write, what proves it, and what will bite.
Read it before each step and update it when reality disagrees — a plan that is
quietly wrong is worse than no plan.

Line references are `src/spectral_joint_policy/` unless stated otherwise.

## Where the port stands

| milestone | state | gate |
|---|---|---|
| 1 — core math | **done** | 45 tests, `rtol=1e-5` |
| 2 — networks | **done** | 21 tests, `rtol=1e-4` |
| 2b steps 1–5 — state + loss | **done** | 26 tests incl. gradient parity |
| 2b steps 6–9 — info, chunk, EMA, validation | **done** | 26 more; 52 in the file |
| device layer (`runtime.py`) | **done** | 22 tests; **every CUDA branch unverified** |
| §5 — optimizer | **done** | 6 tests: pre-clip norms match torch, one step is ~lr and agrees in direction, no-decay leaves still train, `critic_max_grad_norm=None` measures without clipping |
| `configuration_sjfpo.py` | **done** | 9 tests: field set, defaults, the two properties, four load-bearing invariants |
| 3 — collector | **done** | 9 parity tests on `Reacher-v5` with `num_envs=4`: identical `chunk_state` / `chunk_actions` / `action_is_pad` / `step_rewards` across a terminal, an autoreset, and a budget cut |
| 6 — checkpointing + converter | **done**, and it had three silent bugs | 6 tests: weights, Adam moments, converted policy agrees with torch, the name table covers every optimizer parameter |
| 4 — train loop + `scripts/train.py` branch | **done (4.1–4.6)** | every step parity-gated: gradients and applied step match torch tree-for-tree, pre-clip norms match `clip_grad_norm_`, jit == eager, executed actions match step-by-step across a terminal, and the run logs **the same 72 W&B keys torch does** |
| 5 — throughput | not started | **needs a GPU; decides whether the port is worth anything** |

`371 passed` on CPU as of this writing. The loss compiles: `jit(forward_loss)`
matches eager, and `jit(grad)` runs. No GPU exists on the development box, so
every CUDA path stays unverified until a T4 or Blackwell box exists
(`JAX_MIGRATION.md` §0c).

### Milestone 4, step by step

| step | what | where | gate |
|---|---|---|---|
| 4.1 | `generate_rollout`, `compute_gae_diagnostics` | `policy.py`, `gae.py` | every batch key the backup produces matches torch on a seeded rollout; the MC draws are shape-checked, then torch's are injected and the behaviour losses compared |
| 4.2 | `generate_mini_batch_indices`, `sample_minibatch`, `compute_rollout_diagnostics`, `measure_value_drift` | `train_sjfpo.py` | coverage and passthrough contracts, not index-stream parity (the two frameworks draw from different generators) |
| 4.3 | `build_update_step` | `policy.py` | **both halves**: gradients tree-for-tree, and the applied step parameter-for-parameter |
| 4.4 | `select_action` + `ActionBuffer` | `policy.py` | done. Same obs stream one frame per call, with a terminal mid-episode and both buffers cleared: the executed action matches torch at every step. Also end-to-end through `Evaluator` on `Reacher-v5` |
| 4.5 | `train_sjfpo_run` | `train_sjfpo.py` | done. CPU smoke: loss finite, `action_abs_max <= 1.05`, eval in range, artifacts on disk. Reuses `evaluator.py` and `tracking/` as they are -- both are numpy-only |
| 4.6 | `--method sjfpo_jax` | `scripts/train.py` | done. Prefix `--sjfpo_jax.`, `--resume` allowed for both backends. Runs offline end to end, 128 steps then resume to 192 |

### Two things 4.4 settled that are worth writing down

* **The chunk plays in reverse.** torch refills `_action_buffer` with
  `chunk[0, i]` for `i` from `n_execute - 1` **down to 0** and pops from the front,
  so the first action executed is `chunk[0, n_action_steps - 1]` and the last is
  `chunk[0, 0]`. Writing the ring forward and reading forward gets the same set of
  actions in the opposite order, which looks right and is not.
* **A partly filled obs window repeats its *first* frame, and does not rotate.**
  torch's `deque(maxlen=n)` evicts the oldest, which is that repeat, so the
  window is `frames[0] * (n - len)` followed by `frames[:len]`. Zero padding feeds
  the conditioner a window that does not exist. Once the ring is full the write
  cursor is mid-array, so the window has to be **rotated** or the oldest
  `n_obs_steps` observations replay for the rest of the episode.

`ActionBuffer` is a `flax.struct` dataclass of fixed-size arrays, so the step is
one traced program. That means two branches exist in `_chronological_frames` -- a
Python one on a concrete `frame_len` and a `jnp.where` under tracing -- and
`test_the_traced_window_matches_the_eager_one` exists to keep them equal. Note
that a traced cursor cannot be a slice bound: `frames[pos:]` raises
`IndexError` under `jit`, so the rotation is a `jnp.roll` and the fill is a gather
with `clamp(i - (n - len), 0, n - 1)`.

**Resume is answered "yes" but is 4.5 work.** `checkpointing.py` restores the
Adam moments, the EMA shadow and both latches, and `params_to_pure` already
carries `obs_normalizer.{mean,var,count}`, so the normaliser needs no raw frames
on resume -- a claim worth checking against the code before relying on it.

### What 4.5 and 4.6 settled

* **x64 promotes every unpinned call to fp64.** `runtime.configure_device`
  enables `jax_enable_x64` because `RunningMeanStd`'s counts are fp64, and
  flax's `variance_scaling` default follows it. So the conditioner's output came
  back fp64, `sample_action`'s `lax.scan` carry changed dtype mid-scan, and the
  run would not compile -- **the first time anything had built a network with
  x64 on**, because no test enabled it. Both construction sites now pin fp32.
* **The trainer has to write `global_step` into the state.** `checkpoint_tree`
  reads `state.global_step`, and the loop only ever held it in a local, so every
  checkpoint recorded 0 and `--resume` restarted the run from the beginning.
* **`swap_in_ema` returns `(state, displaced)`.** Treating it as the state and
  calling `swap_out_ema(displaced)` reads `ema_shadow` off a tuple.
* **Torch-freedom is a *library* property, not a CLI one.** `sjfpo_jax` and
  everything it imports leave torch and lerobot out of `sys.modules` (gated in a
  subprocess). `scripts/train.py` does not, and cannot without restructuring its
  own module-level torch config import for the `--method sjfpo` branch. What the
  CLI gate asserts is the part it controls: a `sjfpo_jax` dispatch never imports
  `spectral_joint_policy.train_sjfpo`.
* **`CHECKPOINTS_DIR` and `set_seed` are load-bearing local constants.** Both live
  under `lerobot` in the torch tree, and importing either pulls torch in.
* **The dims check moved from the config to `build_state`.** draccus constructs
  the nested config during `parse()`, before the env is known, so raising on
  `obs_dim=0` at construction made `--method sjfpo_jax` unparseable.
* **A fresh actor has 40 of 80 leaves with a structurally zero gradient.** The
  zero-init `adaLN` holds the gates at 0, so attention and the modulation heads
  are off the loss path entirely and their Adam state legitimately stays zero.
  Any gate that asserts "all moments non-zero" is wrong for a from-scratch run;
  the critic's 14 are the ones that must move.

Two deviations from the plan, both forced:

* **`build_update_step(cfg, params)` is a factory, not `update_step(state, ...)`.**
  The chain, the decay mask and the per-leaf lr tree depend only on `cfg` and the
  parameter structure, and `build_optimizer` walks every leaf three times — so
  building it per step would triple the traversal count of the hot path for
  nothing.
* **No `key` argument.** `forward_loss` is a deterministic function of the batch
  (the MC draws arrive in it) and the optimizer is too, so there is nowhere for a
  key to go. `state.py`'s docstring still says `update_step(state, batch, key)`;
  that line is stale.
* **`update_ema` is outside the step.** torch calls it once per rollout
  (`train_sjfpo.py:519`) and `global_step` is a static field on the state, so
  folding it in would retrace the step once per rollout. `update_target_critic`
  *is* inside, because it is per optimizer step.

## The four rules that have bitten already

1. **Verify the API before using it.** `jax.nn.rms_norm`, `jnp.lerp`, `jax.struct`
   and `nnx.vmap`-on-a-call were all absent or differently-shaped in jax 0.11.2 /
   flax 0.12.10. Three of them cost a full debugging cycle.
2. **Nothing converts silently between dtypes.** `jax_enable_x64` changes the
   *default float type*, so a `jnp.zeros(...)` with no dtype becomes float64 and
   `lax.scan`'s carry type no longer matches. Every array pins its dtype.
3. **A forward that matches while its gradients do not is the worst failure this
   port has.** Every loss step is gated on both.
4. **A passing test that cannot fail is worse than a missing one.** Two of the
   losses in this port were green for the wrong reason: the two sides ran
   different configs, and the DCT was never actually truncating. When a gate is
   added, check it fails when the code is wrong.

## Four more, found 2026-10-02 while fixing a red suite

Every one of these was green before it, and each one is the same species: the
test could not fail, or nothing downstream looked at the value.

| what | why it was invisible |
|---|---|
| the DiT block fed the **modulated** `h` back as the residual instead of the un-modulated `x` | at adaLN-Zero `modulate(x, 0, 0) == x`, so the two agree **exactly at initialisation** — and the actor's zero-init output layer means the forward tests compare two zeros regardless. The first optimizer step moves `adaLN.fc2` off zero and the two diverge. `test_actor_forward_with_the_modulation_woken` now wakes `adaLN.fc2` *and* `out`, and fails without the fix |
| `compute_info`'s `cond_stats` used **one** row where torch uses **eight** | with one row `h` is `[16, hidden]` and the effective rank cannot exceed 16, so the Dohare collapse probe was capped and `cond_fused` / `cond_c` reduced over a different population than torch. Also `num_learning_epochs` had drifted to 2 against torch's 1 |
| the converter's flat moment keys were **module-relative** while the optimizer tree is regrouped **by role** | every moment was written to a key nothing read, so a converted run resumed with **all-zero** Adam state. `unflatten_like` now checks shapes, and the moments are matched against `optimizer_params`, not `params_to_pure` |
| Orbax **mangled the optax state** on save/restore | namedtuples came back as dicts, `EmptyState` as `None`, and a 133-leaf Adam state as 66 leaves — a resumed run's optimizer was unusable and nothing noticed. `checkpoint_tree` now stores the opt_state **flattened** and `restore_from_tree` unflattens it onto a fresh `chain.init(params)` |
| the EMA shadow was checkpointed under `shadow` while `load_ema_state_dict` reads `shadow_params`, and the converter filtered on an `actor.` prefix torch never writes | the lookup missed twice over, so a converted or resumed run came back with **no shadow and a cleared latch** — indistinguishable from a policy whose EMA never warmed, so it would re-hard-copy and throw the running average away. The converter's test also passed `ema_state=None`, so `ema.pt` was absent and that whole path never ran |
| `nnx.merge(graphdef, plain_dict)` cannot load a hand-nested tree | it reports "Incorrect number of leaves, expected 40, but got 13" on a tree whose leaf paths match the graphdef exactly; `nnx.State(tree)` fails identically. Only the `assign` walk works, so it now lives once in `state.py` and both the converter and the EMA shadow go through it |
| `RolloutBuffer.add_chunk` guarded on `max_chunks`, but wrote to arrays sized by `chunk_capacity` | `_write` goes through `.at[index].set()`, whose default mode is **promise_in_bounds**, so a write past the end is dropped in silence while `n_chunks` keeps counting. A rollout with enough terminals produced chunks whose keys disagreed on their length, and the chunk backup read that as a length mismatch, far from the write that lost them. torch grows its arrays instead (`ensure_chunk_capacity`); the guard now bounds on the array, and `generate_rollout` sizes `chunk_capacity` from the step budget |
| `wake_actor` leaves 50 of 80 leaves with an **exactly zero** gradient | `adaLN.fc2` is zero-init as well as `out`, so the DiT blocks stay the identity and nothing upstream of them is on the loss path. The update-step gate wakes both heads, and asserts the gradient is dense — otherwise a step compared against 50 zeros agrees on almost nothing |

---

# Steps 6–9: finish `policy.py` — **all done**

Each step's gate is below, with what it turned up. The four traps in each step
were real; the ones in "what the port got wrong" are the reason they are written
down.

## Step 6 — `compute_info`

**Port.** `compute_info` (`modeling_sjfpo.py:1261-1436`) reduced over the
`terms` dict `forward_loss` already returns. **No second forward pass.**

28 named scalars plus `cond_*` from the actor's `return_stats`. The key list is
at `:1403-1432` and is an **interface**: the trainer logs by name
(`train_sjfpo.py:570-600`), so a renamed key silently stops a metric appearing on
W&B.

```python
def compute_info(terms, cfg):
    """Every diagnostic, as 0-dim arrays still on device. The trainer does the
    one device_get per update; inside jit you cannot call tolist()."""
```

No `state` and no `batch`: everything comes out of ``terms``, and the one forward
step ``compute_info`` needs has to happen inside ``forward_loss`` to stay in the
traced program.

Worked from `terms`: `policy_per_chunk`, `value_loss_per_chunk`, `l_old`, `l_new`,
`ratio`, `ratio_token`, `action_is_pad`, `v_star`, `raw_value_pred`, `raw_returns`,
`normalised_advantages`, `returns_rms`, `cond_stats`.

Needs one more input ``terms`` does not yet carry:

- **`cond_stats`** — `temb`, `fused`, `c`, `gate_msa`, `gate_mlp`, `scale_msa`,
  `scale_mlp`, logged as `cond_*`. torch runs the actor once on a single row with
  `return_stats=True` (`:1168`); the port runs it inside `forward_loss` on
  `x_t[:1]` and puts the dict in `terms`, so the reduction stays a pure function
  over already-computed intermediates.

`delta` and the log-ratio statistics are computed inside `compute_info` from
`l_old - l_new`, on the **raw** delta, not the clamped one — the log ratio is
deliberately a clamp-free diagnostic.

**Gate.** Same keys, same values, `rtol=1e-4`, both sides. Check
`explained_var` and `advantage_std_fraction` first: both are ratios of nearly
cancelling quantities, so they are where a silent fp32-vs-TF32 difference shows up
before anything else does.

**Traps.**

- `explained_var` scores the member **mean**, not the `min` the backup uses. `min`
  is biased low, so scoring it flatters the critic.
- `ratio_token` is `[n, M, K]`; the low/high-frequency split is `num_tokens // 2`
  with the remainder to the high half, and a single-token axis must not index out
  of range.
- The scalar `value_huber_max` / `value_pred_max` come off the member mean.

## Step 7 — `predict_action_chunk`

**Port.** `predict_action_chunk` (`modeling_sjfpo.py:567-625`).

```python
def predict_action_chunk(state, states, cfg, *, zero_sampling=True, key=None):
    """[B, T, obs] -> [B, H, action_dim] in real env units."""
```

Order, which is the order that matters:

1. `states = state.obs_normalizer(states)`
2. `condition = module(state.conditioner)(states)`
3. `eps = zeros` when `zero_sampling` (integrating from `eps=0` is what FPO++ uses
   for eval and for the reported metrics), else `sample_noise(key, shape, cfg)`
   — with `num_inference_steps = cfg.num_euler_steps`
4. `sample_action(actor, eps, cond, num_inference_steps)` — the ported `lax.scan`
   Euler sampler (`fpo_core.py:299`)
5. DCT on: `velocity_to_action_steps(x0, cfg.action_horizon)`; DCT off:
   `x0[:, :action_horizon]`
6. `jnp.clip(-1, 1)`
7. `+ action_perturb_std * noise` **only when training and std > 0**
8. unscale: `bias + scale * chunk` from `state.action_low` / `state.action_high`,
   where `scale = (high - low) / 2` and `bias = (high + low) / 2`

**Gate.** Parity against torch with the same weights, **both** noise modes, DCT on
and off, `action_perturb_std` 0 and non-zero. Plus: the whole function is
`jax.jit`-able and the 16 Euler steps run as one compiled program (assert the
velocity callable is entered once, not 16 times).

**Traps.**

- The perturbation goes in **before** the unscale, in normalised action space.
  After it, the perturbation scales with the env's action range instead of being a
  fixed 2% of normalised action.
- `velocity_to_action_steps` reads coeffs as time samples on the half-open grid
  `t_j = j*H/H`. A `linspace(0, 1, K)` time-warps every executed action.
- Clip to `[-1, 1]`, **then** rescale. Reversed, the trainer's
  `action_abs_max > 1.05` guard trips on a perfectly healthy policy.
- `select_action`'s receding chunk buffer is **not** part of this step. The
  collector drives chunks; eval needs it and milestone 4 will own it.

## Step 8 — EMA, Polyak target, behaviour snapshot

**Port.** Four small functions over pytrees. No `torch._foreach_*`: that was a
torch launch-count optimisation with no JAX analogue and no reason to exist.

| function | mirrors | semantics that must survive |
|---|---|---|
| `snapshot_behavior_policy(state)` | `:315-328` | copies `actor` → `old_actor`, `conditioner` → `old_conditioner`. Called **once per rollout, before collection** |
| `update_target_critic(state)` | `:372-400` | first call **hard-copies** from `critic`; after that Polyak per optimizer step at `critic_ema_decay` |
| `update_ema(state, global_step)` | `:330-370` | seeds the shadow on the first call; **hard-copies at warmup end**; decays only *after* warmup; inert when `ema_decay == 0` |
| `swap_in_ema` / `swap_out_ema` | `:412-442` | eval runs on the average; the displaced live weights come back so training resumes on them |

Plus `ema_ready(global_step)` (`:403-409`) and the shadow's export/import
(`:444-460`), because milestone 6 needs them and the shape is decided now.

The shadow is keyed by **pytree path**, not by torch's parameter name. Export as a
`(decay, initialised, shadow_params)` dict so `ema.pt` keeps a stable shape.

**Gate.** Run **N = 5** updates on both sides and compare after *each one*:

- the EMA shadow,
- the target critic,
- `old_actor` after a snapshot,
- `swap_in_ema` then `swap_out_ema` leaves the live actor bit-identical.

The failure this guards is the warmup gate: a one-shot latch behind
`global_step >= ema_warmup_steps`. If the comparison is `==` it is never hit at
rollout granularity, `ema_ready()` stays false forever, and every later call
re-hard-copies and throws the running average away. **A single-call test cannot
see that; an N-step test can.**

**Traps.**

- The EMA is over the **actor only**, not the conditioner and not the critic.
- `ema_ready` requires `decay > 0` **and** `initialised` **and**
  `global_step > ema_warmup_steps` — strictly greater, unlike the warmup latch's
  `>=`.
- `swap_in_ema` returns the displaced weights; the caller holds them across eval
  and hands them back. Returning nothing loses the live weights.

## Step 9 — validation parity

**Port.** torch raises `ValueError` in nine places in the loss path. All nine must
raise in JAX with the same type and a message that names the offending shape.

| # | condition | line |
|---|---|---|
| 1 | `states.shape[-2] != cfg.n_obs_steps` | 1067 |
| 2 | `actions.shape[-2] != cfg.action_horizon` | 1071 |
| 3 | `action_is_pad` not `[n, action_horizon]` | 1076 |
| 4 | `mc_tau` present without `mc_eps` | 1097 |
| 5 | `mc_tau` wrong shape | 1103 |
| 6 | `mc_eps` wrong shape | 1109 |
| 7 | `behavior_cfm_loss` not `[n, M]` | 1147 |
| 8 | `behavior_cfm_loss_token` not `[n, M, tokens]` | 1154 |
| 9 | `raw_value_pred` and `raw_returns` disagree on length | 1193 |

Plus `reduction` outside `{mean, none}` (1230) and the `KeyError` for a missing
behaviour loss (1135), which step 5 already has.

Steps 1–3 and the `reduction` check are already ported; 4–9 are not. Number 8 is
the one that matters most: `tokens` is the action-step grid when the mask moved the
loss onto rows and the frequencies otherwise, and a silent mismatch makes the
ratio compare means over different positions.

**Gate.** One test per condition, both sides, same exception type.

---

# Steps 1–5: build the rest of the learner

Decisions (2026-10-01): **one optax chain with mask-based decay**, **a Python-loop
trainer first** rather than the fused `lax.scan`, and **`--to-jax` only** for the
converter.

Why the Python loop even though `JAX_MIGRATION.md` §10 specifies one compiled
program: this machine is for runtime-error debugging, and a loop is the debuggable
shape. Fuse to `lax.scan` after a GPU shows it pays. A scan bug and a loop bug
look identical from the outside, and there is no GPU here to tell them apart.

## The optimizer's shape, and why

`optax.adamw(..., mask=...)` masks the **whole update**, so using it for the
no-decay group would *freeze* the adaLN and output params instead of exempting
them from decay. Decay is therefore its own masked step:

```python
optax.chain(
    per_role_clip(actor_max, critic_max),            # stashes both pre-clip norms
    optax.adamw(learning_rate=1.0, weight_decay=0.0),  # moment machinery only
    optax.masked(optax.add_decays(-weight_decay), decay_mask),
    scale_by_lr(lr_tree),                            # per-leaf lr
)
```

The lr is applied last, after the decay term, so the result is exactly
`p -= lr*wd*p + lr*m/(sqrt(v)+eps)` -- AdamW, per-leaf lr, masked decay. The
critic's `weight_decay=0.0` (`modeling_sjfpo.py:481-495`) is the all-`False` mask.

`scale_by_lr` follows optax's own naming (`scale_by_adam`, `add_decays`).

The optimiser tree is regrouped to `{"actor": {...flow model + conditioner...},
"critic": {...}}` so one transform can clip per role. The state keeps its own
layout; `build_optimizer` owns the regrouping.

## Step 1 — the optimizer — **done**

`build_optimizer(cfg, params) -> (update, init_state)`, plus
`global_norm_and_clip`, `per_role_clip`, `scale_by_lr`, `decay_mask`,
`optimizer_params`, `pre_clip_norms`.

**Gate met.** Against torch, on the same grads: both pre-clip norms match
`clip_grad_norm_`'s return; one step moves every leaf by about the learning
rate and agrees with torch in direction (cosine > 0.99, not 1.0 -- torch and
optax place bias correction and eps differently); the adaLN and output leaves
train without decaying; `critic_max_grad_norm=None` records the norm and leaves
the step alone.

Three API facts this step turned up, all of which memory would have got wrong:

| assumed | actual in optax 0.2.8 |
|---|---|
| `optax.add_decays` | `optax.add_decayed_weights` |
| ``optax.adamw(..., mask=...)`` exempts a group from decay | it masks the **whole update**, so the no-decay params would freeze instead |
| `clip_by_global_norm.update` takes `extra_args` | three positional args only |

And one real bug the gate caught: torch's rule is a **substring** test on the
parameter name, so `adaLN_out` is covered too. Checking exact path segments left
the output modulation decaying on every step.

## Step 2 — `configuration_sjfpo.py` — **done**

A plain dataclass mirroring `SJFPOConfig` field for field, **minus** the
LeRobot-only fields (`input_features`, `output_features`, `repo_id`,
`push_to_hub`, `private`, `tags`, `license`, `use_peft`, `hidden`, `device`,
`pretrained_path`, `pretrained_revision`, `use_amp`), **plus** `obs_dim: int` and
`action_dim: int` -- torch reads dims off `PolicyFeature` shapes
(`modeling_sjfpo.py:205-206`), which a plain dataclass cannot mirror.

**Gate met.** 9 tests: the field set is exactly torch minus LeRobot plus the two
dims, every overlapping default is identical, the five properties the learner
reads agree, and four load-bearing invariants raise. Stays a dataclass because
`build_tracker` calls `dataclasses.asdict` (`tracking/__init__.py:39`).

**Not ported, deliberately:** torch's full validator suite. Only the invariants
whose violation is a *silent wrong answer* are checked here -- the token axis,
`batch_size` vs `chunks_per_update`, `hidden_dim` divisibility, the discount
range, and the dims being set at all. CLI range errors stay torch's until the
trainer grows its own.

## Step 3 — `collector.py`

Port `collector.py:646-924` (`collect_chunks_vec`) and the `RolloutBuffer`. The
Python loop stays: `env.step` is a Python call, and the awaiting/reset state
machine is a correctness contract, not an optimisation. The buffer moves to jnp,
preallocated, with the growth path dropped (milestone 3).

**Gate.** Given a fixed policy and `envs.reset(seed=...)`, the JAX collector and
the torch collector produce identical `chunk_state` / `chunk_actions` /
`action_is_pad` / `step_rewards` on `Reacher-v5` with `num_envs=4`, across a
terminal **and** an autoreset.

## Step 4 — checkpointing and the converter

`checkpointing.py` (Orbax `CheckpointManager`) and
`scripts/convert_torch_to_jax.py --to-jax`. The weight mapper already exists in
`tests/torch_jax_weights.py` and graduates from there; §8's table of what the
torch checkpoint actually stores is the input spec.

**Gate.** A converted torch checkpoint reproduces the torch eval return. Needs a
real torch checkpoint, so the test trains a tiny CPU run first.

## Step 5 — `train.py` and the CLI branch

`train_sjfpo_run` as a Python loop mirroring `train_sjfpo.py:426-527`, with
`--method sjfpo_jax` in `scripts/train.py`. `generate_rollout` (the batch the
loss consumes) is ported too, since the loop cannot run without it.

**Gate.** A CPU smoke run: loss finite and decreasing over a few updates, the
advantage/backup path exercised, eval through `select_action` producing in-range
actions. The **fps number** needs a GPU and is not claimed here.

---


Not planned here, but the order is:

1. **`state.py`'s optimizer** — `build_optimizer` + `global_norm_and_clip`, the
   per-role clip. One AdamW over actor (decay / no-decay / conditioner) and critic,
   two clip norms, critic clip optional. `JAX_MIGRATION.md` §5.
2. **`configuration_sjfpo.py`** — the plain-dataclass mirror, without the
   LeRobot-only fields, with `obs_dim`/`action_dim`. Needed by the trainer, not by
   the loss.
3. **`collector.py`** — milestone 3. Parity against the torch collector on
   `Reacher-v5` with `num_envs=4`, across a terminal and an autoreset.
4. **`train.py` + the `scripts/train.py` branch** — the fused `update_step`, then
   the first 1000-step run. **Needs a GPU.**
5. **`checkpointing.py` + `scripts/convert_torch_to_jax.py`** — milestone 6. The
   weight mapper already exists in `tests/torch_jax_weights.py` and moves here.

## What steps 6–9 turned up

Four more bugs, all of the "green for the wrong reason" or "assumed API" class:

| what | why it is dangerous |
|---|---|
| `nnx.Param(...)` wrapping the result of a `State`-vs-`State` `tree_map` | an NNX `State` flattens *through* `Param` to the raw array, so the map must return arrays. Wrapping put a `Param` where the next map expects an array, and the **second** call failed — so the first update, the snapshot, and the state-dict round-trip all passed |
| `@lru_cache` over the DCT basis and the sinusoidal table | whichever trace fills the cache first stores a *tracer*; the next trace to read it gets an `UnexpectedTracerError`. Removed. `K` and `half` are static shapes, so inside `jit` both are constant folds anyway |
| `bool(jnp.all(...))` in `decode_coeffs_to_velocity` | `TracerBoolConversionError` under `jit`, and that function is on the loss path, so the fused program would not have compiled. Now `check_query_range`, a no-op under tracing |
| a fresh actor predicts **zero** velocity | `SJFPOFlowModel` zero-inits `out`, so `l_old` does not depend on the frozen weights at all. The snapshot test was comparing two copies of the same constant until `wake_actor` made the output layer non-zero |
