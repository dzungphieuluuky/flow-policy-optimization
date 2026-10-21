# Design A — Per-Token Advantage for Action Chunks

**Status:** plan only. Nothing in this document is implemented.
**Scope:** the "action-level" arm of RLinf-VLA (per-atomic-action reward and
advantage, with a per-position value) adapted to a flow-matching chunk actor
whose policy ratio is the FPO CFM-loss proxy.
**Owner:** the `myfpo` repo (`src/myfpo/`), `action_horizon = n_action_steps = 8`.

---

## 1. Motivation

The current surrogate multiplies a **per-token** ratio by a **per-chunk**
advantage. The two granularities do not match.

- `ratio_token` is `[n, M, H]`: every token of the chunk has its own CFM loss,
  hence its own ratio (`src/myfpo/modeling.py:685-688`).
- `adv` is `[n]` — one scalar per chunk, broadcast over all `H` tokens
  (`src/myfpo/modeling.py:679-680`, `adv_mc.unsqueeze(-1)` at `:680`).

The policy is a **factorised** one, so the mismatch is not benign. Causal
attention (`src/myfpo/attention.py:68`, `is_causal=True`) means token `i` is a
function of the prefix only, so the chunk density factorises as

```
pi(chunk | s) = prod_{i=0}^{H-1} pi(a_i | s, a_{<i})
```

which is exactly the factorisation under which a per-factor advantage is the
well-posed quantity. Multiplying a per-factor gradient by a shared scalar
throws away the credit distinction the factorisation exposes.

### 1.1 What the literature does

| Work | Granularity | Advantage | Value head | Outcome |
|---|---|---|---|---|
| FPO (`arXiv:2507.21053`), FPO-for-VLAs (`arXiv:2510.09976`) | per sample | one scalar per env step; the chunk is one sample | scalar `V(s)` | no token decomposition |
| pi_RL (`arXiv:2510.25889`) | chunk | "chunk-level formulation": summed reward, one shared advantage | scalar `V(s)` | baseline |
| **RLinf-VLA (`arXiv:2510.06710`, RSS 2025)** | **chunk vs action** | chunk: one shared advantage. **action: per atomic action `a_{t,i}` its own reward and advantage** | chunk: `V: S -> R`. **action: `V: S -> R^C`** | **"the action-level formulation consistently leads to better performance"** (Fig. 10: higher success rate *and* lower value loss) |
| ACPPO (`arXiv:2609.36250`, Sep 2026) | chunk | one chunked advantage `A_chunk = sum_j gamma^j delta_{t+j} + gamma^h A_GAE(next)` | scalar `V(s)`, "projected state critic" | **chunked advantage beats stepwise GAE**: "the policy-gradient signal should match the temporal abstraction of the actor" |
| Q-chunking (`arXiv:2507.07969`), DQC (ICLR 2026) | chunk | chunked `Q(s, a_{t:t+h-1})` | grows with the chunk length | harder value learning as the chunk grows |

### 1.2 Why the two results do not contradict

The deciding factor is **whether the actor factorises over the chunk's
sub-actions**.

- ACPPO's actor samples the chunk **jointly** and its ratio is the **joint**
  chunk ratio `prod_k pi_k / pi_k^old`. A per-step advantage against a joint
  ratio is dimensionally inconsistent, so one advantage is the correct choice
  there. Appendix C of that paper gives the trade-off: the chunked estimator
  *telescopes* the intermediate value terms and so leans on fewer value
  estimates, at the price of coarser, higher-variance credit.
- RLinf-VLA's actor **factorises** (`pi(c_t|o_t) = prod_i pi(a_{t,i}|o_t,
  a_{t,:i-1})`), so each factor's log-prob takes its own advantage. Their
  Table II makes the constraint explicit: an action-level advantage requires
  an action-level (or token-level) log-prob; a chunk-level advantage works with
  any log-prob granularity.

**We are in RLinf-VLA's factorised regime and are using their chunk-level
advantage.** That is the gap this design closes.

### 1.3 What is new here, and what is not

Not new: the action-level advantage formulation (RLinf-VLA), the chunked
advantage and its bias/variance analysis (ACPPO), the factorised-policy
advantage (textbook).

**New:** nobody has published a per-token advantage for a *flow-matching chunk
actor whose ratio is the CFM-loss proxy*. RLinf-VLA's action-level arm needs
an **exact** per-action log-probability (AR heads, or a flow with a tractable
likelihood). We cannot compute one: the FPO ratio is a proxy for the
*chunk's* density change, and per token it is a proxy for a *single token's*
density change. The construction is therefore an approximation at a level the
literature has not validated, and the plan treats it accordingly — behind a
flag, with the chunk mode as the frozen reference, and with a credit-quality
diagnostic rather than a return curve as the primary evidence.

### 1.4 A discount warning from the literature

RLinf-VLA's gamma ablation (`arXiv:2510.06710`, App. X-C): with a chunk-level
formulation over an *undiscounted* summed reward, the discount must be
rescaled, `gamma_chunk = gamma_action^(1/c)`; they used `0.99^(1/8) = 0.904`,
and using `gamma_chunk = 0.99` directly is "substantially worse".

**We are already safe.** Our chunk reward is `sum_j gamma^j r_{t+j}` with a
`gamma^length` bootstrap (`src/myfpo/gae.py:80-91`): the discount clock is per
environment step, not per chunk. `length` is 8 for a full chunk and smaller
when the episode ended first. The per-token construction below must
**not** re-discount on top of that. The per-token suffix restarts its own
clock at the token (`gamma^{j-i}`), and the chunk's absolute position stays in
the outer GAE trace. This is the most likely place for a silent bug; it has a
dedicated test (Section 8).

---

## 2. The mathematics

### 2.1 Setting

A chunk starts at environment step `T` and covers `H` steps
(`H = action_horizon = n_action_steps = 8`). Write

- `r_{T+j}` for the reward of environment step `T+j`, `j = 0..H-1`;
- `s_{T+j}` for the state at that step; `s_T` and `s_{T+H}` are stored today
  (`chunk_state`, `chunk_next_state`);
- `a_{T+i}` for the token planned at offset `i`; `a_{T:<i}` for the prefix;
- `f` for the executed-length fraction, the critic's extra input;
- `gamma`, `lambda` for the discount and the GAE trace.

The chunk is the macro-action. The critic is currently a **chunk-Q**:
`Q_phi(s_T, a_{T:T+H-1}, f) -> R`, and the chunk-level advantage is the
multi-step residual of the macro-transition (`src/myfpo/gae.py:86-113`).

### 2.2 The policy gradient for a factorised chunk

For `pi(chunk|s) = prod_i pi_i(a_i | s, a_{<i})` and the episode return
`G = sum_k gamma^k r_k`,

```
grad J = E[ sum_i grad log pi_i(a_i | s, a_{<i}) * A_i ]
```

with the **reward-to-go** advantage

```
A_i = E[ R^to-go_{T+i} | s_T, a_{T:<=i} ]  -  E[ R^to-go_{T+i} | s_T, a_{T:<i} ]
```

where the reward-to-go restarts the clock at the token:

```
R^to-go_{T+i} = sum_{j>=i} gamma^{j-i} r_{T+j}  (plus the bootstrap)
```

The **baseline is the pre-action value**: it conditions on the prefix but not
on `a_i`. Using the post-action value as the baseline would make the
expectation vanish (the residual of a correct critic has zero mean) and the
gradient would die.

### 2.3 The two critic predictions

Introduce, per token position `i`:

```
v_i := E[ R^to-go_{T+i} | s_T, a_{T:<i} ]      (pre-action value)
q_i := E[ R^to-go_{T+i} | s_T, a_{T:<=i} ]     (post-action value)
```

Both are conditional expectations of the **same** random variable, so they
share **one** regression target:

```
R_i := sum_{j=i}^{H-1} gamma^{j-i} r_{T+j}  +  gamma^{H-i} * B
```

where `B` is the bootstrap (`V(s_{T+H})`, injected on truncation, zero on
termination) exactly as the chunk-level path does today
(`src/myfpo/gae.py:86-91`).

The per-token advantage is then

```
A_i = q_i - v_i
```

which is the advantage of the action `a_i` given its prefix. Note the
dueling-DQN shape: two heads, one target, and their difference is the
advantage. `v_i` must not see `a_i`; `q_i` must not see `a_{>i}`.

### 2.4 Invariance check — the chunk mode is the limit

The chunk-level advantage used today is the `H -> 1` limit of the same
construction with one position. The implementation must be arranged so that
`advantage_mode="chunk"` is a *special case* of the same code path, not a
parallel one; then the two modes cannot drift apart in the reward
bookkeeping, the masks, or the trust region.

### 2.5 Interaction with ASPO and the trust region

Two things become per token, and both are deliberate:

- `compute_policy_ratio` (`src/myfpo/fpo_core.py:237-255`) clamps the loss
  for negative advantages. With per-token advantages, the *sign* is per token:
  a token can be "good" inside a bad chunk. This is the point of per-token
  credit, and it is also the biggest behavioural change from today's run.
- `compute_aspo_surrogate` (`src/myfpo/fpo_core.py:196-215`) selects the PPO
  branch for `A > 0` and the SPO branch for `A < 0`. The selection becomes per
  token.

The **ratio** stays exactly as it is: per token, clipped per token. The
alternative — clipping the *joint* chunk ratio, as ACPPO does — is a separate
knob and a separate experiment; it is not part of this design. The flag that
selects the granularity today is `do_average_cfm_loss_in_chunk`, which is
`False` by default, so the token path is the one that runs.

---

## 3. The critic change

### 3.1 Two heads, one trunk, masked inputs

`CriticMember` (`src/myfpo/critic.py:32-139`) currently flattens the chunk
(`parts[1] = self.norm_action(actions.flatten(1))`) and mixes it with the
state. For the prefix semantics to hold, the head that produces `v_i` must not
see `a_i` or anything after it.

`norm_action` is a learned affine, not a norm. It computes no statistics from
the chunk, so the chunk's offset survives to the critic. A plan that masks the
tail must decide whether the mask is applied before or after this affine. The
affine has no statistics, so the order does not change the result. A norm
would have made the order matter.

Implementation:

1. The action input keeps its width (`H * action_dim`) but is **masked per
   head**: for `v_i`, the dims of tokens `>= i` are zeroed; for `q_i`, the dims
   of tokens `> i` are zeroed. The mask is applied to the chunk before the
   affine. The affine has no statistics, so the order does not matter.
2. The output head becomes `Linear(hidden_dim, 2H)`: `v_0..v_{H-1}` then
   `q_0..q_{H-1}`.
3. `Critic.forward` returns `[num_critics, B, 2H]` in action-level mode and
   `[num_critics, B]` in chunk mode, so the chunk path is untouched.

### 3.2 Cost

A masked input per head means `2H` forward passes per member per call
(`2 * 8 = 16` masks, times `num_critics = 2`). The critic is 738k parameters
against the actor's 1.24M for Hopper, and it runs once per minibatch, so this
is affordable — but it must be **measured**, not assumed. Log the critic
step time before and after (the repo already has a `timing/*` family).

Alternative if the cost is material: a **causal token critic** — a small
causal-attention trunk over the `H` chunk tokens that emits `(v_i, q_i)` per
position in one pass. It is `O(1)` instead of `O(H)` and matches the
factorisation natively, but it replaces the TD-MPC2-style critic trunk, which
is a much larger change and a different risk. **Not part of this design**;
recorded as a follow-up.

### 3.3 The exec-fraction input

`f` describes the whole chunk, not a prefix. It is the same for every `i` and
the critic can use it or ignore it. It is kept (it still matters for chunks
cut short by an episode end) and the ambiguity is documented in the head's
docstring.

---

## 4. Configuration surface

New fields in `MyfpoConfig` (`src/myfpo/configuration.py`), each with a
validation rule and a test in `tests/test_config_surface.py`:

| Field | Default | Values | Meaning |
|---|---|---|---|
| `advantage_mode` | `"chunk"` | `chunk`, `action_level` | per-chunk advantage (today) or per-token |
| `token_value_heads` | `False` | bool | shorthand for `advantage_mode="action_level"`; keep only one of the two |
| `token_advantage_normalize` | `"global"` | `global`, `per_position` | standardise the per-token advantages over all valid tokens, or per position |
| `token_advantage_clamp` | `100.0` | float | reuse `advantage_clamp_pos/neg` or add a per-token clamp; decide once |
| `critic_prefix_mask` | `True` | bool | mask the future action dims in the prefix heads; `False` is the (weaker) unmasked variant, kept only as an ablation |

Deliberately **not** in this design: a `clip_granularity` flag (per-token vs
joint chunk ratio) and the within-chunk-GAE variant. Both are separate
experiments (Section 9). The granularity that exists today is
`do_average_cfm_loss_in_chunk`, and it is not renamed.

---

## 5. File-by-file changes

### 5.1 `src/myfpo/critic.py`

- `CriticMember.__init__`: the output head `Linear(hidden_dim, 1)` becomes
  `Linear(hidden_dim, value_head_size)` with `value_head_size in {1, 2H}`.
- `CriticMember.forward`: accept an optional `prefix_masks` argument
  `[B, H, H]` (or `2H` masks) and apply them to the action part before the
  affine. Return `[B, value_head_size]` instead of `[B]`.
- `Critic.forward`: stack over the members, keep the chunk-mode shape
  `[num_critics, B]` when `value_head_size == 1`.

### 5.2 `src/myfpo/gae.py`

- Keep `compute_chunk_advantages` untouched — it is the chunk mode.
- Add `compute_token_targets(...)` returning `[n, H]`: the per-position
  bootstrap targets `R_i` of Section 2.3, computed from the **same** reward
  bookkeeping as the chunk path (the same `gamma**offsets` weights, the same
  truncation injection, the same termination rule), with the clock restarted
  at `i`. The two functions must share one helper for the reward suffix, so
  the discount bookkeeping cannot diverge.

### 5.3 `src/myfpo/modeling.py`

- `_loss_and_info`:
  - load the critic output `[num_critics, B, 2H]` in action-level mode;
  - split into `v` and `q`;
  - the value loss becomes a masked MSE over the `2H` heads against the
    `[n, H]` targets (both `v_i` and `q_i` regress to `R_i`);
  - `A_token = (q - v).detach()` -> `[n, H]`;
  - normalise per `token_advantage_normalize`;
  - `adv_mc = A_token.unsqueeze(1)` -> `[n, 1, H]`, which broadcasts against
    `ratio_token` `[n, M, H]` **without changing the broadcasting code** — the
    `[n, H]` shape convention is what makes this a one-line change;
  - everything downstream (`compute_policy_ratio`, `compute_aspo_surrogate`,
    the per-chunk normalisation `surrogate.sum(dim=(1,2)) / (n_executed * M)`)
    is unchanged in structure and only changes in the advantage's granularity.
- `compute_behavior_cfm_losses` is unaffected: the behaviour CFM loss does not
  depend on the advantage.

### 5.4 `src/myfpo/metrics.py`

- `compute_value_metrics`: in action-level mode, add the per-token explained
  variance and keep the chunk-level series (both are different quantities and
  both deserve a name).
- `compute_ratio_metrics`: unchanged. The ratio is unchanged.

### 5.5 `src/myfpo/runner/train_metrics.py`

- New `train/*` keys (Section 6). The builder is a pure function; the new
  scalars arrive in `info`.

### 5.6 `src/myfpo/rollout.py`

- No change. The batch already carries `action`, `action_is_pad`,
  `raw_advantages`, `raw_returns` and the per-step rewards are inside the
  chunk bookkeeping. The per-token targets are computed from the same data in
  `gae.py`.

---

## 6. Metrics

One key per quantity, no two keys for the same number — the rule enforced by
`tests/test_log_key_family.py`.

| Key | Meaning |
|---|---|
| `train/token_advantage_mean` | mean of `A_i` over the valid tokens |
| `train/token_advantage_std` | spread of `A_i` |
| `train/token_advantage_sign_disagree_frac` | fraction of tokens whose advantage sign differs from the chunk's sign — **the direct measure of how much the per-token signal differs from the current one** |
| `train/token_advantage_corr_with_step_reward` | correlation between `A_i` and the discounted per-step reward at offset `i` — **the primary credit-quality diagnostic** |
| `train/token_value_explained_var` | per-token explained variance of the prefix heads |
| `train/token_advantage_abs_max` | tail magnitude, to catch a diverging head |
| `train/token_ratio_mean_{i}` | the per-position mean ratio, `H` series (the position series of `flow_ratio_mean`, made explicit) |

The two bold rows are the ones that decide whether the design works. A return
improvement without a movement in `token_advantage_corr_with_step_reward` is
noise, not credit assignment.

---

## 7. Files that must not change

- `src/myfpo/fpo_core.py` — the ratio, the ASPO surrogate, the CFM loss. The
  granularity change is upstream of them.
- `src/myfpo/attention.py` — the causal mask is the licence for the
  factorisation and is already pinned by `tests/test_causal_attention.py`.
- `src/myfpo/flow.py` — the actor.

---

## 8. Tests

One test per claim, in the repo's style. A new file
`tests/test_token_advantage.py` plus additions to
`tests/test_config_surface.py` and `tests/test_log_key_family.py`.

1. **Chunk mode is unchanged.** With `advantage_mode="chunk"`, the loss, the
   ratio and the advantage are **bit-identical** to the pre-change code on a
   fixed seed. This is the regression guard for the whole design.
2. **The discount bookkeeping.** On a hand-built chunk with known rewards, the
   per-token target equals `sum_{j>=i} gamma^{j-i} r_j + gamma^{H-i} B`,
   computed by hand. This is the test that catches a double discount.
3. **The prefix masks hold.** `v_i` does not change when `a_i` or any later
   token is perturbed; `q_i` does not change when a later token is perturbed.
   (Note: the critic's head is randomly initialised, so the test must use a
   fixed seed and a non-degenerate head — the same trap as the zero-initialised
   DiT head in the actor.)
4. **The advantage is the head difference.** `A_i = q_i - v_i` exactly, and
   `A_i` is detached from the graph (a backward pass must not reach the critic
   through the advantage).
5. **The masked tokens are inert.** A padded position has zero advantage, a
   ratio of 1, and contributes nothing to the value loss.
6. **The ASPO sign split is per token.** A synthetic chunk with mixed-sign
   token advantages selects the PPO branch for the positive tokens and the SPO
   branch for the negative ones.
7. **The normalisation modes differ only in scale.** `global` and
   `per_position` produce the same *ordering* within a position.
8. **The value head's shape contract.** `[num_critics, B]` in chunk mode,
   `[num_critics, B, 2H]` in action-level mode; the chunk path's shapes are
   asserted so a future change cannot silently alter them.
9. **The gradient reaches the critic's prefix heads** and, through them, the
   action input — but not through the advantage (see 4).

---

## 9. Ablation protocol and the decision rule

Runs: 3 seeds x {Hopper-v5, Humanoid-v5} x `advantage_mode in {chunk,
action_level}`. Everything else fixed. The chunk-mode runs are the frozen
reference from Step 0.

Reported per run: return, `train/flow_ratio_mean`, `train/clipfrac`,
`train/token_advantage_corr_with_step_reward`,
`train/token_advantage_sign_disagree_frac`, `train/cond_effective_rank`, the
critic step time, and the wall clock.

**Decision rule.** Promote `action_level` only if it beats the chunk baseline
on the return **and** `train/token_advantage_corr_with_step_reward` moves in
the expected direction. The repo has no significance tests today
(`AGENTS.md`: "naming/NaN only"), so 3 seeds is a screen, not a proof; a
positive screen should be followed by more seeds before it is claimed.

**Not part of this design** (separate experiments, each with its own knob):

- `clip_granularity="chunk"` — clip the joint chunk ratio, as ACPPO does.
- the within-chunk-GAE variant (ACPPO's estimator decomposed), which needs
  per-step state values and therefore the intermediate state windows; those are
  reconstructible from `step_frames` but are a much larger change.
- the causal token critic (Section 3.2).

---

## 10. Risks and open decisions

1. **The proxy is unvalidated per token.** The per-token CFM-loss difference
   is a proxy for a *single token's* density change, not the chunk's. The
   chunk-level proxy already carries this caveat
   (`src/myfpo/fpo_core.py:234`); the token-level version inherits it and
   amplifies it, because `H` proxies are multiplied by `H` advantages.
2. **The sign flip.** Per-token advantages change the ASPO branch per token.
   If the per-token signal is mostly noise, the branch selection becomes
   noise-driven and the run may be worse than the chunk mode. This is the main
   failure mode, and `token_advantage_sign_disagree_frac` is the early
   warning: if it is near 1, the per-token signal has decorrelated from the
   chunk's and the design is not working.
3. **The prefix heads can collapse.** `v_i` and `q_i` regress to the same
   target; nothing forces them apart except the input mask. If the critic
   ignores the masked-in action dims, `A_i -> 0` and the design degenerates to
   the chunk mode with extra compute. A metric for `|q_i - v_i|` is therefore
   mandatory, not optional.
4. **The critic cost.** `2H` masked forwards per member per call. Measure it.
5. **Variance.** Later positions have shorter effective horizons and lean on
   the next chunk's bootstrap; the per-token advantages are expected to be
   noisier at the tail. `token_advantage_normalize` is the lever.
6. **Open: should the advantage be `q_i - v_i` (dueling) or `q_i - V_i` (a
   per-position state value)?** The former is the correct baseline for a
   factorised policy; the latter is RLinf-VLA's `V: S -> R^C` and is simpler
   but has the projection bias ACPPO discusses (the mid-chunk value depends on
   the chunk variables, not on the state alone). **Recommendation:** the
   dueling form, because it is the textbook baseline for the factorisation
   this actor actually has.
7. **Open: does the per-token advantage need its own clamp?** The chunk
   advantage is clamped at `+/-100`. The per-token advantages have a different
   scale. Reuse the existing clamp, and log the clamp's hit rate.

---

## 11. Rollback

`advantage_mode="chunk"` remains the default until an action-level run wins
on the decision rule. The chunk path must keep working untouched and be
covered by the bit-identical regression test (Section 8, test 1). Every knob is
a config field, so runs stay comparable and the ablation is a CLI flag
difference.

---

## 12. References

- FPO: `arXiv:2507.21053`.
- FPO++: `arXiv:2602.02481`.
- RLinf-VLA (action-level vs chunk-level, the gamma ablation): `arXiv:2510.06710`.
- pi_RL (chunk-level formulation, flow-based VLA RL): `arXiv:2510.25889`.
- FPO for flow-matching VLAs: `arXiv:2510.09976`.
- ACPPO (chunked advantage, projected state critic, App. C): `arXiv:2609.36250`.
- Q-chunking: `arXiv:2507.07969`. Decoupled Q-chunking: ICLR 2026.
- ReinFlow: `arXiv:2505.22094`. Flow-GRPO, DPPO, RISE: the denoising-time axis,
  not the chunk-token axis.
- `literature_review.md` in this repo: the taxonomy and the open problems
  (8.3 chunk-boundary artifacts, 8.2 entropy collapse in the negative-advantage
  regime).
