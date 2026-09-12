# Architecture — Spectral Joint Flow Policy Optimization (SJFPO, MuJoCo-only)

## Data Shapes (`T=16 H=16 K=5 D=128`, `K=round(top_p*T)`)

```
Observation (per step):  flat Box [obs_dim] (env-dependent; Hopper-v5: 11)
State window:            stacked frames [B,T,obs_dim] via deque (T=16)
Action chunk:            [B,H,act_dim] predicted (H=16); training executes all 16
                       open-loop, eval executes the ratio*H prefix (default 8)
```

## Flow (SJFPO path — the only path)

```mermaid
flowchart LR
  O[O_t stacked obs] --> DCT["DCT-II dim=1 → [B,16,obs]\nretain K lowest bins → [B,K,obs]"]
  DCT --> TOK["tokens [B,K,obs] (raw bins, no projector)"]
  TOK --> DIT["SJFOPolicyModel DiT-8 adaLN-Zero\n[state prefix; action tokens] + time/pooled cond"]
  O --> DIT
  DIT --> V["v̂ velocity [B,Ka,Da] (Ka=top_p·H rows)"]
  V --> PAD["zero-pad time → [B,H,Da] → IDCT → clamp"]
  V --> EUL["euler_sample t:1→0 → IDCT → clamp"]
  O --> CE[critic_state_encoder\nLinear + sinus PE → 2× prenorm blocks → Z B,T=16,D=128]
  CE --> CR["Critic V(s,a)\nreal actions +PE query encoded obs"]
  CR --> TD["TD target R + γ^L·V(s',a')"]
  CR --> SIG["Epps-Pulley SIGReg on critic latent\nP=1024, λ=0.1"]
```

Baseline path: `baselines.py` trains skrl 2.1.0 PPO/SAC on the same
frame-stacked env through the shared `baseline_train_loop`
(eval → 3-col CSV → checkpoint → wandb). No spectral/flow code involved.

## Key Contracts (break silently if flipped)

- **t-convention:** `x_t = t·ε + (1−t)·a`, `t=1` noise, `t=0` data. Sampling
  integrates **descending** `x ← x − δ·v̂` 1→0 (`fpo_core.euler_sample`,
  64 train / 16 eval steps). `trainer.py` samples `τ~U[0.005,0.995]`.
- **Spectral order:** DCT-II over time (dim=1), keep the `K` lowest
  low-frequency bins (`K=max(1, round(top_p*T))`, default `0.3*16→5`), then
  with no normalization and no projector (last dim stays `obs_dim`). Tests assert
  invariance `B=1` vs `B=32`. (rFFT is reserved for observation.image.)
- **One encoder:** spectral tokens come straight from the raw obs window
  (no policy trunk); the only `StateEncoder` left is `critic_state_encoder`,
  carrying value-TD + SigReg grads. GAE values come from the critic encoder.
- **Inference:** deterministic zero-sampling eval (`eps=0`), single
  `clamp(-1,1)` after IDCT in `runner.decode_chunk_actions`.
- **Precision:** fp32 everywhere. No `autocast`/`half`/`bfloat16`/`torch.compile`.

## Why These Choices

- Top-`p` low-frequency DCT tokens condition the flow on
  temporal structure of `Z`, not its DC offset.
- Epps-Pulley SIGReg single-home (`sigreg_loss`, P=1024, λ=0.1) — no Barlow/VICReg twins.
