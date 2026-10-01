# Spectral Joint Policy (SJFPO)

On-policy reinforcement learning for continuous control with a **flow-matching**
policy, **open-loop action chunking**, and an optional **DCT spectral encoding**
of the action chunk. Built on [LeRobot](https://github.com/huggingface/lerobot)
and PyTorch.

The policy is a conditional flow-matching velocity field over an action chunk.
A chunk of `H` future actions is the macro-action; one macro-transition is a
macro-transition for the backup; the flow is trained with a Monte-Carlo estimate
of the conditional flow matching (CFM) loss, and PPO-style policy gradient
comes from the *ratio of CFM losses* rather than a Gaussian log-likelihood.

## Status

**Research prototype. Not a released result.** There is no baseline comparison
in this repo, so no claim of matching or beating anything is supported yet. See
[Known limitations](#known-limitations).

## Install

Requires Python 3.12+ and a CUDA-capable GPU for real training (CPU runs work
but are slow).

```bash
uv sync
```

## Train

One run per invocation. Run each environment/seed pair you want.

```bash
# core tier
.venv/bin/python scripts/train.py --env Walker2d-v5 --seed 0 --total_steps 1000000

# override a hyperparameter (flags use underscores)
.venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 \
    --sjfpo.policy_lr 3e-4 --sjfpo.num_kept_tokens 8

# resume
.venv/bin/python scripts/train.py --env Hopper-v5 --seed 0 \
    --resume outputs/sjfpo_Hopper-v5_seed0/checkpoints/step_1000000 \
    --total_steps 2000000
```

### Smoke run

Always run a short smoke train before trusting a change. Use wandb offline so
nothing reaches the network:

```bash
WANDB_MODE=offline .venv/bin/python scripts/train.py \
    --env Reacher-v5 --total_steps 3000 --out_dir ./tmp/smoke \
    --sjfpo.report_to all --sjfpo.eval_max_episodes 3
```

`report_to: none` skips the tracker entirely.

### Precedence

`CLI flags > --config YAML (flat, legacy) > dataclass defaults`

The flat YAML overlay only fills method fields the CLI left at defaults.

### Exit codes

| code | meaning |
|---|---|
0 | success |
1 | usage error |
2 | runtime or divergence abort |
4 | a required artifact was missing |

## Environments

Registered in `configs/envs.yaml` with measured `obs_dim`/`action_dim`, checked
against the live environment space at startup. A mismatch is a loud error, not a
silent shape bug.

| tier | environments |
|---|---|
`toy` / `smoke` | CartPole, Pendulum, MountainCar, InvertedPendulum, InvertedDoublePendulum, Reacher |
`core` | Hopper, HalfCheetah, Walker2d, Swimmer |
`hard` | Pusher, Ant, Humanoid |
`dexterous` | Adroit Hand (Door, Pen, Hammer) |

For this algorithm, prefer the `core` tier: 1000-step episodes give 62 chunks per
rollout, and `action_dim >= 6` is where a flow policy's multimodality actually
has room to show. `action_dim == 1` (Pendulum, InvertedPendulum) and short
episodes (Reacher, 50 steps = 3 chunks) cannot test the thing the method is for.

## Method

- **Interpolant.** `x_t = (1 - t) * eps + t * action`, noise at `t=0`, forward
  Euler sampling to `t=1`. Target velocity `v* = action - eps`.
- **Ratio.** `exp(l_old - l_new)` on per-sample CFM losses, with the *same*
  `(tau, eps)` draws shared between the frozen and live policies. This is FPO
  (arXiv:2507.21053) Eq. 6/16.
- **Trust region.** `min(r*A, clip(r)*A)`, with an asymmetric variant that uses
  a quadratic penalty on negative-advantage samples.
- **Advantage.** One chunk is one macro-transition: rewards are summed and
  discounted by `gamma ** h`, then GAE recurses between consecutive chunks of
  the *same* env.
- **Spectral encoding.** With `use_dct_transform`, the chunk is transformed to
  its first `num_kept_tokens` DCT-II coefficients and the flow runs in
  coefficient space, then inverts back onto the action-step grid so the
  execution mask can index it.

## Configuration

Every hyperparameter is a field on `SJFPOConfig`
(`src/spectral_joint_policy/configuration_sjfpo.py`), grouped by purpose.
`configs/sjfpo.yaml` mirrors the defaults for reading; nothing reads it at
runtime. Regenerate it from the dataclass rather than hand-editing.

## Layout

```
src/spectral_joint_policy/
  configuration_sjfpo.py  every hyperparameter, one dataclass
  fpo_core.py             CFM loss, ratio, trust region, Euler sampler
  modeling_sjfpo.py       the policy: rollout, forward, GAE wiring
  networks.py             DiT actor, StateConditioner, Critic, RunningMeanStd
  collector.py            vectorised chunk collector, RolloutBuffer
  gae.py                  chunk-level GAE
  dct_utils.py            orthonormal DCT-II and its inverse
  evaluator.py            evaluation episodes and divergence checks
  train_sjfpo.py          the training loop
  utils.py                shared stateless helpers
scripts/
  train.py                the entry point
  play.py                 replay a checkpoint
```

## Development

```bash
.venv/bin/python -m pytest tests/ -q     # the only gate
```

See `AGENTS.md` for the working conventions, and `BLACKWELL.md` for the
outstanding GPU measurement plan.

## Known limitations

- **No baselines.** Nothing here is compared against Gaussian PPO, SAC, or
  DPPO on matched seeds, so no state-of-the-art claim is supported.
- **No ablations.** The DCT path and `num_kept_tokens` have not been swept.
- **`num_kept_tokens == action_horizon` makes the DCT a no-op.** At the default
  16 the transform is the full orthonormal DCT, a rotation under which the CFM
  loss is invariant (measured relative difference 1.2e-07). A smaller value is
  what makes it a real experiment.
- **Late-training regression.** Eval return peaks near 215k-260k steps and then
  decays about 32% over the remainder of a 1M-step run. The actor drifts into
  actuator saturation; the critic is not the cause (`explained_var` +0.40, and
  `advantage_std_fraction` 0.94, throughout the warm phase).
- **Environment registry has two non-continuous entries.** `CartPole-v1` and
  `MountainCar-v0` are discrete action spaces.
- **Action-bound guard assumes `[-1, 1]`.** `max_abs > 1.05` is checked against
  postprocessed, real-unit actions, so an environment with a wider range (e.g.
  `Pusher-v5` at `[-2, 2]`) trips it on a saturated policy.

## Citation

This implements published methods; it does not introduce a new algorithm. If
you use it, cite the underlying work:

```bibtex
@article{mcwilliam2025flow,
  title  = {Flow Matching Policy Gradients},
  author = {McAllister, David and Ge, Songwei and Yi, Brent and others},
  journal = {arXiv preprint arXiv:2507.21053},
  year   = {2025}
}
```

## License

See the repository for license terms. The `flow_matching` reference library
(Meta) is CC BY-NC and is **not** a dependency here.
