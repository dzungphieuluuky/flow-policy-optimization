# Behaviour pins

A pin records every number a fixed computation produces. A later run replays
the same computation and compares at `atol=0`.

## Why these exist

The refactor series moved code between files without changing what it
computes. That claim needs a check. `pytest` is not that check. On
2026-10-17 the full suite passed 588 tests while a deliberate change to
parameter construction order silently changed every weight in the actor. The
pins catch it. The suite does not.

## The scripts

| Script | Covers | Baseline |
|---|---|---|
| `pin_train.py` | one `MyfpoPolicy.loss_with_info` call, the whole `info` dict, and the gradient of every parameter, for the three pin arms (`dit`, `resmlp`, `dit.clamp0.5`) | `train_old.pt` |
| `pin_eval.py` | eval summary metrics, the bootstrap interval, the three jerk measures, and both stop guards | `eval_old.pt` |

`_shared.py` holds the setup and the comparison. Do not import anything from
`src` into it beyond what it already builds. A pin that shares code with the
code it checks cannot catch a defect in the shared part.

## The rule

`atol=0`. Two floats are equal only when they are bit-identical. Do not add a
tolerance. A refactor that moves a float by one part in a hundred is a
behaviour change and must fail.

A missing key and a new key are both failures. A key that appears or disappears
means the computation changed shape, which is a behaviour change even when no
value moved.

## Running

```bash
PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py
PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py
```

Exit code 0 means no difference. Any other value means at least one.

Each run takes about 3 to 6 seconds on CPU. Both pins work on CPU and skip CUDA,
so they run anywhere the test suite runs.

## Regenerating a baseline

Only when a change is meant to move the numbers.

```bash
PYTHONPATH=src .venv/bin/python scripts/pins/pin_train.py --write
PYTHONPATH=src .venv/bin/python scripts/pins/pin_eval.py --write
```

The two baselines are tracked differently. `train_old.pt` is checked in and
ships with a clone. `eval_old.pt` is not tracked; the `.gitignore` rule
`scripts/pins/*.pt` covers it. A fresh clone passes `pin_train.py` and fails
`pin_eval.py` with "no baseline. Run with --write first." Generate the eval
baseline on the branch that needs it and record the key counts in the commit
message. A change to `train_old.pt` is a re-pin and belongs in its own commit.

## What the pins do not cover

- Rollout collection and environment stepping. No pin runs an environment.
- Anything under `compute_info=False`. The trainer turns the stats path off when
  the tracker is off, so the metric set depends on the tracking mode. The pins
  pin the full `compute_info=True` set.
- The second optimizer step. The pins cover one forward and one backward. They
  do not cover EMA, the LR schedule, or checkpoint round-trips.