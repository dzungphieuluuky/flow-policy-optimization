# AGENTS.md

Repo-specific. Everything else lives in `~/.pi/agent/AGENTS.md`, which Pi
loads alongside this file.

## Temp Files

Write every scratch file, benchmark script, oracle dump, and run output to
`./tmp`, never to the global `/tmp`.

```
./tmp/bench_collector.py
./tmp/baseline.pt
./tmp/ab/s3407/ne32/
```

`./tmp` is gitignored. Do not commit it, and do not add exceptions.

Why:

- artifacts stay next to the code that produced them, so a later session can
  still read them
- one directory to clean, and `git status` stays quiet
- the global `/tmp` is shared with every other process and session on the box;
  two runs of the same sweep collide, and neither is discoverable from the repo

This covers the large ones too: training output, profiler traces, and
`torch.compile` caches. Point an existing tool at the repo path rather than
letting it default elsewhere.

Smoke and benchmark `scripts/train.py` runs go under `./tmp` with wandb
offline, so nothing reaches the network:

```
WANDB_MODE=offline .venv/bin/python scripts/train.py \
  --env Reacher-v5 --out_dir ./tmp/smoke --sjfpo.report_to all
```

Always run a smoke train after a change that touches the training path. A green
`pytest` does not exercise the collector, the GAE backup, the CFM loss, or the
Euler sampler.

`report_to` defaults to `all`, so the wandb import runs; `WANDB_MODE=offline`
keeps it local (`./wandb/offline-run-*`). The tensorboard and wandb backends
write to `./tensorboard` and `./wandb` in the repo root regardless of
`--out_dir`, so those two directories collect the run output, not
`--out_dir`.

Both directories are tracked in git, and `wandb/` in particular is worse than it
looks: 1783 files are under version control, and merely running `pytest` dirties
`wandb/debug.log`, `wandb/debug-internal.log`, and `wandb/latest-run`, because
`tests/test_tracking.py` instantiates a real tracker. A training run goes
further and can delete tracked `wandb/*.table.json` files.

So after `pytest` or any training run:

```
git checkout -- wandb/
```

That reverts the run's side effects and leaves your own edits alone. Do not
commit anything under `wandb/` or `tensorboard/`. The permanent fix is to
untrack both directories and add them to `.gitignore`, which has not been done
here.

### wandb mode

`WANDB_MODE=offline` is the current and supported mechanism. Verified on
wandb 0.30.0:

- `mode` is `Literal["online", "offline", "shared", "disabled", "dryrun",
  "run"]` in `wandb/sdk/wandb_settings.py`, so `offline` is a first-class
  value, not a legacy alias
- env vars prefixed `WANDB_` override defaults, and `WANDB_MODE` is the one
  for `mode`
- an offline run writes `./wandb/offline-run-<ts>-<id>/` and prints the
  `wandb sync` command to upload it later

Upstream is `github.com/wandb/wandb` (Weights & Biases), actively maintained.
If offline ever stops working, check the installed version and the upstream
repo before assuming the env var moved:

```
.venv/bin/python -c "import wandb; print(wandb.__version__)"
gh api repos/wandb/wandb --jq '{full_name, archived, pushed_at}'
grep -n '"offline"' .venv/lib/python3.12/site-packages/wandb/sdk/wandb_settings.py
```

`--sjfpo.report_to none` is the alternative when no tracker output is wanted at
all; it skips the wandb import entirely.
