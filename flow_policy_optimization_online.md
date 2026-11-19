---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# flow_policy_optimization online

Clone the repo, install it, run the Hopper feasibility gate.

```python
import os

OUTPUT = {"colab": "/content/", "kaggle": "/kaggle/working/", "local": "./"}

def detect_env():
    try:
        if "google.colab" in str(get_ipython()):
            return "colab"
    except NameError:
        pass
    return "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local"

def load_secret(key_name):
    try:
        if ENV_NAME == "colab":
            from google.colab import userdata
            return userdata.get(key_name)
        if ENV_NAME == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key_name)
        return os.getenv(key_name)
    except Exception as exc:
        print(f"load_secret({key_name}) failed: {exc}")
        return None

ENV_NAME = detect_env()
print(f"Env: {ENV_NAME}")
```

```python
%cd $OUTPUT

!rm -rf flow_policy_optimization

token = load_secret("GITHUB_TOKEN")
slug = os.getenv("GITHUB_REPO", "dzungphieuluuky/flow_policy_optimization")
url = f"https://github.com/{slug}.git"
if token:
    url = url.replace("github.com", f"{token}@github.com")
print("Clone:", url.replace(token or "", "<TOKEN>"))

!git clone $url flow_policy_optimization
%cd flow_policy_optimization
!pwd
```

```python
!uv pip install torch numpy einx draccus gymnasium gymnasium-robotics mujoco pyyaml tensorboard wandb
```

```python
# Scripts import the flow_policy_optimization package; a fresh clone needs an install.
import subprocess
import sys

rc = subprocess.call(["uv", "pip", "install", "-e", ".", "--no-deps"])
if rc:
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
if rc:
    raise SystemExit("editable install of flow_policy_optimization failed; see the log above")
print("flow_policy_optimization installed editable")
```

```python
# Guard: every later cell needs this import to resolve.
import sys
subprocess.run([sys.executable, "-c", "import flow_policy_optimization"], check=True)
```

```python
key = load_secret("WANDB_API_KEY")
if key:
    !wandb login $key
else:
    print("No WANDB_API_KEY; runs stay offline.")
```

## Hopper feasibility gate

Does the current method learn on Hopper-v5? Seeds 3407,1,2.
`FEAS_STEPS` is the pilot size; raise it only on the cluster.

- `fpo_pp`: the method as shipped. No flags.
- `fpo_chunk_avg`: plain FPO control (one loss per chunk).

```python
FEAS_ENV = "Hopper-v5"
FEAS_STEPS = 300_000
FEAS_SEEDS = "3407,1,2"
FEAS_WANT = len(FEAS_SEEDS.split(","))

FEAS_ARMS = {
    "fpo_pp": [],
    "fpo_chunk_avg": ["--flow_policy_optimization.do_average_cfm_loss_in_chunk", "true"],
}
```

```python
# Shared helpers for every arm.
import csv
import glob
import os
import statistics as st
import subprocess

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

def read_rows(tag):
    """Summary rows of one arm; [] when the file is absent."""
    path = f"outputs/{tag}/sweep_summary.csv"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def sweep(tag, extra):
    """Run all seeds for one arm; skip if the summary is full."""
    if len(read_rows(tag)) >= FEAS_WANT:
        print(f"{tag}: done, skipping")
        return
    cmd = ["python", "scripts/multiseed.py", "run", "--env", FEAS_ENV,
           "--seeds", FEAS_SEEDS, "--total_steps", str(FEAS_STEPS),
           "--out", f"outputs/{tag}", "--method", "flow_policy_optimization", *extra]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    print(f"=== {tag} exit={subprocess.call(cmd)} ===", flush=True)

def median_last10(tag):
    return [float(r["reward_median_last10"]) for r in read_rows(tag)]

def pooled_se(a, b):
    ses = [st.stdev(v) / len(v) ** 0.5 for v in (a, b) if len(v) > 1]
    return sum(s * s for s in ses) ** 0.5 if ses else float("nan")

def tensorboard_last(run_dir, keys):
    """Last value of the first matching scalar under run_dir/tensorboard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for tb in glob.glob(os.path.join(run_dir, "tensorboard", "*")):
        acc = EventAccumulator(tb, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        for key in keys:
            if key in tags:
                vals = acc.Scalars(key)
                if vals:
                    return float(vals[-1].value)
    return float("nan")
```

```python
for tag, extra in FEAS_ARMS.items():
    sweep(tag, extra)
```

```python
# Gate table: row count (exit 0), random baseline, pooled-SE gap.
scores = {tag: median_last10(tag) for tag in FEAS_ARMS}
for tag, fs in scores.items():
    if len(fs) < FEAS_WANT:
        print(f"{tag:>14}  ({len(fs)}/{FEAS_WANT} seeds)")
        continue
    sd = st.stdev(fs) if len(fs) > 1 else float("nan")
    print(f"{tag:>14}  n={len(fs)}  last10={st.fmean(fs):.1f}  sd={sd:.1f}")

fp = scores.get("fpo_pp")
if fp:
    rows = read_rows("fpo_pp")
    base = st.fmean([tensorboard_last(r["run_dir"], ["eval/random_policy_return"]) for r in rows])
    pmean = st.fmean(fp)
    print(f"\nfpo_pp {pmean:.1f} vs random {base:.1f}")
    print("GATE 2 PASS" if pmean > base else "GATE 2 FAIL: stop, do not scale")

if all(len(v) >= FEAS_WANT for v in scores.values()):
    a, b = scores["fpo_pp"], scores["fpo_chunk_avg"]
    gap, se = st.fmean(a) - st.fmean(b), 2 * pooled_se(a, b)
    ok = abs(gap) > se
    print(f"fpo_pp - fpo_chunk_avg = {gap:+.1f}  (2x pooled SE {se:.1f}; "
          + ("outside noise" if ok else "inside noise") + ")")
```

```python
# Noise floor of fpo_pp. A gain below it decides nothing.
ckpts = sorted(glob.glob("outputs/fpo_pp/*/checkpoints/best"))
if not ckpts:
    print("no fpo_pp best checkpoint yet; run the sweep cell first")
else:
    subprocess.call(["python", "scripts/determinism_probe.py", "--env", FEAS_ENV,
                     "--checkpoint", ckpts[0], "--episodes", "50", "--repeats", "3"])
```
## Performance sweep

Test the current algorithm under different runner configs on Hopper-v5 and
compare frames-per-second against the final return. These numbers are
wall-clock over the whole run (compile warmup included), so they are only
comparable within this notebook at fixed `PERF_STEPS`.

Sweep grid:
- num euler steps: 1, 4, 16 (set via `--flow_policy_optimization.num_steps_zero_start` and
  `--flow_policy_optimization.num_steps_noise_start`)
- mc samples: 8, 16, 64 (set via `--flow_policy_optimization.num_mc_samples`)
- learning rates: 1e-4, 3e-4 (default), 1e-3 (set via `--flow_policy_optimization.policy_lr`)

That is 3 × 3 × 3 = 27 arms, each one seed (3407) at `PERF_STEPS`.

```python
PERF_ENV = "Hopper-v5"
PERF_STEPS = 200_000
PERF_SEEDS = "3407"

EULER_STEPS = [1, 4, 16]
MC_SAMPLES = [8, 16, 64]
LEARNING_RATES = [1e-4, 3e-4, 1e-3]


def perf_arm_tag(euler, mc, lr):
    return f"perf_e{euler}_mc{mc}_lr{lr:g}"


PERF_ARMS = {}
for euler in EULER_STEPS:
    for mc in MC_SAMPLES:
        for lr in LEARNING_RATES:
            tag = perf_arm_tag(euler, mc, lr)
            PERF_ARMS[tag] = [
                "--flow_policy_optimization.num_steps_zero_start", str(euler),
                "--flow_policy_optimization.num_steps_noise_start", str(euler),
                "--flow_policy_optimization.num_mc_samples", str(mc),
                "--flow_policy_optimization.policy_lr", str(lr),
            ]
```

```python
def perf_sweep(tag, extra):
    """Run one arm and print its exit code."""
    if len(read_rows(tag)) >= 1:
        print(f"{tag}: done, skipping")
        return
    cmd = ["python", "scripts/multiseed.py", "run", "--env", PERF_ENV,
           "--seeds", PERF_SEEDS, "--total_steps", str(PERF_STEPS),
           "--out", f"outputs/{tag}", "--method", "flow_policy_optimization", *extra]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    print(f"=== {tag} exit={subprocess.call(cmd)} ===", flush=True)


def perf_rows(tag):
    """Summary rows of one arm; [] when the file is absent."""
    return read_rows(tag)
```

```python
for tag, extra in PERF_ARMS.items():
    perf_sweep(tag, extra)
```

```python
# Compare fps and final return across arms, sorted by reward.
records = []
for tag in PERF_ARMS:
    rows = perf_rows(tag)
    if not rows:
        print(f"{tag:>24}  (no summary yet)")
        continue
    row = rows[0]
    records.append((tag, float(row["fps"]), float(row["reward_mean_last10"]), int(row["global_step"])))
records.sort(key=lambda r: r[2], reverse=True)
print(f"{'arm':>24}  {'fps':>6}  {'reward':>9}  {'step':>7}")
for tag, fps, reward, step in records:
    print(f"{tag:>24}  {fps:>6.0f}  {reward:>9.1f}  {step:>7}")
```

```

## Gate

1. Both arms exit 0; one row per seed.
2. `fpo_pp` beats the random baseline.
3. `fpo_pp` beats `fpo_chunk_avg` by more than 2x the pooled SE and more than
   the probe floor.

All pass: scale up. Any fail: stop. Watch `train/cfm_both_clamped_frac` and
`train/log_ratio_clip_frac` on wandb.
