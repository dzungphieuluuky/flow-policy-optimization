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

```python
import os

OUTPUT_DIR = {
    "colab": "/content/",
    "kaggle": "/kaggle/working/",
    "local": "./",
}
INPUT_DIR = {
    "colab": "/content/",
    "kaggle": "/kaggle/input/",
    "local": "./",
}

def detect_env():
    try:
        if "google.colab" in str(get_ipython()):
            return "colab"
    except NameError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"

def load_secret(key_name):
    try:
        if ENV_NAME == "colab":
            from google.colab import userdata
            return userdata.get(key_name) or None
        if ENV_NAME == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key_name) or None
        return os.getenv(key_name)
    except Exception as exc:
        print(f"load_secret({key_name}) failed: {exc}")
        return None

ENV_NAME = detect_env()
OUTPUT = OUTPUT_DIR[ENV_NAME]
print(f"Environment: {ENV_NAME}")
```

```python
%cd $OUTPUT

!rm -rf myfpo

GITHUB_TOKEN = load_secret("GITHUB_TOKEN")
repo_slug = os.getenv("GITHUB_REPO", "dzungphieuluuky/myfpo")
repo_url = f"https://github.com/{repo_slug}.git"
if GITHUB_TOKEN:
    repo_url = repo_url.replace("github.com", f"{GITHUB_TOKEN}@github.com")
print(f"Cloning from: {repo_url.replace(GITHUB_TOKEN or '', '<TOKEN>') if GITHUB_TOKEN else repo_url}")

!git clone $repo_url myfpo

%cd myfpo

!pwd
```

```python
%%writefile requirements-colab.txt
torch
numpy
einx
draccus
gymnasium
gymnasium-robotics
huggingface-hub
lerobot
mujoco
pyyaml
tensorboard
wandb
```

```python
!uv pip install -r requirements-colab.txt
```

```python
# The scripts import the myfpo package. A fresh clone needs an editable
# install; requirements-colab.txt above only covers the dependencies.
import subprocess
import sys

_install = subprocess.call(["uv", "pip", "install", "-e", ".", "--no-deps"])
if _install != 0:
    _install = subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
if _install != 0:
    raise SystemExit("editable install of myfpo failed; check the log above")
print("myfpo installed editable")
```

```python
# Guard: every later cell depends on this import resolving.
subprocess.run(
    [sys.executable, "-c", "import myfpo, myfpo.training; print('myfpo import OK')"],
    check=True,
)
```

```python
_wandb_key = load_secret("WANDB_API_KEY")
if _wandb_key:
    !wandb login $_wandb_key
else:
    print("WANDB_API_KEY not found; training will run without wandb sync.")
```

```python
import subprocess
import sys

def play(run_dir, env_id="Reacher-v5", ckpt="best", **flags):
    """Evaluate a checkpoint and print the per-episode metrics."""
    cmd = [sys.executable, "scripts/play.py", "--env", env_id,
           "--checkpoint", f"{run_dir}/checkpoints/{ckpt}"]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, check=True)
```

```python
!MPLBACKEND=Agg MUJOCO_GL=egl python scripts/train.py \
    --method myfpo \
    --env Reacher-v5 \
    --seed 3407 \
    --total_steps 1000000 \
    --eval_steps 100 \
    --save_steps 100000
```

```python
play("outputs/myfpo_Reacher-v5_seed3407")
```

```python
ENV = "Humanoid-v5"
SEED = 3407
TOTAL_STEPS = 1_000_000
EVAL_STEPS = 16_384
SAVE_STEPS = 50_000
COMPILE = "false"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

BASE = [
    "--method", "myfpo",
    "--seed", str(SEED),
    "--total_steps", str(TOTAL_STEPS),
    "--eval_steps", str(EVAL_STEPS),
    "--save_steps", str(SAVE_STEPS),
    "--myfpo.report_to", "all",
    "--myfpo.compile", COMPILE,
]

def run_arm(tag, extra=()):
    out = f"outputs/sweep_{tag}"
    cmd = ["python", "scripts/train.py", "--env", ENV, "--out_dir", out, *BASE, *extra]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    print(f"=== {tag} exit={rc} -> {out} ===", flush=True)
    return rc

def arm_dir(tag):
    return f"outputs/sweep_{tag}/myfpo_{ENV}_seed{SEED}"
```

```python
for n in ("2", "4", "8"):
    run_arm(f"exec{n}", ["--myfpo.n_action_steps", n])

for n in ("2", "4", "8"):
    play(arm_dir(f"exec{n}"), ENV, max_steps=300)
```

```python
run_arm("clip020", ["--myfpo.clip_param", "0.2"])
run_arm("spo020", ["--myfpo.spo_clip_param", "0.2"])
run_arm("lr1e3", ["--myfpo.policy_lr", "1e-3", "--myfpo.critic_lr", "1e-3"])

for tag in ("clip020", "spo020", "lr1e3"):
    play(arm_dir(tag), ENV, max_steps=300)
```

```python
import csv
import glob
import statistics

def arm_summary(tag):
    hits = glob.glob(f"outputs/sweep_{tag}/**/eval_history.csv", recursive=True)
    if not hits:
        return None
    with open(hits[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    blocks, cur, prev = [], [], None
    for r in rows:
        ep = int(r["episode"])
        if prev is not None and ep <= prev:
            blocks.append(cur)
            cur = []
        cur.append(r)
        prev = ep
    if cur:
        blocks.append(cur)
    means = [statistics.fmean(float(r["return"]) for r in b) for b in blocks]
    if not means:
        return None
    best_i = max(range(len(means)), key=means.__getitem__)
    return {
        "evals": len(means),
        "best_return": means[best_i],
        "best_at_eval": best_i + 1,
        "final_return": means[-1],
        "decay_pct": 100 * (means[-1] - means[best_i]) / max(abs(means[best_i]), 1e-9),
    }

tags = ["exec2", "exec4", "exec8"]
print(f"{'arm':>12} {'evals':>6} {'best':>10} {'@eval':>6} {'final':>10} {'decay%':>8}")
for tag in tags:
    s = arm_summary(tag)
    if s is None:
        print(f"{tag:>12}  (no eval_history.csv)")
        continue
    print(f"{tag:>12} {s['evals']:>6} {s['best_return']:>10.1f} "
          f"{s['best_at_eval']:>6} {s['final_return']:>10.1f} {s['decay_pct']:>7.1f}%")
```

```python
FORGET_SEEDS = [0, 1, 2]
FORGET_ENV = ENV
FORGET_STEPS = TOTAL_STEPS

FORGET_ARMS = {
    "ctrl": [],
    "actor": ["--myfpo.policy_lr", "5e-6"],
    "critic": ["--myfpo.critic_lr", "5e-5"],
    "mc": ["--myfpo.num_mc_samples", "32"],
}
FORGET_ARMS["critics"] = ["--myfpo.num_critics", "3"]

TB_SNAPSHOT = {}

def tb_dirs(root="."):
    return {
        d for d in glob.glob(os.path.join(root, "**", "tensorboard", "*"),
                             recursive=True)
        if glob.glob(os.path.join(d, "*"))
    }

def run_forget_arm(arm, seed):
    tag = f"forget_{arm}_s{seed}"
    out = f"outputs/{tag}"
    cmd = [
        "python", "scripts/train.py", "--env", FORGET_ENV, "--out_dir", out,
        "--seed", str(seed),
        "--total_steps", str(FORGET_STEPS),
        "--eval_steps", str(EVAL_STEPS),
        "--save_steps", str(SAVE_STEPS),
        "--myfpo.report_to", "all",
        "--myfpo.compile", COMPILE,
        *FORGET_ARMS[arm],
    ]
    before = tb_dirs()
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    new = sorted(tb_dirs() - before)
    if len(new) == 1:
        TB_SNAPSHOT[(arm, seed)] = new[0]
    elif new:
        print(f"  WARNING: {len(new)} new tensorboard dirs, cannot attribute:\n"
              + "\n".join(new), flush=True)
    print(f"=== {tag} exit={rc} tb={new} ===", flush=True)
    return rc

for arm in FORGET_ARMS:
    for seed in FORGET_SEEDS:
        run_forget_arm(arm, seed)
```

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEY_ALIASES = {
    "eval/return_mean": "eval/mean_return",
    "eval/return_min": "eval/min_return",
    "eval/return_max": "eval/max_return",
    "eval/return_std": "eval/std_return",
    "eval/return_best": "eval/best_mean_return",
    "eval/random_policy_return": "eval/sample_random_baseline",
    "train/boot_value_relative_drift": "train/boot_value_delta",
    "train/grad_norm_before_clip": "train/grad_norm_preclip",
}

def tb_series(run_dir, key):
    if not run_dir:
        return []
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    tag = key if key in tags else KEY_ALIASES.get(key)
    if tag is None or tag not in tags:
        return []
    return [(e.step, e.value) for e in acc.Scalars(tag)]

def decline_report(run_dir, label=""):
    mean = tb_series(run_dir, "eval/return_mean")
    if len(mean) < 3:
        return None
    steps, vals = zip(*mean)
    peak_i = max(range(len(vals)), key=vals.__getitem__)
    peak, final = vals[peak_i], vals[-1]
    fall = peak - final

    width = [v for _, v in tb_series(run_dir, "eval/return_ci_width")]
    widths = width[peak_i:] or width or [float("nan")]
    ci = max(widths)

    zero = dict(tb_series(run_dir, "eval/zero_start_return_mean"))
    noise = dict(tb_series(run_dir, "eval/noise_start_return_mean"))
    zero_fall = peak - zero.get(steps[peak_i], float("nan"))
    noise_fall = peak - noise.get(steps[peak_i], float("nan"))

    train_loss = tb_series(run_dir, "train/l_old_mean")
    missing = [
        name for name, series in (
            ("eval/return_ci_width", width),
            ("eval/zero_start_return_mean", list(zero.values())),
            ("eval/noise_start_return_mean", list(noise.values())),
            ("train/l_old_mean", train_loss),
        ) if not series
    ]
    return {
        "label": label or os.path.basename(run_dir),
        "peak": peak, "final": final, "fall": fall,
        "fall_pct": 100 * fall / max(abs(peak), 1e-9),
        "peak_step": steps[peak_i], "ci_width_at_peak": ci,
        "zero_start_fall": zero_fall, "noise_start_fall": noise_fall,
        "train_loss_first": train_loss[0][1] if train_loss else float("nan"),
        "train_loss_last": train_loss[-1][1] if train_loss else float("nan"),
        "missing": missing,
    }
```

```python
r = decline_report(TB_SNAPSHOT.get(("ctrl", FORGET_SEEDS[0])), "ctrl_s0")
if r is None:
    print("no eval/return_mean series yet for ctrl seed", FORGET_SEEDS[0])
else:
    for k, v in r.items():
        print(f"{k:22s} {v}")

    if r["missing"]:
        print("\nMISSING series (this run predates them): " + ", ".join(r["missing"]))

    if not r["missing"] and r["fall"] <= r["ci_width_at_peak"]:
        print("\nVERDICT: the fall is inside the eval interval. "
              "This is noise, not forgetting. Stop here.")

    z, nz = r["zero_start_fall"], r["noise_start_fall"]
    if z == z and nz == nz and min(abs(z), abs(nz)) < 0.25 * r["fall"]:
        print(f"\nVERDICT: one-sided. zero_start fall {z:.1f}, "
              f"noise_start fall {nz:.1f}. Not forgetting; the two "
              "distributions diverged. Move eval_sampling and stop_eval_sampling.")

    tl = r["train_loss_last"] / max(r["train_loss_first"], 1e-9)
    if tl == tl and r["fall"] > r["ci_width_at_peak"] and tl < 0.95:
        print(f"\nVERDICT: train/l_old_mean fell to {tl:.2f} of its start while "
              "eval fell. Training still improved. Suspect the eval path "
              "(swap_in_ema / swap_out_ema), not the policy.")
```

```python
def forget_rows():
    rows = []
    for arm in FORGET_ARMS:
        for seed in FORGET_SEEDS:
            rep = decline_report(TB_SNAPSHOT.get((arm, seed)), f"{arm}_s{seed}")
            if rep:
                rows.append(rep)
    return rows

rows = forget_rows()
if rows:
    print(f"{'arm':>10} {'seed':>5} {'peak':>10} {'final':>10} {'fall%':>8} {'ci@pk':>9}")
    for rep in rows:
        print(f"{rep['label'].split('_s')[0]:>10} {rep['label'].split('_s')[-1]:>5} "
              f"{rep['peak']:>10.1f} {rep['final']:>10.1f} {rep['fall_pct']:>7.1f}% "
              f"{rep['ci_width_at_peak']:>9.1f}")

    print(f"\n{'arm':>10} {'peak mean':>10} {'final mean':>11} {'fall%':>8} {'seed sd':>8}")
    for arm in FORGET_ARMS:
        g = [rep for rep in rows if rep["label"].startswith(arm + "_s")]
        if not g:
            continue
        peaks = [rep["peak"] for rep in g]
        falls = [rep["fall_pct"] for rep in g]
        sd = statistics.stdev(falls) if len(falls) > 1 else float("nan")
        print(f"{arm:>10} {statistics.fmean(peaks):>10.1f} "
              f"{statistics.fmean(rep['final'] for rep in g):>11.1f} "
              f"{statistics.fmean(falls):>7.1f}% {sd:>7.1f}%")

    ctrl = [rep for rep in rows if rep["label"].startswith("ctrl_s")]
    if len(ctrl) > 1:
        spread = max(rep["fall_pct"] for rep in ctrl) - min(rep["fall_pct"] for rep in ctrl)
        if spread > 0:
            print(f"\ncontrol fall spread across seeds: {spread:.1f} points. "
                  "An arm that beats the control by less than that is noise.")
        else:
            print("\ncontrol fall% is identical across seeds. Suspicious: the "
                  "seeds are probably not reaching different rollouts, or "
                  "TOTAL_STEPS is too short for the seeds to diverge.")
```

```python
CKPT = sorted(glob.glob("outputs/forget_ctrl_s0/*/checkpoints/final"))[-1]
print("continuing from", CKPT)
subprocess.call([
    "python", "scripts/train.py", "--env", FORGET_ENV,
    "--out_dir", "outputs/forget_ctrl_restart",
    "--seed", str(FORGET_SEEDS[0]),
    "--total_steps", str(FORGET_STEPS),
    "--eval_steps", str(EVAL_STEPS),
    "--save_steps", str(SAVE_STEPS),
    "--myfpo.report_to", "all",
    "--myfpo.compile", COMPILE,
    "--resume", CKPT,
])
```

```python
from myfpo.flow import MyfpoFlowModel, MyfpoFlowResMLP

OBS_DIM, ACT_DIM = 11, 3
HIDDEN, HEADS, TIME_EMBED = 128, 2, 32
MULT, UNITS = 2.0, 4

def n_params(module):
    return sum(p.numel() for p in module.parameters())

def dit_actor(depth):
    return MyfpoFlowModel(ACT_DIM, 8, HIDDEN, depth, HEADS, MULT, TIME_EMBED)

def resmlp_actor(blocks, units=UNITS, mult=MULT):
    return MyfpoFlowResMLP(ACT_DIM, 8, HIDDEN, blocks, units, mult, HEADS, TIME_EMBED)

print(f"{'actor':>8} {'blocks':>7} {'Dense':>7} {'params':>12} {'x dit_d3':>9}")
base = n_params(dit_actor(3))
for label, knob, actor, dense in (
    ("dit",    1, dit_actor(1), 6),
    ("dit",    3, dit_actor(3), 18),
    ("dit",    5, dit_actor(5), 30),
    ("resmlp", 1, resmlp_actor(1), 4),
    ("resmlp", 2, resmlp_actor(2), 8),
    ("resmlp", 4, resmlp_actor(4), 16),
    ("resmlp", 8, resmlp_actor(8), 32),
):
    p = n_params(actor)
    print(f"{label:>8} {knob:>7} {dense:>7} {p:>12,} {p / base:>8.2f}x")
```

```python
ARCH_SEEDS = [0, 1, 2]
ARCH_ENV = ENV
ARCH_STEPS = TOTAL_STEPS

ARCH_ARMS = {
    "dit_d3":    ("dit",    "--myfpo.dit_depth",     3),
    "resmlp_b2": ("resmlp", "--myfpo.resmlp_blocks", 2),
    "resmlp_b4": ("resmlp", "--myfpo.resmlp_blocks", 4),
}

ARCH_TB = {}

def arch_dirs(root="."):
    return {
        d for d in glob.glob(os.path.join(root, "**", "tensorboard", "*"),
                             recursive=True)
        if glob.glob(os.path.join(d, "*"))
    }

def run_arch_arm(arm, seed):
    arch, flag, value = ARCH_ARMS[arm]
    tag = f"arch_{arm}_s{seed}"
    before = arch_dirs()
    cmd = [
        "python", "scripts/train.py", "--env", ARCH_ENV,
        "--out_dir", f"outputs/{tag}", "--seed", str(seed),
        "--total_steps", str(ARCH_STEPS),
        "--eval_steps", str(EVAL_STEPS),
        "--save_steps", str(SAVE_STEPS),
        "--myfpo.report_to", "all",
        "--myfpo.compile", COMPILE,
        "--myfpo.actor_arch", arch, flag, str(value),
    ]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    new = sorted(arch_dirs() - before)
    if len(new) == 1:
        ARCH_TB[(arm, seed)] = new[0]
    elif new:
        print(f"  WARNING: {len(new)} new tensorboard dirs, cannot attribute:\n"
              + "\n".join(new), flush=True)
    print(f"=== {tag} exit={rc} tb={new} ===", flush=True)
    return rc

for arm in ARCH_ARMS:
    for seed in ARCH_SEEDS:
        run_arch_arm(arm, seed)
```

```python
def arch_rows():
    rows = []
    for arm in ARCH_ARMS:
        for seed in ARCH_SEEDS:
            rep = decline_report(ARCH_TB.get((arm, seed)), f"{arm}_s{seed}")
            if rep:
                rows.append(rep)
    return rows

def paper_depth(arm):
    arch, _, value = ARCH_ARMS[arm]
    return 6 * int(value) if arch == "dit" else int(value) * 4

def actor_params(arm):
    arch, _, value = ARCH_ARMS[arm]
    return n_params(dit_actor(int(value))) if arch == "dit" \
        else n_params(resmlp_actor(int(value)))

rows = arch_rows()
if rows:
    print(f"{'arm':>12} {'arch':>7} {'Dense':>6} {'params':>12} {'peak':>9} "
          f"{'final':>9} {'fall%':>7}")
    for arm in ARCH_ARMS:
        g = [rep for rep in rows if rep["label"].startswith(arm + "_s")]
        if not g:
            continue
        print(f"{arm:>12} {ARCH_ARMS[arm][0]:>7} {paper_depth(arm):>6} "
              f"{actor_params(arm):>12,} "
              f"{statistics.fmean(rep['peak'] for rep in g):>9.1f} "
              f"{statistics.fmean(rep['final'] for rep in g):>9.1f} "
              f"{statistics.fmean(rep['fall_pct'] for rep in g):>6.1f}%")

    print(f"\n{'arm':>12} {'peak sd':>9} {'final sd':>9} {'n':>3}")
    for arm in ARCH_ARMS:
        g = [rep for rep in rows if rep["label"].startswith(arm + "_s")]
        if not g:
            continue
        ps = statistics.stdev([rep["peak"] for rep in g]) if len(g) > 1 else float("nan")
        fs = statistics.stdev([rep["final"] for rep in g]) if len(g) > 1 else float("nan")
        print(f"{arm:>12} {ps:>9.1f} {fs:>9.1f} {len(g):>3}")

    ctrl = [rep for rep in rows if rep["label"].startswith("dit_d3_s")]
    if len(ctrl) > 1:
        print(f"\ncontrol peak sd across seeds: "
              f"{statistics.stdev([rep['peak'] for rep in ctrl]):.1f}. "
              "An arm that beats dit_d3 by less than that is noise.")
```


---

## Kaggle suite — the experiments that gate the next decision

Run the clone/install/wandb cells at the top of this notebook first.
Notes for Kaggle:

- No Blackwell GPU on Kaggle, so `resolve_storage_dtype` returns fp32: the storage
  dtype is consistent between all nodes here. That kills the known local-vs-cluster
  dtype hazard for these runs.
- Two suites from the experiment plan are **not runnable from flags yet**: the
  direct-competitor baselines (ReFPO, Flow-DPPO) and Design A (per-token
  advantage). The ratio-bias quantifier is also not implemented. This notebook
  covers only the arms the repo already ships.
- Budget: Kaggle caps a session at ~9-12 GPU hours. `SUITE_STEPS` is sized for a
  pilot; raise it on the cluster once the pilot decides the direction.

```python
# Suite knobs. One arm = one sweep of SUITE_SEEDS.
import os

SUITE_ENV = "Hopper-v5"
SUITE_STEPS = 300_000
SUITE_SEEDS = "3407,1,2"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

# tag -> extra flags; the default arm (no flags) is FPO++ as validated by the pins.
ARMS = {
    # tag -> extra flags. No flags (fpo_pp) is FPO++ as validated by the pins.
    "fpo_pp":         [],
    # FPO behaviour: one CFM loss per chunk instead of one per token.
    "fpo_chunk_avg":  ["--myfpo.do_average_cfm_loss_in_chunk", "true"],
    # Symmetric PPO clipping: the ASPO ablation.
    "aspo_ppo":       ["--myfpo.trust_region_mode", "ppo"],
    # Credibility ablations (docs/ablation_reference.md).
    "critic_drop0":   ["--myfpo.critic_dropout", "0.0"],
    "critics_1":      ["--myfpo.num_critics", "1"],
    "critics_5":      ["--myfpo.num_critics", "5"],
}

def suite_arm(tag):
    return f"outputs/suite_{tag}"

def suite_summary_path(tag):
    return os.path.join(suite_arm(tag), "sweep_summary.csv")
```

```python
# Run every arm whose summary is missing. Sequential by design: the runner
# already guards against parallel workers contending for cores and RNG.
import os
import subprocess

for tag, extra in ARMS.items():
    if os.path.exists(suite_summary_path(tag)):
        print(f"{tag}: summary exists, skipping")
        continue
    cmd = [
        "python", "scripts/multiseed.py", "run",
        "--env", SUITE_ENV,
        "--seeds", SUITE_SEEDS,
        "--total_steps", str(SUITE_STEPS),
        "--out", suite_arm(tag),
        "--method", "myfpo",
        *extra,
    ]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    print(f"=== {tag} exit={rc} ===", flush=True)
```

```python
# The fpo_pp arm's noise floor. A gain below it decides nothing.
# The probe loads the control arm's best checkpoint.
import csv as _csv
import glob

run_dirs = []
if os.path.exists(suite_summary_path("fpo_pp")):
    with open(suite_summary_path("fpo_pp"), encoding="utf-8") as fh:
        run_dirs = [r["run_dir"] for r in _csv.DictReader(fh)]
ckpt = sorted(glob.glob(f"{run_dirs[0]}/checkpoints/best") if run_dirs else [])
if ckpt:
    subprocess.call([
        "python", "scripts/determinism_probe.py",
        "--env", SUITE_ENV,
        "--checkpoint", ckpt[0],
        "--episodes", "50",
        "--repeats", "3",
    ])
else:
    print("fpo_pp run not found; run the sweep cell first")
```

```python
# One table over every finished arm. reward_mean_last10 is the last-window
# survival read; best_return is the peak read. Compare arms on the last
# window (a peak read rewards checkpoint luck).
import csv
import statistics as st

def suite_rows(tag):
    with open(suite_summary_path(tag), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

readings = {}
for tag in ARMS:
    if not os.path.exists(suite_summary_path(tag)):
        print(f"{tag:>14}  (no summary)")
        continue
    rows = suite_rows(tag)
    readings[tag] = [float(r["reward_mean_last10"]) for r in rows]

print(f"\n{'arm':>14} {'seeds':>6} {'last10 mean':>12} {'sd':>8}")
for tag, fs in readings.items():
    mean = st.fmean(fs)
    sd = st.stdev(fs) if len(fs) > 1 else float("nan")
    print(f"{tag:>14} {len(fs):>6} {mean:>12.1f} {sd:>8.1f}")

if "fpo_pp" in readings and "fpo_chunk_avg" in readings:
    gap = st.fmean(readings["fpo_pp"]) - st.fmean(readings["fpo_chunk_avg"])
    ses = []
    for fs in (readings["fpo_pp"], readings["fpo_chunk_avg"]):
        if len(fs) > 1:
            ses.append(st.stdev(fs) / len(fs) ** 0.5)
    if ses:
        pooled = sum(s * s for s in ses) ** 0.5
        verdict = "outside the seed noise" if abs(gap) > 2 * pooled else "inside the seed noise"
        print(f"\nFPO++ minus chunk-averaged FPO: {gap:+.1f} ({verdict}; "
              f"2x pooled SE = {2 * pooled:.1f})")
```

```python
# Head-to-head comparison of the two main arms.
if os.path.exists(suite_summary_path("fpo_pp")) and \
        os.path.exists(suite_summary_path("fpo_chunk_avg")):
    subprocess.call([
        "python", "scripts/multiseed.py", "compare",
        suite_summary_path("fpo_pp"), suite_summary_path("fpo_chunk_avg"),
    ])
```

Decision checklist (run all arms first, then answer):

1. Is `fpo_pp` above `fpo_chunk_avg` by more than 2x the pooled SE **and** more
   than the determinism-probe floor? If yes, Gate 1 holds for Hopper: scale to
   more envs x 5 seeds on the cluster. If no: quantify the approximate-ratio
   bias before spending more sweeps.
2. Does `aspo_ppo` change the return in a meaningful way? Watch
   `train/value_pred_member_gap_p95` and the clamp guards on wandb.
3. Do `critic_drop0` / `critics_1` / `critics_5` move the value diagnostics
   without moving the return? Then the defaults are defensible ablation notes.
```
