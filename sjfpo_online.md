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

# Spectral Joint Flow (SJFPO) — Online Colab / Kaggle (with internet)

Clone the repo with a GitHub token, install requirements, then run training
via `!python scripts/train.py`. No source is embedded in the notebook.


## 0. Secrets and environment

1. Set `GITHUB_TOKEN` in the environment (Colab secrets or Kaggle secrets).
2. The token is read via `load_secret()` using the runtime environment.
3. Run all cells top to bottom.

```python
# Detect runtime and load secrets from the right store.
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

ENV_NAME = detect_env()
print(f"Environment: {ENV_NAME}")

def load_secret(key_name):
    try:
        if ENV_NAME == "colab":
            from google.colab import userdata
            return userdata.get(key_name) or None
        if ENV_NAME == "kaggle":
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key_name) or None
        return os.getenv(key_name)
    except Exception as e:
        print(f"load_secret({key_name}) failed: {e}")
        return None

ENV_NAME = detect_env()
OUTPUT = OUTPUT_DIR[ENV_NAME]
```

```python
# Clone the repo using the GitHub token from the environment/store.
%cd $OUTPUT
!rm -rf spectral-joint-policy
GITHUB_TOKEN = load_secret("GITHUB_TOKEN")
repo_slug = os.getenv("GITHUB_REPO", "dzungphieuluuky/spectral-joint-policy")
repo_url = f"https://github.com/{repo_slug}.git"
if GITHUB_TOKEN:
    repo_url = repo_url.replace("github.com", f"{GITHUB_TOKEN}@github.com")
print(f"Cloning from: {repo_url.replace(GITHUB_TOKEN or '', '<TOKEN>') if GITHUB_TOKEN else repo_url}")
!git clone $repo_url spectral-joint-policy
!git status
# Move into the cloned repo and run directly (no embedded source).
%cd spectral-joint-policy
!pwd
```

```python
%%writefile requirements-colab.txt
# Deep Learning
torch
torchvision
torchaudio
torchcodec
jax
jaxlib
accelerate
diffusers
datasets
huggingface-hub
safetensors
bitsandbytes
einops
einx

# Reinforcement Learning / Robot Learning
gymnasium
gymnasium-robotics
pettingzoo
mujoco
mujoco-mjx
mujoco-warp
skrl
lerobot
jax[cuda12]
flax
jaxlib
optax
orbax-checkpoint

# Scientific / Vision
numpy
scipy
pandas
matplotlib
scikit-learn
scikit-image
opencv-python-headless
pillow
imageio
imageio-ffmpeg
av
moviepy

# Data / Storage
pyarrow

# Experiment Tracking / Config
wandb
tensorboard
draccus
tqdm
```

```python
# Detect runtime and choose install source.
if ENV_NAME == "colab":
    !uv pip install -r requirements-colab.txt
    # !sudo apt-get update
    # !sudo apt-get install -y pkg-config python3-dev \
    # libavformat-dev libavcodec-dev libavdevice-dev \
    # libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
    !uv pip uninstall av
    !uv pip install "av<15.0.0" --only-binary=:all:
else:
    !uv pip install -r requirements-colab.txt
    # !sudo apt-get update
    # !sudo apt-get install -y pkg-config python3-dev \
    # libavformat-dev libavcodec-dev libavdevice-dev \
    # libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
    !uv pip uninstall av
    !uv pip install "av<15.0.0" --only-binary=:all:
```

```python
# wandb login from secret store (runs before any training cell).
_wandb_key = load_secret("WANDB_API_KEY")
if _wandb_key:
    !wandb login $_wandb_key
else:
    print("WANDB_API_KEY not found; training will run without wandb sync.")
```

```python
# Run training directly from the cloned repo.
!MPLBACKEND=Agg MUJOCO_GL=egl python scripts/train.py \
    --method sjfpo \
    --env Reacher-v5 \
    --seed 3407 \
    --total_steps 1000000 \
    --eval_steps 100 \
    --save_steps 100000 \
    --baseline fpo
```

Note: the original cell passed `--config configs/baselines/fpo.yaml`. That file
was deleted in `d0f97c2`, so the run died with a missing-file or missing-feature
error. `--baseline fpo` is the live equivalent and prints the full diff it
applies to stderr.

## 1. Sweeps

Sequential by design. `num_envs=64` forks a vector-env pool, so run **one arm at
a time** even on a big GPU.

Before running anything: the last logged Humanoid run peaked at 131,072 steps
and decayed 46% by 614,400. Its `train/grad_norm_preclip` read 2291 against a
clip of 1, so the actor/critic split is unknown — `train/actor_grad_norm_preclip`
and `train/critic_grad_norm_preclip` were collapsed into one key in `fc8724b`.
**Restore them before trusting any arm below**; see `docs/DIAGNOSE_DEGRADATION.md`
step 0 for the exact patch. Without it the leading hypothesis is untestable.

```python
# Shared settings. ENV and TOTAL_STEPS are the only two things to change.
ENV = "Humanoid-v5"          # "Reacher-v5" is far cheaper if you just want a smoke
SEED = 3407
TOTAL_STEPS = 200_000        # the peak was at 131k, so this brackets peak + onset
EVAL_STEPS = 16_384          # 12 evals; the original cadence of 8192 gave only 4
SAVE_STEPS = 50_000
COMPILE = "false"            # the logged run used false

import os, subprocess

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

BASE = [
    "--method", "sjfpo",
    "--seed", str(SEED),
    "--total_steps", str(TOTAL_STEPS),
    "--eval_steps", str(EVAL_STEPS),
    "--save_steps", str(SAVE_STEPS),
    "--sjfpo.report_to", "tensorboard",   # no WANDB_API_KEY needed
    "--sjfpo.compile", COMPILE,
]

def run_arm(tag, extra=()):
    out = f"outputs/sweep_{tag}"
    cmd = ["python", "scripts/train.py", "--env", ENV, "--out_dir", out, *BASE, *extra]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    print(f"=== {tag} exit={rc} -> {out} ===", flush=True)
    return rc
```

### Sweep 1 — is the spectral prior doing anything?

The logged run used `freq_weighted_time` with **no white control**, so the prior
and the clip change are confounded. This is the first question.

```python
# Control first. Everything else identical.
run_arm("white", ["--sjfpo.noise_scaling_mode", "white"])

# The spectral arm, matched seed and step budget.
run_arm("spectral", ["--sjfpo.noise_scaling_mode", "freq_weighted_time"])
```

Read `train/noise_band_energy_high_over_low` (renamed from `..._hf_lf_ratio` in
`a001fa8`; the logged run predates it). That run read **0.378** at `beta=1.0`,
which is the check that the colouring actually ran. If it reads ~1.0 the prior
is not being applied and the arm is meaningless.

### Sweep 2 — beta, only if sweep 1 shows the prior helps

```python
for beta in ("0.5", "1.0", "2.0"):
    run_arm(f"beta{beta}", [
        "--sjfpo.noise_scaling_mode", "freq_weighted_time",
        "--sjfpo.noise_scaling_beta", beta,
    ])
```

`noise_scaling_beta=0` is white, so sweep 2 subsumes sweep 1's white arm. If you
only have budget for one sweep, run sweep 2 with `beta` in `0.0, 0.5, 1.0`.

### Sweep 3 — the trust region was inert

The logged run showed `log_ratio_max=0.0717` against a cap of 10 and
`log_ratio_clip_frac=0`, with `flow_ratio_std=0.0098` against `clip_param=0.05`.
Measured on that data, **0.00%** of ratios fell outside the clip band, so neither
the PPO clip nor the SPO quadratic was constraining anything.

```python
# Wider trust region, so the mechanism can actually bind.
run_arm("clip020", ["--sjfpo.clip_param", "0.2"])

# Higher LR, to separate "the region is too tight" from "the steps are too small".
run_arm("lr1e3", ["--sjfpo.policy_lr", "1e-3", "--sjfpo.critic_lr", "1e-3"])
```

Do **not** sweep `cfm_clamp` or `log_ratio_cap`. The logged run had
`cfm_above_clamp_frac=0` and `log_ratio_clip_frac=0`; neither bound, so neither
is a lever.

### Reading the results

`eval_history.csv` is appended once per eval, with columns
`[episode, return, length, success, jerk]`. The `episode` index restarts at 0 on
every eval, so block boundaries are read off the data rather than assumed from
`--sjfpo.eval_max_episodes`.

```python
import csv, glob, statistics

def arm_summary(tag):
    hits = glob.glob(f"outputs/sweep_{tag}/**/eval_history.csv", recursive=True)
    if not hits:
        return None
    with open(hits[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # `episode` restarts at 0 per eval, so a non-increasing step is a new block.
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

tags = ["white", "spectral"]
print(f"{'arm':>12} {'evals':>6} {'best':>10} {'@eval':>6} {'final':>10} {'decay%':>8}")
for tag in tags:
    s = arm_summary(tag)
    if s is None:
        print(f"{tag:>12}  (no eval_history.csv)")
        continue
    print(f"{tag:>12} {s['evals']:>6} {s['best_return']:>10.1f} "
          f"{s['best_at_eval']:>6} {s['final_return']:>10.1f} {s['decay_pct']:>7.1f}%")
```

What to conclude:

| Observation | Meaning |
|---|---|
| `decay_pct` near 0 in both arms | the decay is not clip- or prior-specific; look at the critic |
| spectral best but decaying | the prior helps and something later undoes it |
| white better than spectral | the 1/f prior does not pay on this env — stop spending runs on it |
| `best_at_eval` small in both | both peak early, which is the original symptom |

One arm is not a result. Two seeds minimum before claiming anything, and note
that `seed` is a top-level flag, not `--sjfpo.seed`.
