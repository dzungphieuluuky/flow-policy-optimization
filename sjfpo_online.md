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

# myfpo — Online Colab / Kaggle (with internet)

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
!rm -rf myfpo
GITHUB_TOKEN = load_secret("GITHUB_TOKEN")
repo_slug = os.getenv("GITHUB_REPO", "dzungphieuluuky/myfpo")
repo_url = f"https://github.com/{repo_slug}.git"
if GITHUB_TOKEN:
    repo_url = repo_url.replace("github.com", f"{GITHUB_TOKEN}@github.com")
print(f"Cloning from: {repo_url.replace(GITHUB_TOKEN or '', '<TOKEN>') if GITHUB_TOKEN else repo_url}")
!git clone $repo_url myfpo
!git status
# Move into the cloned repo and run directly (no embedded source).
%cd myfpo
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
jax[cuda13]
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

### Evaluation helper

`scripts/play.py` scores the checkpoint and writes the mp4. This wraps the
call and embeds the file. `MUJOCO_GL=egl` matches the training cells, so the
render path works without a display.

```python
import os
import subprocess
import sys
from pathlib import Path

from IPython.display import Video, display

# play.py renders through rgb_array, which needs a GL backend, not a window.
PLAY_ENV = {**os.environ, "MUJOCO_GL": "egl"}


def play(run_dir, env_id="Reacher-v5", ckpt="best", **flags):
    """Run scripts/play.py on a run directory and embed its video.

    `run_dir` is `<out_dir>/sjfpo_<env>_seed<seed>`. Keywords become play.py
    flags, e.g. `max_steps=300` for `--max_steps 300`.
    """
    mp4 = Path(run_dir) / f"play_{ckpt}.mp4"
    cmd = [sys.executable, "scripts/play.py", "--env", env_id,
           "--checkpoint", f"{run_dir}/checkpoints/{ckpt}", "--video_out", str(mp4)]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, env=PLAY_ENV, check=True)
    display(Video(str(mp4), embed=True))
    return mp4
```

## 1. The algorithm

FPO++ ([arXiv:2602.02481](https://arxiv.org/abs/2602.02481)). A flow-matching actor
trained online from rewards, with no likelihood computation and no reference
model.

The policy plans `action_horizon` steps and executes the first `n_action_steps`
open-loop, then replans from the observation it lands on. That chunk is the
macro-action, so the environment is treated as a macro-MDP.

**The ratio.** For a Gaussian policy the PPO ratio needs `π(a|o)`. For a flow it
would need a divergence integral, which is expensive. FPO substitutes the
conditional flow-matching loss:

```
r = exp(L_CFM_old - L_CFM_new)
```

with `L_CFM` evaluated over `num_mc_samples` Monte-Carlo `(t, eps)` draws,
frozen once per rollout and reused for both the old and the new pass. Those
draws are **not** the noise the behaviour rollout saw: the sampler draws its
own `eps` inside `predict_action_chunk` and discards it, and the rollout buffer
never stores it. Only old-vs-new share draws. The ratio's granularity is two
knobs, as in FPO++: `--sjfpo.do_average_cfm_loss_in_chunk` folds the chunk's
tokens and `--sjfpo.cfm_average_group_size` folds the Monte-Carlo pairs. With
the token axis folded you get one ratio per `(t, eps)` pair, which is FPO++
Eq. 10.

**ASPO, the trust region.** Asymmetric, with two widths. `A > 0` takes PPO
clipping at `--sjfpo.clip_param`, `min(r*A, clip(r)*A)`. `A < 0` takes the SPO
quadratic at its own `--sjfpo.spo_clip_param`. Both default to 0.01, and the two
are separate knobs, so widening one leaves the other alone. The SPO branch keeps
a gradient where PPO's clip saturates to zero. Negative advantages are where a
plain PPO clip stops learning.

**Sampling.** Euler integration, `num_steps_noise_start` steps. Both flow starts
use the same Euler loop, so the step counts differ only in the start point:
`num_steps_zero_start` when the chain starts at `eps=0`, and
`num_steps_noise_start` when it starts at a drawn `eps`.

Eval runs both starts and reports the one `eval_sampling` names. The default is
`noise_start`, because that is the distribution training collects from and the
critic bootstraps from. `stop_eval_sampling` also defaults to `noise_start`, so
the stopping rules read the distribution the run optimizes. Set both to
`zero_start` to measure deployment behaviour instead. `--sjfpo.report_to` is
unrelated: it picks the tracker, not the start.

```python
# Run training directly from the cloned repo.
!MPLBACKEND=Agg MUJOCO_GL=egl python scripts/train.py \
    --method sjfpo \
    --env Reacher-v5 \
    --seed 3407 \
    --total_steps 1000000 \
    --eval_steps 100 \
    --save_steps 100000
```

The defaults are FPO++'s released values where the repo has a claim to them:
16 MC samples, ASPO, 10 sampling steps, and `policy_lr=1e-5`. Several others are
repo-specific, not FPO++ values: `cfm_clamp=20.0`, `log_ratio_cap=10.0`,
`cfm_loss_reduction="mean"`, `do_average_cfm_loss_in_chunk=False`,
`min_timestep`/`max_timestep` of 0.005/0.995 with `timestep_sampling_mode`
`fpo_beta`, and `action_perturb_std=0.0`. An FPO or FPO++ arm is a
`dataclasses.replace` of the live config, so write the overrides out and log the
diff. Do not assume a future default change leaves a comparison intact.

`policy_lr=1e-5` is tuned for fine-tuning a pretrained flow-BC policy,
not for training one from scratch, so on Reacher or Humanoid from scratch expect
a flat curve and read the `lr1e3` arm before blaming the algorithm.

```python
# Evaluate and play the run above. `best` beats `final` when both exist.
play("outputs/sjfpo_Reacher-v5_seed3407")
```

The run directory is `<out_dir>/sjfpo_<env>_seed<seed>`, resolved by
`resolve_run_dir` in `scripts/train.py`. Pass `--out_dir` to `scripts/train.py`
to shorten it.

## 2. Reading the metrics

| Key | Meaning |
|---|---|
| `train/policy_loss` | the ASPO surrogate; near zero means the ratio has collapsed to 1 |
| `train/log_ratio_max` | largest log-ratio seen. Against the cap of 10, anything under ~0.2 means the trust region never binds |
| `train/log_ratio_clip_frac` | fraction of log-ratios past `log_ratio_cap`. 0 means the cap never fired, nothing more |
| `train/clipfrac` | fraction of ratios outside the `clip_param` band. 0 means the PPO clip never bound. Under ASPO it ignores `spo_clip_param`, so it is not the whole trust-region fraction |
| `train/explained_var` | critic fit. Flat below ~0.3 means advantages are critic error, and the policy loss is training on noise |
| `train/grad_norm_before_clip` | against `max_grad_norm`. Saturated means the clip, not the loss, is setting the step size |
| `train/sigreg_raw` | SIGReg penalty on the encoder conditioning. Logged every update, 0.0 when `sigreg_mode` is `none` |
| `eval/return_mean` vs `eval/steps_to_best` | peak location and how late it came |

`train/grad_norm_before_clip` is one key over actor and critic together. If it
saturates, the split between the two is not visible, and which one owns the
budget is the first thing worth instrumenting.

## 3. Sweeps

Sequential by design. `num_envs=64` forks a vector-env pool, so run **one arm at
a time** even on a big GPU.

Each training cell is followed by a `play` cell. Humanoid-v5 at 1000 steps is a
~33 s clip, so pass `max_steps=300` there.

```python
# Shared settings. ENV and TOTAL_STEPS are the only two things to change.
ENV = "Humanoid-v5"          # "Reacher-v5" is far cheaper if you just want a smoke
SEED = 3407
TOTAL_STEPS = 1_000_000
EVAL_STEPS = 16_384          # 61 evals over 1M steps
SAVE_STEPS = 50_000
COMPILE = "false"

import os, subprocess

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

BASE = [
    "--method", "sjfpo",
    "--seed", str(SEED),
    "--total_steps", str(TOTAL_STEPS),
    "--eval_steps", str(EVAL_STEPS),
    "--save_steps", str(SAVE_STEPS),
    "--sjfpo.report_to", "all",   # no WANDB_API_KEY needed
    "--sjfpo.compile", COMPILE,
]

def run_arm(tag, extra=()):
    out = f"outputs/sweep_{tag}"
    cmd = ["python", "scripts/train.py", "--env", ENV, "--out_dir", out, *BASE, *extra]
    print(f"\n=== {tag} ===\n{' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd)
    print(f"=== {tag} exit={rc} -> {out} ===", flush=True)
    return rc


def arm_dir(tag):
    """resolve_run_dir's output for one arm. A failed arm has no checkpoints."""
    return f"outputs/sweep_{tag}/sjfpo_{ENV}_seed{SEED}"
```

### Execution length

`n_action_steps` is the one knob specific to chunking rather than to FPO++. It
trades closed-loop correction against how stale the plan gets.

```python
for n in ("2", "4", "8"):
    run_arm(f"exec{n}", ["--sjfpo.n_action_steps", n])
```

```python
for n in ("2", "4", "8"):
    play(arm_dir(f"exec{n}"), ENV, max_steps=300)
```

`8` is `action_horizon`, i.e. the chunk is planned and executed fully open-loop
with no correction at all.

### Trust region

The logged run showed `log_ratio_max=0.0717` against a cap of 10, so
`log_ratio_clip_frac=0`. It also showed `flow_ratio_std=0.0098`, well inside
`clip_param`. `train/clipfrac` is the key that says whether the band bound, not
`log_ratio_clip_frac`, which only watches the cap.

```python
# Wider trust region, so the mechanism can actually bind. Two widths.
run_arm("clip020", ["--sjfpo.clip_param", "0.2"])
run_arm("spo020", ["--sjfpo.spo_clip_param", "0.2"])

# Higher LR, to separate "the region is too tight" from "the steps are too small".
run_arm("lr1e3", ["--sjfpo.policy_lr", "1e-3", "--sjfpo.critic_lr", "1e-3"])
```

```python
for tag in ("clip020", "spo020", "lr1e3"):
    play(arm_dir(tag), ENV, max_steps=300)
```

Do **not** sweep `cfm_clamp` or `log_ratio_cap`. The logged run had
`cfm_above_clamp_frac=0` and `log_ratio_clip_frac=0`; neither bound, so neither
is a lever.


### Representation regularization (SIGReg)

Weak SIGReg matches the encoder conditioning covariance to identity. It guards
the representation that feeds the actor. It reaches the critic only under
`--sjfpo.critic_share_encoder`, which defaults False. One knob starts it:
`--sjfpo.sigreg_mode weak` with `--sjfpo.sigreg_coef`.

Calibrate before trusting a live value. The log-only arm records
`train/sigreg_raw` without touching the loss. Read its scale first. Then set
the coef so the weighted penalty starts near 1% of the policy loss.

```python
# Log-only: no loss effect, reports the natural penalty scale.
run_arm("sigreg_logonly", ["--sjfpo.sigreg_mode", "weak", "--sjfpo.sigreg_coef", "0.0"])

# Live arms on Humanoid-v5.
run_arm("sigreg_w1e4", ["--sjfpo.sigreg_mode", "weak", "--sjfpo.sigreg_coef", "1e-4"])
run_arm("sigreg_w1e3", ["--sjfpo.sigreg_mode", "weak", "--sjfpo.sigreg_coef", "1e-3"])
```

```python
for tag in ("sigreg_logonly", "sigreg_w1e4", "sigreg_w1e3"):
    play(arm_dir(tag), ENV, max_steps=300)
```

`decorr` standardizes each feature first, so only redundancy pays and column
scales survive. Prefer it if the `weak` arms move `train/sigreg_raw` but not
returns. `--sjfpo.sigreg_sketch_dim` defaults to 64 and clamps to the batch
count automatically, so leave it alone unless the row count drops under it.


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

What to conclude:

| Observation | Meaning |
|---|---|
| `decay_pct` near 0 in every arm | the decay is not arm-specific; look at the critic |
| `best_at_eval` small in every arm | early peak; degradation is post-peak |
| one arm clearly ahead on return | replicate with a second seed before believing it |

One arm is not a result. Two seeds minimum before claiming anything, and note
that `seed` is a top-level flag, not `--sjfpo.seed`.

## 4. Diagnosing a post-peak decline

Use this when `eval/return_mean` rises, peaks, and then falls. The decline has
several possible causes, and they are separated by different experiments, so the
order below matters. Two of them are free and can kill the whole investigation
before you spend GPU time.

The causes, and what isolates each:

| Cause | Signature | Test |
|---|---|---|
| not a real decline | fall is no larger than `eval/return_ci_width` | cell 4.2, free |
| wrong eval distribution | the fall appears under one flow start only | cell 4.2, free |
| eval-path artifact | train metrics improve while eval falls | cell 4.2, free |
| trust region binding | `train/clipfrac` rises before the return falls | cell 4.1 `clip` arm |
| loss-scale saturation | `train/cfm_both_clamped_frac` rises | cell 4.1 `clamp` arm |
| critic misestimation | `train/explained_var` falls, `train/value_pred_member_gap` widens | cell 4.1 `critic` arms |
| critic staleness | `train/boot_value_relative_drift` rises | cell 4.1 `mc` arm |
| representation collapse | `train/cond_effective_rank` falls | cell 4.1 `sigreg` arm |
| actions pushed to the box | `train/action_pre_clamp_frac` trends up | cell 4.2 |
| chunk-boundary jerk | `train/rollout_seam_jerk` rises while `train/rollout_jerk` is flat | cell 4.2 |
| real interference | survives all of the above | section 4.4 |

### 4.1 The arms

Four arms plus a control, three seeds each, so 12 runs. Each arm moves one knob,
so the arm that fixes the decline names the cause.

Run one at a time. `num_envs=64` forks a vector-env pool.

```python
# 12 runs: 4 arms x 3 seeds. Lower it to 1 seed to screen, then widen.
import glob, os, subprocess   # section 4 is self-contained

FORGET_SEEDS = [0, 1, 2]
FORGET_ENV = ENV
FORGET_STEPS = TOTAL_STEPS

# One knob per arm. ctrl is the control and must stay unchanged.
FORGET_ARMS = {
    "ctrl":    [],
    # Halving the actor step separates "the policy is moving too far" from
    # "the critic is lying". The two lr arms are the main discriminator.
    "actor":   ["--sjfpo.actor_lr",  "5e-6"],
    "critic":  ["--sjfpo.critic_lr", "5e-5"],
    # More MC draws lowers the gradient variance without moving a step size.
    "mc":      ["--sjfpo.num_mc_samples", "32"],
}
# Optional fifth arm. num_critics is not in the run name, so the snapshot below
# is what keeps it apart from the others.
# FORGET_ARMS["critics"] = ["--sjfpo.num_critics", "3"]

TB_SNAPSHOT = {}   # (arm, seed) -> tensorboard run dir


def tb_dirs():
    return {d for d in glob.glob("tensorboard/*") if glob.glob(os.path.join(d, "*"))}


def run_forget_arm(arm, seed):
    """One run, and remember which tensorboard dir it wrote to.

    The run dir is `tensorboard/<run_name>` and `run_name` does NOT record
    `critic_lr` or `num_critics`. `run_metadata.json` does not record the
    tensorboard path either. So diff the directory list around the call. That
    is the only association here that does not depend on the name.
    """
    tag = f"forget_{arm}_s{seed}"
    out = f"outputs/{tag}"
    cmd = [
        "python", "scripts/train.py", "--env", FORGET_ENV, "--out_dir", out,
        "--seed", str(seed),
        "--total_steps", str(FORGET_STEPS),
        "--eval_steps", str(EVAL_STEPS),
        "--save_steps", str(SAVE_STEPS),
        "--sjfpo.report_to", "all",
        "--sjfpo.compile", COMPILE,
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

`--sjfpo.report_to all` is not optional. Every `train/*` diagnostic is gated on
`compute_info=tracker.enabled`, so with the tracker off the arms produce no
`clipfrac`, no `explained_var`, and no `cond_*`, and the diagnosis has nothing
to read.

If the run is long and you want the peak before the tail, run `ctrl` first and
read cell 4.2 on it before spending the other nine runs.

### 4.2 Read the decline before you explain it

Free, and it runs on any run that already finished. This cell accepts the old
key spellings, because runs recorded before the rename still carry them.

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# New spelling -> the spelling runs recorded before the rename.
KEY_ALIASES = {
    "eval/return_mean": "eval/mean_return",
    "eval/return_min": "eval/min_return",
    "eval/return_max": "eval/max_return",
    "eval/return_std": "eval/std_return",
    "eval/return_best": "eval/best_mean_return",
    "eval/random_policy_return": "eval/sample_random_baseline",
    "eval/streak_below_random_floor": "eval/consecutive_below",
    "train/boot_value_relative_drift": "train/boot_value_delta",
    "train/advantage_std_over_return_std": "train/advantage_std_fraction",
    "train/chunk_clip_count_mean": "train/chunk_clip_counts",
    "train/grad_norm_before_clip": "train/grad_norm_preclip",
}


def tb_series(run_dir, key):
    """[(step, value)] for `key`, falling back to the pre-rename spelling."""
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
    """The three free checks: real, one-sided, or eval-only.

    Returns None when there is no `eval/return_mean` series at all. Individual
    fields can be nan, which means that run predates the feature. A nan never
    triggers a verdict below, because every comparison against nan is False.
    """
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

Then apply the three verdicts. Each one prints only when its evidence is
present, so a silent cell means the evidence was missing, not that the answer
was clean.

```python
r = decline_report(TB_SNAPSHOT.get(("ctrl", FORGET_SEEDS[0])), "ctrl_s0")
if r is None:
    print("no eval/return_mean series yet for ctrl seed", FORGET_SEEDS[0])
else:
    for k, v in r.items():
        print(f"{k:22s} {v}")

    if r["missing"]:
        print("\nMISSING series (this run predates them): "
              + ", ".join(r["missing"]))
        print("Those checks stay silent below.")

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

### 4.3 Compare the arms

```python
def forget_rows():
    rows = []
    for arm in FORGET_ARMS:
        for seed in FORGET_SEEDS:
            r = decline_report(TB_SNAPSHOT.get((arm, seed)), f"{arm}_s{seed}")
            if r:
                rows.append(r)
    return rows


rows = forget_rows()
if rows:
    print(f"{'arm':>10} {'seed':>5} {'peak':>10} {'final':>10} {'fall%':>8} {'ci@pk':>9}")
    for r in rows:
        print(f"{r['label'].split('_s')[0]:>10} {r['label'].split('_s')[-1]:>5} "
              f"{r['peak']:>10.1f} {r['final']:>10.1f} {r['fall_pct']:>7.1f}% "
              f"{r['ci_width_at_peak']:>9.1f}")

    print(f"\n{'arm':>10} {'peak mean':>10} {'final mean':>11} {'fall%':>8} {'seed sd':>8}")
    for arm in FORGET_ARMS:
        g = [r for r in rows if r["label"].startswith(arm + "_s")]
        if not g:
            continue
        peaks = [r["peak"] for r in g]
        falls = [r["fall_pct"] for r in g]
        sd = statistics.stdev(falls) if len(falls) > 1 else float("nan")
        print(f"{arm:>10} {statistics.fmean(peaks):>10.1f} "
              f"{statistics.fmean(r['final'] for r in g):>11.1f} "
              f"{statistics.fmean(falls):>7.1f}% {sd:>7.1f}%")

    ctrl = [r for r in rows if r["label"].startswith("ctrl_s")]
    if len(ctrl) > 1:
        spread = max(r["fall_pct"] for r in ctrl) - min(r["fall_pct"] for r in ctrl)
        if spread > 0:
            print(f"\ncontrol fall spread across seeds: {spread:.1f} points. "
                  "An arm that beats the control by less than that is noise.")
        else:
            print("\ncontrol fall% is identical across seeds. Suspicious: the "
                  "seeds are probably not reaching different rollouts, or "
                  "TOTAL_STEPS is too short for the seeds to diverge.")
```

Read it this way:

| Observation | Meaning |
|---|---|
| `ctrl` fall% spread across seeds exceeds every arm's improvement | there is no signal. Raise `TOTAL_STEPS` or narrow the eval interval first |
| `actor` fall% collapses, `critic` does not | the policy step size is the cause. Lower `policy_lr`, or tighten `clip_param` |
| `critic` and `mc` improve it, `actor` does not | gradient noise or critic error. More MC draws and more critics both help, so it is variance |
| only `mc` improves it | gradient variance specifically, not critic bias |
| nothing improves it | the cause is not in this set. Go to 4.4 |

### 4.4 If every arm is flat

Real interference is the remaining explanation, and it needs a different
measurement. Do this first, because it needs no new code.

**Restart control.** Run `ctrl` to 500k steps. Then run a second job from
`outputs/forget_ctrl_s0/<run>/checkpoints/final` for another 500k. Compare the
second job's eval curve against the straight 1M control. If the split run ends
higher, the second half of the straight run was destroying earlier progress.
That is the classic interference signature, and the perturbation helping is the
evidence.

To launch the continuation, point `--resume` at the checkpoint directory. It
resolves to `outputs/forget_ctrl_restart_resume`, because a resumed run never
overwrites its base.

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
    "--sjfpo.report_to", "all",
    "--sjfpo.compile", COMPILE,
    "--resume", CKPT,
])
```

`total_steps` stays absolute on a resume. So the continuation must be given the
full 1M budget, not the remaining 500k, or it will stop early. The run dir for
the resumed job is the one the `_resume` suffix produced; point
`decline_report` at its tensorboard dir.

The decisive test for genuine forgetting is a region-split eval, reporting
return per region rather than one aggregate. That needs a new eval mode, because
this repo runs one env per run and has no notion of a region. Build it only if
4.2 is clean on both starts and 4.3 leaves every arm flat.

### Four traps

- `train/cfm_above_clamp_frac` reads the wrong tensor under the default config.
  `compute_policy_ratio` clamps `cfm_loss_new_token` at `[n, MC, H]` when
  `do_average_cfm_loss_in_chunk` is false, but the diagnostic is computed from
  `cfm_loss_new` at `[n, MC]`. Use `train/cfm_both_clamped_frac`, which reads
  the pair that survives. Do not cite the other one as trust-region evidence
  until it is fixed.
- The LR decay is a feedback loop. A decline trips `check_regression`, which
  spends a decay, which can deepen the decline. Treat `train/lr_decays` as a
  covariate: a recovery *after* a decay is not evidence that the policy improved.
- `train/cond_effective_rank` measures the DiT hidden states, not the action
  distribution. For action forensics load `checkpoints/best` and
  `checkpoints/final` and compare action histograms directly.
- Do not put `action_horizon` in this sweep. `collect_chunks_vec` sizes its plan
  by `chunk_size` while `RolloutBuffer` sizes storage by `action_horizon`. They
  agree only because the single call site hardcodes `chunk_size = action_horizon`.
