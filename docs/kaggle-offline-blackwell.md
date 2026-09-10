# Run `spectral_joint_flow.ipynb` on Kaggle (RTX PRO 6000 Blackwell, internet OFF)

> Research journal: `.omo/ulw-research/20260903-225626/` (Wave-1/2 digests, claim graph, execution verification).
> Lineage labels: **MEASURED** = stated in a fetched source or observed by execution here;
> **ASSUMED** = inference, flagged; **DERIVED** = computed, formula shown.

## 0. TL;DR (the whole procedure in 7 steps)

1. **Get the GPU**: RTX PRO 6000 kernels are competition-gated — request `machine_shape:
   NvidiaRtxPro6000` in `kernel-metadata.json` (`kaggle kernels push -p .`); a P100 may
   still be allocated (request-not-guarantee). Free tier is P100/T4×2.
2. **Build a wheelhouse on an internet machine** (Linux x86_64, Python 3.12): recommended
   pin is **torch 2.8.0+cu128 + torchvision 0.23.0+cu128** (+ gym/mujoco/skrl/einx stack).
   Full closure **DERIVED ~4.0 GB**.
3. **Upload the wheels as a private Kaggle Dataset** (ZIP if >50 files; limit 200 GB/dataset).
4. **Attach it** to a new notebook (Add Input → mounts read-only under `/kaggle/input/<slug>/`),
   set **Internet OFF**.
5. **First cell**: `!pip install --no-index --find-links=/kaggle/input/<slug>/… <packages>`.
6. **Upload/run only `spectral_joint_flow.ipynb`** (61 cells, self-contained — no `src/` needed).
   Prepend the `MUJOCO_GL=osmesa` header below; skip the stub cells listed in §5.
7. **Write artifacts to `/kaggle/working`** (≤20 GB MEASURED); Save Version → Save & Run All.

## 1. What the notebook actually needs (verified by execution)

`spectral_joint_flow.ipynb` = 61 cells (35 code), fully self-contained: all 15 `src/mfpo`
modules are inlined as code cells with register-snippets — **no `src/` upload needed**
and the `sys.path` setup is a no-op guard. **MEASURED** via local AST scan (see journal
`verify-closure.md`).

| Dependency | Role | Tested-good pin |
|---|---|---|
| torch (CUDA build, §2) | everything | 2.8.0+cu128 (GPU) |
| torchvision (+cu128, same version line) | ViT scratch path compat | 0.23.0+cu128 |
| gymnasium | env API | 1.2.3 (1.2.x OK) |
| mujoco | physics backend (no GL needed physics-only) | 3.3.7 |
| skrl (exact, asserted in setup cell) | PPO/SAC cells only (lazy import) | ==2.1.0 |
| einx | tensor notation (6 modules) | 0.4.3 |
| numpy | numerics (2.x tolerated despite declared `<2.1`) | pip-resolved 2.x |
| matplotlib (Agg) | analysis figures | any 3.x |

Setup cell asserts: `torch>=2.2`, `gymnasium>=1.0`, `skrl==2.1.0`, fp32 default — this is
the **first failure point** offline; if it fails, your wheel install didn't take.

**Do NOT bundle**: `timm` (guarded `try/except` → scratch ViT fallback; absent in the
verified env), `wandb` (REQUIRED import — set `MFPO_DISABLE_WANDB=1` offline unless you want logging),
`scipy` (banned at runtime by design), extra datasets (absent by design),
`scikit-learn`/`pandas`/`torchvision`-beyond-pin (zero src imports). All **MEASURED**
by import scan + guard analysis.

## 2. GPU / torch pin (Blackwell sm_120)

- Target = **RTX PRO 6000 Blackwell Server Edition**: 96 GB GDDR7, arch **sm_120 = CC 12.0**
  (**MEASURED** via device properties; not sm_100 — never build sm_100-only). Do not confuse
  with RTX 6000 Ada (48 GB, sm_89).
- Floor: **CUDA 12.8 + driver R570+**; first stable torch = **2.7.0+cu128**.
- **Pin cu128** (matches Kaggle's default image torch, proven on sm_120). cu130 (torch ≥2.11,
  driver 580+) is fallback only — and cu128 was removed for torch ≥2.12, so stay on the
  2.7–2.11 line for cu128. torchaudio: omit (not imported; also absent from cu130 ≥2.12).
- `torch+cpu` and `torch+cuXX` **cannot coexist** (same package, different indexes) — ship
  exactly one torch; the CUDA build runs CPU code too, so device-agnostic code is the backup.

Exact wheel filenames (**MEASURED** from index pages, cp312 = Kaggle Python 3.12):

```
torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl      (~848 MB)
torchvision-0.23.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl (~8 MB)
nvidia_cublas_cu12-12.8.4.1-..._x86_64.whl                   (~594 MB)
nvidia_cudnn_cu12-9.10.2.21-..._x86_64.whl                   (~707 MB)
nvidia_nccl_cu12-2.27.3-..._x86_64.whl                       (~322 MB)
nvidia_cusparse_cu12-12.5.8.93 / cusparselt 0.7.1 / cusolver 11.7.3.90 /
  cufft 11.3.3.83 / curand 10.3.9.90 / nvrtc 12.8.93 / nvjitlink 12.8.93 /
  cupti 12.8.90 / cufile 1.13.1.3 / runtime 12.8.90 / nvtx 12.8.90
triton-3.4.0-cp312-..._x86_64.whl                            (~156 MB)
```

NVIDIA wheels are `py3-none` (no cp split); triton/mujoco/torch are cp312-specific —
never substitute a cp311 file. Full Option-A closure **DERIVED ≈ 3.94 GB (~3.7 GiB)**,
so budget **~4 GB** + tens of MB pure-Python transitives.

## 3. Build the wheelhouse (internet machine, Linux x86_64, Python 3.12)

```bash
mkdir -p wheelhouse && cd wheelhouse
pip download \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --only-binary=:all: \
  --python-version 3.12 --platform manylinux_2_28_x86_64 --implementation cp \
  -d . \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  "mujoco==3.3.7" "gymnasium==1.2.0" "skrl==2.1.0" "einx==0.4.3" \
  "matplotlib>=3.8" "scikit-learn>=1.4"
ls *.whl | wc -l   # expect ~30 (torch + 14 nvidia + triton + tv + small pkgs + transitives)
```

Rules (**MEASURED** failure modes): `--only-binary=:all:` is mandatory (skrl/einx/gym/mujoco
all ship sdists → offline source builds fail on missing `setuptools`); `--platform` must
match exactly (never macOS/Windows/alpine wheels); keep the `requirements.txt` inside the
folder as the install map. If you build on the same-arch Linux box you may drop `--platform`.

## 4. Upload + attach + offline install (Kaggle)

1. **Upload**: Datasets → New Dataset → drag the `.whl` files (ZIP first if >50 top-level
   files — ZIPs auto-extract server-side). Private is fine. Limits **MEASURED**: 200 GB per
   dataset, 200 GB private total, 50 top-level files.
2. **Attach**: notebook → Add Input → your wheels dataset (mounts read-only at
   `/kaggle/input/<slug>/`, one folder per dataset). First run after a fresh attach may not
   mount — re-run/save once if the path is empty. Per-notebook attach-count: no official
   number published (gap — keep to wheels + at most 1–2 data/models).
3. **Internet OFF**, then first cell (**MEASURED** canonical recipe):
```bash
!ls /kaggle/input/<slug>/          # confirm mount BEFORE installing
!pip install --no-index --find-links=/kaggle/input/<slug>/ \
  "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  "mujoco==3.3.7" "gymnasium==1.2.0" "skrl==2.1.0" "einx==0.4.3" matplotlib scikit-learn numpy
```
Forgetting `--no-index` makes pip reach for PyPI (fails/hangs offline). Alternative official
path: Dependency Manager (configure online once; installs pre-execution even offline) — but
flag support (`--extra-index-url`) is undocumented, so keep torch on the wheels-dataset path.

## 5. Run the notebook (cell order + skip-list)

**Prepend** (new first cell — Kaggle images ship no EGL userspace, **MEASURED** from
docker-python template + container-toolkit docs; physics-only needs no GL at all):

```python
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# egl path is opt-in only after YOU probe libEGL_nvidia on the live kernel
```

**Probe cell** (run before pinning anything; also closes the runbook's driver gap):

```python
!nvidia-smi
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
print(torch.cuda.get_device_properties(0))  # expect sm_120 / CC (12,0), ~96 GB
```

**Cell order** (spectral notebook has no calibration/sweep/analysis executable cells and no
`/kaggle` paths — nothing Kaggle-specific to skip there): Setup asserts → 15 module cells
→ provenance/architecture markdown → spectral self-test → smoke gate
(prints the `scripts/train.py` pointer note — static cell, safe, keep) →
flag self-test. `IS_KAGGLE` (= `KAGGLE_KERNEL_RUN_TYPE` set) only changes run-output dirs.

**Skip / do-not-touch offline**: any `pip install datasets`, `push_checkpoints_to_dataset()`
(docs-only helper, no code — leave disabled; fetch checkpoints from `/kaggle/working`),
full 45×1M sweep (exceeds kernel time — use `MFPO_TINY_MATRIX=1` or single-seed edits),
online wandb (set `MFPO_DISABLE_WANDB=1` offline — a failed init is fatal, never downgraded silently).
Old doc names (synthetic-env adapter, `calibrate_and_gate_*` variants, 7-col writer,
`SampleEfficiencyBudget`, `per_suite_efficiency`) are stale — all removed in the MuJoCo-only prune and the LIBERO purge (src has `calibrate_and_gate`; runner is MuJoCo-only flat Box).
Eval CSV = 3-col `[episode,return,length]`; the evaluator companions were removed in the prune — only the MuJoCo `Evaluator` path is wired to the runner.

## 6. Real-robot data (out of scope — MuJoCo-only)

This project trains on MuJoCo locomotion only; there is no dataset pipeline. Do not vendor datasets into the repo.

## 7. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `Could not find a version … (from versions: none)` with wheel present | Python-tag mismatch (cp311 wheel on cp312 kernel) → `python -c "import sys,platform;print(sys.version,platform.machine())"`, re-download exact tags |
| `Installing build dependencies … error` / setuptools failure | sdist slipped in → rebuild with `--only-binary=:all:`, or vendor `setuptools`+`wheel` |
| `pip` hangs at install | missing `--no-index` (reaching for PyPI) or wrong `--find-links` path → `os.listdir()` the mount first |
| `gladLoadGL` / GLFW 65537 / `Failed to initialize OpenGL` | GL backend tried to render → keep `osmesa` header; physics-only `mj_step` needs no GL |
| `no kernel image … sm_120` | torch <2.7 or cu≤126 → reinstall cu128 2.7–2.11 line |
| P100 allocated despite RTX request | competition-gating (request-not-guarantee) → re-push `NvidiaRtxPro6000`, check eligibility; OOM on P100 is expected (16 GB vs 96 GB) |
| First run missing `/kaggle/input/<slug>` | attach-mount race → save/re-run once |
| Quiet failures near end of long runs | `/kaggle/working` ~20 GB cap / file-count anecdote → checkpoint often, zip before writing many files |

## 8. Known gaps (carried, not hidden)

1. Kaggle RTX-kernel driver string (cu128 vs cu130 final call) — close with the §5 probe cell.
2. Per-notebook attach-count/size cap — no official number; keep attachments minimal.
3. Dependency Manager flag support — unverified; one install path for torch.
4. cu130 full-closure size (cuda-toolkit meta-package expansion) — recompute if you go cu130.
5. `requirements-local.txt` numpy `<2.1` bound is stale (2.5.2 runs green) — accept 2.x.
