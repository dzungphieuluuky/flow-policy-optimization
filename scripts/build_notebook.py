#!/usr/bin/env python3
"""Build minimal Colab notebooks: spectral_joint_flow.ipynb + flow_matching_vs_ppo_sac.ipynb.

The notebooks carry NO source code. Each one clones this repo (private: GitHub
token in the URL), installs deps, then shells out to scripts/train.py with `!`
(train + eval happen inside train.py via periodic eval → eval_history.csv).
Single source of truth stays in src/.

Usage:
    python scripts/build_notebook.py [--output spectral_joint_flow.ipynb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

REPO_SLUG = "dzungphieuluuky/spectral-joint-policy"
REPO_DIR = "spectral-joint-policy"

TITLE_SPECTRAL = """\
# Spectral Joint Flow (SJFPO) on Colab

Train the spectral flow policy on MuJoCo from a clean Colab runtime.
All code lives in `src/` of the cloned repo — this notebook only clones,
installs, and calls `scripts/train.py` (eval runs inside training and lands
in `eval_history.csv`).
"""

TITLE_COMPARISON = """\
# SJFPO vs PPO/SAC on Colab

Run the SJFPO / PPO / SAC comparison matrix on MuJoCo from a clean Colab
runtime. Same pattern as the spectral notebook: clone, install, then call
`scripts/sweep.sh` with `METHODS/ENVS/SEEDS` axes (eval runs inside training).
"""

SETUP_MD = """\
## 0. Private-repo access

Create a fine-grained GitHub PAT with **Contents: read** on this repo and
paste it below when prompted (or set the `GITHUB_TOKEN` secret — it is
never written into the notebook). Also set the `WANDB_API_KEY` secret
so wandb login happens automatically (no manual paste).
"""

CLONE_CELL = """\
# Clone the private repo (skip if already present), enter it.
# GITHUB_TOKEN comes from the secret store (see cell above); manual paste is the fallback.
import getpass
from pathlib import Path

token = load_secret("GITHUB_TOKEN") or getpass.getpass(
    "GitHub token (PAT with Contents:read on this repo): "
)
repo_url = f"https://{token}@github.com/REPO_SLUG.git"
if not Path("REPO_DIR").exists():
    !git clone $repo_url
%cd REPO_DIR
!git log --oneline -3
""".replace("REPO_SLUG", REPO_SLUG).replace("REPO_DIR", REPO_DIR)

INSTALL_CELL = """\
# Install pinned deps with uv (CPU wheels resolve on Colab; GPU torch comes with them).
!pip install -q uv
!uv pip install -q -r requirements-linux.txt --system
"""

ENV_CELL = """\
# Detect runtime and load secrets from the right store.
import os

def _detect_env():
    try:
        if "google.colab" in str(get_ipython()):
            return "colab"
    except NameError:
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"

ENV_NAME = _detect_env()
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
"""

WANDB_CELL = """\
# wandb login from secret store (runs before any training cell).
_wandb_key = load_secret("WANDB_API_KEY")
if _wandb_key:
    !wandb login $_wandb_key
else:
    print("WANDB_API_KEY not found; training will run without wandb sync.")
"""

SMOKE_CELL = """\
# Smoke: 2k steps, single SJFPO run on Hopper (eval + checkpoints included).
!PYTHONPATH=src python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000
"""

FULL_TRAIN_CELL = """\
# Full run: 1M steps with the matrix budgets (50k eval / 25k ckpt).
!PYTHONPATH=src python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 1000000 --eval_steps 50000 --save_steps 25000
"""

MATRIX_CELL = """\
# Comparison matrix: all methods x Hopper-v5 x seed 0 (extend ENVS/SEEDS for full).
# Each sweep row calls: PYTHONPATH=src python scripts/train.py --method <m> --env <e> --seed <s>
!METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" bash scripts/sweep.sh
"""

RESULTS_CELL = """\
# Eval histories produced by training (one CSV per run).
!find .omo/evidence/train -name eval_history.csv
"""


def md_block(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code_block(source: str):
    return nbf.v4.new_code_cell(source.strip())


def new_notebook_shell():
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        }
    )
    return nb


def build_notebook(spectral: bool = False):
    """Assemble the minimal clone/install/train notebook (no source inlined)."""
    nb = new_notebook_shell()
    cells = [
        md_block(TITLE_SPECTRAL if spectral else TITLE_COMPARISON),
        md_block(SETUP_MD),
        code_block(ENV_CELL),
        code_block(CLONE_CELL),
        code_block(INSTALL_CELL),
        code_block(WANDB_CELL),
    ]
    if spectral:
        cells += [code_block(SMOKE_CELL), code_block(FULL_TRAIN_CELL)]
    else:
        cells += [code_block(MATRIX_CELL), code_block(RESULTS_CELL)]
    nb.cells = cells
    return nb


def parse_build_args() -> argparse.Namespace:
    """Parse the notebook-builder CLI flags."""
    parser = argparse.ArgumentParser(description="Build minimal Colab notebook(s)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output notebook path (default: build both spectral_joint_flow.ipynb + legacy)",
    )
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Only build legacy flow_matching_vs_ppo_sac.ipynb",
    )
    parser.add_argument(
        "--spectral-only",
        action="store_true",
        help="Only build spectral_joint_flow.ipynb",
    )
    return parser.parse_args()


def write_notebook(out: Path, spectral: bool) -> None:
    """Build, validate, and write one notebook."""
    out = Path(out)
    print(f"Building notebook: {out} (spectral={spectral})")
    nb = build_notebook(spectral=spectral)
    nbf.validate(nb)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {len(nb.cells)} cells to {out} spectral={spectral} validated.")


def write_default_notebooks(args: argparse.Namespace) -> None:
    """Build both notebooks, honoring --legacy-only/--spectral-only."""
    if not args.legacy_only:
        write_notebook(ROOT / "spectral_joint_flow.ipynb", True)
    if not args.spectral_only:
        write_notebook(ROOT / "flow_matching_vs_ppo_sac.ipynb", False)


def main():
    args = parse_build_args()
    if args.output:
        out = ROOT / args.output
        write_notebook(out, "spectral" in out.name)
        return
    write_default_notebooks(args)


if __name__ == "__main__":
    main()
