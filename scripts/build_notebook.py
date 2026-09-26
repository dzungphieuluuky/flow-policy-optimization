#!/usr/bin/env python3
"""Build the minimal Colab notebook: spectral_joint_flow.ipynb.

The notebook carries NO source code. It clones this repo (private: GitHub
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


def build_notebook():
    """Assemble the minimal clone/install/train notebook (no source inlined)."""
    nb = new_notebook_shell()
    cells = [
        md_block(TITLE_SPECTRAL),
        md_block(SETUP_MD),
        code_block(ENV_CELL),
        code_block(CLONE_CELL),
        code_block(INSTALL_CELL),
        code_block(WANDB_CELL),
        code_block(SMOKE_CELL),
        code_block(FULL_TRAIN_CELL),
    ]
    nb.cells = cells
    return nb


def parse_build_args() -> argparse.Namespace:
    """Parse the notebook-builder CLI flags."""
    parser = argparse.ArgumentParser(description="Build minimal Colab notebook(s)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output notebook path (default: spectral_joint_flow.ipynb)",
    )
    return parser.parse_args()


def write_notebook(out: Path) -> None:
    """Build, validate, and write one notebook."""
    out = Path(out)
    print(f"Building notebook: {out}")
    nb = build_notebook()
    nbf.validate(nb)
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {len(nb.cells)} cells to {out} validated.")


def main():
    args = parse_build_args()
    write_notebook(ROOT / (args.output or "spectral_joint_flow.ipynb"))


if __name__ == "__main__":
    main()
