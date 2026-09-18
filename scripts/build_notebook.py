#!/usr/bin/env python3
"""Build the standalone Colab notebook spectral_joint_flow.ipynb.

The notebook carries NO source code. It clones this repo (private: GitHub
token in the URL), installs deps, then shells out to scripts/train.py with `!`
(train + eval happen inside train.py via periodic eval → eval_history.csv).
Single source of truth stays in src/.

The notebook is authored as a jupytext percent-format script (in-memory text)
and converted with the jupytext API — no nbformat import here.

Usage:
    python scripts/build_notebook.py [--output spectral_joint_flow.ipynb]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext

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


def md_block(source: str) -> str:
    """Markdown source as a jupytext percent-format cell chunk."""
    lines = ["# %% [markdown]"]
    lines += [("# " + ln) if ln.strip() else "" for ln in source.strip().splitlines()]
    return "\n".join(lines)


def code_block(source: str) -> str:
    """Code source as a jupytext percent-format cell chunk."""
    return "# %%\n" + source.strip()


def build_script() -> str:
    """Assemble the percent-format script (parsed by jupytext, never written)."""
    chunks = [
        md_block(TITLE_SPECTRAL),
        md_block(SETUP_MD),
        code_block(ENV_CELL),
        code_block(CLONE_CELL),
        code_block(INSTALL_CELL),
        code_block(WANDB_CELL),
        code_block(SMOKE_CELL),
        code_block(FULL_TRAIN_CELL),
    ]
    return "\n\n".join(chunks) + "\n"


def build_notebook():
    """Assemble the standalone clone/install/train notebook (no source inlined)."""
    nb = jupytext.reads(build_script(), fmt="py:percent")
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


def parse_build_args() -> argparse.Namespace:
    """Parse the notebook-builder CLI flags."""
    parser = argparse.ArgumentParser(description="Build the standalone Colab notebook")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output notebook path (default: spectral_joint_flow.ipynb)",
    )
    return parser.parse_args()


def check_notebook(nb) -> None:
    """Structural gate: 8 cells in md/md/code×6 order, all non-empty."""
    kinds = [c["cell_type"] for c in nb.cells]
    if kinds != ["markdown", "markdown"] + ["code"] * 6:
        raise ValueError(f"unexpected cell order: {kinds}")
    for i, c in enumerate(nb.cells):
        if not str(c["source"]).strip():
            raise ValueError(f"cell {i} is empty")


def write_notebook(out: Path) -> None:
    """Build, check, and write the notebook via the jupytext API."""
    out = Path(out)
    print(f"Building notebook: {out}")
    nb = build_notebook()
    check_notebook(nb)
    jupytext.write(nb, str(out))
    print(f"Wrote {len(nb.cells)} cells to {out} checked.")


def main():
    args = parse_build_args()
    out = ROOT / args.output if args.output else ROOT / "spectral_joint_flow.ipynb"
    write_notebook(out)


if __name__ == "__main__":
    main()
