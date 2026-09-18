#!/usr/bin/env python3
"""Build the standalone Kaggle notebook spectral_joint_flow.ipynb.

Kaggle runs with internet DISABLED, so the notebook carries the full source:
one `%%writefile` cell per runtime file recreates `src/`, `scripts/train.py`,
`configs/` and `requirements-linux.txt` next to the notebook, then the last
cells install the attached wheels bundle and run
`!python scripts/train.py` as normal. Rebuild after changing any source file.

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

TITLE_SPECTRAL = """\
# Spectral Joint Flow (SJFPO) on Kaggle (offline)

Train the spectral flow policy on MuJoCo with internet disabled. The cells
below recreate the full source tree (`src/`, `scripts/train.py`, `configs/`)
next to this notebook via `%%writefile`, install deps from the attached
wheels dataset, then run `scripts/train.py` (eval runs inside training and
lands in `eval_history.csv`).
"""

SETUP_MD = """\
## 0. Attachments and secrets

1. Attach your wheels dataset (the pinned `requirements-linux.txt` bundle)
   and set `WHEELS_DIR` in the install cell to its `/kaggle/input/...` path.
2. Set the `WANDB_API_KEY` secret so wandb login happens automatically
   (no manual paste). Training also runs without it.
3. Run all cells top to bottom.
"""

MKDIR_CELL = """\
# Recreate the source tree next to the notebook (writefile needs the dirs).
!mkdir -p src/sjfpo scripts configs
"""

INSTALL_CELL = """\
# Offline install from the attached wheels dataset (no internet on Kaggle).
# Edit WHEELS_DIR to the /kaggle/input/... path of your packages dataset.
WHEELS_DIR = "/kaggle/input/sjfpo-packages"
!pip install -q --no-index --find-links=$WHEELS_DIR -r requirements-linux.txt
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
!python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000
"""

FULL_TRAIN_CELL = """\
# Full run: 1M steps with the matrix budgets (50k eval / 25k ckpt).
!python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 1000000 --eval_steps 50000 --save_steps 25000
"""


def embedded_files() -> list[Path]:
    """Runtime files recreated in the notebook, repo-relative order."""
    files = sorted((ROOT / "src" / "sjfpo").glob("*.py"))
    files.append(ROOT / "scripts" / "train.py")
    files += sorted((ROOT / "configs").glob("*.yaml"))
    files.append(ROOT / "requirements-linux.txt")
    return files


def writefile_block(path: Path) -> str:
    """One source file as a jupytext percent-format cell headed by %%writefile.

    A leading ``#!`` line is dropped: jupytext percent-format cannot
    round-trip it, and a shebang is meaningless for writefile materialization
    (files run via ``python scripts/train.py``, never executed directly).
    """
    rel = path.relative_to(ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    content = "\n".join(lines).strip("\n")
    body = f"%%writefile {rel}\n" + content if content else f"%%writefile {rel}"
    return "# %%\n" + body


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
        code_block(MKDIR_CELL),
        code_block(ENV_CELL),
    ]
    chunks += [writefile_block(p) for p in embedded_files()]
    chunks += [
        code_block(INSTALL_CELL),
        code_block(WANDB_CELL),
        code_block(SMOKE_CELL),
        code_block(FULL_TRAIN_CELL),
    ]
    return "\n\n".join(chunks) + "\n"


def build_notebook():
    """Assemble the standalone offline notebook (sources embedded)."""
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
    """Structural gate: md/md/code×N order, writefile cells cover every runtime file."""
    kinds = [c["cell_type"] for c in nb.cells]
    if len(kinds) < 3 or kinds[:2] != ["markdown", "markdown"] or set(kinds[2:]) != {"code"}:
        raise ValueError(f"unexpected cell order: {kinds}")
    sources = [str(c["source"]) for c in nb.cells if c["cell_type"] == "code"]
    for p in embedded_files():
        head = f"%%writefile {p.relative_to(ROOT).as_posix()}"
        if not any(s.splitlines()[0].strip() == head for s in sources):
            raise ValueError(f"missing writefile cell: {head}")
    for i, c in enumerate(nb.cells):
        if not str(c["source"]).strip():
            raise ValueError(f"cell {i} is empty")
    if not any("python scripts/train.py" in s for s in sources):
        raise ValueError("no train cell found")


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
