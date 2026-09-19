#!/usr/bin/env python3
"""Build the online Colab notebook: git clone + !python (no embedded source)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parents[1]

TITLE_ONLINE = """\
# Spectral Joint Flow (SJFPO) — Online Colab / Kaggle (with internet)

Clone the repo with a GitHub token, install requirements, then run training
via `!python scripts/train.py`. No source is embedded in the notebook.
"""

SETUP_MD = """\
## 0. Secrets and environment

1. Set `GITHUB_TOKEN` in the environment (Colab secrets or Kaggle secrets).
2. The token is read via `load_secret()` using the runtime environment.
3. Run all cells top to bottom.
"""

ENV_CELL = """\
# Detect runtime and load secrets from the right store.
import os

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
"""

CLONE_CELL = """\
# Clone the repo using the GitHub token from the environment/store.
GITHUB_TOKEN = load_secret("GITHUB_TOKEN")
repo_slug = os.getenv("GITHUB_REPO", "user/spectral-joint-policy")
repo_url = f"https://github.com/{repo_slug}.git"
if GITHUB_TOKEN:
    repo_url = repo_url.replace("github.com", f"{GITHUB_TOKEN}@github.com")
print(f"Cloning from: {repo_url.replace(GITHUB_TOKEN or '', '<TOKEN>') if GITHUB_TOKEN else repo_url}")
!git clone $repo_url spectral-joint-policy
"""

CD_CELL = """\
# Move into the cloned repo and run directly (no embedded source).
%cd spectral-joint-policy
!pwd
"""

INSTALL_CELL = """\
# Detect runtime and choose install source.
ENV_NAME = detect_env()
if ENV_NAME == "colab":
    !uv pip install -r requirements-colab.txt
else:
    !pip install -q -e . || !pip install -q -r requirements-linux.txt
"""

TRAIN_CELL = """\
# Run training directly from the cloned repo.
!python scripts/train.py \
    --method sjfpo \
    --env Hopper-v5 \
    --seed 3407 \
    --total_steps 200000 \
    --eval_steps 50 \
    --save_steps 10000 \
    --use_per_bin_clip true \
    --top_p_spectral_tokens 1.0
"""

FULL_CELL = """\
# Full run (optional).
!python scripts/train.py \
    --method sjfpo \
    --env Hopper-v5 \
    --seed 3407 \
    --total_steps 200000 \
    --eval_steps 50 \
    --save_steps 10000 \
    --use_per_bin_clip true \
    --top_p_spectral_tokens 1.0
"""


def md_block(source: str) -> str:
    lines = ["# %% [markdown]"]
    lines += [("# " + ln) if ln.strip() else "" for ln in source.strip().splitlines()]
    return "\n".join(lines)


def code_block(source: str) -> str:
    return "# %%\n" + source.strip()


def build_script() -> str:
    chunks = [
        md_block(TITLE_ONLINE),
        md_block(SETUP_MD),
        code_block(ENV_CELL),
        code_block(CLONE_CELL),
        code_block(CD_CELL),
        code_block(INSTALL_CELL),
        code_block(TRAIN_CELL),
        code_block(FULL_CELL),
    ]
    return "\n\n".join(chunks) + "\n"


def build_notebook():
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


def check_notebook(nb) -> None:
    kinds = [c["cell_type"] for c in nb.cells]
    sources = [str(c["source"]) for c in nb.cells if c["cell_type"] == "code"]
    if not any("load_secret" in s for s in sources):
        raise ValueError("missing load_secret / detect_env cell")
    if not any("git clone" in s for s in sources):
        raise ValueError("missing git clone cell")
    if not any("!python scripts/train.py" in s for s in sources):
        raise ValueError("missing train cell")
    for i, c in enumerate(nb.cells):
        if not str(c["source"]).strip():
            raise ValueError(f"cell {i} is empty")


def write_notebook(out: Path) -> None:
    out = Path(out)
    print(f"Building online notebook: {out}")
    nb = build_notebook()
    check_notebook(nb)
    jupytext.write(nb, str(out))
    print(f"Wrote {len(nb.cells)} cells to {out} checked.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the online Colab notebook")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output notebook path (default: sjfpo_online.ipynb)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out = ROOT / args.output if args.output else ROOT / "sjfpo_online.ipynb"
    write_notebook(out)


if __name__ == "__main__":
    main()
