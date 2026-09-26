#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# torch>=2.2,<2.6 ships no wheels for python>=3.13; prefer an interpreter that
# has wheels. Override with PYTHON_BIN=<interpreter> if needed.
if [ -z "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN=python3
  for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/pip install -q -r requirements-local.txt

.venv/bin/python -c '
import importlib.metadata as im
import sys
import torch, gymnasium, mujoco, jupytext, nbformat

def vt(s):
    parts = []
    for p in s.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def V(dist):
    return im.version(dist)

t = V("torch"); g = V("gymnasium"); m = V("mujoco")
assert (2, 2, 0) <= vt(t) < (2, 6, 0), f"torch {t} outside >=2.2,<2.6"
assert (1, 0, 0) <= vt(g) < (1, 3, 0), f"gymnasium {g} outside >=1.0,<1.3"
assert (3, 1, 0) <= vt(m) < (3, 4, 0), f"mujoco {m} outside >=3.1,<3.4"
assert vt(V("scikit-learn")) >= (1, 4, 0), "scikit-learn < 1.4"
assert vt(V("pytest")) >= (8, 0, 0), "pytest < 8"
assert vt(V("jupytext")) >= (1, 16, 0), "jupytext < 1.16"
assert vt(V("nbformat")) >= (5, 10, 0), "nbformat < 5.10"
assert vt(V("matplotlib")) >= (3, 8, 0), "matplotlib < 3.8"
n = V("numpy")
assert (1, 26, 0) <= vt(n) < (2, 1, 0), f"numpy {n} outside >=1.26,<2.1"
print("versions OK: python=%s torch=%s gymnasium=%s mujoco=%s numpy=%s" % (
    sys.version.split()[0], t, g, m, n))
'
