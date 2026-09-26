"""JAX port of SJFPO (parallel tree; ``src/sjfpo`` stays the torch path).

Layout mirrors ``sjfpo`` per phase plan: pure functions (dct/gae/fpo),
NNX models (networks), MJX rollout + jitted Optax update (training),
Orbax checkpoints. Shared backend-agnostic pieces (config dataclasses,
evaluator, wandb) are reused from ``sjfpo``, not duplicated.
"""

from __future__ import annotations
