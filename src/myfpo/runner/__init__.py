"""The pieces of the training loop that do not need the training loop.

``train_sjfpo_run`` holds one optimiser step, one rollout, and one eval window.
The arithmetic around them does not belong inside it. A function that turns a
forward pass into log rows, or that runs an eval window and reports what it
found, can be called and tested on its own.

Nothing in this package reads the policy. Nothing here writes to a tracker.
The caller passes what each function needs and takes back what it produced.
"""

from myfpo.runner.eval import WindowSummary, summarise_window
from myfpo.runner.metrics import build_train_metrics

__all__ = ["WindowSummary", "build_train_metrics", "summarise_window"]
