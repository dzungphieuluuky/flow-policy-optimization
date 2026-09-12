"""Shared pytest fixtures.

Wandb is unconditionally OFF under pytest: ``SJFPO_DISABLE_WANDB=1`` is set
at import time (this file loads before any test module), so no test —
including subprocess CLI smokes, which inherit ``os.environ`` — touches
the network. The autouse wandb mock below is a second layer: enabled-path
tests ``monkeypatch.delenv`` the flag and assert against the mock.
"""

from __future__ import annotations

import os

os.environ["SJFPO_DISABLE_WANDB"] = "1"

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch):
    import sjfpo.wandb_logging as wl

    monkeypatch.setattr(wl, "wandb", MagicMock())
