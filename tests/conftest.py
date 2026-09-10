"""Shared pytest fixtures.

The autouse wandb mock guarantees no test ever touches the network:
``mfpo.wandb_logging.wandb`` is replaced with a MagicMock for every test
(init/log/finish/Table all mocked). Tests asserting disabled behavior set
``MFPO_DISABLE_WANDB=1``; everything else runs default-on against the mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_wandb(monkeypatch):
    import mfpo.wandb_logging as wl

    monkeypatch.setattr(wl, "wandb", MagicMock())
