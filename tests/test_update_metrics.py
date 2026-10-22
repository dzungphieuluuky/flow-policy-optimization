"""The per-update metric average. A key is averaged over the minibatches
that carry it.

The per-position probe runs on some minibatches only, so a probe key is
missing from the others. Reading it from every minibatch raised a KeyError
and stopped the run.
"""

from __future__ import annotations

import torch

from myfpo.training import update_metrics


def _metrics(infos):
    return update_metrics(
        infos,
        [torch.zeros(1)],
        [torch.zeros(1)],
        None,
        [{"total": torch.zeros(1), "actor": torch.zeros(1), "critic": torch.zeros(1)}],
    )


def test_a_key_is_averaged_over_the_minibatches_that_carry_it():
    infos = [
        {"loss": 1.0, "probe": 10.0},
        {"loss": 3.0},
        {"loss": 5.0, "probe": 20.0},
    ]
    out = _metrics(infos).info
    assert out["loss"] == 3.0
    assert out["probe"] == 15.0


def test_a_missing_key_does_not_stop_the_update():
    infos = [{"loss": 1.0, "probe": 10.0}, {"loss": 3.0}]
    out = _metrics(infos).info
    assert out["loss"] == 2.0
    assert out["probe"] == 10.0


def test_no_minibatch_means_no_info():
    assert _metrics([]).info is None


def test_the_probe_key_appears_only_on_its_own_minibatches():
    """The shape of the defect: the first minibatch carries the key."""
    infos = [{"loss": 1.0, "probe": 1.0}, {"loss": 2.0}, {"loss": 3.0}]
    out = _metrics(infos).info
    assert "probe" in out
    assert out["probe"] == 1.0
