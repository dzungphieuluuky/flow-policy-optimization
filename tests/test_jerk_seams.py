"""Seam jerk must compare consecutive chunks of one env, not of two.

Chunk rows are written in closing order, so with more than one env two adjacent
rows are usually two different trajectories. Adjacent-row seams then measured
the jump between unrelated episodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from myfpo.eval import compute_seam_jerk


def _chunks(*rows: list[float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)[..., None]


def test_a_seam_inside_one_env_is_measured():
    a = _chunks([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0])
    pad = np.zeros((2, 4), dtype=bool)
    assert compute_seam_jerk(a, pad, np.array([0, 0])) == 0.0


def test_a_jump_between_two_envs_is_not_a_seam():
    a = _chunks([0.0, 1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0, 1003.0])
    pad = np.zeros((2, 4), dtype=bool)
    assert compute_seam_jerk(a, pad, np.array([0, 1])) is None
    # Read as one row order, that jump is what the metric used to report.
    assert compute_seam_jerk(a, pad) > 1000.0


def test_rows_are_grouped_by_env_before_the_seam_is_taken():
    """Interleaved rows: a same-env seam is smooth, a row-order seam is not."""
    a = _chunks(
        [0.0, 1.0, 2.0, 3.0],       # env 0, first
        [0.0, 10.0, 20.0, 30.0],    # env 1, first
        [4.0, 5.0, 6.0, 7.0],       # env 0, second
        [40.0, 50.0, 60.0, 70.0],   # env 1, second
    )
    pad = np.zeros((4, 4), dtype=bool)
    assert compute_seam_jerk(a, pad, np.array([0, 1, 0, 1])) == 0.0
    assert compute_seam_jerk(a, pad) > 100.0


def test_an_env_id_of_the_wrong_length_is_rejected():
    a = _chunks([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="env_id"):
        compute_seam_jerk(a, np.zeros((2, 3), dtype=bool), np.array([0]))
