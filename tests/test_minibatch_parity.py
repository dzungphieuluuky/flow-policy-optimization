"""Minibatching and the per-update diagnostics, against torch.

Two things are pinned here, and both were once unguarded:

* **Which chunks each minibatch sees.** ``sample_minibatch`` slices the leading
  dim of every per-chunk entry with one index tensor. If an entry that is *not*
  per-chunk got sliced -- ``normalizer_observations`` is one row per executed
  env step -- the minibatch silently covers the wrong frames and the obs
  normaliser is updated from a subset nobody asked for.
* **The diagnostics.** ``compute_rollout_diagnostics`` and
  ``compute_gae_diagnostics`` are reductions with fixed key sets. The trainer logs
  them by name, so a renamed or missing key silently stops a metric appearing on
  W&B.

``generate_mini_batch_indices`` draws from a jax key rather than torch's global
generator, so the index *stream* cannot be compared across frameworks; what is
pinned instead is the coverage contract: every chunk seen exactly once per
epoch, splits proportional to the remainder rule, and reshuffled per epoch.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from sjfpo_jax import train_sjfpo as X
from sjfpo_jax.gae import GAE_DIAGNOSTIC_KEYS

N_CHUNKS, N_MB = 10, 4


def make_batch(n=N_CHUNKS, extra=()):
    batch = {
        "observation.state": jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2),
        "action": jnp.arange(n, dtype=jnp.float32)[:, None],
        "action_is_pad": jnp.zeros((n, 3), dtype=jnp.bool_),
        "raw_advantages": jnp.arange(n, dtype=jnp.float32),
        "chunk_clip_counts": jnp.arange(n, dtype=jnp.int32),
        # One row per executed step, not per chunk: 3*n rows here.
        "normalizer_observations": jnp.arange(3 * n, dtype=jnp.float32)[:, None],
        "gae_diagnostics": {k: jnp.float32(i) for i, k in enumerate(GAE_DIAGNOSTIC_KEYS)},
        "compute_spectral_energy_retained": jnp.float32(0.5),
    }
    batch.update(extra)
    return batch


# ------------------------------------------------------------- mini-batch indices


def test_every_epoch_covers_every_chunk_exactly_once():
    """Each epoch's minibatches concatenate back to the full index set.

    Checked per epoch rather than over the whole stream, because the stream is
    ``num_learning_epochs * n_mb`` minibatches long and a chunk legitimately
    appears once per epoch.
    """
    for epochs in (1, 2, 3):
        splits = list(X.generate_mini_batch_indices(
            N_CHUNKS, N_MB, jax.random.key(0), num_learning_epochs=epochs
        ))
        assert len(splits) == epochs * N_MB, (epochs, len(splits))
        for epoch in range(epochs):
            seen = np.concatenate([
                np.asarray(idx) for idx in splits[epoch * N_MB:(epoch + 1) * N_MB]
            ])
            assert sorted(seen.tolist()) == list(range(N_CHUNKS)), (epochs, sorted(seen.tolist()))


def test_splits_are_proportional_so_the_remainder_spreads():
    """``(i * n) // n_mb``, not a flat ``n // n_mb`` per split.

    10 chunks over 4 minibatches gives starts 0/2/5/7, so 2/3/2/3. A flat
    ``n // n_mb`` would give 2/2/2/2 and drop two chunks per epoch.
    """
    sizes = [
        len(idx) for idx in X.generate_mini_batch_indices(
            N_CHUNKS, N_MB, jax.random.key(0)
        )
    ]
    assert sizes == [2, 3, 2, 3], sizes
    assert sum(sizes) == N_CHUNKS


def test_more_minibatches_than_chunks_collapses_to_one_per_chunk():
    sizes = [
        len(idx) for idx in X.generate_mini_batch_indices(3, 99, jax.random.key(0))
    ]
    assert sizes == [1, 1, 1], sizes


def test_each_epoch_reshuffles():
    """A second epoch must not replay the first epoch's order."""
    first = [np.asarray(i).tolist() for i in X.generate_mini_batch_indices(
        N_CHUNKS, N_MB, jax.random.key(0), num_learning_epochs=1)]
    both = [np.asarray(i).tolist() for i in X.generate_mini_batch_indices(
        N_CHUNKS, N_MB, jax.random.key(0), num_learning_epochs=2)]
    assert first == both[:N_MB]
    assert both[:N_MB] != both[N_MB:]


def test_a_nonpositive_chunk_count_is_rejected():
    with pytest.raises(ValueError, match="must be >= 1"):
        list(X.generate_mini_batch_indices(0, 4, jax.random.key(0)))


# ------------------------------------------------------------------ the slicing


def test_per_chunk_entries_are_sliced_and_the_rest_rides_along():
    batch = make_batch()
    idx = jnp.asarray([3, 1, 4])
    out = X.sample_minibatch(batch, idx)
    for key in ("observation.state", "action", "raw_advantages", "chunk_clip_counts"):
        assert out[key].shape[0] == 3, key
    np.testing.assert_array_equal(
        np.asarray(out["raw_advantages"]), np.asarray(batch["raw_advantages"])[np.asarray(idx)]
    )
    # Not per-chunk: unsliced, and 3*n rows rather than 3.
    for key in X.PASSTHROUGH_KEYS:
        assert out[key] is batch[key], key


def test_reusing_the_out_dict_does_not_accumulate():
    """`out` exists so slices are not reallocated every minibatch."""
    batch = make_batch()
    out = {}
    first = X.sample_minibatch(batch, jnp.asarray([0, 1]), out)
    second = X.sample_minibatch(batch, jnp.asarray([2, 3, 4, 5]), out)
    assert first is second is out
    assert set(out) == set(batch), "a stale key would be logged or indexed later"
    np.testing.assert_array_equal(
        np.asarray(second["raw_advantages"]), np.asarray(batch["raw_advantages"])[2:6]
    )


# ----------------------------------------------------------------- the diagnostics


def test_rollout_diagnostics_keys_and_values():
    batch = make_batch()
    batch["action"] = jnp.asarray([[0.5, -2.0], [0.25, 1.0], [0.0, 0.5]], dtype=jnp.float32)
    batch["action_is_pad"] = jnp.asarray(
        [[False, False, False], [False, False, True], [False, True, True]], dtype=jnp.bool_
    )
    out = X.compute_rollout_diagnostics(batch)
    assert set(out) == {
        "step_delta", "action_abs_max", "action_magnitude", "chunk_clip_counts"
    }
    # 3 x 3 cells, 3 of them padded: 6 executed.
    assert out["step_delta"] == 6, out["step_delta"]
    assert out["action_abs_max"] == pytest.approx(2.0)
    assert out["action_magnitude"] == pytest.approx((0.5 + 2.0 + 0.25 + 1.0 + 0.0 + 0.5) / 6)
    assert out["chunk_clip_counts"] == pytest.approx(np.arange(N_CHUNKS).mean())
    assert all(isinstance(v, float) for v in out.values()), out


def test_rollout_diagnostics_omit_clip_counts_when_absent():
    batch = make_batch()
    del batch["chunk_clip_counts"]
    out = X.compute_rollout_diagnostics(batch)
    assert "chunk_clip_counts" not in out
    assert set(out) == {"step_delta", "action_abs_max", "action_magnitude"}


def test_the_gae_block_keeps_its_key_order_and_becomes_floats():
    diagnostics = {k: jnp.asarray(float(i)) for i, k in enumerate(GAE_DIAGNOSTIC_KEYS)}
    out = X.gae_diagnostics_as_scalars(diagnostics)
    assert list(out) == list(GAE_DIAGNOSTIC_KEYS)
    assert [out[k] for k in GAE_DIAGNOSTIC_KEYS] == [
        float(i) for i in range(len(GAE_DIAGNOSTIC_KEYS))
    ]


def test_value_drift_is_zero_when_the_critic_has_not_moved():
    """The bootstrap values come *from* the critic, so a fresh one agrees."""
    from sjfpo_jax.configuration_sjfpo import SJFPOConfig
    from sjfpo_jax.policy import module
    from sjfpo_jax.state import build_state

    cfg = SJFPOConfig(obs_dim=11, action_dim=2)
    state = build_state(cfg, 11, 2, jax.random.key(0))
    obs = jnp.asarray(
        np.random.default_rng(1).standard_normal((5, cfg.n_obs_steps, 11), dtype=np.float32)
    )
    boot = module(state.critic)(obs).astype(jnp.float32)
    start = jnp.min(boot, axis=0)
    batch = {
        "boot_observations": obs,
        "boot_values_start": start[:, None],
        "raw_returns": jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
    }
    delta, drift = X.measure_value_drift(state, batch, module)
    assert float(delta) == pytest.approx(0.0, abs=1e-6)
    assert float(drift) == pytest.approx(0.0, abs=1e-6)