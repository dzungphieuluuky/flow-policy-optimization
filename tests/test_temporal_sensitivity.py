"""The temporal-sensitivity probe must run, not warn on every interval.

``to_device_batch(grad_inputs=True)`` builds the two probe leaves with
``detach().requires_grad_(True)``, so their ``grad_fn`` is None by
construction. The guard tested ``grad_fn is None and requires_grad``, which is
always true, so the probe warned and returned nothing on every run.
"""

from __future__ import annotations

import warnings

from fixtures import make_batch, make_test_policy


def _info(probe_every: int) -> dict[str, float]:
    policy = make_test_policy(temporal_probe_every=probe_every)
    cfg = policy.config
    batch = make_batch(
        policy,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        horizon=cfg.action_horizon,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = policy.loss_with_info(batch)
    dead = [w for w in caught if "temporal_sensitivity" in str(w.message)]
    assert not dead, dead[0].message
    return info


def test_the_probe_reports_both_axes():
    info = _info(1)
    assert any(k.startswith("critic_action_grad_norm_pos") for k in info), sorted(info)
    assert any(k.startswith("obs_frame_grad_norm_pos") for k in info), sorted(info)


def test_the_probe_stays_off_until_its_interval():
    info = _info(0)
    assert not any("grad_norm_pos" in k for k in info), sorted(info)
