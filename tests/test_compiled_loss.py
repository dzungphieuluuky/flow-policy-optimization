"""The compiled loss path vs the eager one.

The pins cover the eager path at ``atol=0`` and are the source of truth for
behavior. The compiled path may drift at kernel level, so these tests bound
the drift instead of forbidding it: finiteness, loss agreement, an unchanged
gradient coverage set, and an unchanged info-dict key set.
Compile is off by default; nothing here changes the pinned configuration.
"""

import dataclasses

import pytest
import torch

from fixtures import make_batch

from myfpo.configuration import MyfpoConfig
from myfpo.training import make_policy

OBS, ACT, HORIZON, HIDDEN, TIME_EMBED = 6, 3, 8, 32, 16


def _cfg(compile: bool, **kw) -> MyfpoConfig:
    base = dict(
        obs_dim=OBS, action_dim=ACT,
        action_horizon=HORIZON, hidden_dim=HIDDEN, time_embed_dim=TIME_EMBED,
        dit_depth=1, dit_heads=2, n_obs_steps=2,
        # Dropout draws differ between eager and compiled graphs; the math
        # under test is the same. Disable it for equivalence comparisons.
        critic_dropout=0.0,
        compile=compile,
    )
    base.update(kw)
    return dataclasses.replace(MyfpoConfig(), **base)


def _paired_policies(**over):
    # Same seed, same build order: identical weights for both policies.
    torch.manual_seed(0)
    eager = make_policy(
        OBS, ACT, _cfg(False, **over), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    torch.manual_seed(0)
    compiled = make_policy(
        OBS, ACT, _cfg(True, **over), device="cpu",
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    return eager, compiled


def _batch(policy):
    # The zero fixture makes ensemble-correlation metrics degenerate (NaN).
    batch = make_batch(policy, obs_dim=OBS, action_dim=ACT, horizon=HORIZON, n=2)
    g = torch.Generator().manual_seed(1)
    batch["observation.state"] = torch.randn(
        2, policy.config.n_obs_steps, OBS, generator=g
    )
    batch["action"] = torch.randn(2, HORIZON, ACT, generator=g)
    return batch


def test_compiled_loss_matches_eager():
    eager, compiled = _paired_policies()
    batch = _batch(eager)
    loss_eager, info_eager = eager.loss_with_info(batch)
    loss_compiled, info_compiled = compiled.loss_with_info(batch)
    assert torch.isfinite(loss_eager).all()
    assert torch.isfinite(loss_compiled).all()
    assert abs(float(loss_eager.detach()) - float(loss_compiled.detach())) < 1e-4
    assert info_compiled is not None
    assert set(info_compiled) == set(info_eager)
    for key in info_eager:
        a, b = info_eager[key], info_compiled[key]
        if a != a and b != b:  # both NaN: equal by fiat
            continue
        assert abs(a - b) <= 1e-4 + 1e-4 * abs(b), key


def test_compiled_loss_gradient_coverage_matches_eager():
    eager, compiled = _paired_policies()
    batch = _batch(eager)
    eager.loss_with_info(batch)[0].backward()
    compiled.loss_with_info(batch)[0].backward()
    missing_eager = {n for n, p in eager.named_parameters() if p.grad is None}
    missing_compiled = {n for n, p in compiled.named_parameters() if p.grad is None}
    assert missing_compiled == missing_eager
    for (n, p), (n2, p2) in zip(eager.named_parameters(), compiled.named_parameters()):
        assert n == n2
        if p.grad is None:
            continue
        assert torch.allclose(p.grad, p2.grad, atol=1e-4, rtol=1e-4), n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reduce-overhead needs CUDA")
def test_reduce_overhead_loss_matches_eager():
    device = torch.device("cuda")
    torch.manual_seed(0)
    eager = make_policy(
        OBS, ACT, _cfg(False), device=device,
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    torch.manual_seed(0)
    compiled = make_policy(
        OBS, ACT, _cfg(True, compile_mode="reduce-overhead"), device=device,
        action_low=[-1.0] * ACT, action_high=[1.0] * ACT,
    )
    batch = _batch(eager)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    loss_eager, info_eager = eager.loss_with_info(batch)
    loss_compiled, info_compiled = compiled.loss_with_info(batch)
    assert torch.isfinite(loss_compiled).all()
    assert abs(float(loss_eager.detach()) - float(loss_compiled.detach())) < 1e-4
    assert set(info_compiled) == set(info_eager)
