"""The EMA shadow must cover actor and encoder as one unit.

Found from a Humanoid run that peaked at 131k and decayed 46%."""

import warnings

import torch

from fixtures import make_policy
from myfpo.modeling import MyfpoPolicy


def _warm(p: MyfpoPolicy) -> MyfpoPolicy:
    p.update_ema(0)
    p.update_ema(1000)
    return p


# --- EMA covers the encoder -------------------------------------------------.


def test_ema_shadow_covers_the_encoder():
    p = _warm(make_policy())
    enc = {f"encoder.{n}" for n, _ in p.encoder.named_parameters()}
    actor = {f"actor.{n}" for n, _ in p.actor.named_parameters()}
    assert enc, "the encoder has no parameters, test is vacuous"
    assert enc <= set(p.ema_shadow), "encoder params missing from the EMA shadow"
    assert actor <= set(p.ema_shadow)


def test_ema_names_are_namespaced_so_the_two_modules_cannot_collide():
    p = _warm(make_policy())
    for name in p.ema_shadow:
        assert name.startswith(("actor.", "encoder.")), name


def test_swap_round_trips_actor_and_encoder_together():
    p = _warm(make_policy())
    before = {n: q.detach().clone() for n, q in p.ema_named_params().items()}
    shadow = {n: t.clone() for n, t in p.ema_shadow.items()}

    displaced = p.swap_in_ema(1000)
    assert set(displaced) == set(shadow), "swap left a module behind"
    for name, want in shadow.items():
        assert torch.equal(p.ema_named_params()[name], want)

    p.swap_out_ema(displaced)
    for name, want in before.items():
        assert torch.equal(p.ema_named_params()[name], want), name


def test_swap_in_actually_moves_the_encoder():
    """The old bug: only self.actor moved, so the encoder stayed live."""
    p = _warm(make_policy())
    with torch.no_grad():
        for q in p.encoder.parameters():
            q.add_(100.0)
    p.update_ema(2000)
    name = next(n for n in p.ema_named_params() if n.startswith("encoder."))
    live = p.ema_named_params()[name].detach().clone()
    p.swap_in_ema(2000)
    shadow = p.ema_named_params()[name]
    assert not torch.equal(shadow, live), "encoder was not swapped in"
    p.swap_out_ema(p.swap_in_ema(2000))


def test_an_actor_only_shadow_is_reseeded_not_half_applied():
    p = _warm(make_policy())
    legacy = {"initialised": True, "shadow_params": {
        n: torch.zeros_like(q) for n, q in p.actor.named_parameters()
    }}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing should warn at this layer.
        assert p.load_ema_state_dict(legacy) is False
    assert p.ema_shadow == {}
    assert p.ema_initialised is False


def test_a_matching_shadow_is_restored():
    p = _warm(make_policy())
    state = p.ema_state_dict()
    fresh = make_policy()
    assert fresh.load_ema_state_dict(state) is True
    assert set(fresh.ema_shadow) == set(p.ema_shadow)


# --- state dict -----------------------------------------------------------.


def test_the_ema_state_dict_carries_the_encoder():
    p = _warm(make_policy())
    state = p.ema_state_dict()
    assert any(k.startswith("encoder.") for k in state["shadow_params"])


def test_ema_scored_checkpoints_save_the_ema_weights(monkeypatch, tmp_path):
    """The best and final scores are measured on the EMA weights.

    The save must run with the EMA weights in the living model, and must
    put the live weights back after. The equality is exact.
    """
    import myfpo.training as training

    captured: dict[str, dict[str, torch.Tensor]] = {}

    def fake_save(ckpt_dir, policy, **kw):  # noqa: ARG001
        captured["saved"] = {
            n: q.detach().clone() for n, q in policy.ema_named_params().items()
        }

    monkeypatch.setattr(training, "save_policy_checkpoint", fake_save)
    p = _warm(make_policy())
    shadow = {n: t.clone() for n, t in p.ema_shadow.items()}
    with torch.no_grad():
        for q in p.ema_named_params().values():
            q.add_(1.0)
    live = {n: q.detach().clone() for n, q in p.ema_named_params().items()}

    training.save_checkpoint(tmp_path, p, None, 1000, "best", use_ema=True)
    for name, want in shadow.items():
        assert torch.equal(captured["saved"][name], want), name
    after = {n: q.detach().clone() for n, q in p.ema_named_params().items()}
    for name, want in live.items():
        assert torch.equal(after[name], want), name
    assert any(not torch.equal(live[n], shadow[n]) for n in shadow), (
        "live and EMA are identical, test is vacuous"
    )


def test_a_plain_save_keeps_the_live_weights(monkeypatch, tmp_path):
    """The periodic resume checkpoints carry the live weights."""
    import myfpo.training as training

    captured: dict[str, dict[str, torch.Tensor]] = {}

    def fake_save(ckpt_dir, policy, **kw):  # noqa: ARG001
        captured["saved"] = {
            n: q.detach().clone() for n, q in policy.ema_named_params().items()
        }

    monkeypatch.setattr(training, "save_policy_checkpoint", fake_save)
    p = _warm(make_policy())
    live = {n: q.detach().clone() for n, q in p.ema_named_params().items()}

    training.save_checkpoint(tmp_path, p, None, 1000, "step_1000")
    for name, want in live.items():
        assert torch.equal(captured["saved"][name], want), name