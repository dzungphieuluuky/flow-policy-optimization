"""The FPO++ parity changes, each pinned by the property it claims.

The gae re-encoding is the one that could have been silently wrong: it moved the
timeout bootstrap from the target term into the reward, which is the same
arithmetic only if all four terminated/truncated combinations agree. That is
checked against the previous formula, elementwise, at atol=0 -- a tolerance here
would hide exactly the off-by-one-mask bug this is guarding.
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "myfpo"))

from myfpo.configuration_sjfpo import SJFPOConfig
from myfpo.fpo_core import (
    clip_transported,
    compute_cfm_loss,
    compute_policy_ratio,
)
from myfpo.gae import compute_chunk_advantages
from myfpo.layers import SinusoidalEncoding
from myfpo.modeling_sjfpo import build_optimizer, SJFPOPolicy

from train_sjfpo import apply_lr_schedule


def _policy(obs_dim: int = 5, action_dim: int = 2, **kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(**kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (obs_dim,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    return SJFPOPolicy(cfg)


# --- gae: the timeout re-encoding -------------------------------------------


def _gae_inputs(term, trunc, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    assert len(term) == n and len(trunc) == n
    return {
        "step_rewards": torch.rand(24, generator=g),
        "chunk_q": torch.rand(n, generator=g),
        "chunk_next_q": torch.rand(n, generator=g),
        "chunk_starts": torch.arange(n) * 4,
        "chunk_lengths": torch.full((n,), 4),
        "chunk_terminated": torch.tensor(term, dtype=torch.bool),
        "chunk_truncated": torch.tensor(trunc, dtype=torch.bool),
        "chunk_env_id": torch.zeros(n, dtype=torch.long),
        "chunk_size": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }


def _discounted_chunk_rewards(kw):
    """The within-chunk discounted sum, reimplemented for the reference."""
    offsets = torch.arange(kw["chunk_size"])
    window = kw["chunk_starts"].unsqueeze(1) + offsets
    gathered = kw["step_rewards"][window]
    valid = offsets.unsqueeze(0) < kw["chunk_lengths"].unsqueeze(1)
    return (gathered * valid * (kw["gamma"] ** offsets)).sum(dim=1)


def _previous_encoding(kw, term, trunc):
    """The formula this replaced, reimplemented rather than recalled.

    Bootstrap masked by (1 - terminated) only; trace additionally masked by
    (1 - truncated); no reward injection.
    """
    gamma, lam = kw["gamma"], kw["gae_lambda"]
    gamma_h = gamma ** kw["chunk_lengths"]
    q, q_next = kw["chunk_q"], kw["chunk_next_q"]
    bootstrap = gamma_h * (~term)
    delta = _discounted_chunk_rewards(kw) + bootstrap * q_next - q
    trace = bootstrap * (~trunc) * lam
    return delta, trace


def _sweep(delta, trace, succ):
    adv = delta.clone()
    for i in range(len(delta) - 1, -1, -1):
        j = succ[i]
        if j >= 0:
            adv[i] = delta[i] + trace[i] * adv[j]
    return adv


@pytest.mark.parametrize("term,trunc", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("lam", [0.0, 0.95])
def test_timeout_re_encoding_is_bit_identical(term, trunc, lam):
    term_t = torch.zeros(6, dtype=torch.bool)
    trunc_t = torch.zeros(6, dtype=torch.bool)
    term_t[2] = bool(term)
    trunc_t[4] = bool(trunc)

    kw = _gae_inputs(term_t.tolist(), trunc_t.tolist())
    kw["gae_lambda"] = lam
    adv_new, ret_new = compute_chunk_advantages(**kw)

    delta_old, trace_old = _previous_encoding(kw, term_t, trunc_t)
    # One successor: a linear chain, so the trace has something to propagate.
    succ = torch.tensor([i + 1 if i < 5 else -1 for i in range(6)])
    adv_old = _sweep(delta_old, trace_old, succ) if lam > 0 else delta_old

    assert torch.equal(adv_new, adv_old), "advantage diverged from the previous form"
    assert torch.equal(ret_new, adv_old + kw["chunk_q"])


def test_a_timed_out_chunk_still_carries_the_bootstrap():
    """Truncation must not zero the value of the state it cut off in."""
    kw = _gae_inputs([0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0])
    kw["gae_lambda"] = 0.0
    delta, _ = compute_chunk_advantages(**kw)
    gamma_h = kw["gamma"] ** kw["chunk_lengths"]
    want = (
        _discounted_chunk_rewards(kw)[2]
        + gamma_h[2] * kw["chunk_next_q"][2]
        - kw["chunk_q"][2]
    )
    assert delta[2] == pytest.approx(float(want))


def test_a_terminated_chunk_drops_the_bootstrap_entirely():
    kw = _gae_inputs([0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    kw["gae_lambda"] = 0.0
    delta, _ = compute_chunk_advantages(**kw)
    want = _discounted_chunk_rewards(kw)[1] - kw["chunk_q"][1]
    assert delta[1] == pytest.approx(float(want))


def test_terminated_and_truncated_together_keeps_the_terminal_reading():
    """Both flags on one chunk must not inject a value for a finished episode."""
    kw = _gae_inputs([0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])
    kw["gae_lambda"] = 0.0
    delta, _ = compute_chunk_advantages(**kw)
    want = _discounted_chunk_rewards(kw)[1] - kw["chunk_q"][1]
    assert delta[1] == pytest.approx(float(want))


# --- huber CFM loss ---------------------------------------------------------


def test_huber_cfm_loss_equals_mse_inside_the_delta():
    torch.manual_seed(0)
    v_hat = torch.randn(4, 3, 8) * 0.1
    v_star = torch.randn(4, 3, 8) * 0.1
    cfg = SJFPOConfig(cfm_loss_use_huber=True, cfm_loss_huber_delta=0.5)
    huber = compute_cfm_loss(v_hat, v_star, cfg, "token_mean")
    plain = compute_cfm_loss(
        v_hat, v_star, dataclasses.replace(cfg, cfm_loss_use_huber=False), "token_mean"
    )
    # Every |diff| here is below 0.5, so the two branches must agree exactly.
    assert (v_hat - v_star).abs().max() < 0.5
    assert torch.equal(huber, plain)


def test_huber_cfm_loss_is_linear_beyond_the_delta():
    v_star = torch.zeros(1, 1, 1)
    for err in (1.0, 5.0):
        v_hat = torch.full((1, 1, 1), err)
        cfg = SJFPOConfig(cfm_loss_use_huber=True, cfm_loss_huber_delta=0.5)
        got = float(compute_cfm_loss(v_hat, v_star, cfg, "scalar"))
        assert got == pytest.approx(2 * 0.5 * err - 0.5**2)


def test_huber_cfm_loss_keeps_the_mask_renormalisation():
    """A short chunk must keep a full chunk's magnitude."""
    torch.manual_seed(1)
    # [n, mc, horizon, action_dim]; action_is_pad is [n, horizon].
    v_hat, v_star = torch.randn(2, 2, 8, 3), torch.randn(2, 2, 8, 3)
    cfg = SJFPOConfig(cfm_loss_use_huber=True, cfm_loss_huber_delta=0.5)
    half_pad = torch.zeros(2, 8, dtype=torch.bool)
    half_pad[:, 4:] = True
    masked = compute_cfm_loss(v_hat, v_star, cfg, "scalar", half_pad)
    # The same 4 steps with their own pad, so the denominators agree.
    full_pad = torch.zeros(2, 4, dtype=torch.bool)
    trimmed = compute_cfm_loss(
        v_hat[..., :4, :], v_star[..., :4, :], cfg, "scalar", full_pad
    )
    assert float(masked) == pytest.approx(float(trimmed), rel=1e-6)


# --- transported clip -------------------------------------------------------


def test_transported_clip_none_is_a_no_op():
    x = torch.randn(5) * 100
    assert torch.equal(clip_transported(x, None), x)


def test_transported_clip_bounds_both_sides():
    x = torch.tensor([-30.0, -1.0, 0.0, 1.0, 30.0])
    got = clip_transported(x, 2.0)
    assert torch.equal(got, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))


def test_transported_clip_reaches_the_policy_output():
    """Not just the helper: the policy applies it on the prediction path."""
    torch.manual_seed(0)
    policy = _policy(action_dim=2, transported_clip_value=0.01)
    horizon = policy.config.action_horizon
    x = torch.randn(4, horizon, 2) * 50
    tau = torch.full((4, horizon), 0.5)
    cond = torch.randn(4, policy.config.hidden_dim)
    v = policy.compute_predicted_velocity(x, tau, cond)
    assert v.abs().max() <= 0.01 + 1e-6


def test_transported_clip_bounds_the_sampler_output_too():
    """The Euler loop must use the same map, or train and act disagree."""
    torch.manual_seed(0)
    policy = _policy(action_dim=2, transported_clip_value=0.01)
    cfg = policy.config
    x = torch.randn(2, cfg.action_horizon, 2) * 50
    v = policy.compute_predicted_velocity(
        x, torch.full((2, cfg.action_horizon), 0.5),
        torch.randn(2, cfg.hidden_dim),
    )
    assert v.abs().max() <= 0.01 + 1e-6


# --- advantage clamp --------------------------------------------------------


def test_advantage_clamp_is_applied_after_normalisation():
    import gymnasium as gym

    policy = _policy(obs_dim=3, action_dim=1, advantage_clamp=(0.01, 0.01))
    env = gym.make("Pendulum-v1")
    try:
        batch = policy.generate_rollout(env, 32, action_low=-2.0, action_high=2.0)
    finally:
        env.close()
    got = batch["normalised_advantages"]
    assert got.numel() > 0
    assert got.abs().max() <= 0.01 + 1e-6


def test_advantage_clamp_leaves_unclamped_values_alone():
    import gymnasium as gym

    policy = _policy(obs_dim=3, action_dim=1, advantage_clamp=(1e6, 1e6))
    env = gym.make("Pendulum-v1")
    try:
        batch = policy.generate_rollout(env, 32, action_low=-2.0, action_high=2.0)
    finally:
        env.close()
    got = batch["normalised_advantages"]
    assert got.abs().max() > 0.01


# --- lr warmup --------------------------------------------------------------


def test_lr_warmup_ramps_then_holds():
    policy = _policy(
        policy_lr=3e-4, critic_lr=1e-4,
        actor_lr_warmup_steps=4, critic_lr_warmup_steps=2,
    )
    opt = build_optimizer(policy)
    seen = []
    for step in range(1, 7):
        apply_lr_schedule(opt, step, 0, policy.config)
        seen.append([g["lr"] for g in opt.param_groups])
    assert seen[0][0] == pytest.approx(3e-4 / 4)
    assert seen[0][-1] == pytest.approx(1e-4 / 2)  # critic warms twice as fast
    assert seen[1][-1] == pytest.approx(1e-4)       # its ramp is done at step 2
    assert seen[3][0] == pytest.approx(3e-4)        # actor ramp done at step 4
    assert seen[5][0] == pytest.approx(3e-4)        # and held


def test_lr_warmup_and_plateau_decay_compose_without_order():
    """A decay landing mid-warmup must not leave the group permanently scaled."""
    policy = _policy(
        policy_lr=1.0, critic_lr=1.0,
        actor_lr_warmup_steps=4, critic_lr_warmup_steps=4,
    )
    opt = build_optimizer(policy)
    apply_lr_schedule(opt, 2, 1, policy.config)
    mid = opt.param_groups[0]["lr"]
    assert mid == pytest.approx(0.5 * (2 / 4))
    apply_lr_schedule(opt, 4, 1, policy.config)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)


def test_zero_warmup_is_a_no_op():
    policy = _policy(policy_lr=3e-4, actor_lr_warmup_steps=0, critic_lr_warmup_steps=0)
    opt = build_optimizer(policy)
    apply_lr_schedule(opt, 1, 0, policy.config)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4)


# --- optimizer shape and joint clip -----------------------------------------


def test_optimizer_is_one_object_with_two_lr_groups():
    policy = _policy(policy_lr=3e-4, critic_lr=1e-4)
    opt = build_optimizer(policy)
    assert opt.param_groups[0]["lr"] != opt.param_groups[-1]["lr"]
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    live = {id(p) for p in policy.parameters() if p.requires_grad}
    assert live <= in_groups


def _flat_params(opt):
    """The form clip_grad_norm_ accepts. param_groups is a list of dicts."""
    return [p for g in opt.param_groups for p in g["params"]]


def test_joint_clip_is_computed_over_every_group():
    policy = _policy()
    opt = build_optimizer(policy)
    flat = _flat_params(opt)
    for p in flat:
        p.grad = torch.ones_like(p)
    # clip_grad_norm_ scales in place, so the reference must be taken first.
    want = sum(float((p.grad**2).sum()) for p in flat) ** 0.5
    total = float(torch.nn.utils.clip_grad_norm_(flat, 1.0))
    assert total == pytest.approx(want, rel=1e-4)


def test_joint_clip_scales_actor_and_critic_together():
    """The point of a shared budget: one norm, so neither keeps a private one."""
    policy = _policy()
    opt = build_optimizer(policy)
    flat = _flat_params(opt)
    for p in flat:
        p.grad = torch.ones_like(p)
    before = sum(float((p.grad**2).sum()) for p in flat) ** 0.5
    torch.nn.utils.clip_grad_norm_(flat, 1.0)
    after = sum(float((p.grad**2).sum()) for p in flat) ** 0.5
    assert before > 1.0, "the fixture must actually exceed the budget"
    assert after == pytest.approx(1.0, rel=1e-3)


def test_clip_grad_norm_rejects_param_group_dicts():
    """Why the loop passes a flat list: torch raises on the dict form."""
    policy = _policy()
    opt = build_optimizer(policy)
    for p in _flat_params(opt):
        p.grad = torch.ones_like(p)
    with pytest.raises(AttributeError):
        torch.nn.utils.clip_grad_norm_(opt.param_groups, 1.0)


# --- fourier frequencies ----------------------------------------------------


@pytest.mark.parametrize("dim", [32, 64])
def test_frequencies_are_powers_of_two(dim):
    enc = SinusoidalEncoding(dim)
    got = enc.compute_freqs(torch.device("cpu"))
    want = torch.pow(torch.tensor(2.0), torch.arange(dim // 2, dtype=torch.float32))
    assert torch.equal(got, want)


def test_embedding_puts_cos_before_sin():
    """FPO++'s _embed_timestep concatenates cos then sin, not the reverse."""
    enc = SinusoidalEncoding(8)
    t = torch.tensor([0.3])
    ang = t.unsqueeze(-1) * enc.compute_freqs(torch.device("cpu"))
    assert torch.equal(enc.encode_times(t), torch.cat([ang.cos(), ang.sin()], dim=-1))


def test_default_width_matches_fpops_larger_setting():
    """32 is FPO++'s manipulation value; 8 is its isaaclab one. We take 32."""
    assert SJFPOConfig().time_embed_dim == 32


def test_embedding_is_exact_against_a_float64_reference():
    """The channels are not corrupted, only over-wide. Pin that they are fine."""
    for dim in (8, 32, 64):
        enc = SinusoidalEncoding(dim)
        t = torch.linspace(0.0, 1.0, 2048, dtype=torch.float64)
        err = (enc.encode_times(t.float()).double() - enc.encode_times(t)).abs().max()
        assert float(err) < 1e-5, f"dim={dim} drifted from the fp64 reference"


# --- deleted target critic --------------------------------------------------


def test_policy_has_no_target_critic_module():
    policy = _policy()
    assert not hasattr(policy, "target_critic")
    assert not hasattr(policy, "update_target_critic")
    assert not any("target_critic" in n for n, _ in policy.named_parameters())


def test_removed_config_fields_are_gone():
    fields = {f.name for f in dataclasses.fields(SJFPOConfig)}
    for field in (
        "use_target_critic", "critic_ema_decay", "critic_max_grad_norm", "timestep_scale"
    ):
        assert field not in fields


# --- the ratio path is untouched --------------------------------------------


def test_loss_clamps_match_fpops_values():
    """Both are 20 in rl_cfg.py. Moving one alone would strand the other."""
    cfg = SJFPOConfig()
    assert cfg.cfm_clamp == 20.0
    assert cfg.cfm_loss_clamp_negative_advantages_max == 20.0


def test_the_negative_advantage_clamp_is_not_dead_at_the_default():
    """The two must stay equal, or the tighter one silently owns both signs."""
    cfg = SJFPOConfig()
    assert cfg.cfm_loss_clamp_negative_advantages_max == cfg.cfm_clamp


def test_clamp_binds_only_far_outside_the_observed_loss_scale():
    """Guard against tightening past what the trust region actually sees.

    Measured on Pendulum over 8 updates, cfm_loss_new peaks near 1.9 and averages
    ~0.63, so 20 is ~10x headroom. If the reduction changes and the scale
    moves an order of magnitude, this should be re-examined rather than
    inherited.
    """
    torch.manual_seed(0)
    cfg = SJFPOConfig()
    # v_star ~ a - eps with a in [-1,1], eps ~ N(0,1); v_hat ~ 0 at init.
    v_star = torch.randn(512, 8, cfg.action_horizon, 1) * 1.4
    v_hat = torch.zeros_like(v_star)
    l = compute_cfm_loss(v_hat, v_star, cfg, "sample")
    assert float(l.max()) < cfg.cfm_clamp, "clamp would bind on init-scale losses"
    # A genuine runaway still has to hit it, or the guard is decorative.
    assert float((l * 5000).max()) > cfg.cfm_clamp


def test_ratio_still_sees_both_losses_when_clipped():
    cfm_loss_old = torch.tensor([[5.0, 7.0]])
    cfm_loss_new = torch.tensor([[3.0, 4.0]])
    cfg = SJFPOConfig(transported_clip_value=0.5, cfm_loss_use_huber=True)
    ratio = compute_policy_ratio(cfm_loss_old, cfm_loss_new, cfg)
    assert torch.allclose(ratio, torch.exp(torch.tensor([[2.0, 3.0]])), atol=1e-6)