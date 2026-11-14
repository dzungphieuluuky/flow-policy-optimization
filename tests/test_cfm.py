"""CFM math: standard-normal noise draw, behavior-loss agreement."""
import torch

from fixtures import ACT_DIM, OBS_DIM, make_batch, make_test_policy

from myfpo.myfpo import sample_noise


def test_normal_mode_is_plain_randn():
    """White == torch.randn, bit for bit, same generator."""
    shape = (4, 8, 3, 3)
    got = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    want = torch.randn(*shape, generator=g)
    assert torch.equal(got, want)


def test_generated_noise_is_reproducible():
    """Rollouts draw eps from a seeded generator. Twice must match."""
    shape = (2, 3, 8, 3)
    a = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(3))
    b = sample_noise(shape, torch.device("cpu"), torch.Generator().manual_seed(3))
    assert torch.equal(a, b)



# ---- Shared token-loss prologue (from test_cfm_shared_loss) ----

def _policy_batch(n=3):
    torch.manual_seed(0)
    policy = make_test_policy()
    policy.eval()
    batch = make_batch(
        policy, obs_dim=OBS_DIM, action_dim=ACT_DIM,
        horizon=policy.config.action_horizon, n=n,
    )
    return policy, batch


def test_behavior_and_training_read_one_token_loss():
    """A fresh policy reproduces its own behavior loss exactly (ratio == 1).

    The rollout stores the CFM loss at collection time against the frozen
    snapshot; the update pass recomputes it on the same flow points. With no
    update applied, the two must agree bit-for-bit.
    """
    policy, batch = _policy_batch()
    device = next(policy.parameters()).device

    normalized_states = policy.obs_normalizer(batch["observation.state"])
    loss_b, token_b = policy.compute_behavior_cfm_losses(
        normalized_states, batch["action"], batch["mc_t"], batch["mc_eps"], batch["action_is_pad"]
    )
    batch["behavior_cfm_loss"] = loss_b
    batch["behavior_cfm_loss_token"] = token_b

    from myfpo.myfpo import compute_loss

    _, metrics = compute_loss(policy, batch, device)
    assert metrics["cfm_loss_new"] == float(loss_b.mean())
    assert torch.isfinite(token_b).all()

