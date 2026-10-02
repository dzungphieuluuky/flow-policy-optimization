"""torch vs jax parity for ``generate_rollout``: the batch the loss consumes.

What can and cannot be compared, and why:

* **Everything the chunk backup produces** matches, once both sides reset the
  envs from the same seed: ``action``, ``action_is_pad``, ``raw_advantages``,
  ``raw_returns``, ``raw_values``, ``boot_observations``, ``boot_values_start``
  and all five ``gae_diagnostics``. These are reductions over the collected
  chunks, so a seeded rollout is enough.
* **The MC draws cannot match.** ``mc_tau`` / ``mc_eps`` come from torch's global
  generator on one side and a ``jax`` key on the other, and neither
  ``generate_rollout`` takes a seed for them. So they are checked for shape and
  dtype only, and then *torch's* draws are injected into the JAX batch to
  compare ``behavior_cfm_loss`` — which is the reduction that actually depends on
  them.

The env seed is pinned by patching ``modeling_sjfpo.collect_chunks_vec``,
because torch's ``generate_rollout`` does not forward one. The patch asserts it
was reached, so it cannot silently become a no-op.
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

from test_policy_parity import (
    ACT,
    CFG,
    HIGH,
    LOW,
    OBS,
    assert_same,
    pair,
    wake_actor,
)
from torch_jax_weights import to_jax_tree

from sjfpo_jax import policy as J
from sjfpo_jax.state import load_pair
from spectral_joint_policy import modeling_sjfpo as TM

SEED = 7
NUM_ENVS = 4
TOTAL_STEPS = 128
EPISODE = 5  # forces terminations and autoresets inside the rollout
# The batch keys that are reductions over the collected chunks, so both sides can
# agree on them. mc_tau / mc_eps are excluded on purpose: they are drawn, not
# computed, and the two frameworks draw from different generators.
DETERMINISTIC_KEYS = (
    "observation.state",
    "action",
    "action_is_pad",
    "raw_advantages",
    "normalised_advantages",
    "raw_returns",
    "raw_values",
    "normalizer_observations",
    "boot_observations",
    "boot_values_start",
    "chunk_clip_counts",
)
GAE_KEYS = (
    "linked_frac",
    "advantage_std",
    "return_rms",
    "advantage_std_fraction",
    "budget_truncated_chunk_frac",
)


class FakeVecEnv:
    """Deterministic NEXT_STEP vec env at ``pair()``'s dims.

    Reacher-v5 has obs_dim 10, but ``pair()`` builds both policies at OBS=11, so
    the two rollouts cannot share it. This is a fixed function of
    ``(seed + env_id, step)`` with no RNG of its own, so both sides see the same
    trajectory from the same reset seed. ``episode`` forces terminations, and
    autoreset fires because the collector resets in two steps: flag on
    ``done=True``, take the reset obs on the next ``step()``.
    """

    def __init__(self, n=NUM_ENVS, episode=5):
        self.num_envs = n
        self.observation_space = type("S", (), {"shape": (OBS,)})()
        self.single_observation_space = self.observation_space
        self.single_action_space = type("S", (), {"shape": (ACT,)})()
        self.autoreset_mode = type("M", (), {"name": "NEXT_STEP"})()
        self.episode = episode
        self.t = np.zeros(n, dtype=np.int64)
        self.seed = 0
        self.actions_seen: list[np.ndarray] = []

    def _obs(self):
        # A ramp on the step plus a per-env offset, so a window that fails to
        # advance, or a torn one, changes the numbers.
        idx = np.arange(OBS, dtype=np.float32)[None, :]
        env = np.arange(self.num_envs, dtype=np.float32)[:, None]
        return (self.t[:, None].astype(np.float32) * 0.1 + idx * 0.01 + env).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(np.min(seed))
        self.t[:] = 0
        return self._obs(), {}

    def step(self, actions):
        self.actions_seen.append(np.array(actions, copy=True))
        done = self.t >= self.episode
        # The counter restarts on the step *after* the flag, which is what makes
        # done a pulse rather than a level.
        self.t = np.where(done, 0, self.t + 1)
        rew = (self.t.astype(np.float32) * 0.5 - 1.0).astype(np.float32)
        return self._obs(), rew, done, np.zeros_like(done), {}


def make_envs(episode=5):
    return FakeVecEnv(n=NUM_ENVS, episode=episode)


def pin_env_seed(monkeypatch):
    """Force torch's collector to reset from SEED, and prove the patch ran."""
    real = TM.collect_chunks_vec
    calls = []

    def pinned(*args, **kwargs):
        calls.append(kwargs.get("seed"))
        kwargs["seed"] = SEED
        return real(*args, **kwargs)

    monkeypatch.setattr(TM, "collect_chunks_vec", pinned)
    return calls


def run_both(monkeypatch, cfg=CFG, total_steps=TOTAL_STEPS):
    t_policy, state = pair(cfg, seed=0)
    # eval() so torch's predict_action_chunk does not add
    # action_perturb_std * randn: the perturbation is gated on self.training, it
    # is driven by the global torch RNG, and JAX's side is called with
    # perturb=False, so leaving it on makes the two propose different chunks and
    # every downstream comparison measures the noise instead of the math. The
    # perturb on/off contract is gated separately, against predict_action_chunk.
    t_policy.eval()
    # pair() syncs the live critic but not the target critic, and generate_rollout
    # bootstraps from the target when use_target_critic is on. Left unsynced the
    # two read different shadows and every advantage and return diverges.
    state = state.replace(
        target_critic=load_pair(
            state.target_critic,
            to_jax_tree(t_policy.target_critic.state_dict(), stack=["conditioners", "heads"]),
        )
    )
    # wake_actor before the snapshot: SJFPOFlowModel zero-inits its output layer,
    # so an unwoken actor integrates from eps=0 with zero velocity and proposes
    # an all-zero chunk. The rollout would then be a constant, and every
    # comparison below would be two copies of it.
    state = wake_actor(state, t_policy)
    t_policy.snapshot_behavior_policy()
    calls = pin_env_seed(monkeypatch)
    t_batch = t_policy.generate_rollout(
        make_envs(),
        total_steps,
        action_low=LOW,
        action_high=HIGH,
        noisy=False,
    )
    assert calls, "the env-seed patch was never reached, so nothing was pinned"

    j_batch = J.generate_rollout(
        state,
        make_envs(),
        total_steps,
        LOW,
        HIGH,
        cfg,
        jax.random.key(0),
        noisy=False,
        seed=SEED,
    )
    return t_policy, state, t_batch, j_batch


def test_rollout_matches_torch_key_for_key(monkeypatch):
    _, _, t_batch, j_batch = run_both(monkeypatch)
    assert set(j_batch) == set(t_batch), (
        sorted(set(t_batch) - set(j_batch)),
        sorted(set(j_batch) - set(t_batch)),
    )
    for key in DETERMINISTIC_KEYS:
        assert_same(t_batch[key], j_batch[key])
    for key in GAE_KEYS:
        assert_same(t_batch["gae_diagnostics"][key], j_batch["gae_diagnostics"][key])


def test_the_mc_draws_have_the_layout_the_loss_expects(monkeypatch):
    """Shape and dtype only: the values are drawn from different generators."""
    cfg = CFG
    _, _, t_batch, j_batch = run_both(monkeypatch, cfg)
    n_chunks = t_batch["raw_advantages"].shape[0]
    shape = (n_chunks, cfg.num_mc_samples, cfg.num_flow_tokens, ACT)
    assert j_batch["mc_eps"].shape == shape, j_batch["mc_eps"].shape
    assert j_batch["mc_eps"].dtype == jnp.float32
    assert j_batch["mc_tau"].shape == (n_chunks, cfg.num_mc_samples, cfg.num_flow_tokens, 1)
    assert j_batch["mc_eps"].shape == t_batch["mc_eps"].shape
    assert j_batch["mc_tau"].shape == t_batch["mc_tau"].shape


def test_the_behaviour_loss_matches_once_the_draws_are_shared(monkeypatch):
    """The l_old reduction, on torch's draws, in both dialects.

    This is the only part of the batch that depends on the MC samples, and the
    ratio that drives the policy gradient compares it against l_new on the *same*
    samples, so it is the part that most needs to agree.
    """
    cfg = CFG
    _, j_state, t_batch, j_batch = run_both(monkeypatch, cfg)

    shared = dict(j_batch)
    shared["mc_tau"] = jnp.asarray(np.asarray(t_batch["mc_tau"]))
    shared["mc_eps"] = jnp.asarray(np.asarray(t_batch["mc_eps"]))
    l_old, l_old_token = J.compute_behavior_cfm_loss(j_state, shared, cfg)
    assert_same(t_batch["behavior_cfm_loss"], l_old)
    assert_same(t_batch["behavior_cfm_loss_token"], l_old_token)


def test_the_rollout_is_not_a_constant_rollout(monkeypatch):
    """A guard on the guard.

    The actor's output layer is zero-init, so an unwoken policy proposes the
    same chunk for every window and the whole batch is degenerate. If this fails,
    every comparison above is comparing two copies of one constant.
    """
    _, _, _, j_batch = run_both(monkeypatch)
    actions = np.asarray(j_batch["action"])
    assert actions.shape[0] >= NUM_ENVS, actions.shape
    assert np.abs(actions).max() > 0, "every proposed action was zero"
    assert len({tuple(row) for row in np.asarray(j_batch["observation.state"][:, -1])}) > 1, (
        "the obs windows never advanced, so the collector did not step"
    )


def test_the_noisy_rollout_draws_fresh_noise_per_chunk(monkeypatch):
    """noisy=True must not hand every chunk the same eps.

    The propose key is folded in per wavefront. With a fixed key the whole
    rollout shares one eps draw, so the exploration noise is correlated across
    chunks. Asserting on the *actions* would not catch it -- they depend on the
    obs window too, so they differ either way -- so the keys handed to
    ``sample_noise`` are captured instead and required to be pairwise distinct.

    ``noisy=False`` never reads the key, so this is the only place the counter is
    observable.
    """
    t_policy, state = fixture_like()
    del t_policy
    monkeypatch.setattr(J, "compute_gae_diagnostics", _noop_gae)
    monkeypatch.setattr(J, "sample_timesteps", lambda shape, cfg, key: jnp.zeros(shape))

    keys: list = []
    real_noise = J.sample_noise

    def recording_noise(shape, cfg, key):
        keys.append(key)
        # A per-call constant, so a repeated key produces a repeated eps and the
        # actions alone could not tell the two cases apart.
        return jnp.full(shape, float(jax.random.randint(key, (), 0, 1000)), dtype=jnp.float32)

    monkeypatch.setattr(J, "sample_noise", recording_noise)
    J.generate_rollout(state, make_envs(), 96, LOW, HIGH, CFG, jax.random.key(3), noisy=True)

    # One propose call per wavefront, then one draw for the batch's MC eps.
    propose_keys, mc_keys = keys[:-1], keys[-1:]
    assert len(propose_keys) >= 3, f"only {len(propose_keys)} wavefronts, too few to judge"
    assert len({str(k) for k in propose_keys}) == len(propose_keys), (
        "two wavefronts were handed the same noise key"
    )
    assert len(mc_keys) == 1, mc_keys
    assert str(mc_keys[0]) not in {str(k) for k in propose_keys}, (
        "the batch's MC draw reused a wavefront's key"
    )
    assert real_noise is not None


def _noop_gae(*args, **kwargs):
    return {}


def fixture_like(seed=0):
    from test_policy_parity import pair, wake_actor

    t_policy, state = pair(CFG, seed=seed)
    return t_policy, wake_actor(state, t_policy)


def test_a_nonpositive_step_budget_is_rejected(monkeypatch):
    _, j_state, _, _ = run_both(monkeypatch)
    with pytest.raises(ValueError, match="total_steps"):
        J.generate_rollout(
            j_state, make_envs(), 0, LOW, HIGH, CFG, jax.random.key(0), seed=SEED
        )