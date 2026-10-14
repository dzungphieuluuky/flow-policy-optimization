"""DPPO: the denoising MDP (Ren et al., NeurIPS 2024, arXiv:2409.00588).

Not a config flag. DPPO treats the *sampling chain* as an MDP:

    level 1 (inner)  x_0 = eps, x_{k+1} = x_k + dt_k v(x_k, t_k, o) + noise
                     action_k = the increment; observation = (x_k, t_k, o)
    level 2 (outer)  the environment; one macro-action = one full denoise

So the policy ratio is an ordinary Gaussian PPO ratio on ``x_{k+1}`` given
``(x_k, t_k, o)`` -- exact log-probs, no ELBO surrogate -- and the horizon is
``K`` times longer than the environment horizon. Two consequences that make it
*not* FPO:

  * the initial noise eps is part of the state, so the policy must see it;
  * only stochastic sampling works, so training can never start from eps=0.

This module owns the inner MDP. The outer loop (rollout, chunk GAE over
environment rewards, critic) is shared with SJFPO; only the inner step is
new. ``DenoisingPath`` is what the buffer must carry for this to be trainable
on-policy: SJFPO's RolloutBuffer stores the *result* of a chunk, not the path
that produced it, which is exactly the field FPO says it needs and DPPO needs
much more of.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "DenoisingPath",
    "denoising_chain_gae",
    "gaussian_log_prob",
    "gaussian_step_ratio",
    "sample_denoising_path",
]


class DenoisingPath:
    """The full sampling chain for one chunk, kept for the inner PPO backup.

    ``states``  ``[n, K, *flow]``   x_k, the point *before* step k
    ``actions`` ``[n, K, *flow]``   the increment actually applied at step k
    ``mus``     ``[n, K, *flow]``   its mean, so log-prob needs no re-forward
    ``logstd``  ``[n]``             one scalar per path, shared over the chunk
    ``final``   ``[n, *flow]``      x_K, the action handed to the environment

    ``logstd`` is per-path rather than per-(state, action) because a
    state-dependent log-std would need the old policy's log-std evaluated at
    every step to form the ratio; DPPO Alg. 2 carries it as one scalar. That is
    a real modelling simplification inherited from the paper, not a shortcut.
    """

    __slots__ = ("actions", "final", "logstd", "mus", "obs", "states")

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        mus: torch.Tensor,
        logstd: torch.Tensor,
        final: torch.Tensor,
        obs: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.mus = mus
        self.logstd = logstd
        self.final = final
        self.obs = obs

    @property
    def n_steps(self) -> int:
        return int(self.states.shape[1])

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def flatten(self) -> tuple[torch.Tensor, ...]:
        """``[n*K, ...]`` views for the inner PPO minibatch."""
        n, k = self.states.shape[0], self.n_steps
        return (
            self.states.reshape(n * k, *self.states.shape[2:]),
            self.actions.reshape(n * k, *self.actions.shape[2:]),
            self.mus.reshape(n * k, *self.mus.shape[2:]),
            self.logstd[:, None].expand(n, k).reshape(n * k),
        )


def sample_denoising_path(
    predict_velocity,
    condition: torch.Tensor,
    flow_shape: tuple[int, ...],
    *,
    num_steps: int,
    noise_std: float,
    generator: torch.Generator | None = None,
    store: bool = True,
) -> tuple[torch.Tensor, DenoisingPath | None]:
    """Euler-integrate a stochastic denoising chain, optionally recording it.

    ``predict_velocity(x, t, c) -> v`` must match the actor's signature.
    ``condition`` is ``[B, hidden]``. Returns ``(x_final, path)``; ``path`` is
    ``None`` when ``store=False`` (rollouts under :func:`torch.inference_mode`
    that only need the action can skip the memory).

    The chain is ``x_{k+1} = x_k + dt_k v + sigma_k z``, with ``z ~ N(0, I)``.
    DPPO Alg. 2 injects noise at every step; ``noise_std=0`` recovers
    deterministic Euler and exists only for tests, since a deterministic
    sampler has zero-variance Gaussians and therefore no meaningful ratio.

    Note the sign convention. ``mus`` stores the *pre-noise* increment, so the
    stored action is ``mu + sigma z`` and the log-prob is
    ``-||a - mu||^2 / (2 sigma^2) - log sigma - d/2 log 2pi``. Storing the
    increment (rather than the point) is what makes the ratio computable
    without re-running the old policy over the whole chain.
    """
    device = condition.device
    b = condition.shape[0]
    eps = torch.randn(*(b, *flow_shape), device=device, generator=generator)
    x = eps.clone()
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    dts = grid[1:] - grid[:-1]

    if store:
        shape = (b, num_steps, *flow_shape)
        s_buf = torch.empty(shape, device=device)
        a_buf = torch.empty(shape, device=device)
        m_buf = torch.empty(shape, device=device)
    # One scalar per path, matching DPPO Alg. 2's state-independent sigma.
    # log-std, not std: the ratio needs log p, and exp of a stored log-std is
    # what avoids a log(0) at noise_std=0.
    logstd = torch.full(
        (b,), math.log(noise_std) if noise_std > 0.0 else -20.0, device=device
    )

    with torch.inference_mode():
        for k in range(num_steps):
            t = grid[k].reshape(1)
            dt = dts[k]
            v = predict_velocity(x, t, condition)
            mu = dt * v
            if noise_std > 0.0:
                z = torch.randn(x.shape, device=device, generator=generator)
                step = mu + noise_std * z
            else:
                step = mu
            if store:
                s_buf[:, k] = x
                m_buf[:, k] = mu
                a_buf[:, k] = step
            x = x + step

    path = None
    if store:
        path = DenoisingPath(
            states=s_buf, actions=a_buf, mus=m_buf, logstd=logstd, final=x,
            obs=condition,
        )
    return x, path


def gaussian_step_ratio(
    logp_new: torch.Tensor, logp_old: torch.Tensor, clip_param: float
) -> torch.Tensor:
    """PPO ratio for one denoising step. Exact Gaussian log-probs.

    DPPO uses the identical clipped surrogate as PPO; only the *action*
    differs (a noise increment rather than an environment action). Kept
    separate from ``compute_aspo_surrogate`` because that one consumes the
    FPO CFM-loss surrogate and has no log-prob to clip.
    """
    ratio = torch.exp(logp_new - logp_old)
    return torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)


def gaussian_log_prob(
    action: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor
) -> torch.Tensor:
    """Diagonal-Gaussian log-prob, summed over every non-batch axis."""
    diff = (action - mu) * torch.exp(-logstd).unsqueeze(-1)
    d = action[0].numel() if action.dim() > 1 else 1
    return (
        -0.5 * (diff * diff).flatten(1).sum(-1)
        - logstd * d
        - 0.5 * d * math.log(2.0 * math.pi)
    )


def denoising_chain_gae(
    step_rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """GAE over the inner chain, DPPO Eq. 4.

    ``step_rewards`` ``[n, K]``, ``values`` ``[n, K+1]`` = V(s_k), ``dones``
    ``[n, K]`` bool, True when the chain ended (env terminated) at step k.
    Returns ``[n, K]`` advantages.

    ``values`` carries one column more than the chain, because the step at
    ``j = K-1`` bootstraps from ``V(s_K)``; column K is that terminal value,
    not a value of a further step.

    The env's reward lands on the *last* inner step (that is the only step
    where an environment transition happened), so a caller passing chunk rewards
    should scatter them onto the final column rather than the first. That
    asymmetry is DPPO's own accounting and is the single easiest thing to get
    wrong when porting it.
    """
    n, k = step_rewards.shape
    adv = torch.zeros_like(step_rewards)
    last = torch.zeros(n, device=step_rewards.device, dtype=step_rewards.dtype)
    for j in range(k - 1, -1, -1):
        nonterminal = 1.0 - dones[:, j].to(step_rewards.dtype)
        delta = step_rewards[:, j] + gamma * nonterminal * values[:, j + 1] - values[:, j]
        last = delta + gamma * nonterminal * last
        adv[:, j] = last
    return adv