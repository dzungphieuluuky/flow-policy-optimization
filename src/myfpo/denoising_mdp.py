"""DPPO: the denoising MDP (Ren et al., NeurIPS 2024, arXiv:2409.00588).

Not a config flag. DPPO treats the *sampling chain* as an MDP with two levels.

* Level 1 is the inner level. It starts at ``x_0 = eps`` and runs
  ``x_{k+1} = x_k + dt_k v(x_k, t_k, o) + noise``. The action is the increment.
  The observation is ``(x_k, t_k, o)``.
* Level 2 is the outer level. It is the environment. One macro-action is one
  complete denoise.

So the policy ratio is an ordinary Gaussian PPO ratio. It compares ``x_{k+1}``
    given ``(x_k, t_k, o)``. The log-probabilities are exact. There is no ELBO
    surrogate. The horizon is ``K`` times longer than the environment horizon.

    Two consequences make this *not* FPO:

  * The initial noise eps is part of the state. So the policy must see it.
  * Only stochastic sampling works. So training can never start from eps=0.

This module owns the inner MDP. The outer loop is shared with SJFPO. That
    outer loop covers the rollout, the chunk GAE over environment rewards, and
    the critic. Only the inner step is new.

``DenoisingPath`` is the field that the buffer must carry to make this
    trainable on-policy. The RolloutBuffer of SJFPO stores the *result* of a
    chunk. It does not store the path that produced the result. This is exactly
    the field that FPO says it needs, and that DPPO needs much more of.
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
    """Hold the full sampling chain for one chunk. The inner PPO backup uses it.

* ``states`` has the shape ``[n, K, *flow]``. It holds x_k, the point *before*
  step k.
* ``actions`` has the shape ``[n, K, *flow]``. It holds the increment that step
  k applied.
* ``mus`` has the shape ``[n, K, *flow]``. It holds the mean of the increment.
  So the log-prob needs no second forward pass.
* ``logstd`` has the shape ``[n]``. It holds one scalar per path. The scalar is
  shared over the whole chunk.
* ``final`` has the shape ``[n, *flow]``. It holds x_K, the action that the
  environment receives.

    ``logstd`` is per path and not per (state, action). A state-dependent
    log-std would need the old policy to run at every step. The ratio needs that
    value. DPPO Alg. 2 carries ``logstd`` as one scalar instead.

    This is a real modelling simplification. The paper states it. So it is not
    a shortcut.
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

    The argument ``predict_velocity(x, t, c) -> v`` must match the signature
    of the actor. The argument ``condition`` has the shape ``[B, hidden]``.

    The return is ``(x_final, path)``. The value ``path`` is ``None`` when
    ``store=False``. A rollout that runs under :func:`torch.inference_mode` and
    needs only the action can skip the memory.

    The chain runs ``x_{k+1} = x_k + dt_k v + sigma_k z``. The value ``z`` has
    the distribution ``N(0, I)``. DPPO Alg. 2 injects noise at every step.

    The value ``noise_std=0`` recovers the deterministic Euler sampler. That
    case exists only for the tests. A deterministic sampler has Gaussians with
    zero variance. So it has no meaningful ratio.

    Note the sign convention. The value ``mus`` stores the increment *before*
    the noise. So the stored action is ``mu + sigma z``. The log-prob is
    ``-||a - mu||^2 / (2 sigma^2) - log sigma - d/2 log 2pi``.

    Storing the increment, and not the point, is what makes the ratio
    computable. The caller does not run the old policy over the whole chain.
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
    """Return the PPO ratio for one denoising step. The log-probs are the exact
    Gaussian values.

    DPPO uses the same clipped surrogate as PPO. Only the *action* differs. It
    is a noise increment, not an environment action.

    This function stays separate from ``compute_aspo_surrogate``. That function
    consumes the FPO CFM-loss surrogate. It has no log-prob to clip.
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
    """Compute the GAE over the inner chain. This is DPPO Eq. 4.

    The argument ``step_rewards`` has the shape ``[n, K]``. The argument
    ``values`` has the shape ``[n, K+1]`` and holds V(s_k). The argument
    ``dones`` has the shape ``[n, K]`` and holds bool values. It is True when
    the chain ended at step k because the env terminated. The return holds
    ``[n, K]`` advantages.

    The argument ``values`` carries one column more than the chain. The step at
    ``j = K-1`` bootstraps from ``V(s_K)``. So column K holds that terminal
    value. It does not hold a value of a further step.

    The reward of the env lands on the *last* inner step. That is the only step
    where an environment transition happened. So a caller that passes chunk
    rewards should scatter them onto the final column. It should not scatter
    them onto the first column.

    That asymmetry is the accounting of DPPO itself. It is the easiest thing to
    get wrong when someone ports this code.
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