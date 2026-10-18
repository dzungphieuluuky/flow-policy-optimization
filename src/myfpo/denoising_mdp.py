"""DPPO: the denoising MDP (Ren et al., NeurIPS 2024, arXiv:2409.00588)."""

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
    """Hold the full sampling chain for one chunk. The inner PPO backup uses it."""

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
    """Euler-integrate a stochastic denoising chain, optionally recording it."""
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
    """Return the PPO ratio for one denoising step. The log-probs are the exact"""
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
    """Compute the GAE over the inner chain. This is DPPO Eq. 4."""
    n, k = step_rewards.shape
    adv = torch.zeros_like(step_rewards)
    last = torch.zeros(n, device=step_rewards.device, dtype=step_rewards.dtype)
    for j in range(k - 1, -1, -1):
        nonterminal = 1.0 - dones[:, j].to(step_rewards.dtype)
        delta = step_rewards[:, j] + gamma * nonterminal * values[:, j + 1] - values[:, j]
        last = delta + gamma * nonterminal * last
        adv[:, j] = last
    return adv