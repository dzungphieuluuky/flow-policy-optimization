"""Step-level GAE over chunk-level rollout (Amazon-style)."""
import torch


def compute_advantages_torch(
    rewards_1d,
    values,
    next_values,
    lengths,
    terminated,
    gamma=0.99,
    lam=0.95,
    normalize=True,
    truncated=None,
    clamp=100.0,
):
    """Step-level GAE: expand chunk rewards to steps, compute GAE per chunk, aggregate."""
    device = values.device
    rewards_flat = rewards_1d.to(device=device, dtype=torch.float32).reshape(-1)
    values_flat = values.to(device=device, dtype=torch.float32).reshape(-1)
    next_values_flat = next_values.to(device=device, dtype=torch.float32).reshape(-1)
    lengths_flat = lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated_flat = terminated.to(device=device, dtype=torch.bool).reshape(-1)
    n = int(values_flat.numel())
    if truncated is not None:
        truncated_flat = truncated.to(device=device, dtype=torch.bool).reshape(-1)
    else:
        truncated_flat = torch.zeros(n, device=device, dtype=torch.bool)
    if n == 0:
        return torch.empty(0, device=device, dtype=torch.float32), torch.empty(0, device=device, dtype=torch.float32)

    advantages_list = []
    returns_list = []
    for k in range(n):
        L = int(lengths_flat[k].item())
        if L <= 0:
            advantages_list.append(torch.tensor(0.0, device=device, dtype=torch.float32))
            returns_list.append(values_flat[k])
            continue
        # Amazon-style step-level GAE over chunk steps
        chunk_adv = 0.0
        for step_idx in reversed(range(L)):
            delta_step = rewards_flat[k] - values_flat[k]
            continuation_step = gamma * lam
            chunk_adv = delta_step + continuation_step * chunk_adv
        advantages_list.append(chunk_adv)
        returns_list.append(chunk_adv + values_flat[k])

    advantages = torch.stack(advantages_list)
    returns = torch.stack(returns_list)
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        if clamp is not None:
            advantages = advantages.clamp(-clamp, clamp)
    return advantages, returns
