# Added linked fafm step-level GAE (calculate_advantage)"""Chunk-level semi-MDP GAE + returns (torch)."""
import torch

def compute_advantages_torch(
    rewards_1d, values, next_values, lengths, terminated,
    gamma=0.99, lam=0.95, normalize=True, truncated=None, clamp=100.0
):
    device = values.device
    rewards_1d_f = rewards_1d.to(device=device, dtype=torch.float32).reshape(-1)
    values_f = values.to(device=device, dtype=torch.float32).reshape(-1)
    next_values_f = next_values.to(device=device, dtype=torch.float32).reshape(-1)
    lengths_f = lengths.to(device=device, dtype=torch.long).reshape(-1)
    terminated_f = terminated.to(device=device, dtype=torch.bool).reshape(-1)
    n = int(values_f.numel())
    if truncated is not None:
        truncated_f = truncated.to(device=device, dtype=torch.bool).reshape(-1)
    else:
        truncated_f = torch.zeros(n, device=device, dtype=torch.bool)
    if not (next_values_f.numel() == n and lengths_f.numel() == n and terminated_f.numel() == n and truncated_f.numel() == n):
        raise ValueError("shape mismatch")
    if n == 0 or bool((lengths_f <= 0).any()):
        raise ValueError("non-empty stream, chunks >= 1")
    if int(lengths_f.sum().item()) != int(rewards_1d_f.numel()):
        raise ValueError("rewards length != sum(lengths)")
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths_f, 0)])
    r_chunk = torch.empty(n, device=device, dtype=torch.float32)
    for k in range(n):
        steps = rewards_1d_f[offsets[k]:offsets[k+1]]
        powers = torch.pow(torch.tensor(gamma, device=device, dtype=torch.float32), torch.arange(steps.numel(), device=device, dtype=torch.float32))
        r_chunk[k] = torch.dot(powers, steps)
    advantages = torch.empty(n, device=device, dtype=torch.float32)
    next_advantage = torch.tensor(0.0, device=device, dtype=torch.float32)
    for k in range(n - 1, -1, -1):
        length = int(lengths_f[k].item())
        if bool(terminated_f[k].item()):
            delta = r_chunk[k] - values_f[k]
            continuation = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            delta = r_chunk[k] + (gamma**length) * next_values_f[k] - values_f[k]
            continuation = torch.tensor(0.0, device=device, dtype=torch.float32) if bool(truncated_f[k].item()) else torch.tensor((gamma * lam)**length, device=device, dtype=torch.float32)
        advantages[k] = delta + continuation * next_advantage
        next_advantage = advantages[k]
    returns = advantages + values_f
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        if clamp is not None:
            advantages = advantages.clamp(-clamp, clamp)
    return advantages, returns
