"""MC noise is standard-normal, bit for bit."""
import torch

from myfpo.fpo_core import sample_noise


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
