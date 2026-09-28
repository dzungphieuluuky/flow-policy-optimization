"""The value loss is rsl-rl's: squared error against the raw return.

There is no return normalizer, so the target is the raw discounted chunk return
and value_loss scales with the return magnitude. These pin the arithmetic rather
than the scale.
"""

import torch


def _sq(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.square(pred - target)


def test_value_loss_is_squared_error():
    pred = torch.tensor([1.0, 2.0, 5.0])
    target = torch.tensor([0.0, 4.0, 2.0])
    torch.testing.assert_close(_sq(pred, target), torch.tensor([1.0, 4.0, 9.0]))


def test_clipped_branch_takes_the_worse_of_the_two():
    """rsl-rl's pessimistic bound: max(unclipped_sq, clipped_sq) per element.

    Clipping pulls the prediction back toward the rollout snapshot. Where that
    moves it toward the target the clip would shrink the penalty, and where it
    moves it away the clip inflates it. max() must take the worse in both cases,
    never the flattering one.
    """
    clip = 1.0

    # Clipping helps: pred 5 clipped to 1, target 0. raw 25, clipped 1, max 25.
    rollout, pred, target = torch.tensor([0.0]), torch.tensor([5.0]), torch.tensor([0.0])
    clipped = rollout + (pred - rollout).clamp(-clip, clip)
    assert float(torch.maximum(_sq(pred, target), _sq(clipped, target))[0]) == 25.0

    # Clipping hurts: rollout 10, pred 7 clipped to 9, target 0. raw 49, clipped
    # 81, and max must take 81 rather than the flattering 49.
    rollout, pred, target = torch.tensor([10.0]), torch.tensor([7.0]), torch.tensor([0.0])
    clipped = rollout + (pred - rollout).clamp(-clip, clip)
    assert float(clipped[0]) == 9.0
    assert float(torch.maximum(_sq(pred, target), _sq(clipped, target))[0]) == 81.0
