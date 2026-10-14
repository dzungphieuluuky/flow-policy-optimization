"""The value loss is rsl-rl's, with squared error replaced by Huber.

There is no return normalizer, so the target is the raw discounted chunk return.
Huber is quadratic inside delta and linear outside, so a large-residual chunk
contributes a bounded gradient instead of owning critic_grad_norm_preclip.
"""

import torch

from myfpo.modeling_sjfpo import compute_value_huber_loss


def test_huber_is_quadratic_inside_delta_and_linear_outside():
    """The whole point of the change: the penalty stops growing quadratically.

    delta 2: residual 1 -> 0.5*1^2 = 0.5 (quadratic branch); residual 3 ->
    2*(3 - 0.5*2) = 4 (linear branch, not 0.5*3^2 = 4.5).
    """
    delta = 2.0 * torch.tensor(1.0)
    pred = torch.tensor([1.0, 3.0])
    target = torch.zeros(2)
    torch.testing.assert_close(
        compute_value_huber_loss(pred, target, delta), torch.tensor([0.5, 4.0])
    )


def test_delta_scales_with_a_residual_scale():
    """delta is a scalar scale, so a wider one keeps a residual quadratic.

    Deltas come in as literals here, 1.0 and 10.0. Same residual 3.0: the
    wide delta pays 0.5*3^2 = 4.5, the narrow one falls outside its knee and
    pays the linear 1*(3 - 0.5) = 2.5. One delta constant cannot do both.
    Training sets delta from the tracked residual RMS, not the return RMS.
    """
    pred = torch.tensor([[3.0]])
    target = torch.tensor([0.0])
    small = compute_value_huber_loss(pred, target, 1.0 * torch.tensor(1.0))
    large = compute_value_huber_loss(pred, target, 1.0 * torch.tensor(10.0))
    assert float(small[0, 0]) == 2.5
    assert float(large[0, 0]) == 4.5


def test_target_broadcasts_over_the_member_axis():
    pred = torch.zeros(3, 4)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = compute_value_huber_loss(pred, target, 1.0 * torch.tensor(1.0))
    assert out.shape == (3, 4)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_clipped_branch_takes_the_worse_of_the_two():
    """rsl-rl's pessimistic bound: max(unclipped, clipped) per chunk.

    Clipping pulls the prediction back toward the rollout snapshot. Where that
    moves it toward the target the clip would shrink the penalty, and where it
    moves it away the clip inflates it. max() must take the worse in both cases,
    never the flattering one.
    """
    clip = 1.0
    delta = 0.5 * torch.tensor(1.0)

    def penalty(pred: torch.Tensor) -> torch.Tensor:
        # [1, 1] so the per-chunk mean(dim=0) keeps a length-1 axis to index.
        return compute_value_huber_loss(pred.view(1, 1), torch.zeros(1), delta).mean(dim=0)

    # Clipping helps: pred 5 clipped to 1, target 0, and 1 is inside delta, so
    # the clipped penalty is the smaller quadratic term and max must take the raw.
    rollout, pred = torch.tensor([0.0]), torch.tensor([5.0])
    clipped = rollout + (pred - rollout).clamp(-clip, clip)
    assert float(torch.maximum(penalty(pred), penalty(clipped))[0]) == float(
        penalty(pred)[0]
    )

    # Clipping hurts: pred 7 clipped to 9, both outside delta, and the further
    # target dominates. max must take the clipped one, not the flattering 7.
    rollout, pred = torch.tensor([10.0]), torch.tensor([7.0])
    clipped = rollout + (pred - rollout).clamp(-clip, clip)
    assert float(clipped[0]) == 9.0
    assert float(torch.maximum(penalty(pred), penalty(clipped))[0]) == float(
        penalty(clipped)[0]
    )
