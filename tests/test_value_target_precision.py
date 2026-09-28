"""The critic's regression target must survive the rollout buffer in fp32.

resolve_storage_dtype returns bf16 on RTX PRO 1000 Blackwell, so anything cast
to it loses mantissa bits. raw_returns and raw_values are the only stored
tensors whose magnitude is large rather than O(1), and they are the critic's
regression target and clip snapshot, so they must not be cast.
"""

import dataclasses

import torch
from lerobot.configs.policies import PolicyFeature
from lerobot.configs.types import FeatureType

from sjfpo.configuration_sjfpo import SJFPOConfig
from sjfpo.modeling_sjfpo import SJFPOPolicy
from sjfpo.utils import resolve_storage_dtype


def _policy(**kw) -> SJFPOPolicy:
    cfg = dataclasses.replace(
        SJFPOConfig(use_dct_transform=True, num_kept_tokens=4, **kw),
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    return SJFPOPolicy(cfg)


def test_bf16_actually_destroys_a_large_target():
    """Why the dtype matters, independent of which device we are on."""
    x = torch.tensor([1234.5, 87.25, 41.0])
    assert not torch.equal(x.to(torch.bfloat16).to(torch.float32), x)
    # Relative error is ~2^-8, so it is invisible on O(1) and fatal on returns.
    small = torch.tensor([1.0, 0.5])
    assert torch.allclose(small.to(torch.bfloat16).to(torch.float32), small, atol=0.01)


def test_source_does_not_cast_the_value_target_to_storage_dtype():
    """Guard the fix itself.

    A read-side assert is weak, because both the cast and the test would have to
    be wrong together. This pins the source: the value path must not mention
    storage_dtype at all.
    """
    import inspect

    src = inspect.getsource(SJFPOPolicy.forward)
    value_block = src[src.index("raw_value_pred = self.critic") :]
    value_block = value_block[: value_block.index("if reduction ==")]
    assert "storage_dtype" not in value_block, (
        "the critic's value path must stay fp32; it regresses on raw returns"
    )
    assert "torch.float32" in value_block


def test_resolve_storage_dtype_is_fp32_on_cpu():
    assert resolve_storage_dtype(torch.device("cpu")) is torch.float32


def test_generate_rollout_does_not_cast_the_value_target():
    """Write-side guard: the two fp32 tensors must not be cast on the way in.

    The read-side guard below only covers forward. This one pins the other half,
    because a cast on write is invisible to any test that reads the batch back
    and re-derives the dtype itself.
    """
    import inspect

    src = inspect.getsource(SJFPOPolicy.generate_rollout)
    for name in ("raw_returns", "raw_values"):
        assert f"{name} = {name}.to(storage_dtype)" not in src, (
            f"{name} is the critic's regression target in raw reward units and "
            "must not be quantised by the storage dtype"
        )
    # The tensors that legitimately ride storage_dtype must still do so, or this
    # test would pass on a file where the cast was removed wholesale.
    assert "raw_advantages = raw_advantages.to(storage_dtype)" in src
    assert "normalized_advantages = normalized_advantages.to(storage_dtype)" in src
