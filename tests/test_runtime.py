"""Device selection and the precision policy in ``sjfpo_jax.runtime``.

Everything here is a pure function of a device or an env dict, so the logic is
checked on a CPU box even though the hardware it targets is not present. What is
*not* checked here is the CUDA branch: `should_donate()` and the TF32 pin can
only be verified on a real T4 or Blackwell box (JAX_MIGRATION.md §0c).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo_jax import runtime


@dataclass(frozen=True)
class FakeDevice:
    """The three attributes ``runtime`` reads off a real device."""

    platform: str
    id: int
    device_kind: str
    compute_capability: str | None = None


CUDA_DEVICES = [
    FakeDevice("cuda", 0, "NVIDIA GeForce RTX PRO 1000", "10.0"),
    FakeDevice("cuda", 1, "NVIDIA GeForce RTX PRO 1000", "10.0"),
]
T4_DEVICES = [FakeDevice("cuda", 0, "Tesla T4", "7.5")]
CPU_DEVICES = [FakeDevice("cpu", 0, "cpu", None)]


# ------------------------------------------------------------------ capability


@pytest.mark.parametrize(
    "capability, expected",
    [
        (None, False),        # CPU: no attribute at all
        ("7.5", False),       # T4: no TF32 tensor cores
        ("8.0", True),        # Ampere, the first with TF32
        ("9.0", True),        # Hopper
        ("10.0", True),       # Blackwell
        ("12.0", True),
    ],
)
def test_tf32_only_exists_from_ampere(capability, expected):
    assert runtime.has_tf32(capability) is expected


def test_matmul_precision_is_pinned_exactly_where_fp32_could_mean_tf32():
    """T4 and CPU leave it alone; anything with TF32 tensor cores gets 'highest'."""
    assert runtime.matmul_precision_for(None) is None
    assert runtime.matmul_precision_for("7.5") is None
    assert runtime.matmul_precision_for("8.0") == "highest"
    assert runtime.matmul_precision_for("10.0") == "highest"


def test_compute_capability_is_none_on_cpu():
    """The attribute is CUDA-only; reading it directly is an AttributeError here."""
    assert runtime.compute_capability(CPU_DEVICES[0]) is None
    assert runtime.compute_capability(CUDA_DEVICES[0]) == "10.0"


# ----------------------------------------------------------------- device index


def test_gpu_index_env_var_wins_over_local_rank():
    env = {runtime.GPU_INDEX_ENV: "3", runtime.LOCAL_RANK_ENV: "0"}
    assert runtime.device_index_from_env(env) == 3


def test_local_rank_is_the_fallback_so_torchrun_works_unchanged():
    assert runtime.device_index_from_env({runtime.LOCAL_RANK_ENV: "2"}) == 2


@pytest.mark.parametrize("env", [{}, {runtime.GPU_INDEX_ENV: ""}, {runtime.LOCAL_RANK_ENV: ""}])
def test_no_launcher_env_means_a_single_device_run(env):
    assert runtime.device_index_from_env(env) == 0


def test_a_negative_index_is_a_launcher_mistake_not_a_wraparound():
    with pytest.raises(ValueError):
        runtime.device_index_from_env({runtime.GPU_INDEX_ENV: "-1"})


# --------------------------------------------------------------------- selection


def test_each_process_gets_its_own_device():
    """The bug this prevents: four processes, all four on GPU 0."""
    assert runtime.select_device(CUDA_DEVICES, 0) is CUDA_DEVICES[0]
    assert runtime.select_device(CUDA_DEVICES, 1) is CUDA_DEVICES[1]


def test_an_index_past_the_last_device_fails_loudly():
    """Silently clamping to the last device is how a 4-GPU job becomes a 1-GPU job."""
    with pytest.raises(ValueError, match="out of range"):
        runtime.select_device(CUDA_DEVICES, 2)
    with pytest.raises(ValueError, match="out of range"):
        runtime.select_device(T4_DEVICES, 1)


def test_no_visible_device_is_a_runtime_error():
    with pytest.raises(RuntimeError):
        runtime.select_device([], 0)


def test_configure_device_pins_tf32_devices_and_spares_the_others():
    """The whole precision policy, exercised on fakes, then the global restored."""
    original = jax.config.jax_default_matmul_precision
    try:
        runtime.configure_device(T4_DEVICES, 0)
        assert jax.config.jax_default_matmul_precision == original
        runtime.configure_device(CPU_DEVICES, 0)
        assert jax.config.jax_default_matmul_precision == original
        assert runtime.configure_device(CUDA_DEVICES, 1) is CUDA_DEVICES[1]
        assert jax.config.jax_default_matmul_precision == "highest"
    finally:
        jax.config.update("jax_default_matmul_precision", original)


# ------------------------------------------------------------------ this machine


def test_selects_a_real_device_and_runs_a_compiled_program_on_it():
    device = runtime.configure_device()
    assert device.platform == runtime.detect_backend()
    out = jax.jit(lambda x: x * 2)(jax.numpy.arange(4.0))
    assert out.device == device
    assert runtime.local_device_count() >= 1


@pytest.mark.parametrize("backend, expected", [("cuda", True), ("cpu", False)])
def test_donation_follows_the_backend(backend, expected):
    """GPU only. The cuda answer is the policy under test; whether it pays there
    is milestone 5, and needs a GPU box."""
    assert runtime.should_donate(backend) is expected
    assert runtime.should_donate() == (runtime.detect_backend() == "cuda")


def test_describe_reports_what_a_run_actually_used():
    line = runtime.describe(runtime.configure_device())
    for field in ("x64", "matmul_precision", "donate", "devices"):
        assert field in line
    assert "x64=True" in line, "sjfpo_jax/__init__ sets x64 on for the RunningMeanStd count"
