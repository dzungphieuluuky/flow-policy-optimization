"""Device detection and the per-device optimisation policy.

JAX optimises almost everything here for free: XLA fuses, ``lax.scan`` lowers to
a loop, and the whole update step compiles once. Two things are *not* free --
which device a process uses, and what "fp32" means on it. Those are the
decisions in this module, and each one is a pure function of a device so the
logic is testable on a CPU-only box.

Multi-GPU is data-parallel by env (JAX_MIGRATION.md §0d): one process per GPU,
each owning ``num_envs / N`` envs. So the job here is to give each process
exactly one device, and to fail loudly when it cannot.

Nothing else in the package imports this module; the trainer composes it.
"""

from __future__ import annotations

import os

import jax

# TF32 tensor cores arrived with Ampere, compute capability 8.0. Below that --
# a T4 is 7.5 -- an fp32 GEMM is fp32 whatever XLA is asked for.
TF32_MIN_CAPABILITY = (8, 0)

# Launcher-facing names. SJFPO_GPU_INDEX is ours; LOCAL_RANK is what torchrun
# already sets, so a torchrun launch works with no extra configuration.
GPU_INDEX_ENV = "SJFPO_GPU_INDEX"
LOCAL_RANK_ENV = "LOCAL_RANK"


def detect_backend() -> str:
    """``'cuda'`` or ``'cpu'``."""
    return jax.default_backend()


def local_device_count() -> int:
    """Visible devices on this host, which is how many processes to launch."""
    return jax.local_device_count()


def compute_capability(device) -> str | None:
    """``'10.0'`` on a CUDA device, ``None`` on CPU.

    The attribute is CUDA-only, hence the ``getattr``: reading it directly is an
    AttributeError on the development box.
    """
    return getattr(device, "compute_capability", None)


def has_tf32(capability: str | None) -> bool:
    """Could an fp32 GEMM on this device silently run in TF32?"""
    if capability is None:
        return False
    return tuple(int(part) for part in capability.split(".")) >= TF32_MIN_CAPABILITY


def matmul_precision_for(capability: str | None) -> str | None:
    """``'highest'`` where fp32 could mean TF32, ``None`` where it cannot.

    XLA's f32 dot default on Ampere and later is not consistent across versions,
    so pin it rather than infer it. The port is fp32 throughout
    (JAX_MIGRATION.md §7), and a silent TF32 makes the Blackwell parity run
    meaningless -- the cost of pinning is a throughput number we measure, the
    cost of not pinning is numbers nobody can reproduce.
    """
    return "highest" if has_tf32(capability) else None


def device_index_from_env(env=None) -> int:
    """Which GPU this process owns.

    ``SJFPO_GPU_INDEX`` wins, then ``LOCAL_RANK`` so a torchrun launcher works
    unchanged, then device 0 for a single-GPU run. ``ValueError`` on a negative
    index: that is a launcher mistake, and it would otherwise select from the end
    of the list without saying so.
    """
    env = os.environ if env is None else env
    for name in (GPU_INDEX_ENV, LOCAL_RANK_ENV):
        raw = env.get(name)
        if raw not in (None, ""):
            index = int(raw)
            if index < 0:
                raise ValueError(f"{name}={raw!r} must be >= 0")
            return index
    return 0


def select_device(devices=None, index: int | None = None) -> jax.Device:
    """This process's device.

    Without this, four processes on a four-GPU box each see all four devices and
    all four land on GPU 0, which reads as "multi-GPU did nothing" rather than as
    a bug.
    """
    devices = list(jax.devices()) if devices is None else list(devices)
    index = device_index_from_env() if index is None else index
    if not devices:
        raise RuntimeError("no accelerator is visible: jax.devices() is empty")
    if index >= len(devices):
        raise ValueError(
            f"device index {index} is out of range for {len(devices)} visible "
            f"device(s); set {GPU_INDEX_ENV} to 0..{len(devices) - 1}, or unset it "
            "for a single-device run"
        )
    return devices[index]


def configure_device(devices=None, index: int | None = None) -> jax.Device:
    """Pin this process to its device and apply the precision policy."""
    device = select_device(devices, index)
    precision = matmul_precision_for(compute_capability(device))
    if precision is not None:
        jax.config.update("jax_default_matmul_precision", precision)
    return device


def should_donate(backend: str | None = None) -> bool:
    """Donate buffers into the optimizer step, or not?

    GPU only. Donation is a wall-clock win and nothing else -- the donated buffer
    is invalid after the call, so an async Orbax save must not hold a reference
    to it (§8), and on CPU there is no allocator pressure to relieve.
    """
    return (detect_backend() if backend is None else backend) == "cuda"


def describe(device) -> str:
    """One startup line, the same job as the torch branch's policy summary."""
    capability = compute_capability(device)
    parts = [f"{device.platform}:{device.id}", device.device_kind]
    if capability:
        parts.append(f"sm_{capability.replace('.', '')}")
    parts += [
        f"devices={local_device_count()}",
        f"x64={jax.config.jax_enable_x64}",
        f"matmul_precision={jax.config.jax_default_matmul_precision}",
        f"donate={should_donate()}",
    ]
    return " | ".join(parts)
