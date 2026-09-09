"""Central configuration registry for the MFPO vs PPO/SAC benchmark.

Holds every hyperparameter named in plan Scope §Must-have plus the run matrix.
All dataclasses are frozen with immutable (tuple) defaults; RUN_MATRIX is a
tuple of frozen RunSpec instances built by an explicit loop over
methods x envs x seeds.

Module-level mode switches are read from environment variables ONCE at import
time (IS_KAGGLE is the single Kaggle-detection point for the whole project;
tests monkeypatch the env var and re-import).

Tensor conventions (used in shape comments across the codebase):
    B: batch size
    N: context length (default 16)
    H: action-chunk horizon (default 8)
    K: low-K spectral tokens (default 5)
    D: fused latent dimension Z_dim (default 128)
    Dk: spectral projection dimension (default 64)
    Da: action dimension (env-dependent)
    Ds: state dimension (default D = 128)

All tensors are fp32. Run-matrix construction is MuJoCo-only (flat Box).

Retained public API (dead in the src train path, kept per fn-consolidation D1):
``assert_runtime`` (tests/test_config.py dependency-pin + precision-flag tests).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Mode switches (plain module-level constants; read env at import time ONLY)
# ---------------------------------------------------------------------------

IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
TINY_MATRIX = os.environ.get("MFPO_TINY_MATRIX") == "1"
RESUME_ONLY = os.environ.get("MFPO_RESUME_ONLY") is not None
APPROVE_OVERRUN = os.environ.get("MFPO_APPROVE_OVERRUN") == "1"
_legacy_mujoco_env = os.environ.get("MFPO_LEGACY_MUJOCO")
LEGACY_MUJOCO = _legacy_mujoco_env is None or _legacy_mujoco_env == "1"

# ---------------------------------------------------------------------------
# Budget constants (consumed by the task-13 calibration/budget-gate cell)
# ---------------------------------------------------------------------------

BUDGET_TOTAL_HOURS = 120.0
BUDGET_PER_RUN_HOURS = 10.0

# ---------------------------------------------------------------------------
# Run-matrix axes
# ---------------------------------------------------------------------------

METHODS = ("mfpo", "ppo", "sac")
ENVS = ("HalfCheetah-v5", "Walker2d-v5", "Hopper-v5")
SEEDS = (0, 1, 2, 3, 4)

TOTAL_STEPS = 1_000_000
EVAL_INTERVAL = 50_000
CKPT_INTERVAL = 25_000


# ---------------------------------------------------------------------------
# Method hyperparameters (Scope §Must-have, verbatim values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MFPOConfig:
    """FPO++-based flow-matching policy optimization (our contribution).

    Attributes:
        N_mc: Number of (tau, eps) pairs per chunk. Default 16.
        euler_train: Euler steps during training rollouts. Default 64.
        euler_eval: Euler steps at eval (zero-sampling). Default 16.
        chunks_per_update: Chunks per update. Default 512.
        epochs: Training epochs per update. Default 16.
        minibatch: Chunks per minibatch. Default 128.
        lr: AdamW learning rate. Default 1e-4.
        wd: AdamW weight decay. Default 1e-4.
        gamma: Discount factor. Default 0.99.
        gae_lambda: GAE lambda. Default 0.95.
        epsilon_clip: ASPO trust-region width. Default 0.05.
        cfm_clamp: Symmetric clamp on CFM losses. Default 20.0.
        log_ratio_cap: STE cap on log-ratio magnitude. Default 10.0.
        tau_range: Flow time range (t_min, t_max). Default (0.005, 0.995).
        lambda_aux: Aux future-state CFM weight. Default 0.5.
        lambda_sigreg: SigReg (L_var + L_cov) weight. Default 0.1.
        c_value: Value-loss coefficient. Default 1.0.
        ema_decay: EMA target-projection decay. Default 0.99.
        grad_clip: Grad-norm clip. Default 1.0.
        context_len: Frame-stack length N. Default 16.
        chunk_len: Action-chunk horizon H. Default 8.
        K_spectral: Low-K rFFT spectral tokens. Default 5.
        Z_dim: Fused latent dimension D. Default 128.
    """

    N_mc: int = 16  # (tau, eps) pairs per chunk
    euler_train: int = 64  # Euler steps during training rollouts
    euler_eval: int = 16  # Euler steps at eval (zero-sampling)
    chunks_per_update: int = 512
    epochs: int = 16
    minibatch: int = 128  # chunks per minibatch
    lr: float = 1e-4  # AdamW learning rate
    wd: float = 1e-4  # AdamW weight decay
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.05  # ASPO trust-region width
    cfm_clamp: float = 20.0  # symmetric clamp on CFM losses
    log_ratio_cap: float = 10.0  # STE cap on log-ratio magnitude
    tau_range: tuple[float, float] = (0.005, 0.995)
    lambda_aux: float = 0.5  # aux future-state CFM weight
    lambda_sigreg: float = 0.1  # SigReg (L_var + L_cov) weight
    c_value: float = 1.0  # value-loss coefficient
    ema_decay: float = 0.99  # EMA target-projection decay
    grad_clip: float = 1.0  # grad-norm clip
    context_len: int = 16  # frame-stack length N
    chunk_len: int = 8  # action-chunk horizon H
    K_spectral: int = 5  # low-K rFFT spectral tokens K
    Z_dim: int = 128  # fused latent dim D


@dataclass(frozen=True)
class PPOConfig:
    """skrl PPO baseline (skrl 2.1.0 PPO_CFG keys).

    Attributes:
        rollouts: Rollout length. Default 2048.
        learning_epochs: Learning epochs. Default 10.
        mini_batches: Mini-batches. Default 8.
        lr: Learning rate. Default 3e-4.
        kl_threshold: KLAdaptiveLR threshold. Default 0.01.
        grad_norm_clip: Gradient norm clip. Default 1.0.
        discount: Discount factor. Default 0.99.
        gae_lambda: GAE lambda. Default 0.95.
        ratio_clip: PPO clip ratio. Default 0.2.
        value_loss_scale: Value loss scale. Default 1.0.
        hidden: Hidden layer sizes. Default (256, 256, 256).
    """

    rollouts: int = 2048
    learning_epochs: int = 10
    mini_batches: int = 8
    lr: float = 3e-4
    kl_threshold: float = 0.01  # KLAdaptiveLR
    grad_norm_clip: float = 1.0
    discount: float = 0.99
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_loss_scale: float = 1.0
    hidden: tuple[int, ...] = (256, 256, 256)


@dataclass(frozen=True)
class SACConfig:
    """skrl SAC baseline (skrl 2.1.0 SAC_CFG keys).

    Attributes:
        lr: Learning rate. Default 3e-4.
        batch_size: Batch size. Default 256.
        memory_size: Replay buffer size. Default 300_000.
        gradient_steps: Gradient steps per update. Default 1.
        polyak: Polyak averaging factor. Default 0.005.
        discount: Discount factor. Default 0.99.
        initial_random_steps: Initial random steps. Default 5000.
        hidden: Hidden layer sizes. Default (256, 256, 256).
    """

    lr: float = 3e-4
    batch_size: int = 256
    memory_size: int = 300_000
    gradient_steps: int = 1
    polyak: float = 0.005
    discount: float = 0.99
    initial_random_steps: int = 5000
    hidden: tuple[int, ...] = (256, 256, 256)


MFPO_CFG = MFPOConfig()
PPO_CFG = PPOConfig()
SAC_CFG = SACConfig()


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """Single run specification.

    Attributes:
        method: Method name ("mfpo", "ppo" or "sac").
        env: Environment name.
        seed: Random seed.
        total_steps: Total training steps.
        eval_interval: Evaluation interval.
        ckpt_interval: Checkpoint interval.
    """

    method: str
    env: str
    seed: int
    total_steps: int
    eval_interval: int
    ckpt_interval: int


def build_legacy_run_matrix() -> tuple[RunSpec, ...]:
    """Build legacy MuJoCo run matrix (45 runs: 3 methods × 3 envs × 5 seeds).

    Returns:
        Tuple of RunSpec for legacy MuJoCo experiments.
    """
    specs: list[RunSpec] = []
    for method in METHODS:
        for env in ENVS:
            for seed in SEEDS:
                specs.append(
                    RunSpec(
                        method=method,
                        env=env,
                        seed=seed,
                        total_steps=TOTAL_STEPS,
                        eval_interval=EVAL_INTERVAL,
                        ckpt_interval=CKPT_INTERVAL,
                    )
                )
    return tuple(specs)


RUN_MATRIX = build_legacy_run_matrix()  # 45 runs


def build_filtered_matrix(
    methods: Iterable[str],
    envs: Iterable[str],
    seeds: Iterable[int],
    total_steps: int,
    eval_interval: int,
    ckpt_interval: int,
) -> tuple[RunSpec, ...]:
    """Cartesian matrix in RUN_MATRIX order (method x env x seed) for CLI filters."""
    return tuple(
        RunSpec(
            method=m,
            env=e,
            seed=s,
            total_steps=total_steps,
            eval_interval=eval_interval,
            ckpt_interval=ckpt_interval,
        )
        for m in [x for x in METHODS if x in methods]
        for e in [x for x in ENVS if x in envs]
        for s in seeds
    )


if TINY_MATRIX:
    tiny: list[RunSpec] = []
    for method in METHODS:
        tiny.append(
            RunSpec(
                method=method,
                env="Hopper-v5",
                seed=0,
                total_steps=20_000,
                eval_interval=10_000,
                ckpt_interval=10_000,
            )
        )
    RUN_MATRIX = tuple(tiny)  # 3 runs


# ---------------------------------------------------------------------------
# Validation helpers (plan task 2) — raise, never print
# ---------------------------------------------------------------------------


def validate_run_matrix(specs: Iterable[RunSpec]) -> None:
    """Raise AssertionError on any duplicate (method, env, seed) tuple.

    Args:
        specs: Iterable of RunSpec to validate.

    Raises:
        AssertionError: If duplicate (method, env, seed) found.
    """
    seen: set[tuple[str, str, int]] = set()
    for spec in specs:
        key = (spec.method, spec.env, spec.seed)
        assert key not in seen, f"duplicate (method, env, seed) in run matrix: {key}"
        seen.add(key)


def parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse version string to tuple of ints for comparison.

    Args:
        version: Version string like "1.2.3".

    Returns:
        Tuple of ints (major, minor, patch, ...).
    """
    parts: list[int] = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# retained: test-only public API — exercised directly by tests/test_config.py
# (dependency-pin + precision-flag tests); not called by the src train path.
# Keep per fn-consolidation D1.
def assert_runtime(allow_fp16: bool = False) -> None:
    """Fail fast on dependency pins and unsupported precision flags.

    Raises RuntimeError when the installed gymnasium has major.minor < 1.0,
    when skrl is not exactly 2.1.0, or when half/bfloat precision training is
    requested. Versions are resolved through importlib.metadata.version(),
    mirroring scripts/verify_deps.sh.

    Args:
        allow_fp16: If True, raises error (fp16/bf16 unsupported).

    Raises:
        RuntimeError: If dependency versions don't match or fp16 requested.
    """
    import importlib.metadata

    gym_version = importlib.metadata.version("gymnasium")
    if parse_version_tuple(gym_version)[:2] < (1, 0):
        raise RuntimeError(
            f"gymnasium {gym_version} has major.minor < 1.0; "
            "this project requires gymnasium[mujoco]>=1.0,<1.3"
        )

    skrl_version = importlib.metadata.version("skrl")
    if skrl_version != "2.1.0":
        raise RuntimeError(f"skrl {skrl_version} != 2.1.0; exact pin required")

    if allow_fp16:
        raise RuntimeError(
            "fp16/bf16 training requested but unsupported: flow-matching "
            "Euler rollouts and CFM losses require fp32"
        )


validate_run_matrix(RUN_MATRIX)
