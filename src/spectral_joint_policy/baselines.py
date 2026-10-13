"""Faithful baselines: FPO, FPO++, DPPO.

Each entry is a ``dataclasses.replace`` of the live ``SJFPOConfig``, so a
baseline is a config, not a branch. What is *not* overridden stays whatever the
caller set, which is why :func:`apply_baseline` returns the config and the
caller should print the diff -- a baseline that silently inherits the proposed
method's settings is not a baseline.

Provenance for every non-default value is in ``_NOTES``, keyed by baseline, and
the note is written into ``run_metadata.json`` by ``scripts/train.py``.

Sources
-------
FPO    McAllister et al., "Flow Matching Policy Gradients", ICLR 2026,
       arXiv:2507.21053. Eq. 5 (PPO-clip surrogate), Eq. 6 (ratio), Eq. 7-9
       (per-chunk MC mean), Alg. 1.
FPO++  Yi et al., "Flow Policy Gradients for Robot Control",
       arXiv:2602.02481. Eq. 10 (per-action ratio).
DPPO   Ren et al., "Diffusion Policy Policy Optimization", NeurIPS 2024,
       arXiv:2409.00588. Two-level MDP: initial noise is the observation,
       each denoising step is a Gaussian action.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from spectral_joint_policy.configuration_sjfpo import SJFPOConfig

__all__ = ["BASELINES", "apply_baseline", "baseline_report"]

#: Nothing clamps: ``torch.clamp(x, max=inf)`` is the identity, and
#: ``clamp(x, max=inf) >= 0`` passes both validators, so FPO's unclamped
#: objective is reachable without a new code path.
_NO_CLAMP = float("inf")

_NOTES: dict[str, dict[str, str]] = {
    "fpo": {
        "ratio_granularity": "per_chunk: FPO Eq. 9. Mean the MC losses, THEN "
        "exponentiate, so all N_mc (tau, eps) pairs clip together.",
        "trust_region_mode": "ppo: FPO Eq. 5 is min(r*A, clip(r,1+/-e)*A) and "
        "nothing else. The SPO quadratic branch is the proposed method, not FPO.",
        "cfm_loss_reduction": "sum: FPO Eq. 8 is the squared norm over the whole "
        "chunk. SJFPO defaults to 'mean', whose length-invariant trust region is "
        "the proposed change, not FPO.",
        "cfm_clamp": "inf: FPO Alg. 1 has no loss clamp. cfm_clamp exists in "
        "SJFPO to stop a runaway the FPO objective does not have.",
        "log_ratio_cap": "inf: FPO has no log-ratio cap and no straight-through "
        "estimator. The STE in compute_policy_ratio is a SJFPO stabilisation.",
        "cfm_loss_clamp_negative_advantages": "False: same reason.",
        "min_timestep/max_timestep": "[0, 1]: FPO samples tau ~ U(0,1) "
        "(Eq. 1-2, no schedule). SJFPO defaults to [0.005, 0.995].",
        "timestep_sampling_mode": "uniform: FPO Eq. 7 draws tau uniformly. "
        "The 'fpo_beta' default only coincides with uniform at beta=1.",
        "action_perturb_std": "0.0: FPO perturbs the CFM inputs, not the "
        "executed action. SJFPO's default 0.02 is a exploration aid, not FPO.",
        "num_mc_samples": "8: FPO's default N_mc (Alg. 1).",
        "num_euler_steps": "4: matches the live default, pinned so the arms are "
        "matched on NFE by construction rather than by coincidence. The papers "
        "used 10, so this is a deliberate departure from their inference budget, "
        "not a reproduction of it.",
        "num_inference_steps": "4: same rationale, on the eval path.",
    },
    "fpo++": {
        "ratio_granularity": "per_action == FPO++ Eq. 10 (one ratio per (tau, eps) "
        "pair). Pinned explicitly so the baseline cannot drift when the SJFPO "
        "default changes.",
        "trust_region_mode": "aspo: ASPO (PPO-clip on A>=0, SPO quadratic on A<0) "
        "is FPO++'s trust region. Pinned explicitly, same reason.",
        "cfm_loss_reduction": "sum: FPO++ inherits FPO's Eq. 8 chunk norm. Pinned "
        "explicitly; SJFPO's 'mean' default is a proposed change, not the paper.",
        "num_mc_samples": "8: FPO++ keeps FPO's N_mc. Pinned explicitly.",
        "gae_lambda": "0.95: the chunk-level GAE lambda FPO++ reports. This "
        "resolved the open question left by the 0.0 default -- with the default "
        "at 0.95 the baseline inherits the paper's value, but it is pinned so a "
        "future default change cannot silently move it again.",
        "num_euler_steps": "4: matches the live default, pinned so the arms are "
        "matched on NFE by construction rather than by coincidence. The papers "
        "used 10, so this is a deliberate departure from their inference budget, "
        "not a reproduction of it.",
        "num_inference_steps": "4: same rationale, on the eval path.",
    },
    "dppo": {
        "ratio_granularity": "unused: DPPO replaces the FPO surrogate entirely "
        "with a per-denoising-step Gaussian PPO ratio.",
        "num_euler_steps": "10: DPPO is stochastic-only by construction "
        "(Alg. 2 samples noise at every step), so training must start from "
        "sampled noise, never eps=0.",
    },
}

_OVERRIDES: dict[str, dict[str, Any]] = {
    # ---- FPO: McAllister et al., arXiv:2507.21053 -----------------------
    # Note what is NOT overridden. FPO's eps-MSE needs no code change: on the
    # linear-Gaussian path x_tau = tau*a + (1-tau)*eps the velocity->epsilon
    # conversion is eps_hat = a - v_hat, so
    #     ||eps_hat - eps||^2 == ||v_hat - (a - eps)||^2
    # algebraically. Verified to fp32 rounding (max relative diff 3e-5). The
    # paper's eps-vs-u ablation is about which scale clip_param should be
    # calibrated against, not about the loss value.
    "fpo": {
        "ratio_granularity": "per_chunk",
        "trust_region_mode": "ppo",
        "cfm_loss_reduction": "sum",
        "cfm_clamp": _NO_CLAMP,
        "log_ratio_cap": _NO_CLAMP,
        "cfm_loss_clamp_negative_advantages": False,
        "min_timestep": 0.0,
        "max_timestep": 1.0,
        "timestep_sampling_mode": "uniform",
        "action_perturb_std": 0.0,
        "num_mc_samples": 8,
        "num_euler_steps": 4,
        "num_inference_steps": 4,
    },
    # ---- FPO++: Yi et al., arXiv:2602.02481 ------------------------------
    # Pinned explicitly rather than left empty. These four fields are what makes
    # the run FPO++; leaving them inherited made the baseline a moving target,
    # because any default change silently redefined it.
    "fpo++": {
        "ratio_granularity": "per_action",
        "trust_region_mode": "aspo",
        "cfm_loss_reduction": "sum",
        "num_mc_samples": 8,
        "gae_lambda": 0.95,
        "num_euler_steps": 4,
        "num_inference_steps": 4,
    },
    # ---- DPPO: Ren et al., arXiv:2409.00588 ------------------------------
    # Needs denoising_mdp.py; these are the rollout-side settings only.
    "dppo": {
        "num_euler_steps": 10,
    },
}

BASELINES: tuple[str, ...] = ("fpo", "fpo++", "dppo")


def apply_baseline(cfg: SJFPOConfig, name: str) -> tuple[SJFPOConfig, dict[str, Any]]:
    """Return ``cfg`` with ``name``'s overrides applied, plus provenance.

    Raises ``ValueError`` for an unknown baseline. Re-validates, so a baseline
    that would produce an inconsistent config fails here rather than three
    rollouts into a run.
    """
    key = name.strip().lower()
    if key not in _OVERRIDES:
        raise ValueError(
            f"unknown baseline {name!r}; choices: {', '.join(BASELINES)}"
        )
    overrides = _OVERRIDES[key]
    out = dataclasses.replace(cfg, **overrides) if overrides else cfg
    out.validate()
    provenance = {
        "baseline": key,
        "citation": _CITATIONS[key],
        "overrides": dict(overrides),
        "notes": dict(_NOTES[key]),
    }
    return out, provenance


def baseline_report(cfg: SJFPOConfig, name: str) -> str:
    """Human-readable diff of a baseline against the passed-in config."""
    key = name.strip().lower()
    if key not in _OVERRIDES:
        raise ValueError(f"unknown baseline {name!r}; choices: {', '.join(BASELINES)}")
    lines = [f"baseline={key}  ({_CITATIONS[key]})"]
    for field, value in _OVERRIDES[key].items():
        before = getattr(cfg, field)
        same = before == value
        lines.append(
            f"  {field}: {before!r} -> {value!r}" + ("  (unchanged)" if same else "")
        )
        note = _NOTES[key].get(field)
        if note:
            lines.append(f"      {note}")
    if not _OVERRIDES[key]:
        lines.append("  (no overrides: this baseline IS the default config)")
        for note in _NOTES[key].values():
            lines.append(f"      {note}")
    return "\n".join(lines)


_CITATIONS = {
    "fpo": "McAllister et al., Flow Matching Policy Gradients, ICLR 2026 (arXiv:2507.21053)",
    "fpo++": "Yi et al., Flow Policy Gradients for Robot Control, arXiv:2602.02481",
    "dppo": "Ren et al., Diffusion Policy Policy Optimization, NeurIPS 2024 (arXiv:2409.00588)",
}