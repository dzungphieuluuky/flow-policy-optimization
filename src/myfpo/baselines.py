"""Baseline configs: FPO, FPO++, and a ``dppo`` arm that is not DPPO.

FPO and FPO++ are faithful -- config-only, every divergence pinned and noted.
``dppo`` is a rollout-only arm: it flips the step count and runs the FPO++
surrogate. See ``_OVERRIDES``.

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
import warnings
from typing import Any

from myfpo.configuration_sjfpo import SJFPOConfig

__all__ = ["BASELINES", "apply_baseline", "baseline_report"]

#: Nothing clamps: ``torch.clamp(x, max=inf)`` is the identity, and
#: ``clamp(x, max=inf) >= 0`` passes both validators, so FPO's unclamped
#: objective is reachable without a new code path.
_NO_CLAMP = float("inf")

_NOTES: dict[str, dict[str, str]] = {
    "fpo": {
        "cfm_average_group_size": "-1: FPO Eq. 9. One group over every N_mc "
        "(tau, eps) pair, so they all clip together. Paired with "
        "do_average_cfm_loss_in_chunk=True so Eq. 8's chunk norm is averaged "
        "before the exponent.",
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
        "min_timestep": "0.0 and max_timestep 1.0: FPO samples tau ~ U(0,1) "
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
        "do_average_cfm_loss_in_chunk": "True: one ratio per (tau, eps) pair "
        "(FPO++ Eq. 10), tokens averaged valid-masked first. The released "
        "manipulation default is False, which gives one ratio per token; pinned "
        "here so this arm stays per-pair instead of drifting with the config "
        "default.",
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
        # No granularity note: the arm does not override either knob, so
        # baseline_report could never print one. apply_baseline's warning and
        # _assert_dppo_unwired say the same thing where it is acted on.
        "num_euler_steps": "10: matched to the papers' NFE. DPPO itself is "
        "stochastic-only by construction (Alg. 2 samples noise at every step), "
        "so training must start from sampled noise, never eps=0.",
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
        "cfm_average_group_size": -1,
        "do_average_cfm_loss_in_chunk": True,
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
        "do_average_cfm_loss_in_chunk": True,
        "trust_region_mode": "aspo",
        "cfm_loss_reduction": "sum",
        "num_mc_samples": 8,
        "gae_lambda": 0.95,
        "num_euler_steps": 4,
        "num_inference_steps": 4,
    },
    # ---- "dppo": Ren et al., arXiv:2409.00588 ---------------------------
    # NOT an implementation of DPPO. This arm only flips rollout settings and
    # runs the same FPO++ code path, so it is FPO++ at the papers' 10 NFE.
    # denoising_mdp.py holds the per-step sampler but nothing in the training
    # path reaches it; it is exercised only by tests. Implementing DPPO for
    # real means wiring that in, or dropping the name.
    "dppo": {
        "num_euler_steps": 10,
    },
}

BASELINES: tuple[str, ...] = ("fpo", "fpo++", "dppo")


#: Fields the ``dppo`` arm may flip without being wired to
#: :mod:`denoising_mdp`. All rollout-side: a trust-region or ratio override here
#: would relabel whichever code path the caller happened to configure.
_DPPO_ROLLOUT_ONLY = frozenset({"num_euler_steps", "num_inference_steps"})


def _assert_dppo_unwired(overrides: dict[str, Any]) -> None:
    """Keep the ``dppo`` label honest while it is a config-only arm.

    Two failure modes, one guard. An override outside :data:`_DPPO_ROLLOUT_ONLY`
    means someone edited the table below without wiring the sampler the name
    promises, so the arm would claim DPPO while running something else. Failing
    here beats a baseline that is quietly FPO++ at 10 NFE.
    """
    extra = sorted(set(overrides) - _DPPO_ROLLOUT_ONLY)
    if extra:
        raise ValueError(
            f"baseline 'dppo' may only override {sorted(_DPPO_ROLLOUT_ONLY)} "
            f"until denoising_mdp is wired into the training path; got {extra}. "
            "Either wire it in or drop the name: an override here does not "
            "change which surrogate runs."
        )


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
    if key == "dppo":
        _assert_dppo_unwired(overrides)
        warnings.warn(
            "baseline 'dppo' runs the FPO++ surrogate, not DPPO's "
            "per-denoising-step Gaussian ratio, and inherits the CFM "
            "grouping and trust_region_mode from the passed config. denoising_mdp.py "
            "holds the sampler but nothing in the training path calls it.",
            RuntimeWarning,
            stacklevel=2,
        )
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