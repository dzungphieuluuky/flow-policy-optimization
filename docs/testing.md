# Testing

## Suites (273 green, no skips)

| Suite | What it locks | Command |
|---|---|---|
| Full | Everything | `PYTHONPATH=src .venv/bin/python -m pytest -q` |
| Pipeline (7) | CSV schema, matrix axes, wandb gate — **never touch** | `… -m pytest tests/test_pipeline_cpu.py -v` |
| Config | 45-run matrix asserts | `… -m pytest tests/test_config.py -q` |
| Targeted | e.g. spectral, policy, baselines, sweep, wandb | `… -m pytest tests/test_networks.py tests/test_modeling_sjfpo.py -q` |

Full suite tail is `299 passed`.

## Numeric Gates (TDD, not vacuous)

- DCT roundtrip `‖x−IDCT(DCT(x))‖ < 1e-5`.
- Spectral: DCT→low-K→Linear order, invariance `B=1` vs
  `B=32` `atol 1e-6`.
- WindowEncoder mask: garbage tails leave real rows bitwise unchanged
  (`test_window_encoder_mask_hides_tails`).
- Policy frequency slots: identical bin values give distinct rows
  (`test_policy_bin_order_sensitive`); ablated encoding collapses them.
- Flow: trainer 20-step `>1%` (anti-`stays finite` guard).
- Evaluator: CSV header exactly `episode,return,length`; 12-episode default
  (`max_episodes=12`); byte-identical determinism under fixed seed.
- Sweep/analysis: deterministic scan order, diverged excluded, bootstrap
  `n_boot=2000` 95% CI, PNGs >10KB, summary CSV exact header.

## Smokes (hands-on QA)

```bash
# MuJoCo single-run smoke (200 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --total-steps 200 --out-dir .omo/evidence/smoke
# Matrix smoke (explicit filters, 3 methods x Hopper x seed 0)
PYTHONPATH=src .venv/bin/python scripts/train.py --matrix --methods sjfpo ppo sac --envs Hopper-v5 --seeds 0 --total-steps 200 --out-dir .omo/evidence/smoke
```

## Adding a Test

1. Add `tests/test_my_feature.py` (run everything with `PYTHONPATH=src`).
2. Write the failing assertion first and run it to see RED.
3. Implement `src/sjfpo/*.py`, rerun to GREEN, then full `pytest -q` before
   committing. Never touch `test_pipeline_cpu.py`, `test_config.py`
   45-asserts, or `analysis.py` statistics without a plan amendment.
