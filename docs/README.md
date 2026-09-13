# Project Navigation — spectral-joint-policy

> **One-page map for `spectral-joint-policy`:** MuJoCo-only spectral flow
> matching (SJFPO) + skrl PPO/SAC baselines, CUDA-default with CPU fallback, fp32-only.
> Suite: `pytest -q` 273 passed.

## Start Here (30 seconds)

| Want to… | Open → |
|---|---|
| Run something now | [Getting Started](getting-started.md) |
| Understand the tree | [Project Structure](project-structure.md) |
| See how data flows | [Architecture](architecture.md) |
| Jump to a file | [Module Guide](modules.md) |
| Rebuild / run notebooks | [Notebooks](notebooks.md) |
| Flip a flag / change a matrix | [Config & Flags](config-flags.md) |
| Run or add tests | [Testing](testing.md) |
| Fix a weird error | [Troubleshooting](troubleshooting.md) |

## The 10-Second Mental Model

```
stacked obs [T=16,obs_dim] ──► DCT-II(time) ──► K lowest bins (top_p=0.3 → 5) → tokens [5,obs] (raw, no projector)
chunk actions [H=16,Da] ──► DCT ──► DiT-8 velocity (condition = time + pooled tokens) ──► sample t:1→0 ──► IDCT
stacked obs ──► Critic V(s,a, owns state encoder) ──► GAE-returns value target
```

**Binding contracts you cannot flip silently:** `x_t=t·ε+(1−t)·a` with
`t=1` noise → `t=0` data, Euler integrates **descending 1→0**, single
`clamp(-1,1)` after IDCT, `T=16 H=16 K=5 D=128` (`K=round(0.3*16)`; training executes full `H`; eval executes `ratio*H`, default 8).

## One Mode, Three Methods

`scripts/train.py --method {sjfpo,ppo,sac}` on MuJoCo locomotion
(HalfCheetah/Walker2d/Hopper), 45-run matrix (`test_config.py` locks it).

- `sjfpo`: spectral flow path above (`SJFPOPolicy` + `train_sjfpo.py` online loop).
- `ppo` / `sac`: skrl 2.1.0 baselines via shared `baseline_train_loop`;
  eval is always 3-col CSV `[episode,return,length]`.

## Quick Commands (copy-paste)

```bash
# 1. Full test suite (273 passed)
PYTHONPATH=src .venv/bin/python -m pytest -q

# 2. MuJoCo single-run smoke (200 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --total-steps 200 --out-dir .omo/evidence/smoke

# 3. Matrix smoke (explicit filters, proves no regression)
PYTHONPATH=src .venv/bin/python scripts/train.py --matrix --methods sjfpo ppo sac --envs Hopper-v5 --seeds 0 --total-steps 200 --out-dir .omo/evidence/smoke

# 4. Build notebooks
python scripts/build_notebook.py                          # builds both minimal ipynbs (6 cells each, no source)
python scripts/build_notebook.py --spectral-only          # only spectral_joint_flow.ipynb
```
