# Project Navigation — spectral-joint-policy

> **One-page map for `spectral-joint-policy`:** MuJoCo-only spectral flow
> matching (SJFPO) + skrl PPO/SAC baselines, CUDA-default with CPU fallback, fp32-only.
> Suite: `pytest -q` 300 passed.

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
stacked obs [T=16,obs_dim] ──► StateConditioner (window tokens query DCT-II bins) → vector [D=128]
chunk actions [H=16,action_dim] (real, scaled) ──► DCT ──► DiT-8 velocity over Ka low-freq rows (condition `c = time + condition`) ──► sample t:0→1 ──► pad ──► IDCT
stacked obs ──► Critic V(s, owns state encoder) ──► GAE-returns value target
```

**Binding contracts you cannot flip silently:** `x_t=(1−t)·ε+t·a` with
`t=0` noise → `t=1` data, Euler integrates **ascending 0→1**, single
`clamp(-1,1)` after IDCT, `T=16 H=16 Ka=5 D=128` (`Ka=round(0.3*16)`; training executes full `H`; eval executes `ratio*H`, default 8).

## One Mode, Three Methods

`scripts/train.py --method {sjfpo,ppo,sac}` on MuJoCo locomotion
(HalfCheetah/Walker2d/Hopper), 45-run matrix (`test_config.py` locks it).

- `sjfpo`: spectral flow path above (`SJFPOPolicy` + `train_sjfpo.py` online loop).
- `ppo` / `sac`: skrl 2.1.0 baselines via shared `baseline_train_loop`;
  eval is always 3-col CSV `[episode,return,length]`.

## Quick Commands (copy-paste)

```bash
# 1. Full test suite (300 passed)
PYTHONPATH=src .venv/bin/python -m pytest -q

# 2. MuJoCo single-run smoke (2000 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method sjfpo --env Hopper-v5 --seed 0 --total_steps 2000 --out_dir .omo/evidence/smoke

# 3. Matrix smoke (sweep.sh axes, proves no regression)
METHODS="sjfpo ppo sac" ENVS="Hopper-v5" SEEDS="0" TOTAL_STEPS=2000 bash scripts/sweep.sh --out-base .omo/evidence/smoke --python .venv/bin/python

# 4. Build notebook
python scripts/build_notebook.py                          # standalone ipynb (8 cells, no source)
```
