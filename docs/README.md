# Project Navigation — Fourier-Transform Research

> **One-page map for `fourier-transform-research`:** MuJoCo-only spectral joint
> flow matching (MFPO) + skrl PPO/SAC baselines, CPU-first, fp32-only.
> Suite: `pytest -q` 263 passed.

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
stacked obs [N=16,obs_dim] ──► ContextTrunk ──► Z [16,128]
Z ──► Hann(16) ──► rFFT ──► cat(real,imag) ──► K=5 frozen norm ──► Linear(2D→64) → T_FFT [5,64]
chunk actions [H=8,Da] ──► DCT ──► per-branch norm ──┐
chunk states [H,Ds] ──► DCT ──► per-branch norm ─────┤→ X*=[A*,S*] [H,135] ──► JointFlowTransformer(Z,T_FFT) ──► L_flow + VICReg SIGReg
```

**Binding contracts you cannot flip silently:** `z_t=(1-t)z0+tX*` with `t∈[0,1]`
0=noise→1=data, Euler integrates **ascending 0→1**, gripper dims clipped to
`{0,1}`, `λ_A=1.0 λ_S=0.5 λ_SIGReg=0.1`, `N=16 H=8 K=5 D=128 Dk=64`.

## One Mode, Three Methods

`scripts/train.py --method {mfpo,ppo,sac}` on MuJoCo locomotion
(HalfCheetah/Walker2d/Hopper), 45-run matrix (`test_config.py` locks it).

- `mfpo`: spectral joint flow path above (trainer ASPO + value + VICReg).
- `ppo` / `sac`: skrl 2.1.0 baselines via shared `baseline_train_loop`;
  eval is always 3-col CSV `[episode,return,length]`.

## Quick Commands (copy-paste)

```bash
# 1. Full test suite (263 passed)
PYTHONPATH=src .venv/bin/python -m pytest -q

# 2. MuJoCo single-run smoke (200 steps)
PYTHONPATH=src .venv/bin/python scripts/train.py --method mfpo --env Hopper-v5 --total-steps 200 --out-dir .omo/evidence/smoke

# 3. Matrix smoke (explicit filters, proves no regression)
PYTHONPATH=src .venv/bin/python scripts/train.py --matrix --methods mfpo ppo sac --envs Hopper-v5 --seeds 0 --total-steps 200 --out-dir .omo/evidence/smoke

# 4. Build notebooks
python scripts/build_notebook.py                          # builds both ipynbs (61 + 47 cells)
python scripts/build_notebook.py --spectral-only          # only spectral_joint_flow.ipynb
```
