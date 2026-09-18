# Offline Package Bundle — What You Can Do

Pinned wheel bundle (`requirements-linux.txt`, 212 pins). Install offline with
`pip install --no-index --find-links` (or
`uv pip install -r requirements-linux.txt --system`).

## Robotics simulation

Run MuJoCo physics simulations (locomotion, manipulation) with
Gymnasium-style environments (`mujoco`, `gymnasium`, `gym`, `dm-env`,
`pettingzoo`), including headless rendering (`glfw`, `pyopengl`, `pygame`)
and video recording of rollouts.

## Deep learning

Train PyTorch models — including transformers and generative/flow models — on
CUDA GPUs (`torch`, `torchvision`, `torchaudio`, `torchcodec`, `triton`),
with mixed-precision support (pure-torch bf16 autocast), memory-efficient 8-bit optimizers
(`bitsandbytes`), and einops/einx-style tensor notation. Numerics via
`numpy`, `scipy`, `scikit-learn`; GPU stack via the CUDA 12/13 wheels
(`nvidia-cublas/cudnn/nccl/...`, `cuda-toolkit`).

## Robot learning research

Build behavior-cloning, diffusion, and online RL policies (PPO/SAC via
`skrl`) on LeRobot-format datasets (`lerobot`, `datasets`, `gymnasium-robotics`);
save/load checkpoints through the Hugging Face Hub (`huggingface-hub`) in
safetensors format.

## Reinforcement learning experimentation

Prototype custom Gymnasium environments, run vectorized rollouts, compute
advantages (GAE-style), and benchmark algorithms with reproducible pinned
dependencies.

## Data engineering

Load and process large datasets (Parquet/Arrow via `pyarrow`, `zarr`/`h5py`,
`jsonlines`), compose configs (`hydra-core`, `omegaconf`, `draccus`), and
parallelize preprocessing (`multiprocess`, `datasets` pipelines).

## Experiment tracking & testing

Log metrics, videos, and artifacts to Weights & Biases (`wandb`); gate
changes with `pytest`; generate Jupyter notebooks from scripts (`jupytext`,
`nbformat`); fetch shared files from Drive (`gdown`).

## Visualization & media

Plot training curves and analysis figures (`matplotlib`), process video
(`opencv-python`, `moviepy`, `imageio` + ffmpeg, `av`), and handle images
across formats (`pillow`, `scikit-image`, `tifffile`).

## Reproducible installs

Rebuild the exact environment offline anywhere — every transitive dependency
is version-pinned, so results carry over bit-for-bit between machines.
