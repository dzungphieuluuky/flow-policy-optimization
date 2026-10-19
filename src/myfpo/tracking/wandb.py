from __future__ import annotations

import os

import wandb

from .base import build_run_name


class WandbBackend:
    def __init__(
        self,
        *,
        method: str,
        env_name: str,
        seed: int,
        total_steps: int,
        cfg_dict: dict,
    ):
        self.run_name = build_run_name(method, env_name, seed, cfg_dict)
        wandb.init(
            project=method,
            group=f"{method}_{env_name}",
            name=self.run_name,
            tags=[method, "mujoco"],
            config={**cfg_dict, "method": method, "seed": seed, "env": env_name},
            job_type="smoke" if total_steps <= 10000 else "train",
            mode=os.environ.get("WANDB_MODE", "online"),
            reinit="finish_previous",
        )

    def log(self, data: dict[str, float], step: int) -> None:
        wandb.log(data, step=step)

    def finish(self, exit_code: int) -> None:
        wandb.finish(exit_code=exit_code)
