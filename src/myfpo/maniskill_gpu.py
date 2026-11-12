"""ManiSkill 3 on the GPU: a batched torch env with the collector's contract.

Two entry points, on purpose:

* :func:`make_maniskill_env` builds the GPU-parallel simulator the collector steps.
  It hands back torch tensors on the simulator's device, and nothing in the
  rollout path converts them to numpy.
* :func:`register_gym_envs` registers the same tasks under ``MS-*`` ids for the
  single-env CPU path (``scripts/play.py``, the registry test). ``gym.make``
  there is NOT the training path.

The collector speaks the native SAME_STEP contract. ManiSkill resets a
finished env inside the done ``step()``: the done step returns the *reset*
frame of the new episode and reports the true ``o_T`` in
``info["final_observation"]``. ``close_chunks`` stores that final frame as
the chunk's ``next_state`` and adopts the returned frame as the new
episode's first frame instead of stepping a forced zero action.
"""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode

# Registry alias -> ManiSkill task id. The alias is the ``--env`` value and the
# key in ``configs/envs.yaml``; the dims live there, not here.
MANISKILL_TASKS = {
    "MS-PickCube-v1": "PickCube-v1",
}

# Per-task env options. The training env runs the task the way the eval env
# does: the episode ends only at the time limit. PickCube's reward is a dense
# hold reward, so cutting the episode at success would forfeit the remaining
# hold reward, and the eval env already runs this way. A task with a true
# terminal, or with a success reward that only fires once, must set
# ``ignore_terminations`` to False: with it on, the critic never sees a
# terminal, and every chunk bootstraps from V(s_T).
MANISKILL_TASK_OPTIONS = {
    "MS-PickCube-v1": {"ignore_terminations": True},
}

DEFAULT_OBS_MODE = "state"
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"


def is_maniskill_id(env_id: str) -> bool:
    """True for a registry alias this module owns."""
    return env_id in MANISKILL_TASKS




class ManiSkillTorchEnv:
    """The GPU-parallel training env for one ManiSkill task.

    The spaces exposed as ``observation_space``/``action_space`` are the
    *single*-env spaces, not the batched ones: the runner sizes the policy and
    validates the registry dims from them (``scripts/train.py``), and the
    evaluator rejects anything that is not a flat Box.

    ManiSkill is SAME_STEP: the done ``step()`` resets the finished envs
    inside the call, and the returned observation is the first frame of the
    new episode, with the true terminal frame in
    ``info["final_observation"]``. The collector consumes this contract
    natively — see ``close_chunks`` in ``myfpo.rollout``.
    """

    def __init__(
        self,
        alias: str,
        num_envs: int,
        *,
        # None reads the task's own entry in MANISKILL_TASK_OPTIONS. A task
        # with a true terminal keeps the ManiSkill default: the episode ends
        # at the terminal, and the critic learns that the terminal is worth 0.
        ignore_terminations: bool | None = None,
        obs_mode: str = DEFAULT_OBS_MODE,
        control_mode: str = DEFAULT_CONTROL_MODE,
        render_mode: str | None = None,
    ):
        if alias not in MANISKILL_TASKS:
            raise KeyError(f"unknown ManiSkill alias {alias!r}; add it to MANISKILL_TASKS")
        if num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be >= 1")
        import mani_skill.envs  # noqa: F401 - registers the task ids
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        self.alias = alias
        self.task_id = MANISKILL_TASKS[alias]
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        if ignore_terminations is None:
            options = MANISKILL_TASK_OPTIONS.get(alias, {})
            ignore_terminations = bool(options.get("ignore_terminations", False))
        base = gym.make(
            self.task_id,
            num_envs=int(num_envs),
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
        )
        vec = ManiSkillVectorEnv(
            base, auto_reset=True, ignore_terminations=ignore_terminations
        )
        self.vec = vec
        self.num_envs = int(vec.num_envs)
        self.device = torch.device(vec.device)
        self.single_observation_space = vec.single_observation_space
        self.single_action_space = vec.single_action_space
        # What the collector validates against. ManiSkill resets a finished
        # env inside the done step(), so it is declared SAME_STEP and the
        # collector handles the reset frame natively.
        self.autoreset_mode = AutoresetMode.SAME_STEP
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.metadata = {"render_modes": []}
        # `configure_run` names the run from spec.id, and the alias is the id
        # the user passed, not the ManiSkill task id.
        self.spec = types.SimpleNamespace(id=alias)

    def reset(self, *, seed=None, options=None):
        """Reset every row through the wrapped env, with the seed and options passed on."""
        return self.vec.reset(seed=seed, options=options)

    def step_torch(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """One batched step. Every value is a torch tensor on ``self.device``.

        Raw SAME_STEP frames: done rows hold the first frame of the new
        episode, and ``info["final_observation"]`` carries the true terminal
        frame. The collector splits them itself.
        """
        return self.vec.step(
            actions.to(device=self.device, dtype=torch.float32)
        )

    def step(self, actions):
        """Numpy in/out, for callers that are not the collector."""
        obs, reward, terminated, truncated, info = self.step_torch(
            torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        )
        return (
            obs.cpu().numpy(),
            reward.cpu().numpy(),
            terminated.cpu().numpy(),
            truncated.cpu().numpy(),
            info,
        )

    def close(self) -> None:
        """Close the wrapped simulator."""
        self.vec.close()

    def make_eval_env(
        self, *, num_envs: int | None = None, render_mode: str | None = None
    ) -> ManiSkillTorchEnv:
        """A sibling env for evaluation.

        Terminations are ignored so every row stops only at the time limit.
        That keeps the batch in lockstep, which is what lets the evaluator hold
        one shared observation window for all rows.

        ``render_mode="rgb_array"`` costs GPU rendering on every ``render()``,
        and it renders *all* rows at once, so a recording env is built with a
        handful of rows rather than the training width.
        """
        return ManiSkillTorchEnv(
            self.alias,
            self.num_envs if num_envs is None else int(num_envs),
            ignore_terminations=True,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode=render_mode,
        )

    def render(self) -> np.ndarray:
        """RGB frames for every row, ``(num_envs, H, W, 3)`` uint8.

        Capture happens on the GPU and comes back through the human render
        camera, so it works with ``obs_mode="state"``: no sensor is needed.
        """
        if self.render_mode != "rgb_array":
            raise RuntimeError(
                "render() needs render_mode='rgb_array'; pass it to "
                "ManiSkillTorchEnv or make_eval_env"
            )
        frames = self.vec.render()
        if torch.is_tensor(frames):
            frames = frames.detach().cpu().numpy()
        return np.asarray(frames)

    def random_chunk(self, horizon: int) -> torch.Tensor:
        """A raw random chunk ``[num_envs, horizon, action_dim]``."""
        action_dim = int(self.action_space.shape[0])
        shape = (self.num_envs, int(horizon), action_dim)
        return torch.empty(shape, device=self.device).uniform_(-1.0, 1.0)


def make_maniskill_env(alias: str, num_envs: int = 1, **kwargs) -> ManiSkillTorchEnv:
    """The env the training loop collects from."""
    return ManiSkillTorchEnv(alias, num_envs, **kwargs)


def _make_cpu_env(alias: str):
    """One single-env CPU sim behind ``CPUGymWrapper``, for play and tests."""
    import mani_skill.envs  # noqa: F401 - registers the task ids
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    return CPUGymWrapper(
        gym.make(
            MANISKILL_TASKS[alias],
            num_envs=1,
            obs_mode=DEFAULT_OBS_MODE,
            control_mode=DEFAULT_CONTROL_MODE,
        )
    )


def register_gym_envs() -> None:
    """Register the ``MS-*`` aliases for the single-env CPU path.

    Registering is idempotent: a second call is a no-op, so importing this
    module from two entry points cannot fail.
    """
    for alias in MANISKILL_TASKS:
        try:
            gym.register(id=alias, entry_point=lambda a=alias: _make_cpu_env(a))
        except gym.error.Error:
            continue
