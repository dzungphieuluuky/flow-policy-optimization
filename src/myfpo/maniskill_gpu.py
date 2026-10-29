"""ManiSkill 3 on the GPU: a batched torch env with the collector's contract.

Two entry points, on purpose:

* :func:`build_train_env` builds the GPU-parallel simulator the collector steps.
  It hands back torch tensors on the simulator's device, and nothing in the
  rollout path converts them to numpy.
* :func:`register_gym_envs` registers the same tasks under ``MS-*`` ids for the
  single-env CPU path (``scripts/play.py``, the registry test). ``gym.make``
  there is NOT the training path.

The collector requires one thing from autoreset: **a done step must report the
true ``o_{t+1}``**. Its chunks store that frame as ``next_state`` and as the
terminal ``step_frames``. ManiSkill hardcodes SAME_STEP, where the done step
already holds the *reset* frame and the real ``o_{t+1}`` is in
``info["final_observation"]``. The shim below swaps them back. Without it the
truncation bootstrap (``compute_chunk_advantages``) reads the first frame of a
fresh episode, and every terminal frame in the buffer is a foreign episode's.
"""

from __future__ import annotations

import types

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode

# Registry alias -> ManiSkill task id. The alias is the ``--env`` value and the
# key in ``configs/envs.yaml``; the dims live there, not here.
MANISKILL_TASKS = {
    "MS-PickCube-v1": "PickCube-v1",
}

DEFAULT_OBS_MODE = "state"
DEFAULT_CONTROL_MODE = "pd_joint_delta_pos"


def is_maniskill_id(env_id: str) -> bool:
    """True for a registry alias this module owns."""
    return env_id in MANISKILL_TASKS


class NextStepShim:
    """A SAME_STEP torch vector env, re-exposed as NEXT_STEP.

    ManiSkill both resets a finished env inside the done ``step()`` and reports
    the true next observation in ``info["final_observation"]``. The collector
    needs the second half of that: the done step is the only place where a
    chunk's ``next_state`` and terminal frames are read. So the shim swaps the
    reset frame for the final observation on the done step, and leaves the
    following step alone: the env has already advanced one step into the new
    episode, and the collector's awaiting branch takes whatever frame arrives
    next. Holding the reset frame back would skip that frame instead.
    """

    def __init__(self, vec):
        self.vec = vec
        self.num_envs = int(vec.num_envs)
        self.device = torch.device(vec.device)
        self.single_observation_space = vec.single_observation_space
        self.single_action_space = vec.single_action_space
        # What the collector validates against. The done step below is the part
        # of the NEXT_STEP contract the collector's bookkeeping depends on.
        self.autoreset_mode = AutoresetMode.NEXT_STEP

    def reset(self, *, seed=None, options=None):
        return self.vec.reset(seed=seed, options=options)

    def step_torch(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """One batched step. Every value is a torch tensor on ``self.device``."""
        obs, reward, terminated, truncated, info = self.vec.step(
            actions.to(device=self.device, dtype=torch.float32)
        )
        done = terminated | truncated
        if bool(done.any()):
            obs = torch.where(done.unsqueeze(-1), info["final_observation"], obs)
        return obs, reward, terminated, truncated, info

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


class ManiSkillTorchEnv(NextStepShim):
    """The GPU-parallel training env for one ManiSkill task.

    The spaces exposed as ``observation_space``/``action_space`` are the
    *single*-env spaces, not the batched ones: the runner sizes the policy and
    validates the registry dims from them (``scripts/train.py``), and the
    evaluator rejects anything that is not a flat Box.
    """

    def __init__(
        self,
        alias: str,
        num_envs: int,
        *,
        ignore_terminations: bool = False,
        obs_mode: str = DEFAULT_OBS_MODE,
        control_mode: str = DEFAULT_CONTROL_MODE,
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
        base = gym.make(
            self.task_id,
            num_envs=int(num_envs),
            obs_mode=obs_mode,
            control_mode=control_mode,
        )
        vec = ManiSkillVectorEnv(
            base, auto_reset=True, ignore_terminations=ignore_terminations
        )
        super().__init__(vec)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.metadata = {"render_modes": []}
        # `configure_run` names the run from spec.id, and the alias is the id
        # the user passed, not the ManiSkill task id.
        self.spec = types.SimpleNamespace(id=alias)

    def make_eval_env(self) -> ManiSkillTorchEnv:
        """A sibling env for evaluation.

        Terminations are ignored so every row stops only at the time limit.
        That keeps the batch in lockstep, which is what lets the evaluator hold
        one shared observation window for all rows.
        """
        return ManiSkillTorchEnv(
            self.alias,
            self.num_envs,
            ignore_terminations=True,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
        )

    def random_chunk(self, horizon: int) -> torch.Tensor:
        """A raw random chunk ``[num_envs, horizon, action_dim]``."""
        action_dim = int(self.action_space.shape[0])
        shape = (self.num_envs, int(horizon), action_dim)
        return torch.empty(shape, device=self.device).uniform_(-1.0, 1.0)


def build_train_env(alias: str, num_envs: int = 1, **kwargs) -> ManiSkillTorchEnv:
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
