"""Recording an eval rollout to video, without SAPIEN or ffmpeg.

``VectorEvaluator`` captures frames from ``env.render()``. These tests pin *when*
it captures — one frame per step, first round only, nothing at all when no video
was asked for — and the tiling that turns the batch's rows into one image. The
encoder call is stubbed: encoding a real mp4 is exercised by the CLI, not here.
"""

import numpy as np
import pytest
import torch

from myfpo.eval import VectorEvaluator

OBS_STEPS, HORIZON, ACT_DIM, OBS_DIM = 2, 2, 2, 2
EPISODE_LEN = 3
FRAME = 8


class FakeRenderVec:
    """A torch vector env that renders one small image per row."""

    def __init__(self, n: int = 2):
        gym = pytest.importorskip("gymnasium")
        self.num_envs = n
        self.device = torch.device("cpu")
        self.single_action_space = gym.spaces.Box(
            -1.0, 1.0, (ACT_DIM,), np.float32
        )
        self.steps = 0
        self.renders = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.steps = 0
        return torch.zeros(self.num_envs, OBS_DIM), {}

    def step_torch(self, _actions):
        self.steps += 1
        obs = torch.zeros(self.num_envs, OBS_DIM)
        reward = torch.ones(self.num_envs)
        done = self.steps >= EPISODE_LEN
        truncated = torch.full((self.num_envs,), done, dtype=torch.bool)
        info = {"success": torch.zeros(self.num_envs, dtype=torch.bool)}
        return obs, reward, torch.zeros_like(truncated), truncated, info

    def render(self):
        self.renders += 1
        # A distinct constant per row, so the tiling order is observable.
        return np.stack(
            [np.full((FRAME, FRAME, 3), row + 1, np.uint8) for row in range(self.num_envs)]
        )


def _evaluator(env, *, max_episodes=2):
    return VectorEvaluator(
        env,
        num_obs_steps=OBS_STEPS,
        num_action_steps=HORIZON,
        action_horizon=HORIZON,
        eval_seed=0,
        max_episodes=max_episodes,
    )


def _policy(*, horizon=HORIZON):
    return lambda window: torch.zeros(window.shape[0], horizon, ACT_DIM)


def test_one_frame_per_step_plus_the_initial_state(monkeypatch, tmp_path):
    """A 3-step episode is 4 frames: the reset frame, then one per step."""
    written = {}
    monkeypatch.setattr(
        VectorEvaluator,
        "_write_video",
        staticmethod(
            lambda images, video_path, fps: written.update(
                images=images, path=video_path, fps=fps
            )
        ),
    )
    env = FakeRenderVec(n=4)
    _evaluator(env, max_episodes=4).evaluate(
        _policy(), video_path=tmp_path / "step_10.mp4", video_fps=24
    )
    assert len(written["images"]) == EPISODE_LEN + 1
    assert written["path"] == tmp_path / "step_10.mp4"
    assert written["fps"] == 24
    # The four rows are tiled into one 2x2 image, not written as four videos.
    first = written["images"][0]
    assert first.shape == (FRAME * 2, FRAME * 2, 3)
    # Columns are filled top to bottom, so rows 0,1 form the left column.
    assert first[0, 0, 0] == 1 and first[-1, 0, 0] == 2
    assert first[0, -1, 0] == 3 and first[-1, -1, 0] == 4


def test_only_the_first_round_is_recorded(monkeypatch, tmp_path):
    """Four episodes over two rounds still write one video, not two."""
    calls = []
    monkeypatch.setattr(
        VectorEvaluator,
        "_write_video",
        staticmethod(lambda images, _video_path, _fps: calls.append(len(images))),
    )
    env = FakeRenderVec()
    results = _evaluator(env, max_episodes=4).evaluate(
        _policy(), video_path=tmp_path / "v.mp4"
    )
    assert len(results) == 4, "the episode budget must still be spent"
    assert calls == [EPISODE_LEN + 1]


def test_no_video_means_no_render_cost(monkeypatch):
    """Without ``video_path`` the env is never rendered."""
    monkeypatch.setattr(
        VectorEvaluator, "_write_video", staticmethod(lambda *_a, **_kw: None)
    )
    env = FakeRenderVec()
    _evaluator(env).evaluate(_policy())
    assert env.renders == 0
