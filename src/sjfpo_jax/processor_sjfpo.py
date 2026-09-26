"""SJFPO pre/post processors (JAX): obs normalize, action unscale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
class SJFPOObservationProcessor:
    """Normalize ``observation.state`` with running stats."""

    stats: dict[str, Any]

    def normalize(self, states: jnp.ndarray) -> jnp.ndarray:
        mean = self.stats["mean"]
        var = self.stats["var"]
        eps = 1e-8
        norm = (states - mean) * jax.lax.rsqrt(var + eps)
        norm = jnp.clip(norm, -10.0, 10.0).astype(states.dtype)
        return jnp.where(self.stats["uninitialized"], states, norm)

    def __call__(self, batch):
        out = dict(batch)
        out[OBS_STATE] = self.normalize(batch[OBS_STATE])
        return out


@dataclass
class SJFPOActionPostprocessor:
    """Unscale policy action chunks from [-1, 1] to real env bounds."""

    action_low: jnp.ndarray
    action_high: jnp.ndarray

    def __post_init__(self):
        self._scale = 0.5 * (self.action_high - self.action_low)
        self._bias = 0.5 * (self.action_high + self.action_low)

    def unnormalize(self, scaled: jnp.ndarray) -> jnp.ndarray:
        return self._bias + self._scale * scaled

    def __call__(self, batch):
        out = dict(batch)
        out[ACTION] = self.unnormalize(batch[ACTION])
        return out


