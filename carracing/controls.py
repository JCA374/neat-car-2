"""How a model's outputs become CarRacing's three controls (steer, gas, brake).

This is a model's own choice, like its network; the car, the 50 fps physics and the
rulebook's action smoothing (applied to the final controls, every frame) are the same for all.

    raw3      steer, gas, brake as three outputs (CarRacing's own action)
    throttle  steer + one throttle axis in [-1, 1]: gas = max(t, 0), brake = max(-t, 0),
              so gas and brake are never pressed together
    discrete  one of a fixed set of steer x pedal combinations

``repeat`` k: the model decides every k frames and holds its choice in between.
Training uses ``ControlsWrapper`` (fewer, longer steps); the lap check and benchmark use
``HeldControls`` inside a frame-by-frame policy, so both see exactly the same frames.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

MODES = ("raw3", "throttle", "discrete")

STEERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PEDALS = ((1.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 0.8))  # (gas, brake): full, half, coast, brake
DISCRETE_ACTIONS = np.array([(s, g, b) for s, (g, b) in itertools.product(STEERS, PEDALS)], dtype=np.float32)


def action_space(mode: str) -> gym.Space:
    if mode == "raw3":
        return gym.spaces.Box(np.array([-1, 0, 0], dtype=np.float32), np.array([1, 1, 1], dtype=np.float32))
    if mode == "throttle":
        return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    if mode == "discrete":
        return gym.spaces.Discrete(len(DISCRETE_ACTIONS))
    raise ValueError(f"controls must be one of {MODES}, got {mode!r}")


def to_controls(action, mode: str) -> np.ndarray:
    """A model output -> (steer, gas, brake), every control within CarRacing's range."""
    if mode == "discrete":
        return DISCRETE_ACTIONS[int(np.asarray(action).reshape(-1)[0])].copy()
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if mode == "throttle":
        t = float(np.clip(a[1], -1.0, 1.0))
        return np.array([np.clip(a[0], -1.0, 1.0), max(t, 0.0), max(-t, 0.0)], dtype=np.float32)
    if mode == "raw3":
        return np.array([np.clip(a[0], -1.0, 1.0), np.clip(a[1], 0.0, 1.0), np.clip(a[2], 0.0, 1.0)],
                        dtype=np.float32)
    raise ValueError(f"controls must be one of {MODES}, got {mode!r}")


class ControlsWrapper(gym.Wrapper):
    """Training: map the model's action to controls and drive it for ``repeat`` frames.

    Sits on top of CarRacingStateWrapper (which applies the per-frame smoothing). Rewards of the
    held frames are summed; ``info["frames"]`` counts the frames this step took.
    """

    def __init__(self, env: gym.Env, mode: str = "raw3", repeat: int = 1) -> None:
        super().__init__(env)
        self.mode = mode
        self.repeat = max(1, int(repeat))
        self.action_space = action_space(mode)

    def step(self, action):
        controls = to_controls(action, self.mode)
        total = 0.0
        for frame in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(controls)
            total += float(reward)
            if terminated or truncated:
                break
        info["frames"] = frame + 1
        return obs, total, terminated, truncated, info


class HeldControls:
    """Evaluation: called every frame; asks the model every ``repeat`` frames and holds its controls."""

    def __init__(self, mode: str = "raw3", repeat: int = 1) -> None:
        self.mode = mode
        self.repeat = max(1, int(repeat))
        self.reset()

    def reset(self) -> None:
        self.frame = 0
        self.held: Optional[np.ndarray] = None

    def __call__(self, decide: Callable[[], np.ndarray]) -> np.ndarray:
        if self.frame % self.repeat == 0:
            self.held = to_controls(decide(), self.mode)
        self.frame += 1
        return self.held.copy()
