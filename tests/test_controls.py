"""Action interfaces and decision rate (carracing/controls.py): training and evaluation see the same frames."""

from __future__ import annotations

import numpy as np
import pytest

from carracing.controls import DISCRETE_ACTIONS, MODES, ControlsWrapper, HeldControls, action_space, to_controls
from carracing.env import make_env
from carracing.rules import ACTION_SMOOTHING, smooth


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_keeps_controls_in_range(mode):
    space = action_space(mode)
    space.seed(0)
    for _ in range(200):
        raw = space.sample()
        if mode != "discrete":
            raw = raw * 3.0  # PPO's Gaussian goes far outside the box
        steer, gas, brake = to_controls(raw, mode)
        assert -1.0 <= steer <= 1.0 and 0.0 <= gas <= 1.0 and 0.0 <= brake <= 1.0


def test_throttle_never_presses_gas_and_brake_together():
    for t in np.linspace(-2, 2, 41):
        _, gas, brake = to_controls([0.0, t], "throttle")
        assert gas == 0.0 or brake == 0.0
    assert len(DISCRETE_ACTIONS) == 20


@pytest.mark.parametrize("mode,repeat", [("throttle", 4), ("discrete", 2), ("raw3", 3)])
def test_training_wrapper_and_held_policy_drive_the_same_frames(mode, repeat):
    space = action_space(mode)
    space.seed(2)
    decisions = [space.sample() for _ in range(60)]

    # Training: ControlsWrapper on top of the smoothing state wrapper.
    env = make_env(offtrack_penalty=0.0, allow_multiple_laps=False, lap_completion_bonus=0.0, lidar_rays=16,
                   lidar_fov=240, action_smoothing=ACTION_SMOOTHING, controls=mode, action_repeat=repeat,
                   track_seeds="4242")
    assert isinstance(env, ControlsWrapper)
    obs, _ = env.reset()
    train_obs = [obs]
    for a in decisions:
        obs, *_ = env.step(a)
        train_obs.append(obs)

    # Evaluation: a frame-by-frame policy that holds its choice and smooths, as benchmark.drive runs it.
    env2 = make_env(offtrack_penalty=0.0, allow_multiple_laps=False, lap_completion_bonus=0.0, lidar_rays=16,
                    lidar_fov=240)
    obs, _ = env2.reset(seed=4242)
    held, prev, it = HeldControls(mode, repeat), np.zeros(3, dtype=np.float32), iter(decisions)
    eval_obs = []
    for frame in range(len(decisions) * repeat):
        if frame % repeat == 0:
            eval_obs.append(obs)
        prev = smooth(held(lambda: next(it)), prev, ACTION_SMOOTHING)
        obs, *_ = env2.step(prev)
    eval_obs.append(obs)
    np.testing.assert_allclose(np.array(train_obs), np.array(eval_obs), atol=1e-6)


def test_fixed_training_track():
    env = make_env(track_seeds="4242", lidar_rays=16)
    a, _ = env.reset()
    b, _ = env.reset()
    np.testing.assert_array_equal(a, b)
