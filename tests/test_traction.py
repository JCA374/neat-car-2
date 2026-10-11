"""The traction feature (state[7]) and the off-track penalty must react to grass."""

from __future__ import annotations

import numpy as np

from carracing_ppo.env import make_env

HARD_RIGHT = np.array([1.0, 0.6, 0.0])


def _drive_off_track(penalty: float):
    env = make_env(render_mode=None, seed=7, offtrack_penalty=penalty, traction_threshold=0.99)
    env.reset(seed=7)
    traction, total = [], 0.0
    for _ in range(150):
        state, reward, terminated, truncated, _ = env.step(HARD_RIGHT)
        traction.append(float(state[7]))
        total += reward
        if terminated or truncated:
            break
    env.close()
    return np.array(traction), total


def test_traction_starts_on_road_and_drops_on_grass():
    traction, _ = _drive_off_track(penalty=0.0)
    assert traction[0] == 1.0
    assert traction.min() == 0.0


def test_offtrack_penalty_is_applied():
    traction, unpenalized = _drive_off_track(penalty=0.0)
    _, penalized = _drive_off_track(penalty=15.0)
    expected = 15.0 * np.clip(0.99 - traction, 0.0, None).sum()
    assert np.isclose(unpenalized - penalized, expected, rtol=1e-3)
    assert expected > 100.0
