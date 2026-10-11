"""Multi-lap behaviour of CarRacingStateWrapper.

Laps are forced by flipping the base env's ``new_lap`` flag (what CarRacing's
contact listener sets when the car crosses the start tile after ~95% of tiles),
so these run in a second instead of needing a trained driver.
"""

from __future__ import annotations

import numpy as np

from carracing_ppo.env import make_env

COAST = np.array([0.0, 0.0, 0.0])
GAS = np.array([0.0, 0.5, 0.0])
BONUS = 100.0


def _env(**kwargs):
    env = make_env(render_mode=None, seed=7, offtrack_penalty=0.0, lap_completion_bonus=BONUS, **kwargs)
    env.reset(seed=7)
    env.step(COAST)
    return env


def _force_lap(env):
    base = env.unwrapped
    for tile in base.road:
        tile.road_visited = True
    base.tile_visited_count = len(base.track)
    base.new_lap = True


def test_lap_pays_bonus_once_and_keeps_driving():
    env = _env(allow_multiple_laps=True)
    _force_lap(env)

    _, reward, terminated, truncated, info = env.step(COAST)
    assert reward > BONUS - 1.0
    assert not terminated and not truncated
    assert info["laps_completed"] == 1

    base = env.unwrapped
    assert base.new_lap is False
    assert base.tile_visited_count == 0
    assert not any(tile.road_visited for tile in base.road)

    # Next step: no second bonus, episode still alive.
    _, reward, terminated, truncated, info = env.step(COAST)
    assert reward < 1.0
    assert not terminated and not truncated
    assert info["laps_completed"] == 1
    env.close()


def test_tiles_pay_again_after_lap():
    env = _env(allow_multiple_laps=True)
    _force_lap(env)
    env.step(COAST)

    tile_rewards = 0
    for _ in range(80):
        _, reward, terminated, truncated, _ = env.step(GAS)
        assert not terminated and not truncated
        tile_rewards += reward > 1.0
    assert tile_rewards > 0
    assert env.unwrapped.tile_visited_count > 0
    env.close()


def test_out_of_bounds_still_terminates():
    env = _env(allow_multiple_laps=True)
    env.unwrapped.car.hull.position = (1000.0, 1000.0)

    _, reward, terminated, _, info = env.step(COAST)
    assert terminated
    assert reward <= -99.0
    assert info["laps_completed"] == 0
    env.close()


def test_time_limit_wins_over_lap_bonus():
    env = _env(allow_multiple_laps=True, max_episode_steps=50)
    for _ in range(48):
        env.step(COAST)
    _force_lap(env)

    _, _, _, truncated, info = env.step(COAST)  # step 50
    assert truncated
    assert info["laps_completed"] == 1
    env.close()


def test_single_lap_mode_ends_episode_without_bonus():
    env = _env(allow_multiple_laps=False)
    _force_lap(env)

    _, reward, _, truncated, info = env.step(COAST)
    assert truncated
    assert reward < 1.0
    assert info["laps_completed"] == 1
    env.close()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
