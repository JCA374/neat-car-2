"""Step files, the ranking rule, training-time bookkeeping and the lap-time bonus."""

from __future__ import annotations

import numpy as np
import pytest

from carracing.env import make_env
from carracing.steps import all_steps, load_step, neat_train_seconds, ppo_train_seconds, rank_key, train_flags


def test_every_step_file_loads_and_ids_are_unique():
    steps = all_steps()
    assert {"n00_baseline", "p00_baseline"} <= {s.id for s in steps}
    assert len({s.id for s in steps}) == len(steps)
    for s in steps:
        assert s.id[0] == s.model[0]  # n.. for NEAT, p.. for PPO
        if s.parent:
            assert s.parent in {t.id for t in steps}


def test_step_id_must_match_file_name(tmp_path):
    path = tmp_path / "n05_x.toml"
    path.write_text('id = "n05_y"\nmodel = "neat"\ntitle = "t"\n')
    with pytest.raises(ValueError):
        load_step(str(path))


def test_train_flags():
    assert train_flags({"generations": 5, "domain-randomize": True, "allow-multiple-laps": False,
                        "policy-hidden-sizes": [64, 64]}) == [
        "--generations", "5", "--domain-randomize", "--no-allow-multiple-laps", "--policy-hidden-sizes", "64,64"]


def _result(finished, lap, seconds, params):
    return {"summary": {"finished": finished, "lap_s": lap}, "train_seconds": seconds, "size": {"params": params}}


def test_ranking_finished_then_lap_then_time_then_size():
    ordered = [
        _result(5, 20.0, 9000, 500),
        _result(5, 20.0, 600, 900),     # same lap, less training wins
        _result(5, 20.0, 600, 100),     # ... and then the smaller model
        _result(5, 18.0, 99999, 10**6),  # faster lap beats everything after it
        _result(4, 15.0, 1, 1),         # but finishing more tracks comes first
        _result(0, None, 1, 1),
    ]
    ranked = sorted(ordered, key=rank_key)
    assert ranked == [ordered[3], ordered[2], ordered[1], ordered[0], ordered[4], ordered[5]]


def test_neat_train_seconds_handles_resumed_runs(tmp_path):
    csv = tmp_path / "fitness.csv"
    # Old runs restarted time_s at every resume; new runs carry it over.
    csv.write_text("generation,best_fitness,mean_fitness,stdev,time_s\n"
                   "0,1,1,1,100\n1,1,1,1,200\n2,1,1,1,50\n3,1,1,1,120\n")
    assert neat_train_seconds(str(csv)) == 320
    assert neat_train_seconds(str(csv), up_to_generation=2) == 250
    assert neat_train_seconds(str(csv), up_to_generation=0) == 100


def test_ppo_train_seconds_from_monitor(tmp_path):
    mon = tmp_path / "monitor"
    mon.mkdir()
    (mon / "0.monitor.csv").write_text('#{"t_start": 0}\nr,l,t\n1,1000,10.0\n1,1000,30.0\n1,10')
    (mon / "1.monitor.csv").write_text('#{"t_start": 0}\nr,l,t\n1,1000,12.0\n')
    assert ppo_train_seconds(str(tmp_path), at_timestep=2000) == 12.0
    assert ppo_train_seconds(str(tmp_path), at_timestep=2500) == 30.0
    assert ppo_train_seconds(str(tmp_path)) == 30.0


def test_lap_time_bonus_paid_once_per_lap():
    env = make_env(seed=0, offtrack_penalty=0.0, allow_multiple_laps=False, lap_time_bonus=2.0)
    env.reset(seed=0)
    env.step(np.zeros(3))
    for _ in range(9):
        env.step(np.zeros(3))
    env.unwrapped.new_lap = True  # pretend the car just crossed the line after 11 frames
    _, reward, _, truncated, info = env.step(np.zeros(3))
    assert truncated and info["lap_frames"] == 11 and info["laps_completed"] == 1
    assert reward > 2.0 * (1000 - 11) - 1.0


def test_lidar_rays_sets_state_size():
    env = make_env(seed=0, lidar_rays=16)
    state, _ = env.reset(seed=0)
    assert state.shape == (24,) and env.observation_space.shape == (24,)
