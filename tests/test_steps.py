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
        # n.. for NEAT, p.. for PPO; championship cars are named after the model (neat16, ppo32 ...)
        assert s.id.startswith(s.model) if s.championship else s.id[0] == s.model[0]
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


def test_forward_lidar_fan_is_centred_on_the_nose():
    import math

    from carracing.env import lidar_angles

    legacy = lidar_angles(32)
    assert math.isclose(legacy[0], -0.9 * math.pi) and math.isclose(legacy[-1], 0.9 * math.pi)
    fan = lidar_angles(17, fov=240)
    assert math.isclose(fan[8], math.pi / 2)  # middle ray along local +y = the nose
    assert math.isclose(fan[-1] - fan[0], math.radians(240))
    env = make_env(seed=0, lidar_rays=16, lidar_fov=240)
    state, _ = env.reset(seed=0)
    assert state.shape == (24,)
    # On the start straight the forward rays see far more road than the side rays.
    assert state[8 + 7] > state[8]


def test_stuck_car_ends_the_episode_and_pays_for_the_rest():
    env = make_env(seed=0, offtrack_penalty=0.0, allow_multiple_laps=False, max_episode_steps=1000, stuck_patience=30)
    env.reset(seed=0)
    for frame in range(1, 200):
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.0, 1.0]))  # brake: never moves
        if terminated:
            break
    # The start tile counts as reached on frame 1, so the next 30 frames are stuck.
    assert info.get("stuck") and frame <= 32
    assert reward < -0.1 * (1000 - frame) + 0.01  # time penalty for the remaining frames


def test_drop_heading_removes_the_compass_inputs():
    env = make_env(seed=0, lidar_rays=16, lidar_fov=240, drop_heading=True)
    state, _ = env.reset(seed=0)
    assert state.shape == (22,)
    full = make_env(seed=0, lidar_rays=16, lidar_fov=240)
    full_state, _ = full.reset(seed=0)
    assert np.allclose(state, np.delete(full_state, [3, 4]))
    assert env.last_traction == full_state[7]  # traction is still readable, wherever it sits in the state


def test_championship_picks_first_finish_and_fastest(tmp_path, monkeypatch):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_step", "scripts/run_step.py")
    run_step = importlib.util.module_from_spec(spec)
    sys.modules["run_step"] = run_step
    spec.loader.exec_module(run_step)

    step = load_step("experiments/championship/neat16.toml")
    monkeypatch.setattr(type(step), "out_dir", property(lambda self: str(tmp_path)))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "validation_laps.csv").write_text(
        "generation,time_s,finished,mean_lap_s\n"
        "0,60,0,\n1,120,3,21.0\n2,180,5,24.0\n3,240,5,22.5\n4,300,5,22.5\n5,360,4,19.0\n")
    cats = run_step.select_categories(step)
    assert cats["finish"]["selected"] == {"generation": 2} and cats["finish"]["train_seconds"] == 180
    # Fastest = most tracks finished, then the fastest lap, then the earlier checkpoint.
    assert cats["fast"]["selected"] == {"generation": 3} and cats["fast"]["train_seconds"] == 240


def test_budget_reporter_stops_on_time_and_patience():
    from types import SimpleNamespace

    import pytest as _pytest

    from carracing_neat.train import BudgetReporter, TrainingBudgetReached

    clock = SimpleNamespace(t=0.0, elapsed=lambda: clock.t)
    validation = SimpleNamespace(last_laps={"finished": 2, "mean_lap_s": 20.0})
    budget = BudgetReporter(clock, validation, max_hours=1.0, patience_hours=0.5)
    budget.post_evaluate(None, None, None, None)
    clock.t = 1700  # under 30 min since the last improvement
    budget.post_evaluate(None, None, None, None)
    clock.t = 1900  # 31+ min without a better lap check
    with _pytest.raises(TrainingBudgetReached):
        budget.post_evaluate(None, None, None, None)
    late = BudgetReporter(clock, validation, max_hours=0.5, patience_hours=None)
    with _pytest.raises(TrainingBudgetReached):
        late.post_evaluate(None, None, None, None)
