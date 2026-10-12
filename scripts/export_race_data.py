"""
Record every racer on every benchmark track for docs/race.html.

CarRacing simulates one car, so a "race" is a ghost race: each model drives
the track on its own, exactly as in scripts/benchmark.py (raw score, 1000
frames, single lap), and the page replays the recordings side by side.

    python scripts/export_race_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

from benchmark import SEEDS, NEATPolicy, load_policy  # noqa: E402  (scripts/ is on sys.path when run directly)
from carracing_ppo.env import make_env

# Order fixes each racer's colour on the page (colour follows the racer).
RACERS = [
    {"id": "neat", "label": "NEAT run 1", "desc": "Best of run 1 (gen 127)", "stage": "neat"},
    {"id": "neat2", "label": "NEAT run 2", "desc": "Best of run 2 (gen 236), full controls", "stage": "neat2"},
    {"id": "ppo_strict", "label": "PPO best", "desc": "PPO stopped near its peak (Dec 23)", "stage": "strict"},
    {"id": "ppo_baseline", "label": "PPO baseline", "desc": "First PPO agent (Dec 21)", "stage": "baseline"},
    {"id": "neat_g25", "label": "NEAT gen 25", "desc": "Run 1, first champion over 600", "genome": "outputs/neat/checkpoints/best-genome-gen-25.pkl"},
    {"id": "neat_g5", "label": "NEAT gen 5", "desc": "Run 1, first champion over 300", "genome": "outputs/neat/checkpoints/best-genome-gen-5.pkl"},
    {"id": "ppo_collapsed", "label": "PPO collapsed", "desc": "Same PPO run after 4.5M steps", "stage": "collapsed"},
    {"id": "ppo_parked", "label": "PPO parked", "desc": "PPO with the real penalty: learned to stop", "stage": "v2"},
]
EVERY = 2


def r(v: float, nd: int = 1) -> float:
    return round(float(v), nd)


def policy_for(racer: dict):
    if "genome" in racer:
        return NEATPolicy(racer["genome"], legacy_decode=True)
    return load_policy(racer["stage"])


def drive(policy, seed: int) -> dict:
    env = make_env(render_mode=None, offtrack_penalty=0.0, allow_multiple_laps=False, max_episode_steps=1000)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    out = {"x": [], "y": [], "a": [], "v": [], "tiles": [], "grass": []}
    score, finish, steps = 0.0, None, 0
    for steps in range(1, 1001):
        hull = env.unwrapped.car.hull
        if (steps - 1) % EVERY == 0:
            out["x"].append(r(hull.position[0]))
            out["y"].append(r(hull.position[1]))
            out["a"].append(r(hull.angle, 3))
            out["v"].append(r(math.hypot(*hull.linearVelocity)))
            out["tiles"].append(int(env.unwrapped.tile_visited_count))
            out["grass"].append(int(round((1 - float(obs[7])) * 4)))
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        score += float(reward)
        if info.get("laps_completed", 0) >= 1 and finish is None:
            finish = steps
        if terminated or truncated:
            break
    env.close()
    return {"score": r(score), "steps": steps, "finish": finish, "final_tiles": int(env.unwrapped.tile_visited_count), **out}


def track(seed: int) -> dict:
    env = make_env(render_mode=None)
    env.reset(seed=seed)
    base = env.unwrapped
    polys = [[r(c) for xy in poly for c in xy] for poly, _ in base.road_poly]
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    data = {"seed": seed, "tiles": len(base.track), "polys": polys,
            "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "start": [r(base.track[0][2]), r(base.track[0][3])], "start_angle": r(base.track[0][1], 3)}
    env.close()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/assets/data/race.json")
    args = parser.parse_args()
    tracks = [track(seed) for seed in SEEDS]
    for racer in RACERS:
        policy = policy_for(racer)
        for t in tracks:
            run = drive(policy, t["seed"])
            t.setdefault("runs", {})[racer["id"]] = run
        print(racer["label"].ljust(14), " ".join(
            f"t{t['seed']}:{t['runs'][racer['id']]['score']:7.1f}{'*' if t['runs'][racer['id']]['finish'] else ' '}" for t in tracks))
    racers = [{k: v for k, v in rc.items() if k in ("id", "label", "desc")} for rc in RACERS]
    with open(args.out, "w") as fh:
        json.dump({"every": EVERY, "fps": 50, "racers": racers, "tracks": tracks}, fh, separators=(",", ":"))
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
