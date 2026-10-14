"""
Build the project page's data from every step's outputs (no training, no driving).

    python scripts/build_site.py

Reads experiments/*/*.toml and outputs/<model>/<id>/{result,drives}.json and writes
docs/assets/data/:
    leaderboard.json        every step's headline numbers, ranked; champion per model
    tracks.json             the five benchmark tracks (road polygons)
    steps/<id>.json         training curve (+ NEAT: every generation's champion network)
    drives/<id>.json        the benchmark drives (positions) for the race page
    think/<id>/<seed>.json  NEAT only: inputs, node activations and actions per frame
    championship.json       the championship cars, both categories, per-track results
    champ/<id>_<cat>.json   drives of each car's "finish" and "fast" checkpoints
    champ/tracks/<seed>.json  exhibition tracks: road + every car's drives (random / chosen-track races)
    findings.json           experiments/findings.toml
and docs/assets/gifs/<id>.gif.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.benchmark import SEEDS, track_geometry  # noqa: E402
from carracing.exhibition import EXHIBITION_SEEDS, exhibition_path  # noqa: E402
from carracing.steps import FPS, MODELS, all_steps, hardware, rank_key  # noqa: E402

OUT = "docs/assets/data"
HEADLINE = ("id", "model", "title", "why", "outcome", "parent", "lidar_rays", "lidar_fov", "train", "neat_config", "imported", "note",
            "hardware", "commit", "trained_on", "train_seconds", "selected", "size", "summary", "tracks")
# Championship cars: fixed colour per car (categorical tokens) and the PPO cars still to come.
CAR_COLORS = {"neat16": "--c1", "neat32": "--c3", "neatsmart": "--c7", "ppo16": "--c2", "ppo32": "--c4", "pposmart": "--c5"}
PIT_LANE = [
    {"id": "ppo16", "model": "ppo", "title": "PPO 16"},
    {"id": "ppo32", "model": "ppo", "title": "PPO 32"},
    {"id": "pposmart", "model": "ppo", "title": "PPO Smart"},
]
POSITION_KEYS = ("every", "finish", "x", "y", "a", "v", "tiles", "grass")


def dump(path: str, data, pretty: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        if pretty:
            json.dump(data, fh, indent=1, ensure_ascii=False)
        else:
            json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
    results, pending = [], []
    for step in all_steps():
        path = os.path.join(step.out_dir, "result.json")
        if not os.path.exists(path):
            print(f"pending {step.id}: no {path} yet (run scripts/run_step.py {step.path})")
            pending.append({"id": step.id, "model": step.model, "title": step.title, "why": step.why})
            continue
        with open(path) as fh:
            result = json.load(fh)
        result["outcome"] = step.outcome  # from the TOML, so it can be written after the run
        results.append(result)

        dump(os.path.join(OUT, "steps", f"{step.id}.json"),
             {k: result[k] for k in ("id", "curve", "generations", "inputs", "outputs") if k in result})
        with open(os.path.join(step.out_dir, "drives.json")) as fh:
            drives = json.load(fh)
        dump(os.path.join(OUT, "drives", f"{step.id}.json"),
             {seed: {k: rec[k] for k in POSITION_KEYS} for seed, rec in drives.items()})
        gif = os.path.join(step.out_dir, "lap.gif")
        if os.path.exists(gif):
            os.makedirs("docs/assets/gifs", exist_ok=True)
            shutil.copyfile(gif, os.path.join("docs/assets/gifs", f"{step.id}.gif"))
        if step.model == "neat":
            for seed, rec in drives.items():
                dump(os.path.join(OUT, "think", step.id, f"{seed}.json"), rec)

    ranked = sorted(results, key=rank_key)
    champions = {}
    for model in MODELS:
        best = next((r for r in ranked if r["model"] == model), None)
        if best:
            champions[model] = best["id"]
    dump(os.path.join(OUT, "leaderboard.json"), {
        "seeds": SEEDS, "fps": FPS, "hardware": hardware(), "champions": champions,
        "steps": [{k: r.get(k) for k in HEADLINE} for r in ranked],
        "pending": pending,
    }, pretty=True)

    steps = {s.id: s for s in all_steps()}
    by_id = {r["id"]: r for r in results}
    dump(os.path.join(OUT, "championship.json"), championship(steps, by_id), pretty=True)
    with open("experiments/findings.toml", "rb") as fh:
        dump(os.path.join(OUT, "findings.json"), tomllib.load(fh)["finding"], pretty=True)

    tracks_path = os.path.join(OUT, "tracks.json")
    if not os.path.exists(tracks_path):
        dump(tracks_path, [track_geometry(seed) for seed in SEEDS])

    print(f"Site data for {len(ranked)} steps in {OUT}/")
    for i, r in enumerate(ranked, 1):
        s = r["summary"]
        lap = f"{s['lap_s']:6.2f} s" if s["lap_s"] is not None else "   -    "
        print(f"  {i}. {r['id']:<22} {s['finished']}/5 finished  mean lap {lap}  "
              f"trained {r['train_seconds'] / 60:6.0f} min  {r['size']['label']}")


def lineage(step_id: str, steps: dict) -> list:
    """The chain of ladder steps a car was built from, oldest first."""
    chain, cur = [], steps[step_id].parent
    while cur and cur in steps and cur not in chain:
        chain.append(cur)
        cur = steps[cur].parent
    return chain[::-1]


def championship(steps: dict, results: dict) -> dict:
    """Cars, their two checkpoints and benchmark results; drives go to champ/<id>_<cat>.json."""
    cars = []
    for step in sorted((s for s in steps.values() if s.championship), key=lambda s: (s.model, s.id)):
        car = {"id": step.id, "model": step.model, "label": step.title.split(":")[0], "title": step.title,
               "why": step.why, "outcome": step.outcome, "color": CAR_COLORS.get(step.id, "--c8"),
               "lineage": lineage(step.id, steps), "status": "queued", "categories": None}
        result = results.get(step.id)
        if result and result.get("categories"):
            car.update(status="raced", size=result["size"], lap_checks=result.get("lap_checks", []),
                       trained_on=result.get("trained_on"), categories=result["categories"])
            for cat, entry in result["categories"].items():
                drives_path = os.path.join(step.out_dir, f"drives_{cat}.json")
                if entry is None or not os.path.exists(drives_path):
                    continue
                with open(drives_path) as fh:
                    drives = json.load(fh)
                dump(os.path.join(OUT, "champ", f"{step.id}_{cat}.json"),
                     {seed: {k: rec[k] for k in POSITION_KEYS} for seed, rec in drives.items()})
                gif = os.path.join(step.out_dir, "lap.gif" if cat == "fast" else f"lap_{cat}.gif")
                if os.path.exists(gif):
                    shutil.copyfile(gif, os.path.join("docs/assets/gifs", f"champ_{step.id}_{cat}.gif"))
        cars.append(car)
    present = {c["id"] for c in cars}
    for pit in PIT_LANE:
        if pit["id"] not in present:
            cars.append({**pit, "label": pit["title"], "status": "pit", "color": CAR_COLORS[pit["id"]],
                         "why": "PPO is being fixed first: no PPO step has learned to finish a track since the traction fix.",
                         "categories": None, "lineage": []})
    return {"seeds": SEEDS, "fps": FPS, "hardware": hardware(), "budget_hours": 8, "points": [25, 18, 15, 12, 10, 8],
            "cars": cars, "exhibition_seeds": exhibition(steps, [c for c in cars if c["status"] == "raced"])}


def exhibition(steps: dict, cars: list) -> list:
    """One file per exhibition track with the road and every raced car's drives on it."""
    per_seed = {str(seed): {} for seed in EXHIBITION_SEEDS}
    for car in cars:
        for cat in ("fast", "finish"):
            path = exhibition_path(steps[car["id"]], cat)
            if not car["categories"].get(cat) or not os.path.exists(path):
                continue
            with open(path) as fh:
                for seed, entry in json.load(fh).items():
                    if seed in per_seed:
                        per_seed[seed].setdefault(car["id"], {})[cat] = entry
    seeds = []
    for seed, by_car in per_seed.items():
        if not by_car:
            continue
        out = os.path.join(OUT, "champ", "tracks", f"{seed}.json")
        track = None
        if os.path.exists(out):  # the road never changes: reuse it instead of rebuilding the env
            with open(out) as fh:
                track = json.load(fh)["track"]
        dump(out, {"track": track or track_geometry(int(seed)), "cars": by_car})
        seeds.append(int(seed))
    return seeds


if __name__ == "__main__":
    main()
