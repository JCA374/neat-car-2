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
and docs/assets/gifs/<id>.gif.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.benchmark import SEEDS, track_geometry  # noqa: E402
from carracing.steps import FPS, MODELS, all_steps, hardware, rank_key  # noqa: E402

OUT = "docs/assets/data"
HEADLINE = ("id", "model", "title", "why", "outcome", "parent", "lidar_rays", "lidar_fov", "train", "neat_config", "imported", "note",
            "hardware", "commit", "trained_on", "train_seconds", "selected", "size", "summary", "tracks")
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

    tracks_path = os.path.join(OUT, "tracks.json")
    if not os.path.exists(tracks_path):
        dump(tracks_path, [track_geometry(seed) for seed in SEEDS])

    print(f"Site data for {len(ranked)} steps in {OUT}/")
    for i, r in enumerate(ranked, 1):
        s = r["summary"]
        lap = f"{s['lap_s']:6.2f} s" if s["lap_s"] is not None else "   -    "
        print(f"  {i}. {r['id']:<22} {s['finished']}/5 finished  mean lap {lap}  "
              f"trained {r['train_seconds'] / 60:6.0f} min  {r['size']['label']}")


if __name__ == "__main__":
    main()
