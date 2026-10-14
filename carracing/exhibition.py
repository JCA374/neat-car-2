"""Exhibition tracks: extra races for the Championship page, just for fun (no points).

Every championship car's checkpoints drive 25 more tracks that no model trains, validates or is
benchmarked on (training draws seeds >= 1000), so the page can race them on a random or chosen track.
Results are cached in outputs/<model>/<id>/exhibition_<category>.json; only missing seeds are driven.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from carracing.benchmark import drive, load_policy
from carracing.rules import EXHIBITION_SEEDS
from carracing.steps import Step

POSITION_KEYS = ("every", "finish", "x", "y", "a", "v", "tiles", "grass")


def exhibition_path(step: Step, category: str) -> str:
    return os.path.join(step.out_dir, f"exhibition_{category}.json")


def record(step: Step, categories=("fast", "finish")) -> Dict[str, int]:
    """Drive each available championship checkpoint on every exhibition track not yet recorded."""
    driven = {}
    for cat in categories:
        model = step.model_file(cat)
        if not os.path.exists(model):
            continue
        path = exhibition_path(step, cat)
        # Re-drive everything if the checkpoint changed since these drives were recorded.
        fresh = os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(model)
        data = json.load(open(path)) if fresh else {}
        todo = [s for s in EXHIBITION_SEEDS if str(s) not in data]
        if not todo:
            continue
        policy = load_policy(step, model)
        for seed in todo:
            out = drive(policy, seed, step.env_kwargs())
            data[str(seed)] = {"result": out["result"],
                               "drive": {k: out["recording"][k] for k in POSITION_KEYS}}
        with open(path, "w") as fh:
            json.dump(data, fh, separators=(",", ":"))
        driven[cat] = len(todo)
    return driven
