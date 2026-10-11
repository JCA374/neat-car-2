"""
Collect training curves and run summaries for the project page.

Reads each run's monitor CSVs, evaluations.npz and best_model.zip
hyperparameters, and writes docs/assets/data/runs.json.

    python scripts/build_site_data.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# (key, label, log dir, checkpoint dir)
RUNS = [
    ("baseline", "PPO baseline", "outputs/logs", "outputs/checkpoints"),
    ("improved", "Longer run", "outputs/logs_improved", "outputs/checkpoints_improved"),
    ("strict", "Strict penalty", "outputs/logs_strict", "outputs/checkpoints_strict"),
    ("multilap", "Multi-lap (buggy)", "outputs/logs_multilap", "outputs/checkpoints_multilap"),
    ("v2", "Fixed multi-lap (v2)", "outputs/logs_v2", "outputs/checkpoints_v2"),
]
HYPERPARAMS = ("learning_rate", "ent_coef", "n_steps", "batch_size", "n_envs", "gamma", "num_timesteps")


def load_monitor(log_dir: str) -> Optional[pd.DataFrame]:
    """Training episodes only (eval_monitor/ is excluded). Ignores corrupted lines."""
    paths = sorted(glob.glob(os.path.join(log_dir, "monitor", "*.monitor.csv")))
    frames = []
    for path in paths:
        df = pd.read_csv(path, skiprows=1, on_bad_lines="skip")
        for col in ("r", "l", "t"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df.dropna(subset=["r", "l", "t"]))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    return df


def load_hyperparams(ckpt_dir: str) -> Dict[str, float]:
    path = os.path.join(ckpt_dir, "best_model.zip")
    if not os.path.exists(path):
        return {}
    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("data"))
    return {k: data[k] for k in HYPERPARAMS if isinstance(data.get(k), (int, float))}


def downsample(x: np.ndarray, y: np.ndarray, points: int = 200) -> List[List[float]]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(points, len(x))).astype(int))
    return [[int(x[i]), round(float(y[i]), 1)] for i in idx]


def summarize(key: str, label: str, log_dir: str, ckpt_dir: str) -> Optional[dict]:
    df = load_monitor(log_dir)
    if df is None or df.empty:
        return None
    rolling = df["r"].rolling(100, min_periods=10).mean().bfill()
    last = df.tail(400)
    if "laps_completed" in df:
        laps = pd.to_numeric(df["laps_completed"], errors="coerce").fillna(0)
        lap_share = float((laps >= 1).mean())
        mean_laps_last = float(laps.tail(400).mean())
    else:
        # Older runs did not log laps: a lap ends the episode early with a high score.
        lap_share = float(((df["l"] < 1000) & (df["r"] > 600)).mean())
        mean_laps_last = None
    run = {
        "key": key,
        "label": label,
        "episodes": int(len(df)),
        "steps": int(df["steps"].iloc[-1]),
        "hours": round(float(df["t"].iloc[-1]) / 3600, 1),
        "mean_last400": round(float(last["r"].mean()), 1),
        "pct_over_500_last400": round(100 * float((last["r"] > 500).mean()), 1),
        "lap_share": round(100 * lap_share, 1),
        "mean_laps_last400": None if mean_laps_last is None else round(mean_laps_last, 2),
        "curve": downsample(df["steps"].to_numpy(), rolling.to_numpy()),
        "hyperparams": load_hyperparams(ckpt_dir),
    }
    eval_path = os.path.join(log_dir, "evaluations.npz")
    if os.path.exists(eval_path):
        ev = np.load(eval_path)
        means = ev["results"].mean(axis=1)
        run["eval_best"] = round(float(means.max()), 1)
        run["eval_curve"] = [[int(t), round(float(m), 1)] for t, m in zip(ev["timesteps"], means)]
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/assets/data/runs.json")
    args = parser.parse_args()
    runs = [r for r in (summarize(*spec) for spec in RUNS) if r]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"runs": runs}, fh, separators=(",", ":"))
    for r in runs:
        print(
            f"{r['key']:9s} eps={r['episodes']:5d} steps={r['steps']:8d} last400={r['mean_last400']:6.1f} "
            f"laps%={r['lap_share']:5.1f} eval_best={r.get('eval_best')} hp={r['hyperparams']}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
