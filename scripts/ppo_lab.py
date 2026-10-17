"""
PPO lab: screen PPO recipes on short runs, 3 seeds each, before one goes on the ladder.

    nohup python scripts/ppo_lab.py round1 > outputs/ppo/lab/lab.log 2>&1 &
    python scripts/ppo_lab.py --summary          # rewrite outputs/ppo/lab/lab.csv and print it
    python scripts/ppo_lab.py auto               # round1 + round2, then the winner as a ladder step (p14)

Every trial is a normal carracing_ppo.train run under the rulebook (training tracks only, 2000-frame
episodes, 0.1 smoothing) with lap checks on the validation tracks. The benchmark tracks are never used.
Trials go seed by seed (every recipe's seed 0 first), so an early look already compares all recipes.
Finished trials are skipped, so the lab can be stopped and restarted.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.steps import fmt_duration, train_flags  # noqa: E402

LAB = "outputs/ppo/lab"
STATUS = os.path.join(LAB, "status.json")
SUMMARY = os.path.join(LAB, "lab.csv")
PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")

# p11's settings (the best PPO step so far) on 16 forward rays, under the rulebook.
BASE = {
    "entropy-coef": 0.005, "normalize-reward": True, "num-envs": 8, "batch-size": 256,
    "offtrack-penalty": 0.0, "allow-multiple-laps": False, "max-episode-steps": 2000,
    "eval-frequency": 0, "checkpoint-frequency": 0, "lap-check-frequency": 50_000,
    "lidar-rays": 16, "lidar-fov": 240, "action-smoothing": 0.1,
}


def recipe(controls: str = "raw3", repeat: int = 1, **extra) -> dict:
    # n-steps counts decisions: keep the frames per rollout the same at every decision rate.
    return {**BASE, "controls": controls, "action-repeat": repeat, "n-steps": 1024 // repeat, **extra}


ROUNDS = {
    "round0": {"frames": 300_000, "seeds": [0], "recipes": {
        "sanity_fixed_track": recipe(**{"track-seeds": "4242"}),
    }},
    "round1": {"frames": 600_000, "seeds": [0, 1, 2], "recipes": {
        "raw3_r1": recipe("raw3", 1),
        "raw3_r4": recipe("raw3", 4),
        "throttle_r1": recipe("throttle", 1),
        "throttle_r4": recipe("throttle", 4),
        "discrete_r1": recipe("discrete", 1),
        "discrete_r4": recipe("discrete", 4),
    }},
    # Round 1: deciding every 4 frames is what works; deciding every frame fails in every mode. But
    # the exploration noise stays at std ~0.8 and runs fall back after their best lap check.
    "round2": {"frames": 600_000, "seeds": [0, 1, 2], "recipes": {
        "raw3_r4_calm": recipe("raw3", 4, **{"log-std-init": -1.0, "entropy-coef": 0.0}),
        "throttle_r4_calm": recipe("throttle", 4, **{"log-std-init": -1.0, "entropy-coef": 0.0}),
        "raw3_r2": recipe("raw3", 2),
        "raw3_r8": recipe("raw3", 8),
        "raw3_r4_g995": recipe("raw3", 4, gamma=0.995),
    }},
    # Off-road analysis of p14 (outputs/analysis/p14_raw3_r4_g995): PPO never brakes (brake output ~ -1
    # everywhere) and runs wide in sharp turns, though its gas output drops to -1.8 before them. With one
    # pedal axis, lifting off harder becomes braking. The control is round 2's raw3_r4_g995 (same seeds).
    "round3": {"frames": 600_000, "seeds": [0, 1, 2], "recipes": {
        "raw3_r4_g995": recipe("raw3", 4, gamma=0.995),
        "throttle_r4_g995": recipe("throttle", 4, gamma=0.995),
    }},
}
RECIPES = {name: flags for spec in ROUNDS.values() for name, flags in spec["recipes"].items()}
# Step-level settings: a promoted recipe keeps these out of [train] (Step.env_flags adds them).
STEP_KEYS = ("lidar-rays", "lidar-fov", "action-smoothing", "controls", "action-repeat")


def trial_dir(name: str, seed: int) -> str:
    return os.path.join(LAB, name, f"seed{seed}")


def run_trial(name: str, flags: dict, seed: int, frames: int) -> None:
    out = trial_dir(name, seed)
    logs, ckpt = os.path.join(out, "logs"), os.path.join(out, "checkpoints")
    os.makedirs(logs, exist_ok=True)
    cmd = [PYTHON, "-m", "carracing_ppo.train", *train_flags({**flags, "total-timesteps": frames}),
           "--seed", str(seed), "--log-dir", logs, "--checkpoint-dir", ckpt]
    with open(os.path.join(logs, "command.txt"), "w") as fh:
        fh.write(" ".join(cmd) + "\n")
    with open(os.path.join(logs, "train.out"), "w") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if code == 0:
        open(os.path.join(out, "done"), "w").close()
    else:
        print(f"  {name} seed {seed} failed (exit {code}), see {logs}/train.out", flush=True)


# ---------------------------------------------------------------------------- summary

def _lap_checks(logs: str) -> list:
    path = os.path.join(logs, "validation_laps.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{"frames": int(r["timestep"]), "t": float(r["time_s"]), "finished": int(r["finished"]),
                 "lap": float(r["mean_lap_s"]) if r["mean_lap_s"] else None} for r in csv.DictReader(fh)]


def _last_logged(logs: str, key: str):
    path = os.path.join(logs, "train.out")
    if not os.path.exists(path):
        return None
    found = re.findall(rf"\|\s+{key}\s+\|\s+([-0-9.e]+)", open(path, errors="ignore").read())
    return float(found[-1]) if found else None


def _training_laps(logs: str, last: int = 50):
    """Share of the last training episodes (across envs) that finished a lap."""
    rows = []
    for path in glob.glob(os.path.join(logs, "monitor", "*.monitor.csv")):
        with open(path, newline="") as fh:
            next(fh)
            rows += [(float(r["t"]), int(float(r["laps_completed"]))) for r in csv.DictReader(fh)
                     if r.get("t") and r.get("laps_completed")]
    rows.sort()
    rows = rows[-last:]
    return round(100 * sum(1 for _, n in rows if n > 0) / len(rows)) if rows else None


def trial_summary(name: str, seed: int) -> dict | None:
    logs = os.path.join(trial_dir(name, seed), "logs")
    checks = _lap_checks(logs)
    if not checks:
        return None
    best = max(checks, key=lambda c: (c["finished"], -(c["lap"] or 1e9)))
    final = checks[-1]
    first5 = next((c["t"] for c in checks if c["finished"] == 5), None)
    return {
        "recipe": name, "seed": seed, "done": os.path.exists(os.path.join(trial_dir(name, seed), "done")),
        "frames": final["frames"], "minutes": round(final["t"] / 60, 1),
        "best_finished": best["finished"], "best_lap": best["lap"], "best_at_min": round(best["t"] / 60, 1),
        "final_finished": final["finished"], "final_lap": final["lap"],
        "first_5of5_min": round(first5 / 60, 1) if first5 is not None else None,
        "train_lap_pct": _training_laps(logs),
        "std": _last_logged(logs, "std"), "entropy_loss": _last_logged(logs, "entropy_loss"),
    }


def write_summary() -> list:
    rows = [s for d in sorted(glob.glob(os.path.join(LAB, "*", "seed*")))
            if (s := trial_summary(os.path.basename(os.path.dirname(d)), int(d.rsplit("seed", 1)[1])))]
    if rows:
        with open(SUMMARY, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def print_summary(rows: list) -> None:
    by = {}
    for r in rows:
        by.setdefault(r["recipe"], []).append(r)
    print(f"{'recipe':<20} {'seeds':>5}  {'best x/5 per seed':<18} {'final x/5':<12} {'best laps':<22} "
          f"{'train laps %':<14} std")
    for name, rs in by.items():
        best = " ".join(str(r["best_finished"]) for r in rs)
        final = " ".join(str(r["final_finished"]) for r in rs)
        laps = " ".join(f"{r['best_lap']:.1f}" if r["best_lap"] else "–" for r in rs)
        tl = " ".join(str(r["train_lap_pct"]) for r in rs)
        std = " ".join(f"{r['std']:.2f}" if r["std"] is not None else "–" for r in rs)
        print(f"{name:<20} {len(rs):>5}  {best:<18} {final:<12} {laps:<22} {tl:<14} {std}"
              f"   (median best {statistics.median(r['best_finished'] for r in rs):g})")


# ---------------------------------------------------------------------------- promote

def lab_score(rs: list) -> tuple:
    """Reliable first (median best lap check, then how many seeds still hold it at the end), then fast."""
    laps = [r["best_lap"] for r in rs if r["best_lap"]]
    return (statistics.median(r["best_finished"] for r in rs),
            statistics.median(r["final_finished"] for r in rs),
            -(statistics.mean(laps) if laps else 1e9))


def winner(rows: list, seeds: int = 3) -> str:
    by = {}
    for r in rows:
        if r["done"] and r["recipe"] in RECIPES and r["recipe"] != "sanity_fixed_track":
            by.setdefault(r["recipe"], []).append(r)
    full = {name: rs for name, rs in by.items() if len(rs) >= seeds}
    return max(full, key=lambda name: lab_score(full[name]))


def promote(name: str, number: int = 14, frames: int = 3_000_000, suffix: str = "", change: str = "",
            parent: str = "p11_normalize_reward", **overrides) -> str:
    """Write the lab winner (optionally with one change) as a normal two-seed ladder step; return its path."""
    flags = {**RECIPES[name], **overrides}
    step_id = f"p{number:02d}_{name}{suffix}"
    path = os.path.join("experiments", "ppo", f"{step_id}.toml")
    rs = [r for r in write_summary() if r["recipe"] == name]
    seen = ", ".join(f"seed {r['seed']}: best {r['best_finished']}/5, final {r['final_finished']}/5" for r in rs)
    train = {k: v for k, v in flags.items() if k not in STEP_KEYS}
    train.update({"total-timesteps": frames, "lap-check-frequency": 25_000})
    title = f"PPO lab winner: {flags['controls']} controls, decide every {flags['action-repeat']} frames"
    lines = [f'id = "{step_id}"', 'model = "ppo"', f'parent = "{parent}"',
             f'title = "{change.split(".")[0] if change else title}"',
             'why = """', *([change] if change else []),
             f"The PPO lab (scripts/ppo_lab.py, short 600k-frame runs on 3 seeds, validation tracks only) found that "
             f"PPO deciding every frame never learns to finish, in any control mode. Recipe '{name}' scored best "
             f"({seen}). This step trains it for {frames // 1_000_000}M frames on two seeds, benchmarked like every other step.",
             '"""',
             f'lidar_rays = {flags["lidar-rays"]}', f'lidar_fov = {flags["lidar-fov"]}',
             f'action_smoothing = {flags["action-smoothing"]}',
             f'controls = "{flags["controls"]}"', f'action_repeat = {flags["action-repeat"]}', "seeds = [0, 1]",
             "", "[train]"]
    for k, v in train.items():
        lines.append(f"{k} = " + ("true" if v is True else "false" if v is False else
                                  f'"{v}"' if isinstance(v, str) else repr(v)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def run_round(name: str) -> None:
    spec = ROUNDS[name]
    trials = [(recipe_name, seed) for seed in spec["seeds"] for recipe_name in spec["recipes"]]
    status = {"pid": os.getpid(), "round": name, "frames": spec["frames"], "finished": None,
              "trials": [{"recipe": n, "seed": s, "state": "queued", "started": None, "ended": None}
                         for n, s in trials]}
    for entry in status["trials"]:
        if os.path.exists(os.path.join(trial_dir(entry["recipe"], entry["seed"]), "done")):
            entry["state"] = "done"
    save_status(status)
    for entry in status["trials"]:
        if entry["state"] == "done":
            continue
        entry["state"], entry["started"] = "running", time.time()
        save_status(status)
        print(f"{time.strftime('%H:%M')} {entry['recipe']} seed {entry['seed']}", flush=True)
        run_trial(entry["recipe"], spec["recipes"][entry["recipe"]], entry["seed"], spec["frames"])
        entry["ended"] = time.time()
        entry["state"] = "done" if os.path.exists(
            os.path.join(trial_dir(entry["recipe"], entry["seed"]), "done")) else "failed"
        save_status(status)
        print(f"  took {fmt_duration(entry['ended'] - entry['started'])}", flush=True)
        print_summary(write_summary())
    status["finished"] = time.time()
    save_status(status)


# ---------------------------------------------------------------------------- main

def save_status(status: dict) -> None:
    os.makedirs(LAB, exist_ok=True)
    with open(STATUS, "w") as fh:
        json.dump(status, fh, indent=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("round", nargs="?", choices=sorted(ROUNDS) + ["auto"])
    parser.add_argument("--summary", action="store_true", help="Only rewrite and print lab.csv.")
    args = parser.parse_args()
    if args.summary or not args.round:
        print_summary(write_summary())
        return 0

    if args.round == "auto":
        for name in ("round1", "round2"):
            run_round(name)
        best = winner(write_summary())
        p14 = promote(best)
        # One change each on top of the winner, to choose the PPO 32 and PPO Smart championship cars.
        p14_id = os.path.splitext(os.path.basename(p14))[0]
        variants = [
            promote(best, 15, suffix="_32rays", parent=p14_id, **{"lidar-rays": 32},
                    change="32 forward rays instead of 16. Same as p14 otherwise: does PPO use the finer view "
                           "(the NEAT 32 car got faster laps with it)?"),
            promote(best, 16, suffix="_lap_bonus", parent=p14_id, **{"lap-time-bonus": 1.0},
                    change="A lap-time bonus. Same as p14 plus +1 per frame a lap beats 1000 frames, the bonus "
                           "the NEAT cars train with: does it make PPO faster without crashing?"),
        ]
        print(f"{time.strftime('%H:%M')} lab winner: {best} -> {p14}, then {', '.join(variants)}", flush=True)
        for path in [p14, *variants]:
            subprocess.call([PYTHON, "scripts/run_step.py", path])
        return 0
    run_round(args.round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
