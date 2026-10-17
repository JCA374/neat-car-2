"""
Live training progress: the queue, plus details of the step being trained (NEAT or PPO).

    python scripts/progress.py                 # refresh every 20 s (Ctrl+C closes the view, training keeps going)
    python scripts/progress.py --once
    python scripts/progress.py --step n01_lap_time
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.steps import all_steps, fmt_duration  # noqa: E402

QUEUE = "outputs/queue.json"
LAB_STATUS = "outputs/ppo/lab/status.json"


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else pd.DataFrame()


def bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {100 * frac:5.1f}%"


def eta(done: float, total: float, seconds: float) -> str:
    if done <= 0 or seconds <= 0:
        return "–"
    return fmt_duration(max(0.0, total - done) * seconds / done)


def neat_detail(step, rows: int) -> list:
    logs = os.path.join(step.out_dir, "logs")
    fit = _read(os.path.join(logs, "fitness.csv"))
    val = _read(os.path.join(logs, "validation.csv"))
    spc = _read(os.path.join(logs, "species.csv"))
    if fit.empty:
        return ["  Waiting for the first generation ..."]
    total = int(step.train.get("generations", 50))
    done = len(fit)
    elapsed = float(fit["time_s"].iloc[-1])
    per_gen = fit["time_s"].diff().tail(5).mean() if done > 1 else elapsed
    budget = step.train.get("max-hours")
    if budget:  # championship car: the time budget is the limit, not the generation count
        patience = step.train.get("patience-hours")
        lines = [f"  Time budget {bar(elapsed / (budget * 3600))}  {fmt_duration(elapsed)} of {budget:g} h"
                 + (f" (stops early after {patience} h without a faster lap)" if patience else ""),
                 f"  Generation {done}   ~{per_gen / 60:.1f} min/generation"]
    else:
        lines = [f"  Generations {bar(done / total)}  {done}/{total}",
                 f"  Training time {fmt_duration(elapsed)}   ~{per_gen / 60:.1f} min/generation   "
                 f"ETA {fmt_duration(max(0, total - done) * per_gen)}"]
    laps = _read(os.path.join(logs, "validation_laps.csv"))
    if not laps.empty:
        full = laps[laps["finished"] == 5]
        first = f"first finished all 5 at {fmt_duration(full['time_s'].iloc[0])}" if len(full) else "not yet finished all 5"
        best = (f", fastest {full['mean_lap_s'].min():.2f} s" if len(full) else "")
        lines.append(f"  Lap check (5 validation tracks): {first}{best}; latest {int(laps['finished'].iloc[-1])}/5")
    if not val.empty:
        best = val.loc[val["validation_score"].idxmax()]
        lines.append(f"  Best on validation: {best['validation_score']:.1f} (generation {int(best['generation'])}, "
                     f"{int(best['hidden_nodes'])} hidden nodes, {int(best['enabled_connections'])} connections)")
    table = fit[["generation", "best_fitness", "mean_fitness"]].copy()
    if not val.empty:
        table = table.merge(val[["generation", "validation_score", "hidden_nodes"]], on="generation", how="left")
    if not spc.empty:
        table = table.merge(spc[["generation", "species"]], on="generation", how="left")
    names = {"generation": "gen", "best_fitness": "train best", "mean_fitness": "train mean",
             "validation_score": "validation", "hidden_nodes": "hidden", "species": "species"}
    lines += ["", table.tail(rows).rename(columns=names).to_string(index=False, float_format=lambda v: f"{v:.1f}")]
    return lines


def ppo_detail(step, rows: int) -> list:
    logs = os.path.join(step.out_dir, "logs")
    episodes = []
    for path in glob.glob(os.path.join(logs, "monitor", "*.monitor.csv")):
        df = pd.read_csv(path, skiprows=1).dropna(subset=["l", "t"])
        episodes.append(df)
    if not episodes or sum(len(e) for e in episodes) == 0:
        return ["  Waiting for the first episodes ..."]
    ep = pd.concat(episodes).sort_values("t")
    repeat = step.action_repeat  # Monitor lengths count decisions; one decision = `repeat` frames
    total = int(step.train.get("total-timesteps", 500_000))
    done = int(ep["l"].sum()) * repeat
    elapsed = float(ep["t"].iloc[-1])
    budget = step.train.get("max-hours")
    val = _read(os.path.join(logs, "validation_laps.csv"))
    if not val.empty and float(val["time_s"].iloc[-1]) > elapsed:
        # A resumed run: the Monitor CSVs restart, the lap checks carry the whole run's frames and time.
        done, elapsed = int(val["timestep"].iloc[-1]), float(val["time_s"].iloc[-1])
    if budget:  # championship car: the time budget is the limit
        lines = [f"  Time budget {bar(elapsed / (budget * 3600))}  {fmt_duration(elapsed)} of {budget:g} h",
                 f"  {done / 1e6:.2f}M frames   {done / max(elapsed, 1):.0f} frames/s"]
    else:
        lines = [f"  Frames {bar(done / total)}  {done / 1000:,.0f}k/{total / 1000:,.0f}k",
                 f"  Training time {fmt_duration(elapsed)}   {done / max(elapsed, 1):.0f} frames/s   "
                 f"ETA {eta(done, total, elapsed)}"]
    if not val.empty:
        best = val.sort_values(["finished", "mean_lap_s"], ascending=[False, True]).iloc[0]
        five = val[val["finished"] == 5]
        lines.append(f"  Lap checks: {len(val)}, latest {int(val['finished'].iloc[-1])}/5, best {int(best['finished'])}/5"
                     + (f" in {best['mean_lap_s']:.2f} s" if best["finished"] else "")
                     + (f", first 5/5 after {fmt_duration(five['time_s'].iloc[0])}" if len(five) else ""))
    recent = ep.tail(40)
    laps = f", {100 * (recent['laps_completed'] > 0).mean():.0f}% finished a lap" if "laps_completed" in recent else ""
    lines.append(f"  Last {len(recent)} training episodes: mean reward {recent['r'].mean():.1f}, "
                 f"mean length {recent['l'].mean() * repeat:.0f} frames{laps}")
    npz = os.path.join(logs, "evaluations.npz")
    if os.path.exists(npz):
        ev = np.load(npz)
        means = ev["results"].mean(axis=1)
        best = int(np.argmax(means))
        lines.append(f"  Best evaluation: {means[best]:.1f} at {ev['timesteps'][best] / 1000:,.0f}k steps")
        table = pd.DataFrame({"steps": ev["timesteps"], "eval reward": means,
                              "eval length": ev["ep_lengths"].mean(axis=1)})
        lines += ["", table.tail(rows).to_string(index=False, float_format=lambda v: f"{v:.1f}")]
    return lines


def lab_lines(lab: dict, rows: int) -> list:
    """The PPO lab (scripts/ppo_lab.py): its grid of trials and the running one's lap checks."""
    alive = lab["finished"] is None and _alive(lab["pid"])
    lines = [f"PPO lab {lab['round']} ({lab['frames'] / 1000:,.0f}k frames per trial)"
             + ("" if alive else ("  (finished)" if lab["finished"] else "  (not running)"))]
    running = None
    for e in lab["trials"]:
        if e["state"] == "running" and not alive:
            e["state"] = "stopped"
        took = fmt_duration((e["ended"] or time.time()) - e["started"]) if e["started"] else ""
        mark = {"running": ">", "done": "✓", "queued": " "}.get(e["state"], "!")
        lines.append(f"  {mark} {e['recipe']:<20} seed {e['seed']}  {e['state']:<8} {took:>10}")
        if e["state"] == "running":
            running = e
    if running:
        val = _read(f"outputs/ppo/lab/{running['recipe']}/seed{running['seed']}/logs/validation_laps.csv")
        lines += ["", f"  {running['recipe']} seed {running['seed']}: lap checks on the 5 validation tracks"]
        lines += ["  waiting for the first lap check ..."] if val.empty else \
            ["  " + l for l in val.tail(rows).to_string(index=False).splitlines()]
    return lines + [""]


def render(step_id: str | None, rows: int) -> str:
    steps = {s.id: s for s in all_steps()}
    lines = [f"Training progress   {time.strftime('%H:%M:%S')}", ""]
    if os.path.exists(LAB_STATUS) and step_id is None:
        lab = json.load(open(LAB_STATUS))
        if lab["finished"] is None or not os.path.exists(QUEUE) or \
                os.path.getmtime(LAB_STATUS) > os.path.getmtime(QUEUE):
            return "\n".join(lines + lab_lines(lab, rows))
    queue = json.load(open(QUEUE)) if os.path.exists(QUEUE) else None
    current = step_id
    if queue:
        alive = queue["finished"] is None and _alive(queue["pid"])
        lines.append("Queue" + ("" if alive else ("  (finished)" if queue["finished"] else "  (not running)")))
        for e in queue["steps"]:
            if e["state"] == "running" and not alive:
                e["state"] = "stopped"
            took = ""
            if e["started"]:
                took = fmt_duration((e["ended"] or time.time()) - e["started"])
            mark = {"running": ">", "done": "✓", "queued": " "}.get(e["state"], "!")
            lines.append(f"  {mark} {e['id']:<16} {e['state']:<18} {took:>12}   {e['title']}")
            if current is None and e["state"] == "running":
                current = e["id"]
        lines.append("")
    if current is None:
        lines.append("No step is training. Start the queue with:  nohup python scripts/run_queue.py > outputs/queue.log 2>&1 &")
        return "\n".join(lines)
    step = steps[current]
    lines.append(f"{step.model.upper()} {step.id}: {step.title}")
    lines += neat_detail(step, rows) if step.model == "neat" else ppo_detail(step, rows)
    return "\n".join(lines)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default=None, help="Show this step instead of the queue's running one.")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--every", type=float, default=20.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        text = render(args.step, args.rows)
        if args.once:
            print(text)
            return
        print("\033[2J\033[H" + text + "\n\n(Ctrl+C closes this view; training keeps running)", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
