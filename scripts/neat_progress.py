"""
Live summary of a NEAT run: per-generation scores, species and network size.

    python scripts/neat_progress.py                      # outputs/neat_v2/logs, refresh every 20 s
    python scripts/neat_progress.py --log-dir outputs/neat/logs --once
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else pd.DataFrame()


def render(log_dir: str, rows: int) -> str:
    fit = _read(os.path.join(log_dir, "fitness.csv"))
    val = _read(os.path.join(log_dir, "validation.csv"))
    spc = _read(os.path.join(log_dir, "species.csv"))
    if fit.empty:
        return f"Waiting for the first generation in {log_dir} ..."

    table = fit[["generation", "best_fitness", "mean_fitness"]].copy()
    if not val.empty:
        table = table.merge(val[["generation", "validation_score", "hidden_nodes", "enabled_connections"]],
                            on="generation", how="left")
    if not spc.empty:
        table = table.merge(spc[["generation", "species", "threshold"]], on="generation", how="left")
    seconds = fit["time_s"].diff().tail(5).mean()

    lines = [f"NEAT progress  ({log_dir})   {time.strftime('%H:%M:%S')}", ""]
    if not val.empty:
        best = val.loc[val["validation_score"].idxmax()]
        lines.append(f"Best so far: {best['validation_score']:.1f} on the 5 held-out tracks "
                     f"(generation {int(best['generation'])}, {int(best['hidden_nodes'])} hidden nodes)")
    lines.append(f"Generations done: {len(fit)}   last: {int(fit['generation'].iloc[-1])}   "
                 f"~{seconds / 60:.1f} min per generation" if pd.notna(seconds) else f"Generations done: {len(fit)}")
    lines.append("")
    names = {"generation": "gen", "best_fitness": "train best", "mean_fitness": "train mean",
             "validation_score": "validation", "hidden_nodes": "hidden", "enabled_connections": "conns",
             "species": "species", "threshold": "threshold"}
    lines.append(table.tail(rows).rename(columns=names).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs/neat_v2/logs")
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--every", type=float, default=20.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        text = render(args.log_dir, args.rows)
        if args.once:
            print(text)
            return
        print("\033[2J\033[H" + text + "\n\n(Ctrl+C closes this view; training keeps running)", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
