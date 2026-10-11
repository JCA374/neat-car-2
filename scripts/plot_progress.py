"""
Generate PPO training progress plots from Stable-Baselines3 monitor logs.

By default, reads all monitor CSVs under outputs/logs/monitor/ and writes
outputs/plots/ppo_progress.png with episode rewards and lengths (plus rolling
averages).
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def _load_monitor_csvs(log_dir: str) -> pd.DataFrame:
    paths: List[str] = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No monitor CSVs found in {log_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, skiprows=1, names=["r", "l", "t"])
        df[["r", "l", "t"]] = df[["r", "l", "t"]].apply(pd.to_numeric, errors="coerce")
        df["source"] = os.path.basename(path)
        frames.append(df.dropna())

    if not frames:
        raise ValueError(f"No numeric rows found in monitor CSVs under {log_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values("t").reset_index(drop=True)
    return df_all


def plot_progress(df: pd.DataFrame, out_path: str, window: int = 20) -> None:
    df["reward_ma"] = df["r"].rolling(window, min_periods=1).mean()
    df["len_ma"] = df["l"].rolling(window, min_periods=1).mean()

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(df["t"], df["r"], alpha=0.3, label="episode reward")
    ax1.plot(df["t"], df["reward_ma"], label=f"reward (rolling {window})")
    ax1.set_ylabel("Reward")
    ax1.legend()

    ax2.plot(df["t"], df["l"], alpha=0.3, label="episode length")
    ax2.plot(df["t"], df["len_ma"], label=f"length (rolling {window})")
    ax2.set_xlabel("Env timesteps")
    ax2.set_ylabel("Episode length")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO progress from monitor logs.")
    parser.add_argument(
        "--log-dir",
        default="outputs/logs/monitor",
        help="Directory containing monitor CSVs.",
    )
    parser.add_argument(
        "--out",
        default="outputs/plots/ppo_progress.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size for moving averages.",
    )
    return parser.parse_args()


def main() -> int:
    # Use a non-interactive backend to support headless environments.
    matplotlib.use("Agg")
    args = parse_args()
    df = _load_monitor_csvs(args.log_dir)
    plot_progress(df, args.out, window=args.window)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
