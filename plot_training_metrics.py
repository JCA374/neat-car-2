"""
Plot training metrics (not just episode rewards) from Stable-Baselines3 TensorBoard logs.

Reads all event files under a log directory (default: outputs/logs/) and writes
PNG plots to outputs/plots/training_metrics.png showing:
- rollout/ep_rew_mean and ep_len_mean
- train/approx_kl, clip_fraction, entropy_loss
- train/value_loss and policy_gradient_loss
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def _load_scalars(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    event_files = sorted(glob.glob(os.path.join(log_dir, "**", "events.*"), recursive=True))
    if not event_files:
        return {}

    data: Dict[str, List[Tuple[int, float]]] = {}
    for path in event_files:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if not events:
                continue
            series = data.setdefault(tag, [])
            series.extend((e.step, e.value) for e in events)

    # Deduplicate per tag by step (keep latest value for each step)
    for tag, series in data.items():
        series.sort(key=lambda x: x[0])
        dedup: Dict[int, float] = {}
        for step, val in series:
            dedup[step] = val
        data[tag] = sorted(dedup.items())
    return data


def _plot_series(ax, series: List[Tuple[int, float]], label: str, **kwargs):
    if not series:
        return
    xs, ys = zip(*series)
    ax.plot(xs, ys, label=label, **kwargs)


def plot_metrics(data: Dict[str, List[Tuple[int, float]]], out_path: str) -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Rewards/lengths
    _plot_series(axes[0], data.get("rollout/ep_rew_mean", []), "ep_rew_mean")
    _plot_series(axes[0], data.get("rollout/ep_len_mean", []), "ep_len_mean")
    axes[0].set_ylabel("Reward / Ep length")
    axes[0].legend()

    # KL/clip/entropy
    _plot_series(axes[1], data.get("train/approx_kl", []), "approx_kl")
    _plot_series(axes[1], data.get("train/clip_fraction", []), "clip_fraction")
    _plot_series(axes[1], data.get("train/entropy_loss", []), "entropy_loss")
    axes[1].set_ylabel("Policy stats")
    axes[1].legend()

    # Losses
    _plot_series(axes[2], data.get("train/value_loss", []), "value_loss")
    _plot_series(axes[2], data.get("train/policy_gradient_loss", []), "policy_gradient_loss")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Losses")
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO training metrics from TensorBoard logs.")
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory containing TensorBoard event files.")
    parser.add_argument(
        "--out",
        default="outputs/plots/training_metrics.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> int:
    matplotlib.use("Agg")
    args = parse_args()
    data = _load_scalars(args.log_dir)
    if not data:
        print(f"No TensorBoard event files found under {args.log_dir}; skipping plot.")
        return 0
    plot_metrics(data, args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
