"""Plotting utilities for monitoring NEAT training progress and topology."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from typing import List, Tuple

import matplotlib

# Use a non-interactive backend so plots work headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import neat  # noqa: E402


# Dummy classes for unpickling checkpoints that were saved with reporter instances
class _DummyReporter:
    """Placeholder for unpickling checkpoints with reporter instances."""
    def __init__(self, *args, **kwargs):
        pass


# Register dummy classes so pickle can find them
class CSVReporter(_DummyReporter):
    pass


class BestGenomeSaver(_DummyReporter):
    pass


class GenomeStatsReporter(_DummyReporter):
    pass


# Make these available in the module for unpickling
sys.modules['carracing_neat.train'] = sys.modules[__name__]


def _read_log(csv_path: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    generations: List[int] = []
    best: List[float] = []
    mean: List[float] = []
    stdev: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["generation"]))
            best.append(float(row["best_fitness"]))
            mean.append(float(row["mean_fitness"]))
            stdev.append(float(row["stdev"]))
    return generations, best, mean, stdev


def plot_fitness(csv_path: str, output_path: str) -> None:
    generations, best, mean, stdev = _read_log(csv_path)
    if not generations:
        raise RuntimeError(f"No data in log file {csv_path}")

    # Normalize generations to start from 0 for clarity
    gen_offset = min(generations)
    generations_norm = [g - gen_offset for g in generations]

    plt.figure(figsize=(10, 6))
    plt.plot(generations_norm, best, label="Best fitness", linewidth=2, marker='o', markersize=4)
    plt.plot(generations_norm, mean, label="Mean fitness", linestyle="--", linewidth=1.5, marker='s', markersize=3)

    upper = [m + s for m, s in zip(mean, stdev)]
    lower = [m - s for m, s in zip(mean, stdev)]
    plt.fill_between(generations_norm, lower, upper, color="C1", alpha=0.15, label="Mean ± stdev")

    plt.xlabel("Generation (relative)", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title("NEAT CarRacing Training Progress", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    if gen_offset > 0:
        print(f"Note: Generations normalized (actual range: {min(generations)}-{max(generations)})")


def _layered_positions(genome: neat.DefaultGenome, config: neat.Config):
    """Compute layered positions with inputs on left, outputs on right, hidden in middle."""
    inputs = list(config.genome_config.input_keys)
    outputs = list(config.genome_config.output_keys)
    enabled_conns = [
        (a, b)
        for (a, b), cg in genome.connections.items()
        if cg.enabled
    ]

    # Get all hidden nodes
    hidden = list(set(genome.nodes.keys()) - set(inputs) - set(outputs))

    # Try to determine layers using NEAT's built-in function
    try:
        layers = neat.graphs.feed_forward_layers(inputs, outputs, enabled_conns)
        # Separate hidden layers (everything except first and last)
        if len(layers) > 2:
            hidden_layers = layers[1:-1]
        else:
            hidden_layers = [hidden] if hidden else []
    except Exception:
        # Fallback: all hidden in one layer
        hidden_layers = [hidden] if hidden else []

    pos = {}

    # Fixed y-positions for inputs (left side, x=0)
    input_ys = np.linspace(1, -1, num=len(inputs)) if len(inputs) > 1 else [0]
    for i, node in enumerate(sorted(inputs)):
        pos[node] = (0, input_ys[i])

    # Fixed y-positions for outputs (right side, x=3)
    output_ys = np.linspace(0.5, -0.5, num=len(outputs)) if len(outputs) > 1 else [0]
    for i, node in enumerate(sorted(outputs)):
        pos[node] = (3, output_ys[i])

    # Hidden nodes in the middle (x=1 to 2, distributed by layer)
    if hidden_layers:
        num_hidden_layers = len(hidden_layers)
        for layer_idx, layer in enumerate(hidden_layers):
            if not layer:
                continue
            x = 1.0 + (layer_idx / max(num_hidden_layers - 1, 1))
            ys = np.linspace(1, -1, num=len(layer)) if len(layer) > 1 else [0]
            for y, node in zip(ys, sorted(layer)):
                pos[node] = (x, y)

    # Ensure any stray nodes are placed
    for node in set(genome.nodes.keys()) | set(inputs) | set(outputs):
        if node not in pos:
            pos[node] = (1.5, -1.5)

    return pos, enabled_conns, inputs, outputs


def plot_network(genome: neat.DefaultGenome, config: neat.Config, output_path: str, title: str = "NEAT Network Topology") -> None:
    pos, enabled_conns, inputs, outputs = _layered_positions(genome, config)

    # Count nodes and connections
    hidden = set(genome.nodes.keys()) - set(inputs) - set(outputs)
    num_enabled = len(enabled_conns)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    colors = {}
    for n in pos:
        if n in inputs:
            colors[n] = "#3498db"  # Blue for inputs
        elif n in outputs:
            colors[n] = "#e74c3c"  # Red for outputs
        else:
            colors[n] = "#2ecc71"  # Green for hidden

    # Draw connections with varying opacity based on weight
    for (a, b) in enabled_conns:
        conn = genome.connections.get((a, b))
        weight = conn.weight if conn else 0.0
        abs_weight = abs(weight)
        alpha = min(0.3 + abs_weight * 0.1, 0.8)
        color = "red" if weight < 0 else "blue"

        # Draw arrow
        ax.annotate(
            "",
            xy=pos[b],
            xytext=pos[a],
            arrowprops=dict(arrowstyle="->", alpha=alpha, lw=1.5, color=color),
        )

        # Add weight label on connection (only for significant weights)
        if abs_weight > 0.5:  # Only show weights > 0.5
            mid_x = (pos[a][0] + pos[b][0]) / 2
            mid_y = (pos[a][1] + pos[b][1]) / 2
            ax.text(mid_x, mid_y, f"{weight:.1f}",
                   fontsize=5, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                   zorder=2)

    # Draw nodes
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=400, color=colors.get(n, "C3"), zorder=3, edgecolors='black', linewidths=1.5)
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=8, fontweight='bold', zorder=4)

    # Add labels for inputs and outputs with fixed positions
    input_labels = [
        "Speed (norm)", "Vx (local)", "Vy (local)", "cos(θ)", "sin(θ)",
        "ω (ang.vel)", "Steer (cur)", "Traction"
    ] + [f"L{i} (lidar)" for i in range(12)]  # 8 dynamics + 12 lidar

    output_labels = ["Steer (out)", "Gas (out)", "Brake (out)"]

    for i, node in enumerate(sorted(inputs)):
        x, y = pos[node]
        label = input_labels[i] if i < len(input_labels) else f"In{i}"
        ax.text(x - 0.35, y, label, ha="right", va="center", fontsize=6.5, style='italic')

    for i, node in enumerate(sorted(outputs)):
        x, y = pos[node]
        label = output_labels[i] if i < len(output_labels) else f"Out{i}"
        ax.text(x + 0.3, y, label, ha="left", va="center", fontsize=9, fontweight='bold')

    # Add layer labels
    ax.text(0, 1.4, "INPUTS", ha="center", va="bottom", fontsize=11, fontweight='bold', color='#3498db')
    ax.text(3, 0.9, "OUTPUTS", ha="center", va="bottom", fontsize=11, fontweight='bold', color='#e74c3c')
    if hidden:
        ax.text(1.5, 1.4, "HIDDEN", ha="center", va="bottom", fontsize=11, fontweight='bold', color='#2ecc71')

    ax.set_xlim(-0.8, 3.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_axis_off()

    # Add info text
    info_text = f"Nodes: {len(inputs)} inputs + {len(hidden)} hidden + {len(outputs)} outputs | Connections: {num_enabled}"
    plt.title(f"{title}\n{info_text}", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def _load_genome(genome_path: str):
    with open(genome_path, "rb") as f:
        return pickle.load(f)


def visualize_all_checkpoints(checkpoint_dir: str, config_path: str, output_dir: str) -> None:
    """Generate network visualizations for all saved best genomes."""
    import glob as glob_module
    import re

    # Look for best-genome-gen-N.pkl files
    genome_pattern = os.path.join(checkpoint_dir, "best-genome-gen-*.pkl")
    genome_files = sorted(glob_module.glob(genome_pattern),
                          key=lambda x: int(re.search(r'gen-(\d+)', x).group(1)) if re.search(r'gen-(\d+)', x) else 0)

    if not genome_files:
        print(f"No per-generation genome files found in {checkpoint_dir}")
        print("These files are generated during training with the updated train.py")
        return

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(genome_files)} generation genome files. Generating visualizations...")

    for genome_path in genome_files:
        # Extract generation number from filename
        basename = os.path.basename(genome_path)
        match = re.search(r'best-genome-gen-(\d+)\.pkl', basename)
        if not match:
            continue
        gen_num = match.group(1)

        try:
            # Load the genome
            genome = _load_genome(genome_path)

            if genome.fitness is None:
                print(f"  Skipping generation {gen_num} (no fitness)")
                continue

            # Generate visualization
            output_path = os.path.join(output_dir, f"network_gen_{gen_num}.png")
            title = f"Generation {gen_num} - Best Fitness: {genome.fitness:.2f}"
            plot_network(genome, config, output_path, title=title)
            print(f"  ✓ Generated {output_path}")

        except Exception as e:
            print(f"  ✗ Failed to process generation {gen_num}: {e}")

    print(f"\nAll visualizations saved to {output_dir}/")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot fitness curves and network topologies from NEAT training.")
    parser.add_argument("--log", default="outputs/logs/fitness.csv", help="Input CSV log path.")
    parser.add_argument("--out", default="outputs/plots/fitness.png", help="Output image path.")
    parser.add_argument(
        "--network-pkl",
        help="Path to a pickled genome (e.g., outputs/checkpoints/best-genome-final.pkl) to render.",
    )
    parser.add_argument(
        "--network-config",
        default="neat_config/neat.cfg",
        help="Path to NEAT config matching the genome.",
    )
    parser.add_argument(
        "--network-out",
        default="outputs/plots/network.png",
        help="Output path for the network topology image.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Generate network visualizations for ALL checkpoints in checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
        help="Directory containing NEAT checkpoints.",
    )
    parser.add_argument(
        "--networks-dir",
        default="outputs/plots/networks",
        help="Directory for checkpoint network visualizations.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Generate fitness plot
    plot_fitness(args.log, args.out)
    print(f"✓ Saved fitness plot to {args.out}")

    # Generate all checkpoint visualizations if requested
    if args.all_checkpoints:
        visualize_all_checkpoints(args.checkpoint_dir, args.network_config, args.networks_dir)

    # Generate single network visualization if specified
    if args.network_pkl:
        genome = _load_genome(args.network_pkl)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            args.network_config,
        )
        plot_network(genome, config, args.network_out)
        print(f"✓ Saved network plot to {args.network_out}")


if __name__ == "__main__":
    raise SystemExit(main())
