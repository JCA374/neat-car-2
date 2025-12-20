# NEAT CarRacing (CPU-only)

This repo wraps Gymnasium's `CarRacing-v2` with a compact state vector (dynamics + lidar-style distances) and trains a NEAT controller. It is CPU-only and keeps the per-generation workload low by default.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m carracing_neat.train \
  --generations 10 \
  --num-workers 2 \
  --max-steps 500 \
  --episodes 1
```

Key files:

- `outputs/checkpoints/`: NEAT checkpoints every N generations (`neat-checkpoint-*`) plus the latest/best genome pickles.
- `outputs/logs/fitness.csv`: per-generation best/mean/std fitness for monitoring.
- `outputs/logs/genome_stats.csv`: per-generation best-genome node and enabled-connection counts.
- `outputs/plots/fitness.png`: optional plot produced with the command below.
- `outputs/plots/network.png`: optional network topology image of a saved genome.

The trainer resumes from the newest checkpoint by default (`--resume auto`). Pass `--resume /path/to/ckpt` to pick a specific checkpoint or `--resume ""` to start fresh.

## Visualize Training Progress

```bash
# Generate fitness plot and all network visualizations
python -m carracing_neat.viz --all-checkpoints

# Just the fitness plot
python -m carracing_neat.viz --log outputs/logs/fitness.csv --out outputs/plots/fitness.png

# Visualize a specific genome
python -m carracing_neat.viz \
  --network-pkl outputs/checkpoints/best-genome-final.pkl \
  --network-config neat_config/neat.cfg \
  --network-out outputs/plots/network.png
```

The visualization generates:
- `outputs/plots/fitness.png`: Training progress with best/mean fitness over generations
- `outputs/plots/networks/network_gen_N.png`: Network topology for each generation showing NEAT evolution

Network visualizations show:
- **Inputs** (left, blue): 8 car dynamics + 12 forward-focused lidar sensors (L0-L11) with labels
- **Hidden nodes** (middle, green): Evolved internal representation
- **Outputs** (right, red): Steer, Gas, Brake (labeled)
- Connection colors: Blue = positive weight, Red = negative weight

## Replay Best Genome

```bash
# Visual replay (opens window)
python -m carracing_neat.play \
  --config neat_config/neat.cfg \
  --max-steps 1000 \
  --render-mode human

# Headless evaluation
python -m carracing_neat.play --render-mode None
```

`play.py` will search `outputs/checkpoints` in order: `best-genome-final.pkl`, `best-genome.pkl`, then the latest `best-genome-gen-*.pkl`, so you usually don’t need to pass `--genome`.

## Notes

- State vector: 8 car-dynamics scalars + 12 forward-focused lidar distances derived from road polygons (length 20, matching `neat_config/neat.cfg`).
- Lidar sensors span -108° to +108° (forward ¾ circle), providing spatial awareness without rear-facing redundancy.
- Defaults favor low compute (population 30, 1 episode/genome, 500 steps). Increase gradually once the run is stable.
- Action smoothing and light noise are applied during evaluation to keep control stable and exploratory without using a GPU.
