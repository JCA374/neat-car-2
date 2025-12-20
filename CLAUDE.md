# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CPU-only NEAT (NeuroEvolution of Augmenting Topologies) implementation for training agents on Gymnasium's CarRacing-v2 environment. The project uses a **state-based observation** (compact vector) instead of raw pixels to make NEAT training feasible on CPU.

## Development Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Basic training run
python -m carracing_neat.train --generations 10 --num-workers 2 --max-steps 500 --episodes 1

# Resume from checkpoint (automatic)
python -m carracing_neat.train --resume auto

# Resume from specific checkpoint
python -m carracing_neat.train --resume outputs/checkpoints/neat-checkpoint-50

# Start fresh (ignore existing checkpoints)
python -m carracing_neat.train --resume ""

# With domain randomization
python -m carracing_neat.train --domain-randomize
```

### Visualization
```bash
# Generate all visualizations (fitness + network evolution)
python -m carracing_neat.viz --all-checkpoints

# Just the fitness plot
python -m carracing_neat.viz --log outputs/logs/fitness.csv --out outputs/plots/fitness.png

# Visualize a specific genome
python -m carracing_neat.viz \
  --network-pkl outputs/checkpoints/best-genome-final.pkl \
  --network-config neat_config/neat.cfg \
  --network-out outputs/plots/network.png
```

The `--all-checkpoints` flag generates network topology visualizations for every generation, showing how NEAT evolves the network structure over time. Each visualization shows:
- Input nodes (left, blue): Labeled with sensor names
- Hidden nodes (middle, green): Internal representation
- Output nodes (right, red): Labeled with action names
- Connections: Blue for positive weights, red for negative

### Replay/Testing
```bash
# Headless replay
python -m carracing_neat.play --genome outputs/checkpoints/best-genome-final.pkl --render-mode None

# Visual replay (opens window)
python -m carracing_neat.play --genome outputs/checkpoints/best-genome-final.pkl --render-mode human

# With domain randomization
python -m carracing_neat.play --genome outputs/checkpoints/best-genome-final.pkl --domain-randomize
```

## Architecture Overview

### State Representation (`carracing_neat/env_state.py`)

The core innovation is the **CarRacingStateWrapper** that converts pixel observations into a fixed 20-dimensional state vector:

- **8 car dynamics features**:
  - `speed_norm`: normalized speed
  - `local_vx`, `local_vy`: velocities in car's local frame
  - `cos(heading)`, `sin(heading)`: orientation
  - `angular_velocity`: rotation rate
  - `mean_steer`: current steering angle
  - `traction_fraction`: wheel traction (0-1)

- **12 forward-focused lidar distances**: Normalized distances (0-1) computed by raycasting against road polygons. Rays span -108° to +108° (forward ¾ circle), providing spatial awareness for forward driving without rear-facing redundancy. The lidar computation:
  1. Extracts road polygons from the environment on reset
  2. Steps along rays from the car's position
  3. Stops when the sample point leaves the road
  4. Normalizes by `lidar_max_dist` (default 45.0)

### Network Architecture

- **Inputs**: 20 (state vector length: 8 dynamics + 12 lidar, defined in [neat_config/neat.cfg:9](neat_config/neat.cfg#L9))
- **Outputs**: 3 continuous actions (steer, gas, brake)
- **Topology**: Feed-forward, starts fully connected with 60 initial connections (20 inputs × 3 outputs)
- **Activation**: tanh for all nodes
- **Evolution**: NEAT adds/removes nodes and connections via mutation

### Action Decoding (`carracing_neat/eval.py`)

Network outputs are mapped to CarRacing's action space:
```python
steer = tanh(output[0])                    # [-1, 1]
gas = (tanh(output[1]) + 1) / 2           # [0, 1]
brake = (tanh(output[2]) + 1) / 2         # [0, 1]
```

**Action smoothing** reduces jitter: `action_t = α * action_{t-1} + (1-α) * action_raw` (default α=0.2)

**Action noise** aids exploration during training (default std=0.05)

### Training Loop (`carracing_neat/train.py`)

1. **Population initialization**: 30 genomes (configurable in [neat_config/neat.cfg:4](neat_config/neat.cfg#L4))
2. **Parallel evaluation**: Uses multiprocessing to evaluate genomes across CPU cores
3. **Fitness calculation**: Average episode reward minus light shaping penalties:
   - `-0.02 * brake_action` (discourage excessive braking)
   - `-0.01 * |steer_action|` (discourage jitter)
4. **Evolution**: NEAT applies mutation, crossover, and speciation
5. **Checkpointing**: Saves every N generations (default 5) and best genome continuously

### Important Implementation Details

**Environment caching**: Each worker process caches one environment instance to avoid reconstruction overhead (see `_WORKER_ENVS` in [carracing_neat/eval.py:22](carracing_neat/eval.py#L22))

**Checkpoint resumption**: The trainer automatically finds and resumes from the newest checkpoint in `outputs/checkpoints/` when `--resume auto` is used. Log files are truncated to maintain monotonic generation numbers. **Important**: After changing sensor configuration, old checkpoints are incompatible - use `--resume ""` to start fresh.

**State size validation**: On startup, the trainer validates that the environment's state vector length matches `num_inputs` in the NEAT config to catch configuration mismatches early.

**Lidar geometry extraction**: Road polygons are extracted from `env.unwrapped.road` or `env.unwrapped.road_poly` (version-dependent). The wrapper handles both tuple formats `(vertices, color)` and raw vertex lists.

**Sensor configuration**: 12 forward-focused lidar rays (-108° to +108°) provide adequate spatial awareness for track driving while reducing network complexity by 38% compared to 24-sensor full-circle configuration.

## Configuration

[neat_config/neat.cfg](neat_config/neat.cfg) defines all NEAT hyperparameters:
- Population size: 30
- Mutation rates: node_add=0.08, conn_add=0.2, weight_mutate=0.8
- Speciation threshold: 3.0
- Elitism: top 2 genomes preserved each generation

When tuning, increase `max_steps` and `episodes` before increasing population size to improve signal quality without excessive compute cost.

## Output Files

**Checkpoints:**
- `outputs/checkpoints/neat-checkpoint-N`: Full population snapshots every N generations
- `outputs/checkpoints/best-genome.pkl`: Best genome seen so far (updated continuously)
- `outputs/checkpoints/best-genome-gen-N.pkl`: Best genome from generation N (for evolution visualization)
- `outputs/checkpoints/best-genome-final.pkl`: Winner genome after training completes

**Logs:**
- `outputs/logs/fitness.csv`: Per-generation best/mean/stdev fitness (generations start from checkpoint)
- `outputs/logs/genome_stats.csv`: Best genome's node and connection counts per generation

**Visualizations:**
- `outputs/plots/fitness.png`: Training progress plot (generations normalized to start from 0)
- `outputs/plots/networks/network_gen_N.png`: Network topology for each generation
- Network visualizations use a left-to-right layout with fixed y-positions for easy comparison across generations

## Common Modifications

**Changing state vector size**:
1. Update `lidar_rays` in `CarRacingStateWrapper.__init__()` (default 12)
2. Update `num_inputs` in [neat_config/neat.cfg:9](neat_config/neat.cfg#L9) to match `8 + lidar_rays`
3. Update visualization labels in [carracing_neat/viz.py:191](carracing_neat/viz.py#L191) to match sensor count
4. Start fresh training with `--resume ""` after config changes

**Adjusting lidar range/resolution**:
- `lidar_max_dist`: maximum ray distance (default 45.0)
- `lidar_step`: ray sample spacing (default 3.0, smaller = more accurate but slower)
- `lidar_angles`: computed in `CarRacingStateWrapper.__init__()` as `linspace(-0.6π, 0.6π, lidar_rays)` (forward ¾ circle)
  - For more rear coverage: increase to `-0.7π, 0.7π` or `-0.8π, 0.8π`
  - For narrower forward focus: decrease to `-0.5π, 0.5π`

**Fitness shaping**: Modify penalties in `evaluate_genome()` at [carracing_neat/eval.py:101-102](carracing_neat/eval.py#L101-L102)

**Network topology**: To start with hidden nodes or change connection pattern, modify `num_hidden` and `initial_connection` in [neat_config/neat.cfg:11-13](neat_config/neat.cfg#L11-L13)
