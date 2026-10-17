"""The rulebook: what is the same for every model, NEAT or PPO.

Each method may train its own way (reward shaping, hyperparameters, network), but these
rules and the measurement are shared, so lap times and training times are comparable.
"""

from __future__ import annotations

import numpy as np

# Tracks (CarRacing track seeds). Nothing trains on validation, championship or exhibition tracks.
TRAIN_SEED_RANGE = (1_000, 1_000_000)         # training tracks are drawn from here
VALIDATION_SEEDS = [101, 202, 303, 404, 505]  # choosing which checkpoint to keep
BENCHMARK_SEEDS = [3, 11, 21, 34, 55]         # the championship / ladder benchmark
EXHIBITION_SEEDS = list(range(700, 725))      # extra races on the Championship page, no points
RESERVED_SEEDS = set(VALIDATION_SEEDS) | set(BENCHMARK_SEEDS) | set(EXHIBITION_SEEDS)

# Completing a track: CarRacing's own rule (every road tile touched), within this many frames.
MAX_FRAMES = 2000          # 40 s at 50 fps; training episodes, lap checks and the benchmark alike
FPS = 50

# Controls: the same action smoothing for every new model (blend with the previous action),
# applied to the final steer/gas/brake every frame. How a model's outputs become those three
# controls, and how often it decides (carracing/controls.py), is the model's own choice, like its network.
ACTION_SMOOTHING = 0.1

# Model selection and budget.
LAP_CHECK_PPO_STEPS = 25_000   # PPO lap check interval (NEAT checks every generation, ~1-2 min)
BUDGET_HOURS = 8               # championship cars, wall-clock on this laptop, all threads available


def training_seed(rng: np.random.Generator) -> int:
    """A training track: random, and never one of the reserved tracks."""
    while True:
        seed = int(rng.integers(*TRAIN_SEED_RANGE))
        if seed not in RESERVED_SEEDS:
            return seed


def smooth(action: np.ndarray, previous: np.ndarray, blend: float) -> np.ndarray:
    """Blend an action with the previous one and clip to CarRacing's control ranges."""
    action = np.asarray(action, dtype=np.float32)
    if blend > 0.0:
        action = blend * previous + (1.0 - blend) * action
    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    return action
