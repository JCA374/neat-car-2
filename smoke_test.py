"""
Lightweight smoke test to confirm the CarRacing NEAT pipeline runs headlessly.

Checks:
- Imports all project modules.
- Instantiates the wrapped CarRacing env and verifies state shape.
- Runs a few random steps to ensure the wrapper does not crash.
- Runs a tiny genome evaluation (10 steps) to validate config + evaluator wiring.
"""

from __future__ import annotations

import math
import os
import random

import neat
import numpy as np

from carracing_neat.env_state import make_env
from carracing_neat.eval import EvalSettings, evaluate_genome


def assert_close(val, expected, name):
    if val != expected:
        raise AssertionError(f"{name}: expected {expected}, got {val}")


def test_env_basic():
    env = make_env(render_mode=None, domain_randomize=False, seed=123)
    state, _ = env.reset()
    assert len(state) == env.observation_space.shape[0] == 40
    for _ in range(5):
        action = np.array(
            [
                random.uniform(-1.0, 1.0),  # steer
                random.uniform(0.0, 1.0),  # gas
                random.uniform(0.0, 1.0),  # brake
            ],
            dtype=np.float32,
        )
        state, reward, terminated, truncated, _ = env.step(action)
        assert math.isfinite(reward)
        assert len(state) == 40
        if terminated or truncated:
            break
    env.close()


def test_eval_tiny():
    cfg_path = os.path.join("neat_config", "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )
    # Ensure config input size matches wrapper output.
    env = make_env(render_mode=None, domain_randomize=False, seed=0)
    state, _ = env.reset()
    assert_close(len(state), config.genome_config.num_inputs, "state_size_vs_config")
    env.close()

    # Build a fresh genome and run a minimal evaluation.
    genome = neat.DefaultGenome(0)
    genome.configure_new(config.genome_config)
    fitness = evaluate_genome(
        genome,
        config,
        env_kwargs={"render_mode": None, "domain_randomize": False},
        settings=EvalSettings(max_steps=10, episodes=1, action_noise=0.0, action_smoothing=0.1),
    )
    assert math.isfinite(fitness)


if __name__ == "__main__":
    test_env_basic()
    test_eval_tiny()
    print("Smoke test passed.")
