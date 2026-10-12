"""
Lightweight smoke test for the PPO pipeline.

Checks:
- Instantiates the wrapped CarRacing env and verifies state shape.
- Runs a few random steps to ensure the wrapper does not crash.
- Executes a tiny PPO training loop to ensure learning pipeline wiring.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from carracing.env import make_env
from carracing_ppo import train as train_mod


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


def test_tiny_ppo():
    env = DummyVecEnv([lambda: make_env(render_mode=None, domain_randomize=False, seed=321)])
    env = VecMonitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        verbose=0,
        seed=1,
    )
    model.learn(total_timesteps=32)
    env.close()


def test_train_entrypoint():
    # Runs the real train entrypoint with tiny settings to catch missing deps (tensorboard/tqdm/rich).
    args = [
        "--total-timesteps",
        "8",
        "--num-envs",
        "1",
        "--n-steps",
        "8",
        "--batch-size",
        "8",
        "--eval-frequency",
        "0",
        "--checkpoint-frequency",
        "0",
        "--log-dir",
        "outputs/logs_test",
        "--checkpoint-dir",
        "outputs/checkpoints_test",
    ]
    rc = train_mod.main(args)
    assert rc == 0
    # Train and eval envs must log to separate monitor files.
    assert os.path.exists("outputs/logs_test/monitor/0.monitor.csv")
    assert os.path.exists("outputs/logs_test/eval_monitor/0.monitor.csv")


def test_allow_multiple_laps_flag():
    assert train_mod.parse_args([]).allow_multiple_laps is True
    assert train_mod.parse_args(["--no-allow-multiple-laps"]).allow_multiple_laps is False


if __name__ == "__main__":
    test_env_basic()
    test_tiny_ppo()
    test_train_entrypoint()
    test_allow_multiple_laps_flag()
    print("Smoke test passed.")
