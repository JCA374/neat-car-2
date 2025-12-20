"""Evaluation utilities for NEAT + CarRacing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

# Allow running as a script without installing the package.
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import neat
import numpy as np

from carracing_neat.env_state import CarRacingStateWrapper, make_env

# Cache one environment per OS process to avoid repeated construction overhead.
_WORKER_ENVS: Dict[int, CarRacingStateWrapper] = {}


@dataclass
class EvalSettings:
    """Lightweight configuration for evaluating a genome."""

    max_steps: int = 800
    episodes: int = 2
    action_noise: float = 0.0
    action_smoothing: float = 0.1
    domain_randomize: bool = False
    render_mode: Optional[str] = None
    seed: Optional[int] = None


def _decode_action(output: np.ndarray, prev_action: np.ndarray, smoothing: float) -> np.ndarray:
    """Map network outputs to CarRacing's continuous control range."""
    steer_raw = math.tanh(output[0])
    gas_raw = (math.tanh(output[1]) + 1.0) / 2.0
    brake_raw = (math.tanh(output[2]) + 1.0) / 2.0

    action = np.array([steer_raw, gas_raw, brake_raw], dtype=np.float32)
    if smoothing > 0.0:
        blend_prev = float(np.clip(smoothing, 0.0, 0.95))
        action = blend_prev * prev_action + (1.0 - blend_prev) * action

    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1] = float(np.clip(action[1], 0.0, 1.0))
    action[2] = float(np.clip(action[2], 0.0, 1.0))
    return action


def _get_env(env_kwargs) -> CarRacingStateWrapper:
    pid = os.getpid()
    env = _WORKER_ENVS.get(pid)
    if env is None:
        env = make_env(**env_kwargs)
        _WORKER_ENVS[pid] = env
    return env


def evaluate_genome(
    genome,
    config: neat.Config,
    env_kwargs: dict,
    settings: EvalSettings,
) -> float:
    """
    Evaluate one genome and return its fitness (higher is better).

    The fitness is the average reward over the requested number of episodes with
    light action smoothing and optional noise for exploration.
    """
    env = _get_env(env_kwargs)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness_total = 0.0
    for episode in range(settings.episodes):
        reset_kwargs = {}
        if settings.seed is not None:
            reset_kwargs["seed"] = settings.seed + episode
        state, _ = env.reset(**reset_kwargs)
        prev_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        episode_reward = 0.0

        for step in range(settings.max_steps):
            output = np.asarray(net.activate(state), dtype=np.float32)
            action = _decode_action(output, prev_action, smoothing=settings.action_smoothing)
            if settings.action_noise > 0.0:
                action += np.random.normal(0.0, settings.action_noise, size=action.shape).astype(np.float32)
                action[0] = float(np.clip(action[0], -1.0, 1.0))
                action[1:] = np.clip(action[1:], 0.0, 1.0)

            prev_action = action
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            # Mild shaping: encourage smooth but forward driving.
            episode_reward -= 0.01 * float(action[2])  # braking penalty
            episode_reward -= 0.01 * float(abs(action[0]))  # steering jitter penalty

            # Forward progress bonus using local forward velocity (state[1] * speed_scale).
            local_forward = float(max(0.0, state[1] * 50.0))
            episode_reward += 0.05 * local_forward

            # Encourage throttle to keep moving.
            episode_reward += 0.01 * float(action[1])

            # Light off-road penalty using traction fraction (state[7]).
            traction = float(state[7]) if len(state) > 7 else 1.0
            if traction < 0.8:
                episode_reward -= 0.05 * (0.8 - traction)

            if terminated or truncated:
                break

        fitness_total += episode_reward

    return fitness_total / float(settings.episodes)
