"""Evaluate NEAT genomes on CarRacing using the shared state wrapper."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import neat
import numpy as np

from carracing_ppo.env import CarRacingStateWrapper, make_env

# Cache one environment per OS process to avoid repeated construction overhead.
_WORKER_ENVS: Dict[Tuple, CarRacingStateWrapper] = {}


@dataclass
class EvalSettings:
    """How a genome is scored.

    Fitness is the mean CarRacing reward over ``seeds`` (the same tracks for every
    genome in a generation), optionally with the wrapper's off-track penalty.
    """

    max_steps: int = 1000
    action_smoothing: float = 0.1
    offtrack_penalty: float = 0.0
    traction_threshold: float = 0.95
    domain_randomize: bool = False
    # Stop an episode early when no new tile is reached for this many steps.
    patience: int = 100
    seeds: Sequence[int] = field(default_factory=lambda: [0])


def decode_action(output: Sequence[float], prev_action: np.ndarray, smoothing: float) -> np.ndarray:
    """Map network outputs to CarRacing's continuous control range."""
    action = np.array(
        [
            math.tanh(output[0]),
            (math.tanh(output[1]) + 1.0) / 2.0,
            (math.tanh(output[2]) + 1.0) / 2.0,
        ],
        dtype=np.float32,
    )
    if smoothing > 0.0:
        blend = float(np.clip(smoothing, 0.0, 0.95))
        action = blend * prev_action + (1.0 - blend) * action
    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    return action


def _get_env(settings: EvalSettings, render_mode: Optional[str] = None) -> CarRacingStateWrapper:
    key = (os.getpid(), render_mode, settings.offtrack_penalty, settings.traction_threshold,
           settings.domain_randomize, settings.max_steps)
    env = _WORKER_ENVS.get(key)
    if env is None:
        env = make_env(
            render_mode=render_mode,
            domain_randomize=settings.domain_randomize,
            offtrack_penalty=settings.offtrack_penalty,
            traction_threshold=settings.traction_threshold,
            allow_multiple_laps=False,
            max_episode_steps=settings.max_steps,
        )
        _WORKER_ENVS[key] = env
    return env


def run_episode(net, settings: EvalSettings, seed: int, render_mode: Optional[str] = None,
                stop_when_stuck: bool = True) -> Dict:
    """Drive one track. Returns reward, steps, laps and (if rendering) frames."""
    env = _get_env(settings, render_mode)
    state, _ = env.reset(seed=seed)
    prev_action = np.zeros(3, dtype=np.float32)
    total, steps, since_tile, laps = 0.0, 0, 0, 0
    frames: List[np.ndarray] = []
    for steps in range(1, settings.max_steps + 1):
        action = decode_action(net.activate(state), prev_action, settings.action_smoothing)
        prev_action = action
        tiles_before = env.unwrapped.tile_visited_count
        state, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        laps = info.get("laps_completed", laps)
        if render_mode == "rgb_array":
            frames.append(env.render())
        since_tile = 0 if env.unwrapped.tile_visited_count > tiles_before else since_tile + 1
        if terminated or truncated:
            break
        if stop_when_stuck and since_tile >= settings.patience:
            # A parked or lost car only bleeds -0.1/step from here; charge the rest
            # of the time penalty so stopping early never beats driving on.
            total -= 0.1 * (settings.max_steps - steps)
            break
    return {"reward": total, "steps": steps, "laps": laps, "frames": frames}


def evaluate_genome(genome, config: neat.Config, settings: EvalSettings) -> float:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rewards = [run_episode(net, settings, seed)["reward"] for seed in settings.seeds]
    return float(np.mean(rewards))
