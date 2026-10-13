"""Evaluate NEAT genomes on CarRacing using the shared state wrapper."""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import neat
import numpy as np

from carracing.env import CarRacingStateWrapper, make_env

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
    # Genomes evolved before Oct 12 2026 expect the old double-tanh decoding.
    legacy_decode: bool = False
    lidar_rays: int = 32
    lidar_fov: Optional[float] = None
    drop_heading: bool = False
    # Reward per frame a lap finishes under 1000 frames (0 = raw score only).
    lap_time_bonus: float = 0.0
    seeds: Sequence[int] = field(default_factory=lambda: [0])


def decode_action(output: Sequence[float], prev_action: np.ndarray, smoothing: float,
                  legacy: bool = False) -> np.ndarray:
    """Map network outputs (already tanh, so in [-1, 1]) to CarRacing controls.

    ``legacy`` reproduces the original decoding, which applied tanh a second
    time: that capped steering at 0.76 and gas at 0.881 and left the brake at
    0.119 or more on every frame.
    """
    o = [math.tanh(v) for v in output[:3]] if legacy else output[:3]
    action = np.array([o[0], (o[1] + 1.0) / 2.0, (o[2] + 1.0) / 2.0], dtype=np.float32)
    if smoothing > 0.0:
        blend = float(np.clip(smoothing, 0.0, 0.95))
        action = blend * prev_action + (1.0 - blend) * action
    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    return action


def _get_env(settings: EvalSettings, render_mode: Optional[str] = None) -> CarRacingStateWrapper:
    key = (os.getpid(), render_mode, settings.offtrack_penalty, settings.traction_threshold,
           settings.domain_randomize, settings.max_steps, settings.lidar_rays, settings.lap_time_bonus, settings.lidar_fov,
           settings.drop_heading)
    env = _WORKER_ENVS.get(key)
    if env is None:
        env = make_env(
            render_mode=render_mode,
            domain_randomize=settings.domain_randomize,
            offtrack_penalty=settings.offtrack_penalty,
            traction_threshold=settings.traction_threshold,
            allow_multiple_laps=False,
            max_episode_steps=settings.max_steps,
            lidar_rays=settings.lidar_rays,
            lidar_fov=settings.lidar_fov,
            drop_heading=settings.drop_heading,
            lap_time_bonus=settings.lap_time_bonus,
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
        action = decode_action(net.activate(state), prev_action, settings.action_smoothing, settings.legacy_decode)
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


def load_config(config_path: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )


def load_genome(path: str, config_path: str):
    with open(path, "rb") as f:
        genome = pickle.load(f)
    return genome, load_config(config_path)
