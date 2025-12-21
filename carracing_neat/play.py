"""Replay a saved NEAT genome in CarRacing."""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
from typing import Optional, Sequence

import neat
import numpy as np

# Allow running as a script without installing the package.
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from carracing_neat.env_state import make_env


def _decode_action(output: np.ndarray, prev_action: np.ndarray, smoothing: float) -> np.ndarray:
    steer_raw = math.tanh(output[0])
    gas_raw = (math.tanh(output[1]) + 1.0) / 2.0
    brake_raw = (math.tanh(output[2]) + 1.0) / 2.0

    action = np.array([steer_raw, gas_raw, brake_raw], dtype=np.float32)
    if smoothing > 0.0:
        blend_prev = float(np.clip(smoothing, 0.0, 0.95))
        action = blend_prev * prev_action + (1.0 - blend_prev) * action

    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1:] = np.clip(action[1:], 0.0, 1.0)
    return action


def _resolve_genome_path(requested: str, checkpoint_dir: str) -> str:
    """Resolve the genome path, falling back to best-known pickles in a dir."""
    if requested and os.path.exists(requested):
        return requested

    search_order = []
    if checkpoint_dir:
        search_order.extend(
            [
                os.path.join(checkpoint_dir, "best-genome-final.pkl"),
                os.path.join(checkpoint_dir, "best-genome.pkl"),
            ]
        )
        gen_pkls = glob.glob(os.path.join(checkpoint_dir, "best-genome-gen-*.pkl"))
        if gen_pkls:
            gen_pkls.sort(key=lambda p: int(os.path.basename(p).split("-")[-1].split(".")[0]))
            search_order.extend(reversed(gen_pkls))  # highest generation first

    for cand in search_order:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Genome file not found. Tried requested '{requested}' and checkpoints in '{checkpoint_dir}'."
    )


def run_episode(
    genome_path: str,
    config_path: str,
    max_steps: int,
    render_mode: Optional[str],
    action_smoothing: float,
    action_noise: float,
    domain_randomize: bool,
    track_seed: Optional[int],
):

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = make_env(
        render_mode=render_mode,
        domain_randomize=domain_randomize,
        continuous=True,
        seed=track_seed,
    )
    reset_kwargs = {}
    if track_seed is not None:
        reset_kwargs["seed"] = track_seed
    state, _ = env.reset(**reset_kwargs)
    prev_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    total_reward = 0.0

    for step in range(max_steps):
        output = np.asarray(net.activate(state), dtype=np.float32)
        action = _decode_action(output, prev_action, smoothing=action_smoothing)
        if action_noise > 0.0:
            action += np.random.normal(0.0, action_noise, size=action.shape).astype(np.float32)
            action[0] = float(np.clip(action[0], -1.0, 1.0))
            action[1:] = np.clip(action[1:], 0.0, 1.0)

        prev_action = action
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    env.close()
    return total_reward, step + 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play a saved NEAT genome in CarRacing.")
    parser.add_argument(
        "--genome",
        default="outputs/checkpoints/best-genome-final.pkl",
        help="Path to a pickled genome to replay.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
        help="Directory to search for best genome if --genome is not found.",
    )
    parser.add_argument(
        "--config",
        default="neat_config/neat.cfg",
        help="Path to NEAT config matching the genome.",
    )
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument(
        "--render-mode",
        default="human",
        help='Gymnasium render mode ("human", "rgb_array", or None). Default: "human" to display.',
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.2,
        help="Blend factor for previous action [0, 0.95].",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Gaussian action noise (std) for stochastic playback.",
    )
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="Enable Gymnasium domain_randomize flag.",
    )
    parser.add_argument(
        "--track-seed",
        type=int,
        default=123,
        help="Fixed seed to lock the track layout (None = random each run).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    render_mode = args.render_mode
    if isinstance(render_mode, str):
        render_mode_lower = render_mode.lower()
        if render_mode_lower == "none":
            render_mode = None
        elif render_mode_lower in ("human", "rgb_array"):
            render_mode = render_mode_lower
        else:
            print(f"Unrecognized render mode '{render_mode}', defaulting to 'human'.")
            render_mode = "human"

    total_reward, steps = run_episode(
        genome_path=_resolve_genome_path(args.genome, args.checkpoint_dir),
        config_path=args.config,
        max_steps=args.max_steps,
        render_mode=render_mode,
        action_smoothing=args.action_smoothing,
        action_noise=args.action_noise,
        domain_randomize=args.domain_randomize,
        track_seed=args.track_seed,
    )
    print(f"Episode finished: reward={total_reward:.2f}, steps={steps}")


if __name__ == "__main__":
    raise SystemExit(main())
