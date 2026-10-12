"""Replay a saved NEAT genome on CarRacing (window, headless or GIF)."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Optional, Sequence

import neat

from carracing_neat.eval import EvalSettings, run_episode
from carracing_ppo.play import _save_gif


def load_genome(path: str, config_path: str):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    with open(path, "rb") as f:
        genome = pickle.load(f)
    return genome, config


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved NEAT genome.")
    parser.add_argument("--genome", default="outputs/neat/checkpoints/best-genome.pkl", help="Pickled genome.")
    parser.add_argument("--config", default="neat_config/neat.cfg", help="NEAT config used for training.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None, help="Track seed (default: random).")
    parser.add_argument("--render-mode", default="human", help='"human", "rgb_array" or "none".')
    parser.add_argument("--action-smoothing", type=float, default=0.1, help="Same as training.")
    parser.add_argument("--legacy-decode", action="store_true",
                        help="Old double-tanh output decoding (genomes evolved before Oct 12 2026).")
    parser.add_argument("--gif", default=None, help="Record the first episode to this GIF path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    genome, config = load_genome(args.genome, args.config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    render_mode = None if args.render_mode.lower() == "none" else args.render_mode
    if args.gif:
        render_mode = "rgb_array"
    settings = EvalSettings(max_steps=args.max_steps, action_smoothing=args.action_smoothing,
                            legacy_decode=args.legacy_decode)

    print(f"Genome {genome.key} from {args.genome}")
    for ep in range(args.episodes):
        seed = (args.seed + ep) if args.seed is not None else int.from_bytes(os.urandom(2), "little")
        result = run_episode(net, settings, seed, render_mode=render_mode, stop_when_stuck=False)
        print(f"Episode {ep + 1} (track {seed}): reward={result['reward']:.1f}, steps={result['steps']}, laps={result['laps']}")
        if args.gif and ep == 0:
            _save_gif(result["frames"], args.gif, every=3, width=288)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
