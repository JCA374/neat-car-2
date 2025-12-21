"""Load a saved PPO checkpoint and roll out CarRacing episodes."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from carracing_ppo.env import make_env


def _resolve_model(path: str, checkpoint_dir: str) -> str:
    candidates = [path]
    if checkpoint_dir:
        candidates.extend(
            [
                os.path.join(checkpoint_dir, "best_model.zip"),
                os.path.join(checkpoint_dir, "ppo_carracing_best_model.zip"),
                os.path.join(checkpoint_dir, "ppo_latest.zip"),
            ]
        )
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"No model found. Tried {candidates}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a saved PPO model in CarRacing-v2.")
    parser.add_argument(
        "--model",
        default="outputs/checkpoints/best_model.zip",
        help="Path to a PPO checkpoint zip. Falls back to best/ppo_latest in checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
        help="Directory to search for best/latest checkpoints.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument(
        "--render-mode",
        default="human",
        help='Gymnasium render mode ("human", "rgb_array", or None). Default "human".',
    )
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    parser.add_argument("--domain-randomize", action="store_true", help="Enable domain randomization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional environment seed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    render_mode = args.render_mode
    if isinstance(render_mode, str) and render_mode.lower() == "none":
        render_mode = None

    model_path = _resolve_model(args.model, args.checkpoint_dir)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device="auto")

    env = DummyVecEnv(
        [lambda: make_env(render_mode=render_mode, domain_randomize=args.domain_randomize, seed=args.seed)]
    )

    for ep in range(args.episodes):
        obs = env.reset()
        total_reward = 0.0
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            if done[0]:
                break
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={step + 1}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
