"""Load a saved PPO checkpoint and roll out CarRacing episodes."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
from stable_baselines3 import PPO

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
    # Reward shaping: match the training run so printed rewards are comparable.
    parser.add_argument("--offtrack-penalty", type=float, default=1.0, help="Same as train.py.")
    parser.add_argument("--traction-threshold", type=float, default=0.95, help="Same as train.py.")
    parser.add_argument(
        "--allow-multiple-laps", action=argparse.BooleanOptionalAction, default=True, help="Same as train.py."
    )
    parser.add_argument("--lap-completion-bonus", type=float, default=100.0, help="Same as train.py.")
    parser.add_argument(
        "--gif",
        default=None,
        help="Record the first episode to this GIF path (forces render-mode rgb_array).",
    )
    parser.add_argument("--gif-every", type=int, default=2, help="Keep every Nth frame in the GIF.")
    parser.add_argument("--gif-width", type=int, default=320, help="GIF width in pixels.")
    return parser.parse_args(argv)


def _save_gif(frames: List[np.ndarray], path: str, every: int, width: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    kept = []
    for frame in frames[::every]:
        img = Image.fromarray(frame)
        height = int(img.height * width / img.width)
        kept.append(np.asarray(img.resize((width, height), Image.BILINEAR)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, kept, duration=1000 * every / 50, loop=0)
    print(f"Saved {len(kept)} frames to {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    render_mode = args.render_mode
    if isinstance(render_mode, str) and render_mode.lower() == "none":
        render_mode = None

    model_path = _resolve_model(args.model, args.checkpoint_dir)
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, device="auto")

    if args.gif:
        render_mode = "rgb_array"
    env_kwargs = dict(
        render_mode=render_mode,
        domain_randomize=args.domain_randomize,
        seed=args.seed,
        offtrack_penalty=args.offtrack_penalty,
        traction_threshold=args.traction_threshold,
        allow_multiple_laps=args.allow_multiple_laps,
        lap_completion_bonus=args.lap_completion_bonus,
        max_episode_steps=args.max_steps,
    )
    env = make_env(**env_kwargs)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed if ep == 0 else None)
        frames: List[np.ndarray] = []
        total_reward = 0.0
        laps = 0
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            laps = info.get("laps_completed", laps)
            if args.gif and ep == 0:
                frames.append(env.render())
            if terminated or truncated:
                break
        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={step + 1}, laps={laps}")
        if frames:
            _save_gif(frames, args.gif, args.gif_every, args.gif_width)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
