"""Watch any step's model drive.

    python -m carracing.play n00_baseline --seed 3          # window
    python -m carracing.play p00_baseline --seed 11 --gif out.gif
"""

from __future__ import annotations

import argparse
import glob
import os

from carracing.benchmark import load_policy, save_gif
from carracing.env import make_env
from carracing.steps import load_step


def find_step(step_id: str):
    matches = glob.glob(os.path.join("experiments", "*", f"{step_id}.toml"))
    if not matches:
        raise SystemExit(f"No experiments/*/{step_id}.toml")
    return load_step(matches[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", help="Step id, e.g. n00_baseline")
    parser.add_argument("--seed", type=int, default=3, help="Track seed.")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--gif", default=None, help="Record to this GIF instead of opening a window.")
    args = parser.parse_args(argv)

    step = find_step(args.step)
    policy = load_policy(step)
    env = make_env(render_mode="rgb_array" if args.gif else "human", offtrack_penalty=0.0,
                   allow_multiple_laps=args.laps > 1, lap_completion_bonus=0.0,
                   max_episode_steps=1000 * args.laps + 1000, **step.env_kwargs())
    obs, _ = env.reset(seed=args.seed)
    policy.reset()
    frames, score, steps, info = [], 0.0, 0, {}
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        score += float(reward)
        steps += 1
        if args.gif:
            frames.append(env.render())
        if terminated or truncated or info.get("laps_completed", 0) >= args.laps:
            break
    env.close()
    print(f"{step.id} on track {args.seed}: {steps} frames ({steps / 50:.1f} s), "
          f"laps {info.get('laps_completed', 0)}, score {score:.1f}")
    if args.gif:
        save_gif(frames, args.gif)
        print(f"Saved {args.gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
