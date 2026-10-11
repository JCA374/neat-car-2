"""
Apples-to-apples benchmark of every stage's best model.

Each model (PPO checkpoint or NEAT genome) drives the same fixed tracks for up to 1000 steps and is scored with
the *raw* CarRacing-v2 reward (no off-track penalty, no lap bonus, single lap),
so runs trained with different reward shaping can be compared. Also records a
GIF per stage on the first track.

    python scripts/benchmark.py                 # all stages
    python scripts/benchmark.py --stages v2     # update one stage
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO

from carracing_ppo.env import make_env
from carracing_ppo.play import _save_gif

MODELS = {
    "baseline": "outputs/checkpoints/best_model.zip",
    "improved": "outputs/checkpoints_improved/best_model.zip",
    "collapsed": "outputs/checkpoints_improved/ppo_latest.zip",
    "strict": "outputs/checkpoints_strict/best_model.zip",
    "multilap": "outputs/checkpoints_multilap/best_model.zip",
    "v2": "outputs/checkpoints_v2/best_model.zip",
    "neat": "outputs/neat/checkpoints/best-genome.pkl",
}
# Trained before the traction fix: they only ever saw state[7] == 1.0.
LEGACY_TRACTION = {"baseline", "improved", "collapsed", "strict", "multilap"}
SEEDS = [3, 11, 21, 34, 55]
GIF_SEED = SEEDS[0]


class PPOPolicy:
    def __init__(self, path: str, legacy: bool) -> None:
        self.model = PPO.load(path, device="cpu")
        self.legacy = legacy

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        if self.legacy:
            obs[7] = 1.0
        return self.model.predict(obs, deterministic=True)[0]


class NEATPolicy:
    def __init__(self, path: str, config_path: str = "neat_config/neat.cfg") -> None:
        import neat

        from carracing_neat.eval import EvalSettings, decode_action
        from carracing_neat.play import load_genome

        genome, config = load_genome(path, config_path)
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.decode = decode_action
        self.smoothing = EvalSettings().action_smoothing
        self.reset()

    def reset(self) -> None:
        self.prev = np.zeros(3, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.prev = self.decode(self.net.activate(obs), self.prev, self.smoothing)
        return self.prev


def load_policy(stage: str):
    path = MODELS[stage]
    if path.endswith(".pkl"):
        return NEATPolicy(path)
    return PPOPolicy(path, legacy=stage in LEGACY_TRACTION)


def run_episode(policy, seed: int, record: bool):
    env = make_env(
        render_mode="rgb_array" if record else None,
        offtrack_penalty=0.0,
        allow_multiple_laps=False,
        max_episode_steps=1000,
    )
    obs, _ = env.reset(seed=seed)
    policy.reset()
    total, frames, info, steps = 0.0, [], {}, 0
    traction = []
    for steps in range(1, 1001):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        traction.append(float(obs[7]))
        if record:
            frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return {
        "seed": seed,
        "score": round(total, 1),
        "steps": steps,
        "lap": int(info.get("laps_completed", 0) >= 1),
        "offtrack_pct": round(100 * float(np.mean(np.array(traction) < 0.99)), 1),
    }, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="*", default=list(MODELS))
    parser.add_argument("--out", default="docs/assets/data/benchmark.json")
    parser.add_argument("--gif-dir", default="docs/assets/gifs")
    parser.add_argument("--gif-steps", type=int, default=600, help="Frames of the episode to keep in the GIF.")
    args = parser.parse_args()

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = json.load(fh)["stages"]

    for stage in args.stages:
        path = MODELS[stage]
        if not os.path.exists(path):
            print(f"skip {stage}: {path} missing")
            continue
        policy = load_policy(stage)
        episodes = []
        for seed in SEEDS:
            ep, frames = run_episode(policy, seed, record=seed == GIF_SEED)
            episodes.append(ep)
            print(stage, ep, flush=True)
            if frames:
                _save_gif(frames[: args.gif_steps], os.path.join(args.gif_dir, f"{stage}.gif"), every=3, width=288)
        scores = [e["score"] for e in episodes]
        results[stage] = {
            "model": path,
            "mean": round(float(np.mean(scores)), 1),
            "std": round(float(np.std(scores)), 1),
            "laps": sum(e["lap"] for e in episodes),
            "offtrack_pct": round(float(np.mean([e["offtrack_pct"] for e in episodes])), 1),
            "episodes": episodes,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"seeds": SEEDS, "stages": results}, fh, indent=1)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
