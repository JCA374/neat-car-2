"""Benchmark a step's model on the five reserved tracks and record the drives.

Every model drives the same tracks with the *raw* CarRacing-v2 reward (no
off-track penalty, no bonuses), deterministically, for up to 2000 frames,
stopping at the end of its first lap. The headline number is the lap time.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from carracing.env import make_env
from carracing.rules import BENCHMARK_SEEDS, FPS, MAX_FRAMES, VALIDATION_SEEDS, smooth
from carracing.steps import Step

SEEDS = BENCHMARK_SEEDS       # never used for training or validation
GIF_SEED = SEEDS[0]
RECORD_EVERY = 2              # keep every 2nd frame in recordings (25 fps on the page)


def _r(v: float, nd: int = 1) -> float:
    return round(float(v), nd)


class PPOPolicy:
    def __init__(self, path: str, legacy_traction: bool = False, action_smoothing: float = 0.0,
                 controls: str = "raw3", action_repeat: int = 1) -> None:
        from stable_baselines3 import PPO

        from carracing.controls import HeldControls

        self.model = PPO.load(path, device="cpu")
        # Trained before the traction fix: these models only ever saw state[7] == 1.0.
        self.legacy_traction = legacy_traction
        # Same smoothing the training env applied (rules.ACTION_SMOOTHING for new steps).
        self.smoothing = action_smoothing
        # The step's action interface and decision rate (carracing/controls.py), frame by frame.
        self.held = HeldControls(controls, action_repeat)
        self.reset()

    def reset(self) -> None:
        self.prev = np.zeros(3, dtype=np.float32)
        self.held.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.legacy_traction:
            obs = obs.copy()
            obs[7] = 1.0
        action = self.held(lambda: self.model.predict(obs, deterministic=True)[0])
        if self.smoothing:
            action = self.prev = smooth(action, self.prev, self.smoothing)
        return action

    def size(self) -> Dict:
        n = sum(p.numel() for p in self.model.policy.parameters())
        arch = self.model.policy.net_arch
        hidden = arch["pi"] if isinstance(arch, dict) else arch
        return {"params": int(n), "label": f"{'×'.join(str(h) for h in hidden)} MLP"}


class NEATPolicy:
    def __init__(self, path: str, config_path: str, legacy_decode: bool = False) -> None:
        import neat

        from carracing_neat.eval import EvalSettings, decode_action, load_genome

        self.genome, self.config = load_genome(path, config_path)
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        self.decode = decode_action
        self.legacy_decode = legacy_decode
        self.smoothing = EvalSettings().action_smoothing
        self.reset()

    def reset(self) -> None:
        self.prev = np.zeros(3, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.prev = self.decode(self.net.activate(obs), self.prev, self.smoothing, self.legacy_decode)
        return self.prev

    def size(self) -> Dict:
        outputs = set(self.config.genome_config.output_keys)
        hidden = len(set(self.genome.nodes) - outputs)
        conns = sum(1 for c in self.genome.connections.values() if c.enabled)
        return {"params": conns + len(self.genome.nodes), "hidden": hidden, "connections": conns,
                "label": f"{hidden} hidden, {conns} links"}


def load_policy(step: Step, model_path: Optional[str] = None):
    path = model_path or step.model_path
    if step.model == "neat":
        return NEATPolicy(path, os.path.join(step.out_dir, "neat.cfg"), step.legacy_decode)
    return PPOPolicy(path, step.legacy_traction, step.action_smoothing, step.controls, step.action_repeat)


def drive(policy, seed: int, env_kwargs: Optional[Dict] = None, gif: bool = False, activations: bool = False,
          give_up: Optional[int] = None) -> Dict:
    """One benchmark lap. Returns the result plus a recording for the race page.

    ``env_kwargs`` holds the step's sensor settings (``Step.env_kwargs()``). ``give_up`` ends the drive
    after that many frames without a new tile (lap checks only; the benchmark always runs to the end).
    """
    env = make_env(render_mode="rgb_array" if gif else None, offtrack_penalty=0.0, allow_multiple_laps=False,
                   lap_completion_bonus=0.0, max_episode_steps=MAX_FRAMES, **(env_kwargs or {}))
    obs, _ = env.reset(seed=seed)
    policy.reset()
    base = env.unwrapped
    rec = {"x": [], "y": [], "a": [], "v": [], "tiles": [], "grass": []}
    if activations:
        node_keys = [k for k, *_ in policy.net.node_evals]
        rec.update(inputs=[], nodes=[], actions=[])
    frames: List[np.ndarray] = []
    score, offtrack, finish, frame, info = 0.0, 0, None, 0, {}
    since_tile, tiles_seen = 0, 0
    for frame in range(1, MAX_FRAMES + 1):
        action = policy(obs)
        if (frame - 1) % RECORD_EVERY == 0:
            hull = base.car.hull
            rec["x"].append(_r(hull.position[0]))
            rec["y"].append(_r(hull.position[1]))
            rec["a"].append(_r(hull.angle, 3))
            rec["v"].append(_r(math.hypot(*hull.linearVelocity)))
            rec["tiles"].append(int(base.tile_visited_count))
            rec["grass"].append(int(round((1 - env.last_traction) * 4)))  # wheels on grass
            if activations:
                rec["inputs"].append([_r(v, 2) for v in obs])
                rec["nodes"].append([_r(policy.net.values[k], 2) for k in node_keys])
                rec["actions"].append([_r(v, 2) for v in action])
        obs, reward, terminated, truncated, info = env.step(action)
        score += float(reward)
        offtrack += env.last_traction < 0.99
        if gif:
            frames.append(env.render())
        if info.get("laps_completed", 0) >= 1:
            finish = frame
        if terminated or truncated:
            break
        since_tile = 0 if base.tile_visited_count > tiles_seen else since_tile + 1
        tiles_seen = base.tile_visited_count
        if give_up and since_tile >= give_up:
            break
    tiles_total = len(base.track)
    reached = int(base.tile_visited_count) if finish is None else tiles_total
    env.close()
    result = {
        "seed": seed,
        "finished": finish is not None,
        "lap_s": _r(finish / FPS, 2) if finish else None,
        "frames": frame,
        "score": _r(score),
        "offtrack_pct": _r(100 * offtrack / frame),
        "avg_speed": _r(np.mean(rec["v"])),
        "reached_pct": _r(100 * reached / tiles_total),
        "crashed": bool(terminated) and finish is None,
    }
    if activations:
        rec["node_keys"] = node_keys
    return {"result": result, "recording": {"every": RECORD_EVERY, "finish": finish, **rec}, "frames": frames}


def lap_check(policy, env_kwargs: Optional[Dict] = None) -> Dict:
    """Lap check on the validation tracks in the benchmark's conditions (used to pick checkpoints)."""
    laps = [drive(policy, seed, env_kwargs, give_up=150)["result"] for seed in VALIDATION_SEEDS]
    done = [r["lap_s"] for r in laps if r["finished"]]
    return {"finished": len(done), "mean_lap_s": round(float(np.mean(done)), 3) if done else None}


def track_geometry(seed: int) -> Dict:
    env = make_env(render_mode=None)
    env.reset(seed=seed)
    base = env.unwrapped
    polys = [[_r(c) for xy in poly for c in xy] for poly, _color in base.road_poly]
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    data = {"seed": seed, "tiles": len(base.track), "polys": polys, "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "start": [_r(base.track[0][2]), _r(base.track[0][3])], "start_angle": _r(base.track[0][1], 3)}
    env.close()
    return data


def save_gif(frames: List[np.ndarray], path: str, every: int = 3, width: int = 288) -> None:
    import imageio.v2 as imageio
    from PIL import Image

    kept = []
    for frame in frames[::every]:
        img = Image.fromarray(frame)
        height = int(img.height * width / img.width)
        kept.append(np.asarray(img.resize((width, height), Image.BILINEAR)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, kept, duration=1000 * every / FPS, loop=0)
