"""
Export NEAT networks and recorded drives for docs/network.html.

Writes docs/assets/data/network.json with
  - every generation's champion genome per run (nodes, enabled connections,
    layer for layout, which hidden nodes actually reach an output),
  - a few milestone individuals driving the same track, recorded frame by
    frame (position, heading, speed, tiles, all node activations, actions),
  - that track's road polygons.

    python scripts/export_network_data.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional

import neat
import numpy as np
import pandas as pd

from carracing_neat.eval import decode_action
from carracing_neat.play import load_genome
from carracing_ppo.env import make_env

CONFIG = "neat_config/neat.cfg"
TRACK_SEED = 3  # same track as the page's GIFs
RUNS = [
    # key, label, checkpoint dir, log dir, legacy decoding (genomes before Oct 12 2026)
    ("run1", "Run 1: capped gas/brake, single species", "outputs/neat/checkpoints", "outputs/neat/logs", True),
    ("run2", "Run 2: full controls, speciation", "outputs/neat_v2/checkpoints", "outputs/neat_v2/logs", False),
]
DYNAMICS = ["Speed", "Vx (sideways)", "Vy (forward)", "cos(heading)", "sin(heading)", "Spin rate", "Steering", "Traction"]
OUTPUTS = ["Steer", "Gas", "Brake"]


def r(v: float, nd: int = 2) -> float:
    return round(float(v), nd)


def input_spec(config: neat.Config) -> List[dict]:
    n_lidar = config.genome_config.num_inputs - len(DYNAMICS)
    angles = np.degrees(np.linspace(-0.9 * math.pi, 0.9 * math.pi, n_lidar))
    spec = [{"key": -(i + 1), "label": name, "group": "dynamics"} for i, name in enumerate(DYNAMICS)]
    for j, deg in enumerate(angles):
        # The wrapper measures ray angles from the car's local +x axis (its right
        # side); the nose is local +y. Label relative to the nose, + = left.
        ahead = (deg - 90.0 + 180.0) % 360.0 - 180.0
        spec.append({"key": -(len(DYNAMICS) + j + 1), "label": f"Lidar {ahead:+.0f}°", "group": "lidar",
                     "angle": r(deg, 1), "ahead": r(ahead, 1)})
    return spec


def genome_graph(genome, config: neat.Config) -> dict:
    """Nodes with layer (longest path from inputs) and whether they reach an output."""
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    conns = [(i, o, w.weight) for (i, o), w in genome.connections.items() if w.enabled]
    used = neat.graphs.required_for_output(config.genome_config.input_keys, config.genome_config.output_keys,
                                          [(i, o) for i, o, _ in conns])
    layer: Dict[int, int] = {k: 0 for k in inputs}
    hidden = [k for k in genome.nodes if k not in outputs]
    for _ in range(len(genome.nodes) + 1):  # relax longest-path layers (graph is feed-forward)
        changed = False
        for i, o, _ in conns:
            if i in layer and layer.get(o, 0) < layer[i] + 1 and o not in outputs:
                layer[o] = layer[i] + 1
                changed = True
        if not changed:
            break
    nodes = []
    for k in sorted(hidden):
        g = genome.nodes[k]
        nodes.append({"k": k, "b": r(g.bias), "layer": layer.get(k, 1), "used": k in used, "type": "hidden"})
    for k in sorted(outputs):
        nodes.append({"k": k, "b": r(genome.nodes[k].bias), "type": "output"})
    return {"nodes": nodes, "conns": [[i, o, r(w)] for i, o, w in conns],
            "disabled": sum(1 for c in genome.connections.values() if not c.enabled)}


def champions(ckpt_dir: str) -> Dict[int, str]:
    found = {}
    for path in glob.glob(os.path.join(ckpt_dir, "best-genome-gen-*.pkl")):
        found[int(re.search(r"gen-(\d+)\.pkl$", path).group(1))] = path
    return dict(sorted(found.items()))


def record(genome, config, legacy: bool, every: int = 2, max_steps: int = 1000) -> dict:
    """Drive TRACK_SEED and record what the car does and what every node computes."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = make_env(render_mode=None, offtrack_penalty=0.0, allow_multiple_laps=False, max_episode_steps=max_steps)
    state, _ = env.reset(seed=TRACK_SEED)
    prev = np.zeros(3, dtype=np.float32)
    eval_keys = [n for n, *_ in net.node_evals]
    frames = {"x": [], "y": [], "a": [], "v": [], "tiles": [], "inputs": [], "nodes": [], "actions": []}
    total, offtrack, steps, laps = 0.0, 0, 0, 0
    for steps in range(1, max_steps + 1):
        out = net.activate(state)
        prev = decode_action(out, prev, 0.1, legacy=legacy)
        hull = env.unwrapped.car.hull
        if (steps - 1) % every == 0:
            frames["x"].append(r(hull.position[0], 1))
            frames["y"].append(r(hull.position[1], 1))
            frames["a"].append(r(hull.angle, 3))
            frames["v"].append(r(math.hypot(*hull.linearVelocity), 1))
            frames["tiles"].append(int(env.unwrapped.tile_visited_count))
            frames["inputs"].append([r(v) for v in state])
            frames["nodes"].append([r(net.values[k]) for k in eval_keys])
            frames["actions"].append([r(v) for v in prev])
        state, reward, terminated, truncated, info = env.step(prev)
        total += float(reward)
        offtrack += float(state[7]) < 0.99
        laps = info.get("laps_completed", laps)
        if terminated or truncated:
            break
    tiles_total = len(env.unwrapped.track)
    env.close()
    return {
        "score": r(total, 1), "steps": steps, "laps": laps, "tiles_total": tiles_total,
        "offtrack_pct": r(100 * offtrack / steps, 1), "avg_speed": r(np.mean(frames["v"]), 1),
        "every": every, "node_keys": eval_keys, **frames,
    }


def track_geometry() -> dict:
    env = make_env(render_mode=None)
    env.reset(seed=TRACK_SEED)
    base = env.unwrapped
    polys = [[r(c, 1) for xy in poly for c in xy] for poly, _color in base.road_poly]
    start = base.track[0]
    env.close()
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    return {"seed": TRACK_SEED, "polys": polys, "bounds": [min(xs), min(ys), max(xs), max(ys)],
            "start": [r(start[2], 1), r(start[3], 1)]}


def pick_milestones(log_dir: str, thresholds=(300, 600)) -> List[int]:
    val = pd.read_csv(os.path.join(log_dir, "validation.csv"))
    gens = [int(val["generation"].iloc[0])]
    for t in thresholds:
        hit = val[val["validation_score"] > t]
        if len(hit):
            gens.append(int(hit["generation"].iloc[0]))
    gens.append(int(val.loc[val["validation_score"].idxmax(), "generation"]))
    return sorted(set(gens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/assets/data/network.json")
    args = parser.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, CONFIG)
    data = {"inputs": input_spec(config),
            "outputs": [{"key": k, "label": OUTPUTS[k]} for k in config.genome_config.output_keys],
            "track": track_geometry(), "runs": [], "individuals": []}

    for key, label, ckpt_dir, log_dir, legacy in RUNS:
        champs = champions(ckpt_dir)
        val_path = os.path.join(log_dir, "validation.csv")
        if not champs or not os.path.exists(val_path):
            print(f"skip {key}: no champions in {ckpt_dir}")
            continue
        val = pd.read_csv(val_path).set_index("generation")["validation_score"]
        gens = []
        for gen, path in champs.items():
            genome, _ = load_genome(path, CONFIG)
            gens.append({"gen": gen, "id": genome.key,
                         "validation": r(val.get(gen, float("nan")), 1) if gen in val.index else None,
                         **genome_graph(genome, config)})
        data["runs"].append({"key": key, "label": label, "legacy": legacy, "generations": gens})

        milestones = pick_milestones(log_dir)
        if key == "run2":  # run 1 already covers the early milestones; show run 2's best
            milestones = milestones[-1:]
        for gen in milestones:
            genome, _ = load_genome(champs[gen], CONFIG)
            rec = record(genome, config, legacy)
            data["individuals"].append({
                "id": f"{key}-gen{gen}", "run": key, "gen": gen, "genome_id": genome.key,
                "validation": r(val.get(gen), 1), "graph": genome_graph(genome, config), **rec})
            print(f"{key} gen {gen}: score {rec['score']} on track {TRACK_SEED}, {rec['steps']} frames, "
                  f"laps {rec['laps']}, off-track {rec['offtrack_pct']}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(data, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
