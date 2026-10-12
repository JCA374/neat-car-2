"""Describe NEAT genomes as plain JSON for the NEAT page's network explorer."""

from __future__ import annotations

import math
from typing import Dict, List

import neat
import numpy as np

DYNAMICS = ["Speed", "Vx (sideways)", "Vy (forward)", "cos(heading)", "sin(heading)", "Spin rate", "Steering", "Traction"]
OUTPUTS = ["Steer", "Gas", "Brake"]


def _r(v: float, nd: int = 2) -> float:
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
                     "angle": _r(deg, 1), "ahead": _r(ahead, 1)})
    return spec


def output_spec(config: neat.Config) -> List[dict]:
    return [{"key": k, "label": OUTPUTS[k]} for k in config.genome_config.output_keys]


def genome_graph(genome, config: neat.Config) -> Dict:
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
        nodes.append({"k": k, "b": _r(g.bias), "layer": layer.get(k, 1), "used": k in used, "type": "hidden"})
    for k in sorted(outputs):
        nodes.append({"k": k, "b": _r(genome.nodes[k].bias), "type": "output"})
    return {"nodes": nodes, "conns": [[i, o, _r(w)] for i, o, w in conns],
            "disabled": sum(1 for c in genome.connections.values() if not c.enabled)}
