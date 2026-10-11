"""The vectorised lidar must match the original ray-marching implementation."""

from __future__ import annotations

import math

import numpy as np

from carracing_ppo.env import make_env


def _point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def _reference_lidar(wrapper, origin, heading):
    """The original per-ray, per-sample loop (before vectorisation)."""
    out = []
    for rel in wrapper.lidar_angles:
        angle = heading + rel
        distance = wrapper.lidar_max_dist
        travelled = wrapper.lidar_step
        while travelled <= wrapper.lidar_max_dist:
            x = origin[0] + math.cos(angle) * travelled
            y = origin[1] + math.sin(angle) * travelled
            if not any(_point_in_poly(x, y, poly) for poly in wrapper._road_polygons):
                distance = travelled
                break
            travelled += wrapper.lidar_step
        out.append(min(1.0, distance / wrapper.lidar_max_dist))
    return np.array(out, dtype=np.float32)


def test_vectorised_lidar_matches_reference():
    env = make_env(render_mode=None, seed=5, offtrack_penalty=0.0)
    env.reset(seed=5)
    rng = np.random.default_rng(1)
    mismatched_rays = total_rays = 0
    for step in range(300):
        env.step(np.array([rng.uniform(-1, 1), rng.uniform(0.2, 1), 0.0]))
        if step % 10:
            continue
        hull = env.unwrapped.car.hull
        origin = np.array([hull.position[0], hull.position[1]], dtype=np.float64)
        fast = env._lidar_distances(origin, float(hull.angle))
        slow = _reference_lidar(env, origin, float(hull.angle))
        mismatched_rays += int((np.abs(fast - slow) > 1e-6).sum())
        total_rays += len(fast)
    env.close()
    # Float rounding can flip a sample sitting exactly on a tile edge; allow a sliver.
    assert mismatched_rays <= total_rays * 0.01, (mismatched_rays, total_rays)
