"""
Compact state wrapper for Gymnasium's CarRacing-v2 tailored for PPO training.

The wrapper exposes a low-dimensional observation vector:
    - 8 car-dynamics scalars
    - N lidar-style distances sampled against road polygons (default N=32)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception as exc:  # pragma: no cover - handled at runtime by the user
    raise ImportError(
        "gymnasium is required. Install with `pip install gymnasium[box2d]`."
    ) from exc


def _point_in_poly(point: Tuple[float, float], poly: np.ndarray) -> bool:
    """Check if a 2D point is inside a polygon using a winding test."""
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = (yi > y) != (yj > y) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class CarRacingStateWrapper(gym.Wrapper):
    """
    Wraps CarRacing to emit a fixed-length, low-dimensional state vector.

    Base features (8 values):
        speed_norm, local_vx, local_vy, cos(heading), sin(heading),
        angular_velocity, mean_steer, traction_fraction
    Lidar: N normalized distances in [0, 1]
    """

    def __init__(
        self,
        env: gym.Env,
        lidar_rays: int = 32,
        lidar_max_dist: float = 45.0,
        lidar_step: float = 3.0,
    ) -> None:
        super().__init__(env)
        self.lidar_rays = lidar_rays
        self.lidar_max_dist = lidar_max_dist
        self.lidar_step = lidar_step

        self.lidar_angles = np.linspace(-0.9 * math.pi, 0.9 * math.pi, lidar_rays)
        self.base_feature_count = 8
        self.state_size = self.base_feature_count + lidar_rays

        # Re-declare observation space to match the state vector.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        self._road_polygons: List[np.ndarray] = []
        self._road_bounds: List[Tuple[float, float, float, float]] = []

    # ---------------------------------------------------------------------- #
    # Gym API
    # ---------------------------------------------------------------------- #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._extract_road_geometry()
        state = self._build_state(obs)
        return state, info

    def step(self, action):
        # Cast to float64 to satisfy Box2D bindings.
        action = np.asarray(action, dtype=np.float64)
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._build_state(obs)
        return state, reward, terminated, truncated, info

    # ------------------------------------------------------------------ utils
    def _build_state(self, obs) -> np.ndarray:
        """
        Construct the state vector from car dynamics and road-aware lidar.

        The observation itself is not used directly; the wrapper pulls data
        from the underlying Box2D simulation to remain CPU-friendly.
        """
        car = getattr(self.env.unwrapped, "car", None)
        hull = getattr(car, "hull", None)
        wheels = getattr(car, "wheels", None) or []

        if hull is None:
            return np.zeros(self.state_size, dtype=np.float32)

        vel = hull.linearVelocity
        heading = float(hull.angle)

        # Transform velocities into the car's local frame for stability.
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        local_vx = vel.x * cos_h + vel.y * sin_h
        local_vy = -vel.x * sin_h + vel.y * cos_h
        speed = math.sqrt(vel.x**2 + vel.y**2)

        speed_scale = 50.0
        angular_scale = 5.0
        steer_limit = 0.8  # matches Car.steer_limit in most gym versions

        traction_values: List[float] = []
        steer_angles: List[float] = []
        for wheel in wheels:
            traction = getattr(wheel, "ground_traction", None)
            if traction is not None:
                traction_values.append(float(traction))
            joint = getattr(wheel, "joint", None)
            if joint is not None:
                steer_angles.append(float(getattr(joint, "angle", 0.0)))
            elif hasattr(wheel, "steer_angle"):
                steer_angles.append(float(getattr(wheel, "steer_angle")))

        traction_fraction = (
            float(np.clip(np.mean(traction_values), 0.0, 1.0))
            if traction_values
            else 1.0
        )
        mean_steer = float(np.clip(np.mean(steer_angles) / steer_limit, -1.0, 1.0)) if steer_angles else 0.0

        base_features = np.array(
            [
                np.clip(speed / speed_scale, 0.0, 1.0),
                np.clip(local_vx / speed_scale, -1.0, 1.0),
                np.clip(local_vy / speed_scale, -1.0, 1.0),
                cos_h,
                sin_h,
                np.clip(hull.angularVelocity / angular_scale, -1.0, 1.0),
                mean_steer,
                traction_fraction,
            ],
            dtype=np.float32,
        )

        lidar = self._lidar_distances(
            origin=np.array([hull.position[0], hull.position[1]], dtype=np.float32),
            heading=heading,
        )

        state = np.concatenate([base_features, lidar]).astype(np.float32)
        return state

    # ---------------------------------------------------------------- lidar --
    def _extract_road_geometry(self) -> None:
        """Cache track polygons from the underlying env for cheap point tests."""
        self._road_polygons.clear()
        self._road_bounds.clear()

        env = self.env.unwrapped
        candidates: Sequence = []
        for attr in ("road_poly", "road"):
            if hasattr(env, attr):
                road = getattr(env, attr)
                if road is not None:
                    candidates.append(road)

        for container in candidates:
            for item in container:
                vertices: Optional[Iterable] = None
                # Some gym versions store (poly, color) tuples.
                if isinstance(item, (list, tuple)):
                    if len(item) == 2 and isinstance(item[0], (list, tuple, np.ndarray)):
                        vertices = item[0]
                    elif len(item) >= 3 and isinstance(item[0], (list, tuple, np.ndarray)):
                        vertices = item
                if vertices is None:
                    continue
                poly = np.asarray(vertices, dtype=np.float32)
                if poly.ndim == 2 and poly.shape[0] >= 3 and poly.shape[1] >= 2:
                    self._road_polygons.append(poly[:, :2])

        for poly in self._road_polygons:
            xs = poly[:, 0]
            ys = poly[:, 1]
            self._road_bounds.append((float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())))

    def _lidar_distances(self, origin: np.ndarray, heading: float) -> np.ndarray:
        """Sample ray distances against cached road polygons."""
        if not self._road_polygons:
            return np.ones(self.lidar_rays, dtype=np.float32)

        margin = self.lidar_max_dist + 2.0
        candidates = [
            idx
            for idx, (xmin, xmax, ymin, ymax) in enumerate(self._road_bounds)
            if (origin[0] >= xmin - margin)
            and (origin[0] <= xmax + margin)
            and (origin[1] >= ymin - margin)
            and (origin[1] <= ymax + margin)
        ]
        if not candidates:
            return np.ones(self.lidar_rays, dtype=np.float32)

        distances: List[float] = []
        for rel_angle in self.lidar_angles:
            angle = heading + rel_angle
            dir_x = math.cos(angle)
            dir_y = math.sin(angle)

            distance = self.lidar_max_dist
            step = self.lidar_step
            travelled = step
            while travelled <= self.lidar_max_dist:
                sample = origin + np.array([dir_x * travelled, dir_y * travelled], dtype=np.float32)
                if not self._point_on_road(sample, candidates):
                    distance = travelled
                    break
                travelled += step

            distances.append(np.clip(distance / self.lidar_max_dist, 0.0, 1.0))

        return np.asarray(distances, dtype=np.float32)

    def _point_on_road(self, point: np.ndarray, candidates: List[int]) -> bool:
        """Return True if the point lies within any cached road polygon."""
        px, py = point
        for idx in candidates:
            xmin, xmax, ymin, ymax = self._road_bounds[idx]
            if px < xmin - 1.0 or px > xmax + 1.0 or py < ymin - 1.0 or py > ymax + 1.0:
                continue
            if _point_in_poly((px, py), self._road_polygons[idx]):
                return True
        return False


def make_env(
    render_mode: Optional[str] = None,
    continuous: bool = True,
    domain_randomize: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> CarRacingStateWrapper:
    """
    Convenience helper to construct a wrapped CarRacing environment.

    Parameters
    ----------
    render_mode: Optional[str]
        None for headless training; "human" or "rgb_array" for visualisation.
    continuous: bool
        CarRacing expects continuous control; keep True unless debugging.
    domain_randomize: bool
        Toggle Gymnasium's domain randomization option.
    seed: Optional[int]
        Optional environment seed for reproducibility.
    kwargs:
        Passed through to gymnasium.make("CarRacing-v2", ...).
    """
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        continuous=continuous,
        domain_randomize=domain_randomize,
        **kwargs,
    )
    if seed is not None:
        env.reset(seed=seed)
    return CarRacingStateWrapper(env)
