"""
Compact state wrapper for Gymnasium's CarRacing-v2 shared by the NEAT and PPO learners.

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


def lidar_angles(rays: int, fov: Optional[float] = None) -> np.ndarray:
    """Ray angles in radians, relative to the car's local +x axis (its right side).

    The nose points along local +y (+pi/2). ``fov=None`` is the original layout:
    rays spread over 324 degrees centred on the right side, so many point backwards
    and a 36 degree blind gap sits on the left. With ``fov`` (degrees) the fan is
    centred on the nose and spans +-fov/2.
    """
    if fov is None:
        return np.linspace(-0.9 * math.pi, 0.9 * math.pi, rays)
    half = math.radians(fov) / 2
    return math.pi / 2 + np.linspace(-half, half, rays)


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
        lidar_fov: Optional[float] = None,
        drop_heading: bool = False,
        lidar_max_dist: float = 45.0,
        lidar_step: float = 3.0,
        offtrack_penalty: float = 1.0,
        traction_threshold: float = 0.95,
        allow_multiple_laps: bool = True,
        lap_completion_bonus: float = 100.0,
        lap_time_bonus: float = 0.0,
        lap_time_budget: int = 1000,
        stuck_patience: Optional[int] = None,
        track_seeds: Optional[str] = None,
        action_smoothing: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.lidar_rays = lidar_rays
        self.lidar_max_dist = lidar_max_dist
        self.lidar_step = lidar_step
        self.offtrack_penalty = offtrack_penalty
        self.traction_threshold = traction_threshold
        self.allow_multiple_laps = allow_multiple_laps
        self.lap_completion_bonus = lap_completion_bonus
        self.lap_time_bonus = lap_time_bonus
        self.lap_time_budget = lap_time_budget
        self.stuck_patience = stuck_patience
        # "train": every reset without an explicit track draws one from rules.TRAIN_SEED_RANGE
        # (a seed passed to reset only seeds that draw), so training never sees a reserved track.
        self.track_seeds = track_seeds
        self._track_rng: Optional[np.random.Generator] = None
        self.action_smoothing = action_smoothing
        self._prev_action = np.zeros(3, dtype=np.float32)

        self.lidar_angles = lidar_angles(lidar_rays, lidar_fov)
        # cos/sin(heading) is the car's absolute compass direction: no use on a random track.
        self.drop_heading = drop_heading
        self.base_feature_count = 6 if drop_heading else 8
        self.last_traction = 1.0  # fraction of wheels on the road, whatever the state layout
        self.state_size = self.base_feature_count + lidar_rays

        # Re-declare observation space to match the state vector.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        self._road_polygons: List[np.ndarray] = []
        self._road_bounds: List[Tuple[float, float, float, float]] = []
        self._poly_array: Optional[np.ndarray] = None
        self._bounds_array: Optional[np.ndarray] = None
        self._laps_completed = 0
        self._steps = 0
        self._lap_start = 0
        self._since_tile = 0
        self._tiles_seen = 0
        spec = getattr(env, "spec", None)
        self._max_steps: Optional[int] = getattr(spec, "max_episode_steps", None)

    # ---------------------------------------------------------------------- #
    # Gym API
    # ---------------------------------------------------------------------- #
    def reset(self, **kwargs):
        if self.track_seeds == "train":
            from carracing.rules import training_seed

            if kwargs.get("seed") is not None or self._track_rng is None:
                self._track_rng = np.random.default_rng(kwargs.get("seed"))
            kwargs["seed"] = training_seed(self._track_rng)
        self._prev_action = np.zeros(3, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        info["track_seed"] = kwargs.get("seed")
        self._extract_road_geometry()
        self._laps_completed = 0
        self._steps = 0
        self._lap_start = 0
        self._since_tile = 0
        self._tiles_seen = 0
        state = self._build_state(obs)
        return state, info

    def step(self, action):
        if self.action_smoothing:
            from carracing.rules import smooth

            action = self._prev_action = smooth(action, self._prev_action, self.action_smoothing)
        # Cast to float64 to satisfy Box2D bindings.
        action = np.asarray(action, dtype=np.float64)
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._build_state(obs)

        # Encourage staying on track using the traction fraction.
        if self.offtrack_penalty > 0.0:
            delta = max(0.0, self.traction_threshold - self.last_traction)
            reward -= self.offtrack_penalty * delta

        self._steps += 1
        if self._lap_finished():
            self._laps_completed += 1
            # Speed reward: lap_time_bonus per frame the lap came in under budget.
            lap_frames = self._steps - self._lap_start
            self._lap_start = self._steps
            reward += self.lap_time_bonus * max(0, self.lap_time_budget - lap_frames)
            info["lap_frames"] = lap_frames
            if self.allow_multiple_laps:
                # CarRacing reports a finished lap as truncation. Pay the bonus,
                # re-arm every tile so the next lap earns tile rewards again, and
                # keep driving unless the time limit was hit on this same step.
                # Out-of-bounds (terminated, -100) is left untouched.
                reward += self.lap_completion_bonus
                self._start_new_lap()
                truncated = self._max_steps is not None and self._steps >= self._max_steps
        info["laps_completed"] = self._laps_completed

        if self.stuck_patience and not (terminated or truncated):
            # End the episode once no new tile has been reached for stuck_patience
            # frames (lost on the grass, parked, driving backwards). Charge the time
            # penalty the rest of the episode would have cost, so ending early never pays.
            tiles = int(getattr(self.env.unwrapped, "tile_visited_count", 0))
            self._since_tile = 0 if tiles > self._tiles_seen else self._since_tile + 1
            self._tiles_seen = tiles  # drops to 0 when a new lap re-arms the tiles
            if self._since_tile >= self.stuck_patience:
                terminated = True
                info["stuck"] = True
                if self._max_steps is not None:
                    reward -= 0.1 * max(0, self._max_steps - self._steps)

        return state, reward, terminated, truncated, info

    def _lap_finished(self) -> bool:
        base = self.env.unwrapped
        track = getattr(base, "track", None) or []
        return bool(getattr(base, "new_lap", False)) or (
            len(track) > 0 and getattr(base, "tile_visited_count", 0) == len(track)
        )

    def _start_new_lap(self) -> None:
        base = self.env.unwrapped
        base.new_lap = False
        base.tile_visited_count = 0
        for tile in getattr(base, "road", None) or []:
            tile.road_visited = False

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
            # Gymnasium's car_dynamics treats a wheel touching no road tile as on
            # grass. (Wheels have no `ground_traction` attribute, which once made
            # this feature a constant 1.0 and silently disabled the off-track penalty.)
            tiles = getattr(wheel, "tiles", None)
            if tiles is not None:
                traction_values.append(1.0 if tiles else 0.0)
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
        self.last_traction = traction_fraction

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

        if self.drop_heading:
            base_features = np.delete(base_features, [3, 4])
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

        # Stacked copies for the vectorised lidar. Shorter polygons are padded by
        # repeating their last vertex: a zero-length edge never counts as a crossing.
        if self._road_polygons:
            n_vertices = max(len(poly) for poly in self._road_polygons)
            padded = [
                np.vstack([poly, np.repeat(poly[-1:], n_vertices - len(poly), axis=0)])
                for poly in self._road_polygons
            ]
            self._poly_array = np.stack(padded).astype(np.float64)
            self._bounds_array = np.asarray(self._road_bounds, dtype=np.float64)
        else:
            self._poly_array = None
            self._bounds_array = None

    def _lidar_distances(self, origin: np.ndarray, heading: float) -> np.ndarray:
        """Distance along each ray to the first sample point that is off the road.

        All sample points (rays x steps) are tested against every nearby road
        polygon at once with a vectorised crossing-number test.
        """
        if self._poly_array is None:
            return np.ones(self.lidar_rays, dtype=np.float32)

        ox, oy = float(origin[0]), float(origin[1])
        bounds = self._bounds_array
        margin = self.lidar_max_dist + 2.0
        near = (
            (ox >= bounds[:, 0] - margin)
            & (ox <= bounds[:, 1] + margin)
            & (oy >= bounds[:, 2] - margin)
            & (oy <= bounds[:, 3] + margin)
        )
        if not near.any():
            return np.ones(self.lidar_rays, dtype=np.float32)

        polys = self._poly_array[near]                       # (P, V, 2)
        steps = np.arange(self.lidar_step, self.lidar_max_dist + 1e-9, self.lidar_step)  # (S,)
        angles = heading + self.lidar_angles                 # (R,)
        px = ox + np.cos(angles)[:, None] * steps[None, :]   # (R, S)
        py = oy + np.sin(angles)[:, None] * steps[None, :]

        x = px.reshape(-1, 1, 1)                             # (R*S, 1, 1)
        y = py.reshape(-1, 1, 1)
        xi, yi = polys[None, :, :, 0], polys[None, :, :, 1]  # (1, P, V)
        prev = np.roll(polys, 1, axis=1)                     # edge i -> i-1
        xj, yj = prev[None, :, :, 0], prev[None, :, :, 1]
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        on_road = ((crosses.sum(axis=2) % 2) == 1).any(axis=1).reshape(px.shape)  # (R, S)

        off_road = ~on_road
        hit = off_road.any(axis=1)
        first_off = off_road.argmax(axis=1)
        distance = np.where(hit, steps[first_off], self.lidar_max_dist)
        return np.clip(distance / self.lidar_max_dist, 0.0, 1.0).astype(np.float32)


def make_env(
    render_mode: Optional[str] = None,
    continuous: bool = True,
    domain_randomize: bool = False,
    seed: Optional[int] = None,
    offtrack_penalty: float = 1.0,
    traction_threshold: float = 0.95,
    allow_multiple_laps: bool = True,
    lap_completion_bonus: float = 100.0,
    max_episode_steps: Optional[int] = None,
    lidar_rays: int = 32,
    lap_time_bonus: float = 0.0,
    lidar_fov: Optional[float] = None,
    stuck_patience: Optional[int] = None,
    drop_heading: bool = False,
    track_seeds: Optional[str] = None,
    action_smoothing: float = 0.0,
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
    offtrack_penalty: float
        Penalty scale applied when traction falls below traction_threshold.
    traction_threshold: float
        Traction value above which no penalty is applied.
    allow_multiple_laps: bool
        If True, episodes continue after lap completion allowing multiple laps.
    lap_completion_bonus: float
        Bonus reward given when completing a lap (when allow_multiple_laps=True).
    max_episode_steps: Optional[int]
        Override the 1000-step time limit (a lap takes ~850 steps, so a second
        lap needs a longer limit or a faster driver).
    lidar_rays: int
        Lidar distances in the state (state size = 8 + lidar_rays).
    lidar_fov: Optional[float]
        None = original 324-degree layout; degrees = forward fan of +-fov/2 around the nose.
    drop_heading: bool
        Leave cos/sin(heading) out of the state (6 car readings instead of 8).
    stuck_patience: Optional[int]
        End the episode (with the remaining time penalty) after this many frames without a new tile.
    lap_time_bonus: float
        Reward per frame a finished lap comes in under 1000 frames (0 = off).
    kwargs:
        Passed through to gymnasium.make("CarRacing-v2", ...).
    """
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(
        "CarRacing-v2",
        render_mode=render_mode,
        continuous=continuous,
        domain_randomize=domain_randomize,
        **kwargs,
    )
    if render_mode is None:
        # The wrapper reads Box2D state, never pixels, so skip CarRacing's per-step
        # 96x96 software render (~2/3 of step time). A constant blank frame keeps
        # Gymnasium's observation checks happy.
        blank = np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)
        env.unwrapped._render = lambda mode, *a, **k: blank if mode == "state_pixels" else None
    if seed is not None:
        env.reset(seed=seed)
    return CarRacingStateWrapper(
        env,
        offtrack_penalty=offtrack_penalty,
        traction_threshold=traction_threshold,
        allow_multiple_laps=allow_multiple_laps,
        lap_completion_bonus=lap_completion_bonus,
        lidar_rays=lidar_rays,
        lidar_fov=lidar_fov,
        lap_time_bonus=lap_time_bonus,
        stuck_patience=stuck_patience,
        drop_heading=drop_heading,
        track_seeds=track_seeds,
        action_smoothing=action_smoothing,
    )
