"""Step files, the ranking rule and training-time bookkeeping.

A *step* is one rung on a model's improvement ladder: ``experiments/<model>/<id>.toml``
holds what changed and why, and ``outputs/<model>/<id>/`` holds what came out
(``model.pkl``/``model.zip``, ``result.json``, ``drives.json``, logs).
"""

from __future__ import annotations

import csv
import glob
import os
import platform
import tomllib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from carracing.rules import FPS  # noqa: F401  (re-exported: CarRacing runs at 50 frames per second)

MODELS = ("neat", "ppo")


@dataclass
class Step:
    id: str
    model: str
    title: str
    why: str = ""
    outcome: str = ""                 # what happened, written after the run (shown on the page)
    parent: Optional[str] = None
    lidar_rays: int = 32
    lidar_fov: Optional[float] = None  # None = original 324-degree layout
    drop_heading: bool = False         # leave cos/sin(heading) out of the state
    championship: bool = False         # pick two checkpoints from the lap check: first to finish, fastest
    action_smoothing: float = 0.0      # PPO: blend with the previous action (NEAT does this in its decoder)
    seeds: List[int] = field(default_factory=list)  # PPO: train once per seed, keep the best on validation
    # Compatibility with models trained before Oct 12 2026 (n00/p00 only).
    legacy_decode: bool = False
    legacy_traction: bool = False
    train: Dict = field(default_factory=dict)        # --flags for the trainer
    neat_config: Dict = field(default_factory=dict)  # "Section.key" -> value overrides of neat.cfg
    imported: Dict = field(default_factory=dict)     # [import]: existing artefacts instead of training
    path: str = ""
    out_override: str = ""             # set for one seed's run of a multi-seed step

    @property
    def out_dir(self) -> str:
        return self.out_override or os.path.join("outputs", self.model, self.id)

    def env_kwargs(self) -> Dict:
        """The sensor settings this step's model was trained with (for make_env)."""
        return {"lidar_rays": self.lidar_rays, "lidar_fov": self.lidar_fov, "drop_heading": self.drop_heading}

    def env_flags(self) -> List[str]:
        """The same settings as trainer flags."""
        flags = ["--lidar-rays", str(self.lidar_rays)]
        if self.lidar_fov is not None:
            flags += ["--lidar-fov", str(self.lidar_fov)]
        if self.drop_heading:
            flags.append("--drop-heading")
        if self.model == "ppo" and self.action_smoothing:
            flags += ["--action-smoothing", str(self.action_smoothing)]
        return flags

    @property
    def state_inputs(self) -> int:
        return (6 if self.drop_heading else 8) + self.lidar_rays

    def model_file(self, category: str) -> str:
        """model_finish.pkl / model_fast.zip ... for championship checkpoints."""
        return os.path.join(self.out_dir, f"model_{category}." + ("pkl" if self.model == "neat" else "zip"))

    @property
    def model_path(self) -> str:
        return os.path.join(self.out_dir, "model.pkl" if self.model == "neat" else "model.zip")


def load_step(path: str) -> Step:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    model = raw["model"]
    if model not in MODELS:
        raise ValueError(f"{path}: model must be one of {MODELS}, got {model!r}")
    expected = os.path.splitext(os.path.basename(path))[0]
    if raw["id"] != expected:
        raise ValueError(f"{path}: id {raw['id']!r} must match the file name {expected!r}")
    return Step(
        id=raw["id"], model=model, title=raw["title"], why=raw.get("why", "").strip(),
        outcome=raw.get("outcome", "").strip(),
        parent=raw.get("parent"), lidar_rays=int(raw.get("lidar_rays", 32)),
        lidar_fov=float(raw["lidar_fov"]) if "lidar_fov" in raw else None,
        drop_heading=bool(raw.get("drop_heading", False)),
        championship=bool(raw.get("championship", False)),
        action_smoothing=float(raw.get("action_smoothing", 0.0)),
        seeds=[int(s) for s in raw.get("seeds", [])],
        legacy_decode=bool(raw.get("legacy_decode", False)),
        legacy_traction=bool(raw.get("legacy_traction", False)),
        train=dict(raw.get("train", {})), neat_config=dict(raw.get("neat_config", {})),
        imported=dict(raw.get("import", {})), path=path,
    )


def all_steps(root: str = "experiments") -> List[Step]:
    steps = [load_step(p) for p in sorted(glob.glob(os.path.join(root, "*", "*.toml")))]
    return [s for s in steps if s.model in MODELS]


def train_flags(train: Dict) -> List[str]:
    """{"generations": 100, "domain-randomize": true} -> ["--generations", "100", "--domain-randomize"]."""
    flags: List[str] = []
    for key, value in train.items():
        if value is True:
            flags.append(f"--{key}")
        elif value is False:
            flags.append(f"--no-{key}")
        elif isinstance(value, list):
            flags += [f"--{key}", ",".join(str(v) for v in value)]
        else:
            flags += [f"--{key}", str(value)]
    return flags


# --------------------------------------------------------------- ranking ---
def summarise(tracks: List[Dict]) -> Dict:
    """Headline numbers from per-track benchmark results."""
    laps = [t["lap_s"] for t in tracks if t["finished"]]
    return {
        "finished": len(laps),
        "lap_s": round(sum(laps) / len(laps), 2) if laps else None,
        "score": round(sum(t["score"] for t in tracks) / len(tracks), 1),
        "offtrack_pct": round(sum(t["offtrack_pct"] for t in tracks) / len(tracks), 1),
    }


def rank_key(result: Dict):
    """Sort key: most tracks finished, then fastest mean lap, then least training, then smallest."""
    s = result["summary"]
    return (-s["finished"], s["lap_s"] if s["lap_s"] is not None else float("inf"),
            result["train_seconds"], result["size"]["params"])


# --------------------------------------------------------- training time ---
def neat_train_seconds(fitness_csv: str, up_to_generation: Optional[int] = None) -> float:
    """Wall-clock seconds spent training up to (and including) a generation.

    ``time_s`` counts from the start of the training process; older runs restart
    it from zero on every resume, so add up the maximum of each segment.
    """
    total, segment, prev = 0.0, 0.0, -1.0
    with open(fitness_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if up_to_generation is not None and int(row["generation"]) > up_to_generation:
                break
            t = float(row["time_s"])
            if t < prev:
                total += segment
            segment = prev = t
    return round(total + segment, 1)


def ppo_train_seconds(log_dir: str, at_timestep: Optional[int] = None) -> float:
    """Wall-clock seconds until the training envs had made ``at_timestep`` steps.

    Uses the Monitor CSVs: each row is an episode with its length ``l`` and the
    time ``t`` since the run started. Without ``at_timestep``, the whole run.
    """
    episodes = []
    for path in glob.glob(os.path.join(log_dir, "monitor", "*.monitor.csv")):
        with open(path, newline="") as fh:
            next(fh)  # '#{"t_start": ...}' header
            # A run stopped mid-write can leave a truncated last row.
            episodes += [(float(r["t"]), int(r["l"])) for r in csv.DictReader(fh) if r.get("t") and r.get("l")]
    if not episodes:
        raise FileNotFoundError(f"No monitor CSVs in {log_dir}/monitor")
    episodes.sort()
    done = 0
    for t, length in episodes:
        done += length
        if at_timestep is not None and done >= at_timestep:
            return round(t, 1)
    return round(episodes[-1][0], 1)


def hardware() -> Dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(line.split(":", 1)[1].strip() for line in fh if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "threads": os.cpu_count(), "gpu": False}


def fmt_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"
