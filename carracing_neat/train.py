"""Training entrypoint for NEAT on Gymnasium's CarRacing."""

from __future__ import annotations

import argparse
import dataclasses
import glob
import multiprocessing
import os
import pickle
import sys
import time
from typing import List, Optional, Tuple

import neat
import numpy as np

from carracing_neat.eval import EvalSettings, evaluate_genome, load_config, run_episode
from carracing.env import make_env

# Held-out tracks for picking the best genome. Kept apart from the
# carracing/benchmark.py tracks so the page's benchmark stays unseen.
VALIDATION_SEEDS = [101, 202, 303, 404, 505]
LAP_FRAMES = 2000  # same frame limit as the benchmark
RESERVED_SEEDS = set(VALIDATION_SEEDS) | {3, 11, 21, 34, 55}


def generation_seeds(generation: int, count: int, base_seed: int) -> List[int]:
    """Tracks shared by every genome in one generation (fresh each generation)."""
    rng = np.random.default_rng(base_seed + generation)
    seeds: List[int] = []
    while len(seeds) < count:
        seed = int(rng.integers(1_000, 1_000_000))
        if seed not in RESERVED_SEEDS and seed not in seeds:
            seeds.append(seed)
    return seeds


def _validate_genome(genome, config, settings: EvalSettings) -> float:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return run_episode(net, settings, settings.seeds[0], stop_when_stuck=False)["reward"]


def _lap_genome(genome, config, settings: EvalSettings) -> Tuple[bool, int]:
    """(finished, frames) for one validation lap in the benchmark's conditions."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    result = run_episode(net, settings, settings.seeds[0], stop_when_stuck=True)
    return result["laps"] >= 1, result["steps"]


class TrainingBudgetReached(Exception):
    """Raised by BudgetReporter to end a run cleanly at its time budget or patience."""


class SharedTrackEvaluator:
    """Evaluates a generation in parallel; all genomes drive the same tracks."""

    def __init__(self, population: neat.Population, settings: EvalSettings, num_workers: int,
                 tracks_per_generation: int, base_seed: int) -> None:
        self.population = population
        self.settings = settings
        self.tracks = tracks_per_generation
        self.base_seed = base_seed
        self.pool = multiprocessing.get_context("fork").Pool(num_workers) if num_workers > 1 else None

    def __getstate__(self):
        # neat's Checkpointer pickles the species set, which references the
        # reporters (and through ValidationReporter, this evaluator). Pools and
        # the live population can't or shouldn't be pickled.
        state = self.__dict__.copy()
        state["pool"] = None
        state["population"] = None
        return state

    def _map(self, fn, arg_list):
        if self.pool is None:
            return [fn(*args) for args in arg_list]
        return self.pool.starmap(fn, arg_list)

    def evaluate(self, genomes, config) -> None:
        seeds = generation_seeds(self.population.generation, self.tracks, self.base_seed)
        settings = dataclasses.replace(self.settings, seeds=seeds)
        fitnesses = self._map(evaluate_genome, [(g, config, settings) for _, g in genomes])
        for (_, genome), fitness in zip(genomes, fitnesses):
            genome.fitness = fitness

    def validate(self, genome, config) -> float:
        """Mean full-length reward of one genome on the held-out tracks."""
        jobs = [(genome, config, dataclasses.replace(self.settings, seeds=[s])) for s in VALIDATION_SEEDS]
        return float(np.mean(self._map(_validate_genome, jobs)))

    def validate_laps(self, genome, config) -> dict:
        """Lap check on the held-out tracks, in the benchmark's conditions: raw reward, 2000 frames."""
        raw = dataclasses.replace(self.settings, offtrack_penalty=0.0, lap_time_bonus=0.0, max_steps=LAP_FRAMES)
        laps = self._map(_lap_genome, [(genome, config, dataclasses.replace(raw, seeds=[s])) for s in VALIDATION_SEEDS])
        done = [frames for finished, frames in laps if finished]
        return {"finished": len(done), "mean_lap_s": round(float(np.mean(done)) / 50.0, 3) if done else None}

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


def _latest_checkpoint(path: str) -> Optional[str]:
    pattern = os.path.join(path, "neat-checkpoint-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _gen_and_mtime(p: str) -> Tuple[int, float]:
        fname = os.path.basename(p)
        parts = fname.split("-")
        gen = -1
        if parts and parts[-1].isdigit():
            gen = int(parts[-1])
        return gen, os.path.getmtime(p)

    candidates.sort(key=lambda p: _gen_and_mtime(p))
    return candidates[-1]


class CSVReporter(neat.reporting.BaseReporter):
    """Per-generation fitness stats. ``time_s`` is training wall-clock so far,
    carried over across resumes (see carracing.steps.neat_train_seconds)."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        dir_path = os.path.dirname(csv_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("generation,best_fitness,mean_fitness,stdev,time_s\n")
        offset = 0.0
        with open(csv_path, encoding="utf-8") as f:
            rows = [line for line in f.readlines()[1:] if line.strip()]
        if rows:
            offset = float(rows[-1].split(",")[4])
        self.start_time = time.time() - offset
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitnesses:
            return
        best = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        stdev = (sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
        elapsed = time.time() - self.start_time
        generation = self.generation
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{generation},{best:.3f},{mean:.3f},{stdev:.3f},{elapsed:.1f}\n")


class ValidationReporter(neat.reporting.BaseReporter):
    """Score each generation's champion on held-out tracks; keep the best one.

    Training fitness comes from a few random tracks, so a high value can be luck.
    The saved ``best-genome.pkl`` is chosen by validation score instead.
    """

    def __init__(self, evaluator: SharedTrackEvaluator, checkpoint_dir: str, csv_path: str, every: int,
                 clock: Optional["CSVReporter"] = None, laps_csv: Optional[str] = None) -> None:
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.csv_path = csv_path
        self.every = max(1, every)
        self.generation = 0
        self.best_score: Optional[float] = None
        self.cache = {}  # genome key -> validation score (elites survive many generations)
        self.lap_cache = {}
        self.clock = clock
        self.laps_csv = laps_csv
        self.last_laps = None
        if laps_csv and not os.path.exists(laps_csv):
            with open(laps_csv, "w", encoding="utf-8") as f:
                f.write("generation,time_s,finished,mean_lap_s\n")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if os.path.dirname(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("generation,train_fitness,validation_score,hidden_nodes,enabled_connections\n")
        else:
            with open(csv_path, encoding="utf-8") as f:
                scores = [float(line.split(",")[2]) for line in f.readlines()[1:] if line.strip()]
            self.best_score = max(scores) if scores else None

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome is None or self.generation % self.every:
            return
        if best_genome.key not in self.cache:
            self.cache[best_genome.key] = self.evaluator.validate(best_genome, config)
        score = self.cache[best_genome.key]
        hidden = len(set(best_genome.nodes) - set(config.genome_config.output_keys))
        enabled = sum(1 for c in best_genome.connections.values() if c.enabled)
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{self.generation},{best_genome.fitness:.3f},{score:.3f},{hidden},{enabled}\n")
        with open(os.path.join(self.checkpoint_dir, f"best-genome-gen-{self.generation}.pkl"), "wb") as f:
            pickle.dump(best_genome, f)
        marker = ""
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            with open(os.path.join(self.checkpoint_dir, "best-genome.pkl"), "wb") as f:
                pickle.dump(best_genome, f)
            marker = "  (new best)"
        laps = ""
        if self.laps_csv:
            if best_genome.key not in self.lap_cache:
                self.lap_cache[best_genome.key] = self.evaluator.validate_laps(best_genome, config)
            self.last_laps = self.lap_cache[best_genome.key]
            # Validation counts as training time: stamp it after the check has run.
            with open(self.laps_csv, "a", encoding="utf-8") as f:
                lap = self.last_laps["mean_lap_s"]
                f.write(f"{self.generation},{self.clock.elapsed():.1f},{self.last_laps['finished']},"
                        f"{'' if lap is None else lap}\n")
            laps = f", laps {self.last_laps['finished']}/5" + (f" in {lap:.2f} s" if lap else "")
        print(f"Validation gen {self.generation}: {score:.1f} on {len(VALIDATION_SEEDS)} held-out tracks{laps}{marker}")


def lap_key(laps: dict) -> Tuple[int, float]:
    """Higher is better: more validation tracks finished, then a faster mean lap."""
    return laps["finished"], -(laps["mean_lap_s"] or 1e9)


class BudgetReporter(neat.reporting.BaseReporter):
    """Stop at a wall-clock budget, or once the lap check has not improved for a while."""

    def __init__(self, clock: "CSVReporter", validation: ValidationReporter,
                 max_hours: Optional[float], patience_hours: Optional[float]) -> None:
        self.clock = clock
        self.validation = validation
        self.max_s = max_hours * 3600 if max_hours else None
        self.patience_s = patience_hours * 3600 if patience_hours else None
        self.best = None
        self.best_at = 0.0

    def post_evaluate(self, config, population, species, best_genome):
        now = self.clock.elapsed()
        laps = self.validation.last_laps
        if laps is not None and (self.best is None or lap_key(laps) > self.best):
            self.best, self.best_at = lap_key(laps), now
        if self.max_s and now >= self.max_s:
            raise TrainingBudgetReached(f"time budget of {self.max_s / 3600:g} h reached")
        if self.patience_s and self.best is not None and now - self.best_at >= self.patience_s:
            raise TrainingBudgetReached(f"no faster validation lap for {self.patience_s / 3600:g} h")


class SpeciesTargetReporter(neat.reporting.BaseReporter):
    """Steer compatibility_threshold toward a target number of species.

    With a fully connected start, genomes differ by small distances relative to
    their size, so a fixed threshold left the whole population in one species
    (no protection for new structure). After each evaluation, nudge the
    threshold used by the next speciation step and log species counts.
    """

    def __init__(self, target: int, step: float, csv_path: str, minimum: float = 0.3) -> None:
        self.target = target
        self.step = step
        self.minimum = minimum
        self.csv_path = csv_path
        self.generation = 0
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("generation,species,threshold,sizes\n")

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        count = len(species.species)
        cfg = config.species_set_config
        sizes = " ".join(str(len(s.members)) for s in species.species.values())
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{self.generation},{count},{cfg.compatibility_threshold:.3f},{sizes}\n")
        if count < self.target:
            cfg.compatibility_threshold = max(self.minimum, cfg.compatibility_threshold - self.step)
        elif count > self.target:
            cfg.compatibility_threshold += self.step


def _apply_speciation_settings(config: neat.Config, config_path: str) -> None:
    """A restored checkpoint carries its own pickled config. Re-apply the
    speciation/stagnation/reproduction settings from the config file in place,
    because the live species set, stagnation and reproduction objects hold
    references to these exact config objects."""
    fresh = load_config(config_path)
    for section in ("species_set_config", "stagnation_config", "reproduction_config"):
        getattr(config, section).__dict__.update(getattr(fresh, section).__dict__)



def _build_population(config_path: str, checkpoint: Optional[str]) -> neat.Population:
    if checkpoint and os.path.exists(checkpoint):
        print(f"Restoring from checkpoint: {checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        pop = neat.Population(load_config(config_path))
    return pop


def _validate_state_size(config: neat.Config, settings: EvalSettings) -> None:
    env = make_env(render_mode=None, lidar_rays=settings.lidar_rays, lidar_fov=settings.lidar_fov,
                   drop_heading=settings.drop_heading)
    state, _ = env.reset(seed=0)
    env.close()
    if len(state) != config.genome_config.num_inputs:
        raise ValueError(
            f"Config expects {config.genome_config.num_inputs} inputs, "
            f"but environment produced {len(state)} (car readings + --lidar-rays)."
        )


def _drop_rows_from(log_paths, generation: int) -> None:
    """neat-python restores a checkpoint under the generation it was saved at,
    so that generation runs again; drop its old rows to keep logs monotonic."""
    for path in log_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            header, *rows = f.readlines()
        kept = [r for r in rows if r.strip() and int(r.split(",")[0]) < generation]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines([header, *kept])


def _reset_logs(log_paths):
    """Remove existing log files when starting a fresh run to keep generations monotonic."""
    for path in log_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train NEAT on CarRacing (CPU-only).")
    parser.add_argument("--config", default="neat_config/neat.cfg", help="Path to NEAT config file.")
    parser.add_argument("--generations", type=int, default=50, help="Generations to run (added to a resumed run).")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Parallel workers.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--tracks-per-generation", type=int, default=3,
                        help="Random tracks every genome drives each generation (shared within a generation).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the per-generation track draw.")
    parser.add_argument("--patience", type=int, default=100,
                        help="End an episode after this many steps without reaching a new tile.")
    parser.add_argument("--offtrack-penalty", type=float, default=0.0, help="Wrapper off-track penalty (0 = raw score).")
    parser.add_argument("--traction-threshold", type=float, default=0.95, help="Traction below which the penalty applies.")
    parser.add_argument("--action-smoothing", type=float, default=0.1, help="Blend factor for previous action [0, 0.95].")
    parser.add_argument("--domain-randomize", action="store_true", help="Enable Gymnasium domain randomization.")
    parser.add_argument("--lidar-rays", type=int, default=32, help="Lidar inputs (config num_inputs must be 8 + this).")
    parser.add_argument("--lidar-fov", type=float, default=None,
                        help="Forward lidar fan in degrees (default: original 324-degree layout).")
    parser.add_argument("--drop-heading", action="store_true", help="Leave cos/sin(heading) out of the state.")
    parser.add_argument("--lap-time-bonus", type=float, default=0.0,
                        help="Fitness per frame a lap finishes under 1000 frames (0 = raw score).")
    parser.add_argument("--legacy-decode", action="store_true",
                        help="Old double-tanh output decoding (only to reproduce runs before Oct 12 2026).")
    parser.add_argument("--target-species", type=int, default=6,
                        help="Adjust compatibility_threshold each generation toward this many species (0 = fixed).")
    parser.add_argument("--threshold-step", type=float, default=0.1, help="Threshold change per generation.")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate the champion every N generations.")
    parser.add_argument("--max-hours", type=float, default=None, help="Stop after this much training wall-clock.")
    parser.add_argument("--patience-hours", type=float, default=None,
                        help="Stop when the validation lap check has not improved for this long.")
    parser.add_argument("--checkpoint-dir", default="outputs/neat/dev/checkpoints", help="Checkpoints and best genomes.")
    parser.add_argument("--log-dir", default="outputs/neat/dev/logs", help="fitness.csv, validation.csv, species.csv.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N generations.")
    parser.add_argument("--resume", default="auto", help='Checkpoint path, "auto" (newest in checkpoint-dir) or "none".')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    fitness_log = os.path.join(args.log_dir, "fitness.csv")
    validation_log = os.path.join(args.log_dir, "validation.csv")
    species_log = os.path.join(args.log_dir, "species.csv")
    laps_log = os.path.join(args.log_dir, "validation_laps.csv")

    resume_ckpt = None
    if args.resume == "auto":
        resume_ckpt = _latest_checkpoint(args.checkpoint_dir)
        print(f"Auto-resume: {resume_ckpt}" if resume_ckpt else "Auto-resume: no checkpoint found, starting fresh.")
    elif args.resume and args.resume != "none":
        resume_ckpt = args.resume
    if resume_ckpt is None:
        _reset_logs([fitness_log, validation_log, species_log, laps_log])

    pop = _build_population(args.config, resume_ckpt)
    if resume_ckpt is not None:
        _drop_rows_from([fitness_log, validation_log, species_log, laps_log], pop.generation)
        _apply_speciation_settings(pop.config, args.config)
    config = pop.config

    settings = EvalSettings(
        max_steps=args.max_steps,
        action_smoothing=args.action_smoothing,
        offtrack_penalty=args.offtrack_penalty,
        traction_threshold=args.traction_threshold,
        domain_randomize=args.domain_randomize,
        patience=args.patience,
        legacy_decode=args.legacy_decode,
        lidar_rays=args.lidar_rays,
        lidar_fov=args.lidar_fov,
        drop_heading=args.drop_heading,
        lap_time_bonus=args.lap_time_bonus,
    )
    _validate_state_size(config, settings)
    evaluator = SharedTrackEvaluator(pop, settings, args.num_workers, args.tracks_per_generation, args.seed)

    clock = CSVReporter(fitness_log)
    if args.max_hours and clock.elapsed() >= args.max_hours * 3600:
        print(f"Time budget of {args.max_hours:g} h already used ({clock.elapsed() / 3600:.2f} h); nothing to train.")
        evaluator.close()
        return 0
    validation = ValidationReporter(evaluator, args.checkpoint_dir, validation_log, args.validate_every,
                                    clock=clock, laps_csv=laps_log)
    reporters = [
        neat.StdOutReporter(show_species_detail=False),
        neat.StatisticsReporter(),
        clock,
        neat.Checkpointer(
            generation_interval=args.checkpoint_interval,
            filename_prefix=os.path.join(args.checkpoint_dir, "neat-checkpoint-"),
        ),
        validation,
    ]
    if args.target_species > 0:
        reporters.append(SpeciesTargetReporter(args.target_species, args.threshold_step, species_log))
    if args.max_hours or args.patience_hours:
        reporters.append(BudgetReporter(clock, validation, args.max_hours, args.patience_hours))
    for rep in reporters:
        pop.add_reporter(rep)

    target = pop.generation + args.generations
    print(f"Running generations {pop.generation}..{target - 1} with {args.num_workers} workers, "
          f"{args.tracks_per_generation} shared tracks per generation.")
    try:
        winner = pop.run(evaluator.evaluate, args.generations)
    except TrainingBudgetReached as stop:
        print(f"Stopped: {stop}. Best validated genome: {os.path.join(args.checkpoint_dir, 'best-genome.pkl')}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Latest checkpoint and best genome preserved.", file=sys.stderr)
        return 1
    finally:
        evaluator.close()

    with open(os.path.join(args.checkpoint_dir, "best-genome-final.pkl"), "wb") as f:
        pickle.dump(winner, f)
    print(f"Done. Best validated genome: {os.path.join(args.checkpoint_dir, 'best-genome.pkl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
