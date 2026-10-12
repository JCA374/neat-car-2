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

from carracing_neat.eval import EvalSettings, evaluate_genome, run_episode
from carracing_ppo.env import make_env

# Held-out tracks for picking the best genome. Kept apart from the
# scripts/benchmark.py tracks so the page's benchmark stays unseen.
VALIDATION_SEEDS = [101, 202, 303, 404, 505]
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
    """Writes per-generation fitness stats to CSV for later plotting."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        dir_path = os.path.dirname(csv_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("generation,best_fitness,mean_fitness,stdev,time_s\n")
        self.start_time = time.time()
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

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

    def __init__(self, evaluator: SharedTrackEvaluator, checkpoint_dir: str, csv_path: str, every: int) -> None:
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir
        self.csv_path = csv_path
        self.every = max(1, every)
        self.generation = 0
        self.best_score: Optional[float] = None
        self.cache = {}  # genome key -> validation score (elites survive many generations)
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
        print(f"Validation gen {self.generation}: {score:.1f} on {len(VALIDATION_SEEDS)} held-out tracks{marker}")


class GenomeStatsReporter(neat.reporting.BaseReporter):
    """Log node and enabled-connection counts of the best genome per generation."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        dir_path = os.path.dirname(csv_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("generation,nodes,enabled_connections\n")
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome is None:
            return
        enabled_conns = [
            cg for cg in best_genome.connections.values() if cg.enabled
        ]
        input_keys = set(config.genome_config.input_keys)
        output_keys = set(config.genome_config.output_keys)
        hidden_keys = set(best_genome.nodes.keys()) - input_keys - output_keys
        node_count = len(input_keys) + len(output_keys) + len(hidden_keys)
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{self.generation},{node_count},{len(enabled_conns)}\n")


class VisualizationReporter(neat.reporting.BaseReporter):
    """Generate fitness and network plots after each generation."""

    def __init__(self, fitness_log: str, fitness_plot: str, checkpoint_dir: str, config_path: str, networks_dir: str) -> None:
        self.fitness_log = fitness_log
        self.fitness_plot = fitness_plot
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path
        self.networks_dir = networks_dir
        self.generation = 0
        os.makedirs(os.path.dirname(fitness_plot), exist_ok=True)
        os.makedirs(networks_dir, exist_ok=True)

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        try:
            # Import viz functions here to avoid circular imports
            from carracing_neat.viz import plot_fitness, plot_network

            # Generate fitness plot
            if os.path.exists(self.fitness_log):
                plot_fitness(self.fitness_log, self.fitness_plot)

            # Generate network plot for this generation's best genome
            if best_genome is not None and best_genome.fitness is not None:
                network_path = os.path.join(self.networks_dir, f"network_gen_{self.generation}.png")
                if not os.path.exists(network_path):
                    title = f"Generation {self.generation} - Best Fitness: {best_genome.fitness:.2f}"
                    plot_network(best_genome, config, network_path, title=title)
        except Exception as e:
            # Don't crash training if visualization fails
            print(f"Warning: Visualization failed for generation {self.generation}: {e}")


def _build_population(config_path: str, checkpoint: Optional[str]) -> neat.Population:
    if checkpoint and os.path.exists(checkpoint):
        print(f"Restoring from checkpoint: {checkpoint}")
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        pop = neat.Population(config)
    return pop


def _validate_state_size(config: neat.Config) -> None:
    env = make_env(render_mode=None)
    state, _ = env.reset(seed=0)
    env.close()
    if len(state) != config.genome_config.num_inputs:
        raise ValueError(
            f"Config expects {config.genome_config.num_inputs} inputs, "
            f"but environment produced {len(state)}."
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
    parser.add_argument("--legacy-decode", action="store_true",
                        help="Old double-tanh output decoding (only to reproduce runs before Oct 12 2026).")
    parser.add_argument("--validate-every", type=int, default=1, help="Validate the champion every N generations.")
    parser.add_argument("--checkpoint-dir", default="outputs/neat/checkpoints", help="Checkpoints and best genomes.")
    parser.add_argument("--log-dir", default="outputs/neat/logs", help="fitness.csv, genome_stats.csv, validation.csv.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N generations.")
    parser.add_argument("--resume", default="auto", help='Checkpoint path, "auto" (newest in checkpoint-dir) or "none".')
    parser.add_argument("--auto-visualize", action="store_true", help="Write fitness and network plots each generation.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    fitness_log = os.path.join(args.log_dir, "fitness.csv")
    genome_log = os.path.join(args.log_dir, "genome_stats.csv")
    validation_log = os.path.join(args.log_dir, "validation.csv")

    resume_ckpt = None
    if args.resume == "auto":
        resume_ckpt = _latest_checkpoint(args.checkpoint_dir)
        print(f"Auto-resume: {resume_ckpt}" if resume_ckpt else "Auto-resume: no checkpoint found, starting fresh.")
    elif args.resume and args.resume != "none":
        resume_ckpt = args.resume
    if resume_ckpt is None:
        _reset_logs([fitness_log, genome_log, validation_log])

    pop = _build_population(args.config, resume_ckpt)
    if resume_ckpt is not None:
        _drop_rows_from([fitness_log, genome_log, validation_log], pop.generation)
    config = pop.config
    _validate_state_size(config)

    settings = EvalSettings(
        max_steps=args.max_steps,
        action_smoothing=args.action_smoothing,
        offtrack_penalty=args.offtrack_penalty,
        traction_threshold=args.traction_threshold,
        domain_randomize=args.domain_randomize,
        patience=args.patience,
        legacy_decode=args.legacy_decode,
    )
    evaluator = SharedTrackEvaluator(pop, settings, args.num_workers, args.tracks_per_generation, args.seed)

    reporters = [
        neat.StdOutReporter(show_species_detail=False),
        neat.StatisticsReporter(),
        CSVReporter(fitness_log),
        GenomeStatsReporter(genome_log),
        neat.Checkpointer(
            generation_interval=args.checkpoint_interval,
            filename_prefix=os.path.join(args.checkpoint_dir, "neat-checkpoint-"),
        ),
        ValidationReporter(evaluator, args.checkpoint_dir, validation_log, args.validate_every),
    ]
    if args.auto_visualize:
        plots = os.path.join(os.path.dirname(args.log_dir.rstrip("/")) or ".", "plots")
        reporters.append(
            VisualizationReporter(
                fitness_log=fitness_log,
                fitness_plot=os.path.join(plots, "fitness.png"),
                checkpoint_dir=args.checkpoint_dir,
                config_path=args.config,
                networks_dir=os.path.join(plots, "networks"),
            )
        )
    for rep in reporters:
        pop.add_reporter(rep)

    target = pop.generation + args.generations
    print(f"Running generations {pop.generation}..{target - 1} with {args.num_workers} workers, "
          f"{args.tracks_per_generation} shared tracks per generation.")
    try:
        winner = pop.run(evaluator.evaluate, args.generations)
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
