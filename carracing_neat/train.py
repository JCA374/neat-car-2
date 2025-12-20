"""Training entrypoint for NEAT on Gymnasium's CarRacing."""

from __future__ import annotations

import argparse
import functools
import glob
import os
import pickle
import sys
import time
from typing import Optional, Tuple

# Allow running as a script without installing the package.
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import neat
from neat import parallel as neat_parallel

from carracing_neat.env_state import CarRacingStateWrapper, make_env
from carracing_neat.eval import EvalSettings, evaluate_genome


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


class BestGenomeSaver(neat.reporting.BaseReporter):
    """Persist the best genome seen so far for resumable evaluation."""

    def __init__(self, path: str, save_all: bool = False, checkpoint_dir: str = None) -> None:
        self.path = path
        self.save_all = save_all
        self.checkpoint_dir = checkpoint_dir
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.best_fitness = None
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome is None or best_genome.fitness is None:
            return

        # Save the best genome of this generation if save_all is enabled
        if self.save_all and self.checkpoint_dir:
            gen_path = os.path.join(self.checkpoint_dir, f"best-genome-gen-{self.generation}.pkl")
            with open(gen_path, "wb") as f:
                pickle.dump(best_genome, f)

        # Save the overall best genome
        if self.best_fitness is None or best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            with open(self.path, "wb") as f:
                pickle.dump(best_genome, f)


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


def _validate_state_size(config: neat.Config, env_kwargs) -> None:
    env = make_env(**env_kwargs)
    state, _ = env.reset(seed=env_kwargs.get("seed", 0))
    env.close()
    if len(state) != config.genome_config.num_inputs:
        raise ValueError(
            f"Config expects {config.genome_config.num_inputs} inputs, "
            f"but environment produced {len(state)}."
        )


def _patch_parallel_cleanup():
    """Guard ParallelEvaluator.__del__ to avoid errors when pool creation fails."""

    def _safe_del(self):
        pool = getattr(self, "pool", None)
        try:
            if pool is not None:
                pool.close()
        except Exception:
            pass

    neat_parallel.ParallelEvaluator.__del__ = _safe_del


def _reset_logs(log_paths):
    """Remove existing log files when starting a fresh run to keep generations monotonic."""
    for path in log_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _truncate_logs(log_paths, cutoff_generation: int):
    """Drop log rows with generation >= cutoff_generation when resuming from a checkpoint."""
    for path in log_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue
            header, data = lines[0], lines[1:]
            kept = []
            for line in data:
                parts = line.strip().split(",")
                if not parts:
                    continue
                try:
                    gen_val = int(float(parts[0]))
                except Exception:
                    continue
                if gen_val < cutoff_generation:
                    kept.append(line)
            with open(path, "w", encoding="utf-8") as f:
                f.write(header)
                f.writelines(kept)
        except Exception:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train NEAT on CarRacing (CPU-only).")
    parser.add_argument("--config", default="neat_config/neat.cfg", help="Path to NEAT config file.")
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Generations to run from the current generation (adds to checkpoint generation).",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel evaluator workers.")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per genome evaluation.")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints", help="Where to store checkpoints.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Checkpoint every N generations.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help='Checkpoint path or "auto" to resume from the newest checkpoint if present.',
    )
    parser.add_argument("--log-file", default="outputs/logs/fitness.csv", help="CSV log output path.")
    parser.add_argument(
        "--genome-log-file",
        default="outputs/logs/genome_stats.csv",
        help="CSV log for best-genome node/connection counts.",
    )
    parser.add_argument("--action-noise", type=float, default=0.0, help="Gaussian action noise (std).")
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.1,
        help="Blend factor for previous action [0, 0.95].",
    )
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="Enable Gymnasium domain_randomize flag.",
    )
    parser.add_argument(
        "--track-seed",
        type=int,
        default=123,
        help="Fixed seed to lock the track layout during training (None = random each run).",
    )
    parser.add_argument(
        "--auto-visualize",
        action="store_true",
        help="Generate fitness and network plots automatically after each generation.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    _patch_parallel_cleanup()
    args = parse_args(argv)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    resume_ckpt = None
    if args.resume == "auto":
        resume_ckpt = _latest_checkpoint(args.checkpoint_dir)
        if resume_ckpt:
            print(f"Auto-resume: using checkpoint {resume_ckpt}")
        else:
            print("Auto-resume: no checkpoint found, starting fresh.")
    elif args.resume:
        resume_ckpt = args.resume

    if resume_ckpt is None:
        _reset_logs([args.log_file, args.genome_log_file])

    pop = _build_population(args.config, resume_ckpt)

    env_kwargs = {
        "render_mode": None,
        "domain_randomize": args.domain_randomize,
        "continuous": True,
        "seed": args.track_seed,
    }
    try:
        _validate_state_size(pop.config, env_kwargs)
    except ValueError as exc:
        if resume_ckpt:
            print(
                f"Checkpoint config mismatch ({exc}). Starting fresh without checkpoint.",
                file=sys.stderr,
            )
            pop = _build_population(args.config, None)
            _reset_logs([args.log_file, args.genome_log_file])
            _validate_state_size(pop.config, env_kwargs)
            resume_ckpt = None
        else:
            raise

    config = pop.config

    eval_settings = EvalSettings(
        max_steps=args.max_steps,
        episodes=args.episodes,
        action_noise=args.action_noise,
        action_smoothing=args.action_smoothing,
        domain_randomize=args.domain_randomize,
    )

    eval_fn = functools.partial(
        evaluate_genome, env_kwargs=env_kwargs, settings=eval_settings
    )

    def eval_population(genomes, cfg):
        for _, genome in genomes:
            genome.fitness = eval_fn(genome, cfg)

    reporter_list = [
        neat.StdOutReporter(show_species_detail=False),
        neat.StatisticsReporter(),
        CSVReporter(args.log_file),
        GenomeStatsReporter(args.genome_log_file),
        neat.Checkpointer(
            generation_interval=args.checkpoint_interval,
            filename_prefix=os.path.join(args.checkpoint_dir, "neat-checkpoint-"),
        ),
        BestGenomeSaver(
            os.path.join(args.checkpoint_dir, "best-genome.pkl"),
            save_all=True,
            checkpoint_dir=args.checkpoint_dir
        ),
    ]

    if args.auto_visualize:
        reporter_list.append(
            VisualizationReporter(
                fitness_log=args.log_file,
                fitness_plot="outputs/plots/fitness.png",
                checkpoint_dir=args.checkpoint_dir,
                config_path=args.config,
                networks_dir="outputs/plots/networks"
            )
        )

    for rep in reporter_list:
        pop.add_reporter(rep)

    target_generation = pop.generation + args.generations
    if pop.generation > 0:
        print(
            f"Resuming at generation {pop.generation}; running {args.generations} more "
            f"(target generation {target_generation})."
        )
    else:
        print(f"Starting fresh for {args.generations} generations (target {target_generation}).")

    try:
        if args.num_workers > 1:
            try:
                pe = neat.parallel.ParallelEvaluator(args.num_workers, eval_fn)
                winner = pop.run(pe.evaluate, target_generation)
            except Exception as exc:
                print(
                    f"Parallel workers unavailable ({exc}); falling back to serial.",
                    file=sys.stderr,
                )
                winner = pop.run(eval_population, target_generation)
        else:
            winner = pop.run(eval_population, target_generation)
    except KeyboardInterrupt:
        print("Interrupted. Latest checkpoint preserved.", file=sys.stderr)
        return 1

    best_path = os.path.join(args.checkpoint_dir, "best-genome-final.pkl")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(best_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Training complete. Best genome saved to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
