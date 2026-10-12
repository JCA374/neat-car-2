"""NEAT pipeline: fair (shared-track) evaluation and checkpoint-safe evaluator."""

from __future__ import annotations

import dataclasses
import pickle

import neat

from carracing_neat.eval import EvalSettings, evaluate_genome
from carracing_neat.train import RESERVED_SEEDS, SharedTrackEvaluator, generation_seeds

CONFIG = "neat_config/neat.cfg"


def _config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)


def _genome(config, key=1):
    genome = neat.DefaultGenome(key)
    genome.configure_new(config.genome_config)
    return genome


def test_fitness_is_deterministic_per_track():
    config = _config()
    genome = _genome(config)
    settings = EvalSettings(max_steps=60, seeds=[1234])
    assert evaluate_genome(genome, config, settings) == evaluate_genome(genome, config, settings)


def test_generation_tracks_are_shared_fresh_and_never_reserved():
    first = generation_seeds(0, 3, base_seed=0)
    assert first == generation_seeds(0, 3, base_seed=0)
    assert first != generation_seeds(1, 3, base_seed=0)
    for gen in range(50):
        assert not RESERVED_SEEDS & set(generation_seeds(gen, 3, base_seed=0))


def test_all_genomes_in_a_generation_drive_the_same_tracks():
    config = _config()
    population = neat.Population(config)
    evaluator = SharedTrackEvaluator(population, EvalSettings(max_steps=40), num_workers=1,
                                     tracks_per_generation=2, base_seed=0)
    genomes = list(population.population.items())[:3]
    evaluator.evaluate(genomes, config)
    seeds = generation_seeds(population.generation, 2, base_seed=0)
    for _, genome in genomes:
        expected = evaluate_genome(genome, config, EvalSettings(max_steps=40, seeds=seeds))
        assert genome.fitness == expected


def test_evaluator_survives_checkpoint_pickling():
    population = neat.Population(_config())
    evaluator = SharedTrackEvaluator(population, EvalSettings(), num_workers=2, tracks_per_generation=1, base_seed=0)
    try:
        restored = pickle.loads(pickle.dumps(evaluator))
        assert restored.pool is None and restored.settings == evaluator.settings
    finally:
        evaluator.close()


def test_settings_replace_keeps_other_fields():
    settings = EvalSettings(max_steps=123, offtrack_penalty=2.0)
    assert dataclasses.replace(settings, seeds=[7]).max_steps == 123


def test_decode_reaches_full_control_range():
    import numpy as np

    from carracing_neat.eval import decode_action

    zero = np.zeros(3, dtype=np.float32)
    assert np.allclose(decode_action([-1.0, 1.0, -1.0], zero, 0.0), [-1.0, 1.0, 0.0])
    assert np.allclose(decode_action([1.0, -1.0, 1.0], zero, 0.0), [1.0, 0.0, 1.0])
    # The old double-tanh decoding could never release the brake or floor the gas.
    legacy = decode_action([-1.0, 1.0, -1.0], zero, 0.0, legacy=True)
    assert np.allclose(legacy, [-0.7616, 0.8808, 0.1192], atol=1e-3)


def test_species_target_moves_threshold(tmp_path):
    from types import SimpleNamespace

    from carracing_neat.train import SpeciesTargetReporter

    config = _config()
    start = config.species_set_config.compatibility_threshold
    reporter = SpeciesTargetReporter(target=6, step=0.1, csv_path=str(tmp_path / "species.csv"))
    one = SimpleNamespace(species={1: SimpleNamespace(members={1: None, 2: None})})
    reporter.post_evaluate(config, None, one, None)
    assert config.species_set_config.compatibility_threshold == start - 0.1
    many = SimpleNamespace(species={i: SimpleNamespace(members={}) for i in range(9)})
    reporter.post_evaluate(config, None, many, None)
    assert abs(config.species_set_config.compatibility_threshold - start) < 1e-9
    assert (tmp_path / "species.csv").read_text().count("\n") == 3


def test_resume_reapplies_speciation_settings_in_place():
    from carracing_neat.train import _apply_speciation_settings

    config = _config()
    stagnation_cfg = config.stagnation_config
    stagnation_cfg.max_stagnation = 3
    config.species_set_config.compatibility_threshold = 9.0
    _apply_speciation_settings(config, CONFIG)
    assert config.stagnation_config is stagnation_cfg  # live objects keep their reference
    assert stagnation_cfg.max_stagnation == 15
    assert config.species_set_config.compatibility_threshold == 1.5
