# NEAT step settings

Check the live list with `python -m carracing_neat.train --help`. It's the source of truth.

## Top-level TOML keys (`carracing/steps.py:Step`)
| Key | Meaning |
|---|---|
| `id`, `model = "neat"`, `parent`, `title`, `why`, `outcome` | bookkeeping, shown on the page |
| `lidar_rays` (default 32) | rays in the state, and sets `num_inputs` |
| `lidar_fov` | forward fan in degrees (240 = ±120°). Missing = old 324° layout |
| `drop_heading` | leave cos/sin(heading) out of the state |
| `championship = true` | championship car (see the `championship` skill) |
| `legacy_decode` | n00 only |
| `[import]` | n00 only: existing artefacts instead of training |

## `[train]` → `carracing_neat/train.py` flags
`true` → `--flag`, `false` → `--no-flag`, a list → comma-joined.

| Flag | Default | Notes |
|---|---|---|
| `generations` | 50 | added to a resumed run |
| `tracks-per-generation` | 3 | every genome drives the same random training tracks |
| `max-steps` | 1000 | **use 2000** (rulebook; n04 showed 1000 hurts long tracks) |
| `lap-time-bonus` | 0 | fitness per frame a lap beats 1000 frames (1.0 is the proven value) |
| `offtrack-penalty`, `traction-threshold` | 0, 0.95 | grass penalty (off in the good recipes) |
| `patience` | 100 | end an episode after N frames without a new tile |
| `action-smoothing` | 0.1 | rulebook value |
| `target-species`, `threshold-step` | 6, 0.1 | speciation control |
| `validate-every` | 1 | champion lap check on validation every N generations |
| `max-hours`, `patience-hours` | none | budget, and a stop when nothing improves (the championship uses no patience) |
| `seed` | 0 | base seed for the track draw |
| `num-workers` | cpus − 1 | leave it alone |

## `[neat_config]` overrides
`"Section.key" = value` on top of `neat_config/neat.cfg`, written to `outputs/neat/<id>/neat.cfg`. Sections: `NEAT`, `DefaultGenome`, `DefaultSpeciesSet`, `DefaultStagnation`, `DefaultReproduction`.

Examples:
```toml
[neat_config]
"DefaultGenome.initial_connection" = "unconnected"   # n03: too sparse, never finished
"NEAT.pop_size" = 150
"DefaultGenome.conn_add_prob" = 0.5
```
Defaults worth knowing: pop 100, `initial_connection = full_direct`, tanh only, feed-forward, 3 outputs.
