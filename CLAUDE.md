# CLAUDE.md

Guidance for working on this repository.

## Project Overview

NEAT vs PPO on Gymnasium's `CarRacing-v2`, learned from a compact state vector (8 car-dynamics scalars + N lidar distances, default N=32). Both models improve along a **ladder of steps**. The goal is the fastest lap for the least training wall-clock time on this laptop CPU (i5-8250U, 8 threads, no GPU), and a simpler model wins when two are equally fast.

- **NEAT** (`carracing_neat/`): neat-python, pop 100. Every genome in a generation drives the same random tracks, and fitness is the raw score (+ optional lap-time bonus). The champion is picked on 5 held-out validation tracks (seeds 101–505).
- **PPO** (`carracing_ppo/`): Stable-Baselines3. The selected model is the best eval checkpoint.
- Benchmark seeds (3, 11, 21, 34, 55) are reserved: never train or validate on them.

## The step workflow

```bash
python scripts/run_step.py experiments/neat/n01_lap_time.toml     # train (timed) → select → benchmark → build_site
python scripts/run_step.py experiments/ppo/p00_baseline.toml --benchmark-only
python scripts/build_site.py                                      # rebuild docs/assets/data from outputs/*/*/result.json
nohup python scripts/run_queue.py > outputs/queue.log 2>&1 &        # train all untrained steps in sequence
python scripts/progress.py                                        # live progress: queue + current NEAT/PPO step
python -m carracing.play n00_baseline --seed 3                    # watch a step drive
pytest tests/
```

- A step = `experiments/<model>/<id>.toml` (`id` must equal the file name; `n..` = NEAT, `p..` = PPO). `[train]` keys become trainer `--flags` (`true` → `--flag`, `false` → `--no-flag`, lists → comma-joined). `[neat_config]` holds `"Section.key"` overrides written into `outputs/neat/<id>/neat.cfg`. `lidar_rays` sets both the env and NEAT `num_inputs`.
- To add a step, copy the previous TOML, change one thing, and write *why*. Don't edit an old step that has already run; add a new one instead.
- `[import]` (only n00/p00) points at existing artefacts instead of training. n00 needs `legacy_decode` (old double-tanh outputs) and p00 needs `legacy_traction` (trained with traction stuck at 1.0).
- Outputs: `outputs/<model>/<id>/` holds `model.*`, `result.json`, `drives.json`, `lap.gif` and `neat.cfg`, which are tracked. `checkpoints/` and `logs/` are git-ignored.

## Benchmark & ranking (`carracing/benchmark.py`, `carracing/steps.py`)

- Raw env (no penalty or bonuses), deterministic, up to 2000 frames, and the run stops at the end of the first lap. Lap time = frames / 50.
- Rank: tracks finished ↓, mean lap time ↑, training seconds ↑, parameter count ↑ (`rank_key`).
- Training time = wall-clock to the *selected* model. NEAT reads it from `fitness.csv` `time_s` (cumulative across resumes now; older runs reset per process, so `neat_train_seconds` sums segment maxima). PPO uses the Monitor CSVs at the best eval timestep.

## Env wrapper (`carracing/env.py`)

- State[7] (traction) is the fraction of wheels on a road tile.
- `offtrack_penalty × max(0, threshold − traction)` per frame.
- `lap_time_bonus × max(0, 1000 − lap_frames)` when a lap finishes (`info["lap_frames"]`).
- With `allow_multiple_laps`, the wrapper pays `lap_completion_bonus`, re-arms the tiles and continues. CarRacing reports a finished lap as `truncated`.
- Headless envs skip the pixel render, and the lidar is vectorised (~1200 steps/s). `tests/test_lidar.py` checks the lidar against a reference loop.

## Site (`docs/`, GitHub Pages)

- `index.html` (overview: champions, lap vs training-time scatter, leaderboard), `neat.html` (ladder, curves, network explorer, watch-it-think), `ppo.html` (ladder, curves), `race.html` (ghost race of any steps).
- Shared `assets/site.css` (tokens, dark mode) and `assets/site.js` (helpers, `Site.ladder`). Chart.js comes from the jsDelivr CDN.
- The data is generated, so never hand-edit `docs/assets/data/`.
