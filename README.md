# NEAT vs PPO on CarRacing (CPU only)

Two learners drive Gymnasium's `CarRacing-v2` from a compact 40-number state (8 car readings + 32 lidar distances) instead of pixels:

- **NEAT** (`carracing_neat/`): evolves small networks with neat-python.
- **PPO** (`carracing_ppo/`): Stable-Baselines3 with a fixed MLP.

Both improve on a **ladder of small steps**. The goal is the **fastest lap for the least training time** on one laptop CPU (i5-8250U, no GPU). When two models drive equally fast, the simpler one wins.

**Project page:** <https://jca374.github.io/neat-car-2/> (`docs/`, GitHub Pages) has four tabs: Overview, NEAT, PPO and Race.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # PyTorch CPU wheel
pytest tests/
```

## How a step works

A step is one TOML file: `experiments/<model>/<id>.toml`. It holds the title, *why* the change was made, the parent step and the trainer flags.

```bash
cp experiments/neat/n01_lap_time.toml experiments/neat/n02_my_idea.toml   # edit id/parent/title/why/settings
python scripts/run_step.py experiments/neat/n02_my_idea.toml               # train (timed), benchmark, update site data
python scripts/neat_progress.py                                            # (other terminal) live NEAT progress
python -m carracing.play n02_my_idea --seed 3                               # watch it drive
```

`run_step.py` writes `outputs/<model>/<id>/` and then runs `scripts/build_site.py`, which refreshes `docs/assets/data/`. The files written to the step folder:
- `model.pkl` / `model.zip`: the selected model (NEAT: best on 5 held-out validation tracks; PPO: best eval checkpoint).
- `result.json`: settings, training time, size, lap time per benchmark track, training curve.
- `drives.json`: the recorded benchmark drives, which feed the race and "watch it think" views.
- `lap.gif`, plus `neat.cfg` for NEAT steps.
- `checkpoints/` and `logs/`, which stay local (git-ignored).

Step file reference:

```toml
id = "n02_my_idea"            # must match the file name; n.. = NEAT, p.. = PPO
model = "neat"                # neat | ppo
parent = "n01_lap_time"       # the page shows deltas against it
title = "Short name"
why = "What changed and what we expect."
lidar_rays = 16               # optional, default 32 (state size = 8 + rays)
[train]                       # passed through as --flags to carracing_neat.train / carracing_ppo.train
generations = 100
lap-time-bonus = 1.0
[neat_config]                 # NEAT only: overrides of neat_config/neat.cfg
"NEAT.pop_size" = 50
```

`python scripts/run_step.py <toml> --benchmark-only` re-scores a step without training it.

## The benchmark and the ranking

- Five reserved tracks (seeds 3, 11, 21, 34, 55). No step trains or validates on them.
- Raw CarRacing reward, deterministic actions, up to 2000 frames, and the run ends when the lap does.
- Ranking: most tracks finished, then mean lap time, then training wall-clock time to the selected model, then parameter count.

## Layout

```
carracing/         shared: env wrapper (env.py), benchmark + recording (benchmark.py), step files + ranking + timing (steps.py), play.py
carracing_neat/    NEAT training (train.py), genome evaluation (eval.py), network → JSON (graph.py)
carracing_ppo/     PPO training (train.py)
experiments/       one TOML per step, per model
outputs/           one folder per step
scripts/           run_step.py, build_site.py, neat_progress.py
docs/              the project page (index, neat, ppo, race + assets/)
neat_config/       base NEAT config
tests/             pytest
```

Step 0 for each model (`n00_baseline`, `p00_baseline`) is imported from the project's earlier history. The bug-hunting write-up from that period is in git history (commit `cd099f3`).
