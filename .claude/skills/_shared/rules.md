# The rulebook (shared by every project skill)

The source of truth is `carracing/rules.py`. If the numbers here and that file disagree, the file is right, so fix this note.

## Tracks (CarRacing seeds)
| Set | Seeds | Use |
|---|---|---|
| Training | random from `TRAIN_SEED_RANGE` (1 000–1 000 000) via `rules.training_seed()` | training only |
| Validation | 101, 202, 303, 404, 505 | picking a checkpoint (lap checks, NEAT champion) |
| Benchmark | 3, 11, 21, 34, 55 | the ladder and championship score, **nothing else** |
| Exhibition | 700–724 | fun races on the Championship page, no points |

- Never train or validate on a benchmark or exhibition seed. Never tune on benchmark results: look at validation first.
- Analysis and debugging drives use validation seeds or fresh training-range seeds.

## Shared conditions
- `MAX_FRAMES = 2000` (40 s at 50 fps) for training episodes, lap checks and the benchmark.
- `ACTION_SMOOTHING = 0.1` for every new model. How outputs become steer/gas/brake (`carracing/controls.py`) and how often a model decides is the model's own choice.
- The benchmark runs the raw env: no penalty or bonus, deterministic, and it stops at the end of the first lap. Lap time = frames / 50.
- Championship budget: `BUDGET_HOURS = 8` of wall-clock on this laptop, all threads.

## Ranking (`carracing/steps.py:rank_key`)
Tracks finished ↓, mean lap time ↑, training seconds ↑, parameter count ↑.

## Training time is honest
- Wall-clock to the *selected* model, not to the end of the run. NEAT: `fitness.csv` `time_s` (`neat_train_seconds`). PPO: Monitor CSVs at the selected timestep (`ppo_train_seconds`).
- Run one training job at a time. A second job on the CPU inflates the time, so note it in the `outcome` if it happens (as p05 did).
- A new step trains from a fresh start, so its time is the whole cost.

## Steps
- One change per step, with the reason in `why`. Copy the parent TOML. `id` = file name, and `parent` = the step it changes.
- Never edit a step that has already run (apart from adding `outcome`). Add a new step instead.
- After a run, write an `outcome` note in plain language: what happened, the numbers, and what it means for the next step.
