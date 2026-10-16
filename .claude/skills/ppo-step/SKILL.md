---
name: ppo-step
description: Screen PPO recipes in the PPO lab and turn the winner into a PPO ladder step (experiments/ppo/pNN_*.toml) for the CarRacing NEAT vs PPO project. Use when trying a PPO idea, changing Stable-Baselines3 hyperparameters, controls or action repeat, reading PPO logs/lap checks, or writing a PPO step's outcome note.
---

# PPO lab → PPO ladder step

Read `.claude/skills/_shared/rules.md` first, and `references/ppo-lessons.md` before proposing anything. PPO has many dead ends already.

## When
- "Try X with PPO", "fix PPO", "next PPO step", "why did pNN fail", "write the outcome for pNN".

## Why screen first
A single PPO run depends a lot on its seed (p08 vs p06). A 3M-frame step on two seeds costs about an hour or more. The lab (`scripts/ppo_lab.py`) runs 600k-frame trials on 3 seeds, using validation lap checks only, so it can compare several ideas for the price of one step.

## Steps
1. **State the hypothesis** and check it against `references/ppo-lessons.md` and the latest `outcome`s in `experiments/ppo/`.
2. **Screen in the lab** (when the idea is a recipe or hyperparameter change):
   - Add a new round to `ROUNDS` in `scripts/ppo_lab.py`. Build recipes with `recipe(controls, repeat, **extra)` on top of `BASE`. Always include the current best recipe as a control in the round.
   - Run `nohup python scripts/ppo_lab.py <round> > outputs/ppo/lab/lab.log 2>&1 &` (one CPU job at a time, see the `train-queue` skill).
   - Read `python scripts/ppo_lab.py --summary`. Per trial: best/final validation laps finished, lap time, `first_5of5_min`, `train_lap_pct`, `std`. Rank by consistency across the 3 seeds (`lab_score`), not by one lucky seed. A best that later falls back (best ≫ final) is a warning sign.
   - Finished trials are skipped, so the lab can be stopped and restarted.
3. **Promote the winner to a step.** `promote()` in ppo_lab writes one, or copy the latest `experiments/ppo/pNN_*.toml`:
   - `id` = file name, `parent`, `title`, and a `why` that quotes the lab numbers for each seed.
   - Step-level keys stay out of `[train]`: `lidar_rays`, `lidar_fov`, `action_smoothing` (0.1), `controls` (`raw3|throttle|discrete`), `action_repeat`, `seeds = [0, 1]`.
   - `[train]`: flags of `carracing_ppo/train.py` (`python -m carracing_ppo.train --help`). Keep the rulebook ones: `max-episode-steps = 2000`, `offtrack-penalty = 0.0`, `allow-multiple-laps = false`, `eval-frequency = 0`, `checkpoint-frequency = 0`, `lap-check-frequency = 25000`.
   - Change one thing from the parent.
4. **Check it parses:** `pytest tests/test_steps.py -q`.
5. **Train:** `python scripts/run_step.py experiments/ppo/pNN_x.toml` via the queue. Each seed trains in `outputs/ppo/<id>/seed<k>/`, and the best on validation is kept. A restarted multi-seed step reuses the seeds that already finished.
6. **Assess:** `result.json` → `summary`, `tracks`, `seed_runs` (every seed's benchmark: is it luck?), `kept_seed`, `validation`, `train_seconds`. Logs: `seed<k>/logs/validation_laps.csv`, `train.out` (SB3 table: `std`, `entropy_loss`, `approx_kl`), `monitor/*.csv`. Watch it: `python -m carracing.play <id> --seed 3`.
7. **Write it up:** add an `outcome` to the TOML, add a lesson to `references/ppo-lessons.md` and, if it's general, to `experiments/findings.toml`. Then run `python scripts/build_site.py`.

## Pitfalls
- `--action-repeat N`: `total-timesteps` and `lap-check-frequency` count **frames**, while `n-steps` counts **decisions** (the lab uses `n-steps = 1024 // repeat`).
- The trainer defaults are *not* the rulebook (`offtrack-penalty` 1.0, multiple laps on, eval on, `action-smoothing` 0). Set them explicitly.
- `legacy_traction` is for p00 only.
- Training time comes from the Monitor CSVs at the kept checkpoint's timestep.

## Done when
`result.json` exists with both seeds, the step has an `outcome`, `ppo-lessons.md` is updated, and the user has a short comparison with the parent and with the best NEAT step.
