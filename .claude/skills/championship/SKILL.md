---
name: championship
description: Set up, train, score and present the championship cars (experiments/championship/*.toml) - NEAT and PPO cars with an 8 h budget, two categories (quickest to finish, fastest lap), exhibition races. Use when adding or entering a championship car, re-recording exhibition tracks, or updating the Championship page or its findings.
---

# Championship

Read `.claude/skills/_shared/rules.md` first.

## The format
- Each car is a recipe trained **from scratch** for up to `BUDGET_HOURS = 8` of wall-clock (`max-hours = 8`, **no** `patience-hours`: a single lucky lap check stopped NEAT 16 at 3.5 h while it was still improving).
- Lap checks on the validation tracks pick two checkpoints per car (`scripts/run_step.py:select_categories`):
  - **finish**: the first checkpoint that laps all 5 validation tracks, scored on how little training it took.
  - **fast**: the fastest validation mean lap (most tracks finished first), scored on lap time.
- Both are benchmarked on 3/11/21/34/55. A car that never laps all 5 validation tracks has no "finish" entry.
- Exhibition: every car's checkpoints also drive tracks 700–724 for fun races (`carracing/exhibition.py`, cached in `outputs/<model>/<id>/exhibition_<cat>.json`).
- The plan: 3 NEAT cars (`neat16`, `neat32`, `neatsmart`) and 3 matching PPO cars (16 rays, 32 rays, "smart"). **PPO cars enter only once a PPO ladder step finishes reliably** (check the PPO ladder before proposing one).

## Add a car
1. Copy `experiments/championship/neat16.toml` to `experiments/championship/<model><name>.toml`. The id must start with the model name (`neat…`, `ppo…`), as `tests/test_steps.py` checks.
2. Set `championship = true`, `parent` = the ladder step whose recipe it uses, and a `why` saying which recipe and why. Keep the `[train]` flags of that recipe and add `max-hours = 8` (NEAT: a large `generations` such as 1000. PPO: a large `total-timesteps`, and lap checks are needed). Don't set `seeds`: a championship car is trained once (`run_step.py` skips `run_seeds` for championship steps).
3. `pytest tests/test_steps.py -q`.
4. Train through the `train-queue` skill. An 8 h run blocks the CPU, so tell the user the ETA. `run_step.py` picks the checkpoints, benchmarks both, and records the exhibition tracks.
5. Re-record the exhibition only if needed: `python scripts/record_exhibition.py [<id>]` (only drives the seeds that are missing).
6. Write the `outcome` (both categories, with times and tracks), update the championship `[[finding]]` in `experiments/findings.toml`, and run `python scripts/build_site.py`. Check `championship.html` with the `site` skill.

## Reading results
`outputs/<model>/<id>/result.json` → `categories.finish|fast` (`selected`, `train_seconds`, `validation`), plus the benchmark per category. The drives are in `drives_<cat>.json`, and `docs/assets/data/championship.json` holds the page view.

## Done when
Both categories are benchmarked (or explained as missing), the exhibition is recorded, the outcome and finding are written, and the site is rebuilt.
