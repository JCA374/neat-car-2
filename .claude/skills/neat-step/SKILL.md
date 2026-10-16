---
name: neat-step
description: Design, run and assess a NEAT ladder step (experiments/neat/nNN_*.toml) for the CarRacing NEAT vs PPO project. Use when adding a NEAT experiment, changing neat.cfg or NEAT trainer flags, reading NEAT training logs (fitness/validation/species CSVs), or writing a NEAT step's outcome note.
---

# NEAT ladder step

Read `.claude/skills/_shared/rules.md` first: seeds, one change per step, honest time.

## When
- "Try X with NEAT", "next NEAT step", "why did nNN do worse", "write the outcome for nNN".

## Steps
1. **Pick the parent.** Usually the best-ranked NEAT step. Check the ranking:
   `python -c "import json;[print(r['id'],r['summary']['finished'],r['summary']['lap_s'],r['train_seconds']) for r in json.load(open('docs/assets/data/leaderboard.json'))['steps'] if r['id'].startswith('n')]"`
   (the list is ranked and includes the championship cars). Read the parent's `why`/`outcome` and `experiments/findings.toml`, so we don't retest a dead end.
2. **Write the step.** `cp experiments/neat/<parent>.toml experiments/neat/nNN_<short_name>.toml`, then set:
   - `id` (= file name), `parent`, `title` (short, plain), and `why`: the hypothesis, the evidence behind it, and what result would confirm it.
   - Change **one** thing. Remove any `outcome` copied from the parent.
   - Settings: see `references/neat-config.md` (TOML keys, `[train]` flags, `[neat_config]` overrides).
3. **Check it parses:** `python -c "from carracing.steps import load_step; s=load_step('experiments/neat/nNN_x.toml'); print(s)"`, and `pytest tests/test_steps.py -q`.
4. **Train.** Use the `train-queue` skill. Run one job at a time, in the background:
   `nohup python scripts/run_step.py experiments/neat/nNN_x.toml > outputs/neat/nNN_x.log 2>&1 &`
   (make the `outputs/neat/nNN_x/` dir or put the log elsewhere first) and watch it with `python scripts/progress.py --once`.
5. **Assess** (when `outputs/neat/<id>/result.json` exists):
   - `result.json`: `summary` (finished, lap_s, offtrack_pct), `tracks` per seed, `train_seconds`, `selected` generation, `size`.
   - `logs/fitness.csv` (per generation: fitness stats, `time_s`), `logs/validation.csv` (champion on validation tracks), `logs/species.csv`, `logs/validation_laps.csv`.
   - Compare with the parent: tracks finished, mean lap, training time, params. Did the best generation come late (still improving) or early (plateau)?
   - Watch a drive: `python -m carracing.play <id> --seed 3` (or `--gif /tmp/x.gif` when headless).
   - For "why does it crash", use the `car-analysis` skill.
6. **Write it up.** Add an `outcome = """..."""` to the step TOML (2–4 plain sentences with the numbers, and what it means next). If it taught something general, add or extend a `[[finding]]` in `experiments/findings.toml` with the step ids. Then `python scripts/build_site.py`.

## Pitfalls
- `lidar_rays` sets both the env and the NEAT `num_inputs` (8 + rays, or 6 + rays with `drop_heading`). Don't set `num_inputs` by hand.
- `lidar_fov` missing means the old 324° layout. Every step since n02 uses `lidar_fov = 240` (forward).
- `legacy_decode` is for n00 only.
- `--resume auto` (the default) resumes from the newest checkpoint in `outputs/neat/<id>/checkpoints/`. To restart from scratch, delete that step's `checkpoints/` and `logs/` (they're git-ignored) only after checking with the user.
- Speciation: `--target-species 6` adjusts the compatibility threshold every generation. Changing pop size or compatibility settings interacts with it.
- `--generations` is *added* to a resumed run, and `--max-hours` caps wall-clock.
- `fitness.csv` `time_s` is cumulative across resumes now. Older runs reset it, and `neat_train_seconds` handles both.

## Done when
`result.json` exists, the step has an `outcome`, findings are updated if needed, `build_site.py` has run, and the user has the parent vs step comparison in a few lines.
