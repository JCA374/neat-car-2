---
name: car-analysis
description: Failure analysis for a CarRacing step's car - record crashes, DNFs and slow sections on validation/fresh tracks, test candidate input signals, and build a self-contained HTML report that suggests the next ladder step. Use when asked why a NEAT or PPO car crashes, leaves the road, is slow, or what new input/feature or reward could help. Project version of the user's failure-analysis-report skill.
---

# Car failure analysis

This is the project version of `~/.claude/skills/failure-analysis-report/SKILL.md` (the method the user liked in the Tetris project). Read that file for the general method and the report style. This skill covers what's specific to the car.

Read `.claude/skills/_shared/rules.md`: **never use benchmark seeds (3, 11, 21, 34, 55) or exhibition seeds**. Use the validation seeds (101–505) plus fresh seeds from `rules.training_seed()` (≥ 50 tracks so the statistics mean something).

## Code layout
Put scripts in `scripts/analysis/` (`record_drives.py`, `analyze_drives.py`) and outputs in `outputs/analysis/<step id>/` (git-ignored, except the final `report.html` and `summary.json` if the user wants them kept).

## 1. Record
Load a step's policy with `carracing.benchmark.load_policy(step)` and drive it with the same env settings as the benchmark (`carracing.env.make_env(..., **step.env_kwargs())`, raw reward, `MAX_FRAMES`). Copy the loop in `carracing/benchmark.py:drive` rather than changing it. Per frame, keep:
- `obs` (8 dynamics: speed, vx, vy, cos/sin heading, spin rate, steering, traction, then the lidar rays), the action, `env.last_traction`, hull x/y/angle, the tile index, and the track curvature ahead (from `base.track`).

Per drive, save: the outcome (`finished`, `lap_s`, crash or DNF, `reached_pct`), a rolling window of the **last 150 frames before the first grass excursion** (traction < 0.5) or the final stop, and snapshots every 100 frames as the "normal" baseline. Format: `.jsonl.gz`, with a header line and then one drive per line.

Events to analyse (pick with the user if it's unclear):
- **Off-road**: traction drops below 0.5 (the main one).
- **DNF**: the car times out or gets lost.
- **Slow section**: the lap time per tile is much worse than the median (for "why is it slow").

## 2. Candidate signals (numpy, vectorised)
The model's current inputs (blue in the plots), plus hypotheses (orange), for example:
speed × curvature ahead, minimum of the forward 3 rays, left/right ray asymmetry, change in ray distance (closing rate), spin rate vs steering (understeer or oversteer), steering saturation (|steer| ≥ 0.95), gas and brake together, distance to the road centre, heading error relative to the tile direction, frames since the last new tile.

## 3. Report (one self-contained HTML, one figure per question)
1. What does a failure look like? A gallery of track maps with the path coloured by speed, failure point marked, sorted by outcome.
2. Where does it happen? Failures by curvature and by turn direction, compared with normal driving.
3. What drifts before it? Median + IQR band over the last 150 frames, with a dashed normal level.
4. What predicts it? The AUC for "goes off-road within 25 frames", and the partial correlation with time-to-failure after regressing out the current inputs.
5. What causes it? Share of failures on left vs right turns, entry speed, braking late vs not at all, compared with chance.
6. How does it start? The last healthy frame, then +10 and +30 frames (car, rays, controls).

Put a summary table at the top, and also write `summary.json` with every number. Use the `dataviz` skill for the charts. Publish the report as an Artifact (private) and give the user the link.

## 4. Findings are leads
End with **one** proposed ladder step that tests the strongest lead (new input, reward term, control mode or setting). Write its `why` with the evidence, ready for `neat-step` or `ppo-step`. Validation is the paired benchmark comparison that the step itself does.
