---
name: train-queue
description: Start, queue, monitor, pause and resume training runs on this laptop CPU for the CarRacing project (scripts/run_step.py, scripts/run_queue.py, scripts/ppo_lab.py, scripts/progress.py). Use when asked to train, queue steps, check training progress, see what's running, stop or continue a run.
---

# Running training on the laptop

The CPU (i5-8250U, 8 threads, no GPU) is shared by everything, and training time is part of the score. So **run one training job at a time**.

## Before starting anything
```bash
pgrep -af "carracing_neat.train|carracing_ppo.train|run_step.py|run_queue.py|ppo_lab.py"
python scripts/progress.py --once
```
If something is running, don't start a second job. Tell the user what's running and offer to queue the new work after it.

## Start
| What | Command |
|---|---|
| Every step without a `result.json`, in id order | `nohup python scripts/run_queue.py > outputs/queue.log 2>&1 &` |
| Chosen steps, in this order | `nohup python scripts/run_queue.py n06_x p17_y > outputs/queue.log 2>&1 &` |
| One step | `nohup python scripts/run_step.py experiments/<model>/<id>.toml > outputs/<id>.log 2>&1 &` |
| Re-score without training | `python scripts/run_step.py experiments/<model>/<id>.toml --benchmark-only` |
| PPO lab round | `nohup python scripts/ppo_lab.py <round> > outputs/ppo/lab/lab.log 2>&1 &` |

`run_queue.py` writes `outputs/queue.json` (queued / running / done / failed). A failed step is recorded and the queue moves on to the next one. Every `run_step.py` ends with `build_site.py`.

## Watch
- `python scripts/progress.py --once` shows the queue, plus details of the step being trained (NEAT generations and validation, or PPO frames and lap checks, with an ETA). Use `--step <id>` for a specific step.
- Raw logs: `tail -n 30 outputs/queue.log`, `outputs/<model>/<id>/logs/` (`fitness.csv`, `validation_laps.csv`, `train.out`).
- To check in on long runs, use the Monitor tool on the log, or a ScheduledWakeup of 20–30 min. Don't busy-poll or `sleep`.

## Pause and continue
- Pause: kill the queue/run_step process tree (`pkill -f run_queue.py; pkill -f run_step.py; pkill -f carracing_` after confirming with the user).
- Continue NEAT: rerun the same step. `--resume auto` picks the newest checkpoint, and `time_s` keeps counting across resumes.
- Continue a multi-seed PPO step: rerun it. Seeds with a `seed<k>/result.json` are reused, and the unfinished seed starts again from scratch.
- The PPO lab skips finished trials (`done` marker), so rerun the same round.
- Never delete `checkpoints/` or `logs/` without asking. They're git-ignored, so nothing can bring them back.

## Honest timing
- Anything else heavy (another training job, a big analysis, pytest of the env) while a step trains inflates its `train_seconds`. Avoid it, or write it into the `outcome` (as p05 did).
- Imported steps (`[import]`) and steps that already have a `result.json` are skipped by the queue.

## Done when
The user knows what's running, what's queued, the ETA, and where the log is.
