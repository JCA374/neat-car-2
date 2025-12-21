# CLAUDE.md

Guidance for working on this repository.

## Project Overview

PPO baseline for Gymnasium's `CarRacing-v2` using a compact state vector (8 car-dynamics scalars + 32 lidar distances). CPU-friendly defaults; Stable-Baselines3 handles policy/value updates and checkpointing.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Requirements use the PyTorch CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid CUDA downloads.

## Training

```bash
# Default-ish run
python -m carracing_ppo.train \
  --total-timesteps 500000 \
  --num-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --eval-frequency 25000 \
  --checkpoint-frequency 50000

# Quicker smoke run (CPU-light)
python -m carracing_ppo.train --total-timesteps 100000 --num-envs 2 --n-steps 512 --batch-size 256

# Resume from saved PPO zip
python -m carracing_ppo.train --resume outputs/checkpoints/ppo_latest.zip
```

## Playback / Evaluation

```bash
# Render a saved model (falls back to best/ppo_latest in checkpoint_dir)
python -m carracing_ppo.play --model outputs/checkpoints/best_model.zip --render-mode human

# Headless rollouts
python -m carracing_ppo.play --render-mode None --episodes 3
```

## Tests

```bash
python smoke_test.py
```
- Validates the state wrapper shape and runs a tiny PPO learn loop to ensure wiring is intact.

## Architecture Notes

- Observation: 40-d vector (8 dynamics + 32 lidar). Wrapper lives in `carracing_ppo/env.py`.
- Policy/Value: shared MLP, two 256-unit layers by default (configurable via `--policy-hidden-sizes`).
- Actions: SB3 PPO continuous policy with tanh-squashed outputs → steer [-1, 1], gas/brake [0, 1].
- Training: PPO (clip 0.2, GAE λ 0.95, γ 0.99, entropy 0.01). Eval + checkpoint callbacks built into `carracing_ppo/train.py`.

## Key Files

- `carracing_ppo/env.py` — state wrapper (dynamics + lidar)
- `carracing_ppo/train.py` — PPO training entrypoint, multi-env setup, eval/checkpoint callbacks
- `carracing_ppo/play.py` — roll out saved models
- `smoke_test.py` — pipeline sanity test
- `README.md` — usage, quick start, and model summary

## Outputs

- Checkpoints/Logs: `outputs/checkpoints/` (best/periodic zips), `outputs/logs/` (monitor/TensorBoard).
- Playbacks resolve models in order: `--model` arg, then `best_model.zip`, `ppo_carracing_best_model.zip`, `ppo_latest.zip` in `checkpoint_dir`.

## Common Tweaks

- More stability: lower `--entropy-coef` (e.g., 0.005) or reduce `--num-envs`.
- Faster throughput: increase `--n-steps` and `--num-envs` while keeping `batch-size` divisible by `num-envs * n-steps`.
- Domain randomization: add `--domain-randomize` to train/eval for robustness.
