# PPO CarRacing (CPU-friendly)

This repo trains a PPO policy on Gymnasium's `CarRacing-v2` using a compact state vector (car dynamics + lidar-style distances) instead of pixels. It is CPU-only by default and keeps rollout settings modest so you can iterate quickly.

## Model Summary

- Observation: 40-dim vector (8 dynamics scalars + 32 lidar distances, forward-biased rays).
- Policy/Value: shared MLP with two 256-unit hidden layers (tanh by default).
- Actions: steer in [-1, 1], gas/brake in [0, 1] with SB3’s tanh-squashed outputs.
- Training: PPO with clip 0.2, GAE λ 0.95, γ 0.99, entropy coef 0.01, eval + checkpoints built-in.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- The requirements use the PyTorch CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid pulling GPU CUDA packages.

## Quick Start

```bash
# 1) Sanity-check the pipeline
python smoke_test.py

# 2) Kick off a short training run (CPU-friendly)
python -m carracing_ppo.train --total-timesteps 100000 --num-envs 2 --n-steps 512 --batch-size 256

# 3) Play the latest policy (headless or render)
python -m carracing_ppo.play --model outputs/checkpoints/ppo_latest.zip --render-mode human
```

## Train

```bash
python -m carracing_ppo.train \
  --total-timesteps 500000 \
  --num-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --eval-frequency 25000 \
  --checkpoint-frequency 50000
```

Notes:
- Observations: 8 car-dynamics scalars + 32 lidar distances (40-dim vector).
- Policy: MLP (defaults to two 256-unit layers for policy/value).
- Checkpoints + logs land in `outputs/checkpoints/` and `outputs/logs/` (TensorBoard-ready).
- Add `--domain-randomize` to use Gymnasium's random track parameters during training.

## Play a Saved Policy

```bash
python -m carracing_ppo.play \
  --model outputs/checkpoints/best_model.zip \
  --render-mode human \
  --episodes 1
```

`--model` falls back to `outputs/checkpoints/ppo_latest.zip` if `best_model.zip` is missing. Use `--render-mode None` for headless evaluation.

## Repo Structure

- `carracing_ppo/env.py`: CarRacing state wrapper (dynamics + lidar vector).
- `carracing_ppo/train.py`: PPO training entrypoint with evaluation/checkpoint callbacks.
- `carracing_ppo/play.py`: Roll out a saved PPO policy (human or headless).
- `smoke_test.py`: Minimal pipeline sanity check.
- `outputs/`: Ignored by git; holds checkpoints, logs, and monitor files.

## Recommended Hyperparameters

- PPO defaults in `train.py` are good starters: clip `0.2`, entropy coeff `0.01`, GAE λ `0.95`, γ `0.99`.
- Increase `--n-steps` and `--total-timesteps` once runs are stable; keep `batch-size` dividing `num-envs * n-steps`.
- If training destabilizes, try lowering `--entropy-coef` to `0.005` or using fewer parallel envs.

## What Changed vs. the NEAT version

- Swapped NEAT for PPO with a compact state vector and SB3 tooling.
- Removed NEAT configs/visualizers; added PPO training and playback scripts.
- Kept the lightweight lidar/dynamics wrapper for fast CPU-only iteration.
