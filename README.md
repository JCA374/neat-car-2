# PPO CarRacing (CPU-friendly)

This repo trains a PPO policy on Gymnasium's `CarRacing-v2` using a compact state vector (car dynamics + lidar-style distances) instead of pixels. It is CPU-only by default and keeps rollout settings modest so you can iterate quickly.

## Model Summary

- Observation: 40-dim vector (8 dynamics scalars + 32 lidar distances, forward-biased rays).
- Policy/Value: shared MLP with two 256-unit hidden layers (tanh by default).
- Actions: steer in [-1, 1], gas/brake in [0, 1] with SB3's tanh-squashed outputs.
- Training: PPO with clip 0.2, GAE λ 0.95, γ 0.99, entropy coef 0.005-0.01, eval + checkpoints built-in.
- Multi-lap support: Episodes continue after lap completion, rewarding fast drivers with higher scores.

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
python tests/smoke_test.py

# 2) Kick off a short training run (CPU-friendly)
python -m carracing_ppo.train \
  --total-timesteps 100000 \
  --num-envs 2 \
  --n-steps 512 \
  --batch-size 256 \
  --offtrack-penalty 5.0 \
  --traction-threshold 0.98

# 3) Play the latest policy (headless or render)
python -m carracing_ppo.play --model outputs/checkpoints/ppo_latest.zip --render-mode human
```

## Train

### Recommended Configuration (12-hour training)

```bash
python -m carracing_ppo.train \
  --total-timesteps 4500000 \
  --num-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --offtrack-penalty 15.0 \
  --traction-threshold 0.99 \
  --entropy-coef 0.005 \
  --domain-randomize \
  --allow-multiple-laps \
  --lap-completion-bonus 100.0 \
  --eval-frequency 50000 \
  --checkpoint-frequency 100000 \
  --checkpoint-dir outputs/checkpoints_multilap \
  --log-dir outputs/logs_multilap \
  --seed 42
```

### Key Training Parameters

**Reward Shaping:**
- `--offtrack-penalty 15.0`: Strong penalty for leaving the track (default: 1.0)
- `--traction-threshold 0.99`: Strict traction requirement (default: 0.95)
- `--lap-completion-bonus 100.0`: Bonus for completing each lap

**Multi-Lap Racing:**
- `--allow-multiple-laps`: Episodes continue after lap completion (default: enabled)
- Fast drivers can complete 2-3 laps per episode for higher scores
- Slow drivers complete 1 lap with lower total reward

**Training Stability:**
- `--entropy-coef 0.005`: Lower entropy for stable exploitation (default: 0.01)
- `--domain-randomize`: Random track layouts for better generalization

**Notes:**
- Observations: 8 car-dynamics scalars + 32 lidar distances (40-dim vector).
- Policy: MLP (defaults to two 256-unit layers for policy/value).
- Checkpoints + logs land in `outputs/checkpoints/` and `outputs/logs/` (TensorBoard-ready).
- 4.5M timesteps ≈ 12 hours on CPU with 4 parallel environments.

## Play a Saved Policy

```bash
python -m carracing_ppo.play \
  --model outputs/checkpoints/best_model.zip \
  --render-mode human \
  --episodes 1
```

`--model` falls back to `outputs/checkpoints/ppo_latest.zip` if `best_model.zip` is missing. Use `--render-mode None` for headless evaluation.

## Repo Structure

- `carracing_ppo/env.py`: CarRacing state wrapper (dynamics + lidar vector, multi-lap support).
- `carracing_ppo/train.py`: PPO training entrypoint with evaluation/checkpoint callbacks.
- `carracing_ppo/play.py`: Roll out a saved PPO policy (human or headless).
- `tests/`: Test files (smoke test, multi-lap verification).
- `docs/`: Analysis reports and documentation.
- `outputs/`: Ignored by git; holds checkpoints, logs, and monitor files.

## Recommended Hyperparameters

**For Track Adherence:**
- `--offtrack-penalty 15.0` with `--traction-threshold 0.99` encourages clean driving
- Lower penalties (5.0-10.0) for more aggressive but still on-track driving
- Default penalties (1.0) allow too much off-track cutting

**For Training Stability:**
- `--entropy-coef 0.005` for stable exploitation (recommended for long runs)
- `--entropy-coef 0.01` for more exploration (default, good for early training)
- Lower entropy reduces jittery steering and improves smoothness

**For Performance:**
- Increase `--n-steps` and `--total-timesteps` for better convergence
- Keep `batch-size` dividing `num-envs * n-steps`
- Use `--domain-randomize` for better generalization across tracks

## What Changed vs. the NEAT version

- Swapped NEAT for PPO with a compact state vector and SB3 tooling.
- Removed NEAT configs/visualizers; added PPO training and playback scripts.
- Kept the lightweight lidar/dynamics wrapper for fast CPU-only iteration.
