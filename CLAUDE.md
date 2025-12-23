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
# Recommended production run (12 hours, multi-lap with strong track adherence)
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

# Moderate run (balanced track adherence, ~4 hours)
python -m carracing_ppo.train \
  --total-timesteps 1500000 \
  --num-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --offtrack-penalty 5.0 \
  --traction-threshold 0.98 \
  --entropy-coef 0.005 \
  --domain-randomize

# Quicker smoke run (CPU-light, basic testing)
python -m carracing_ppo.train \
  --total-timesteps 100000 \
  --num-envs 2 \
  --n-steps 512 \
  --batch-size 256

# Resume from saved PPO zip
python -m carracing_ppo.train \
  --resume outputs/checkpoints/ppo_latest.zip \
  --total-timesteps 1000000
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
# Smoke test (validates pipeline)
python tests/smoke_test.py

# Multi-lap functionality test
python tests/test_multilap.py
```
- `smoke_test.py` validates the state wrapper shape and runs a tiny PPO learn loop to ensure wiring is intact.
- `test_multilap.py` verifies multi-lap functionality and lap completion bonuses.

## Architecture Notes

- **Observation**: 40-d vector (8 dynamics + 32 lidar). Wrapper lives in `carracing_ppo/env.py`.
- **Policy/Value**: Shared MLP, two 256-unit layers by default (configurable via `--policy-hidden-sizes`).
- **Actions**: SB3 PPO continuous policy with tanh-squashed outputs → steer [-1, 1], gas/brake [0, 1].
- **Training**: PPO (clip 0.2, GAE λ 0.95, γ 0.99, entropy 0.005-0.01). Eval + checkpoint callbacks built into `carracing_ppo/train.py`.
- **Multi-lap Racing**: Episodes continue after lap completion (when `--allow-multiple-laps` is enabled, default). Fast drivers complete multiple laps within the 1000-step limit for higher scores.

### Reward Structure

**Base CarRacing-v2 Rewards:**
- Tile visit bonus: ~+10 per new tile visited (~1000 total for complete lap)
- Time penalty: -0.1 per frame (encourages speed)
- Out-of-bounds: -100 + episode termination

**Custom Wrapper Rewards (`carracing_ppo/env.py`):**
- Off-track penalty: `-offtrack_penalty × max(0, traction_threshold - traction_actual)` per frame
  - Recommended: `offtrack_penalty=15.0`, `traction_threshold=0.99` for strict track adherence
  - Default: `offtrack_penalty=1.0`, `traction_threshold=0.95` (too lenient, allows corner cutting)
- Lap completion bonus: `+lap_completion_bonus` when completing a lap (default: 100.0)
  - Only applies when `allow_multiple_laps=True`
  - Episode continues after lap, agent can start lap 2, 3, etc.

**Scoring Example:**
- Slow driver (1 lap, 900 steps): ~900 points (tiles - time penalties)
- Fast driver (2 laps, 800 steps): ~2000+ points (2× tiles + 2× lap bonus - time penalties)
- Super fast driver (3 laps, 900 steps): ~3100+ points

## Key Files

- `carracing_ppo/env.py` — State wrapper (dynamics + lidar, multi-lap support, reward shaping)
- `carracing_ppo/train.py` — PPO training entrypoint, multi-env setup, eval/checkpoint callbacks
- `carracing_ppo/play.py` — Roll out saved models
- `tests/smoke_test.py` — Pipeline sanity test
- `tests/test_multilap.py` — Multi-lap functionality verification
- `docs/training_analysis_report.txt` — Training analysis and reward structure recommendations
- `README.md` — Usage, quick start, and model summary

## Outputs

- Checkpoints/Logs: `outputs/checkpoints/` (best/periodic zips), `outputs/logs/` (monitor/TensorBoard).
- Playbacks resolve models in order: `--model` arg, then `best_model.zip`, `ppo_carracing_best_model.zip`, `ppo_latest.zip` in `checkpoint_dir`.

## Common Tweaks

### Training Stability
- **Reduce jittery steering**: Lower `--entropy-coef` to `0.005` or `0.001` for smoother, more exploitative policies
- **Prevent training collapse**: Reduce `--num-envs` to 2-3 if training becomes unstable
- **Better convergence**: Increase `--total-timesteps` to 3M-5M for longer runs

### Track Adherence
- **Too aggressive (off-track)**: Increase `--offtrack-penalty` to 15.0-25.0 and `--traction-threshold` to 0.99
- **Too conservative (slow)**: Decrease `--offtrack-penalty` to 3.0-5.0 and `--traction-threshold` to 0.97
- **Balance**: Use `--offtrack-penalty 10.0 --traction-threshold 0.98` for moderate adherence

### Performance
- **Faster throughput**: Increase `--n-steps` (e.g., 2048) and `--num-envs` while keeping `batch-size` divisible by `num-envs * n-steps`
- **Better generalization**: Add `--domain-randomize` for varied track layouts
- **Multi-lap scoring**: Adjust `--lap-completion-bonus` (50-200) to tune the speed vs safety tradeoff

### Debugging
- **Disable multi-lap** (for comparison): Add `--no-allow-multiple-laps` flag
- **Check reward balance**: Monitor off-track penalties vs tile bonuses in TensorBoard
- **Visual inspection**: Use `--render-mode human` during eval to observe driving behavior
