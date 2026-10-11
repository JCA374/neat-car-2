# CLAUDE.md

Guidance for working on this repository.

## Project Overview

Learns Gymnasium's `CarRacing-v2` from a compact state vector (8 car-dynamics scalars + 32 lidar distances). Two learners share one env wrapper (`carracing_ppo/env.py`):
- **NEAT** (`carracing_neat/`, main approach): neat-python, pop 100, all genomes in a generation drive the same random tracks, raw-score fitness, champion validated on 5 held-out tracks (seeds 101–505). Benchmark seeds (3, 11, 21, 34, 55) are reserved and never used for training or validation.
- **PPO** (`carracing_ppo/`, baseline): Stable-Baselines3.

```bash
python -m carracing_neat.train --generations 150 --tracks-per-generation 3 --auto-visualize   # resumes from outputs/neat/checkpoints
python -m carracing_neat.play --seed 3                     # best validated genome
python scripts/build_site_data.py && python scripts/benchmark.py --stages neat   # refresh the project page
```
Headless envs skip CarRacing's unused pixel render and the lidar is vectorised (~1200 steps/s single process); `tests/test_lidar.py` checks it against the original loop.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Requirements use the PyTorch CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu` to avoid CUDA downloads.

## Training

```bash
# Recommended production run (v2 recipe, ~12 hours, multi-lap with strong track adherence)
python -m carracing_ppo.train \
  --total-timesteps 4500000 \
  --num-envs 4 \
  --n-steps 1024 \
  --batch-size 256 \
  --offtrack-penalty 15.0 \
  --traction-threshold 0.99 \
  --entropy-coef 0.005 \
  --lr-schedule linear \
  --target-kl 0.02 \
  --domain-randomize \
  --allow-multiple-laps \
  --lap-completion-bonus 100.0 \
  --max-episode-steps 2000 \
  --eval-frequency 50000 \
  --checkpoint-frequency 100000 \
  --checkpoint-dir outputs/checkpoints_v2 \
  --log-dir outputs/logs_v2 \
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
pytest tests/
```
- `smoke_test.py` validates the state wrapper shape, a tiny PPO learn loop, the train entrypoint (separate `monitor/` and `eval_monitor/`) and the `--[no-]allow-multiple-laps` flag.
- `test_multilap.py` forces laps via the base env's `new_lap` flag: bonus paid once, tiles re-armed, out-of-bounds still terminates, time limit wins.
- `test_traction.py` drives onto grass and checks the traction feature and the exact off-track penalty.

## Architecture Notes

- **Observation**: 40-d vector (8 dynamics + 32 lidar). Wrapper lives in `carracing_ppo/env.py`.
- **Policy/Value**: Shared MLP, two 256-unit layers by default (configurable via `--policy-hidden-sizes`).
- **Actions**: SB3 PPO continuous policy with tanh-squashed outputs → steer [-1, 1], gas/brake [0, 1].
- **Training**: PPO (clip 0.2, GAE λ 0.95, γ 0.99, entropy 0.005-0.01). Eval + checkpoint callbacks built into `carracing_ppo/train.py`.
- **Traction (state[7])**: fraction of wheels touching a road tile (`wheel.tiles` non-empty). Gymnasium wheels have no `ground_traction`; reading that attribute made traction a constant 1.0 and disabled the penalty in every run before Oct 2026.
- **Multi-lap Racing**: CarRacing-v2 (gymnasium 0.29.1) reports a finished lap as `truncated`. With `--allow-multiple-laps` (default) the wrapper pays the bonus, resets `new_lap`, `tile_visited_count` and each tile's `road_visited`, and keeps the episode alive. A lap takes ~850 steps, so use `--max-episode-steps 2000` for more than one lap.

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
  - Episode continues after lap, agent can start lap 2, 3, etc. `info["laps_completed"]` is logged in the monitor CSVs.

**Scoring Example (2000-step episodes):**
- One clean lap in ~850 steps, then partial lap 2: ~1000 (tiles) + 100 (bonus) + more tiles − 0.1/step
- Two laps: ~2000 (tiles) + 200 (bonus) − time penalties

## Key Files

- `carracing_ppo/env.py` — State wrapper (dynamics + lidar, multi-lap support, reward shaping)
- `carracing_ppo/train.py` — PPO training entrypoint, multi-env setup, eval/checkpoint callbacks
- `carracing_ppo/play.py` — Roll out saved models (`--gif` to record, reward flags match train)
- `tests/` — pytest: smoke, multi-lap, traction
- `scripts/build_site_data.py`, `scripts/benchmark.py` — generate `docs/assets/data/*.json` and GIFs for the project page
- `docs/index.html` — GitHub Pages project story (static, Chart.js from CDN)
- `notes/` — technical documentation and the Dec 2025 training analysis report
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
- **Late-run collapse** (reward peaks then falls, approx KL / clip fraction rising): use `--lr-schedule linear --target-kl 0.02`
- **Check reward balance**: Monitor off-track penalties vs tile bonuses in TensorBoard
- **Visual inspection**: Use `--render-mode human` during eval to observe driving behavior
