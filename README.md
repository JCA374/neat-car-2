# NEAT + PPO CarRacing (CPU-friendly)

Learns to drive Gymnasium's `CarRacing-v2` from a compact state vector (car dynamics + lidar-style distances) instead of pixels, either by evolving networks with **NEAT** (the main approach) or with **PPO** (the comparison baseline). CPU-only.

**Project story:** <https://jca374.github.io/neat-car-2/> (source: [`docs/index.html`](docs/index.html)) describes where the project started (NEAT), each step that changed results, and the bugs found along the way. It is published with GitHub Pages from `docs/`.

## Model Summary

- Observation: 40-dim vector (8 dynamics scalars incl. on-road traction + 32 lidar distances).
- Policy/Value: MLP with two 256-unit hidden layers.
- Actions: steer in [-1, 1], gas/brake in [0, 1].
- Training: SB3 PPO (clip 0.2, GAE λ 0.95, γ 0.99), optional linear LR decay and KL early-stop, eval + checkpoint callbacks.
- Reward shaping: off-track penalty from wheel traction, plus optional multi-lap episodes with a lap bonus.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```
The requirements use the PyTorch CPU wheel (`--extra-index-url https://download.pytorch.org/whl/cpu`).

## NEAT (main approach)

```bash
# Evolve (resumes automatically from outputs/neat/checkpoints)
python -m carracing_neat.train --generations 150 --tracks-per-generation 3 --auto-visualize

# Watch the best validated genome
python -m carracing_neat.play --genome outputs/neat/checkpoints/best-genome.pkl --seed 3
python -m carracing_neat.play --seed 3 --gif neat.gif
```
- Every genome in a generation drives the same `--tracks-per-generation` random tracks (new ones each generation); fitness is the raw game score.
- Each generation's champion is also scored on 5 held-out tracks; `best-genome.pkl` is the best by that validation score.
- Logs in `outputs/neat/logs/`: `fitness.csv`, `validation.csv`, `genome_stats.csv`. Config: `neat_config/neat.cfg` (pop 100).

## PPO (baseline)

```bash
# Quick smoke run
python -m carracing_ppo.train --total-timesteps 100000 --num-envs 2 --n-steps 512 --batch-size 256

# Recommended (v2 recipe, ~12 h on a 4-core CPU)
python -m carracing_ppo.train \
  --total-timesteps 4500000 --num-envs 4 --n-steps 1024 --batch-size 256 \
  --offtrack-penalty 15.0 --traction-threshold 0.99 --entropy-coef 0.005 \
  --lr-schedule linear --target-kl 0.02 \
  --domain-randomize --allow-multiple-laps --lap-completion-bonus 100.0 --max-episode-steps 2000 \
  --eval-frequency 50000 --checkpoint-frequency 100000 \
  --checkpoint-dir outputs/checkpoints_v2 --log-dir outputs/logs_v2 --seed 42

# Resume
python -m carracing_ppo.train --resume outputs/checkpoints_v2/ppo_latest.zip --total-timesteps 1000000
```

### Key parameters

| Flag | Default | Notes |
|---|---|---|
| `--offtrack-penalty` | 1.0 | Per-frame penalty × `max(0, threshold − traction)`; 15 is strict |
| `--traction-threshold` | 0.95 | Fraction of wheels that must be on road; 0.99 = all four |
| `--entropy-coef` | 0.01 | 0.015 made the baseline's policy noisier over time; 0.005 works well |
| `--lr-schedule` | constant | `linear` decays LR to 0; prevents the late-run collapse seen with constant LR |
| `--target-kl` | none | e.g. `0.02` stops an update early when the policy moves too far |
| `--[no-]allow-multiple-laps` | on | After a lap: +bonus, tiles re-armed, keep driving |
| `--lap-completion-bonus` | 100 | Bonus per completed lap |
| `--max-episode-steps` | 1000 | A lap takes ~850 steps; use 2000 so multi-lap matters |
| `--domain-randomize` | off | Randomised colours for robustness |

Logs: training monitor CSVs in `<log-dir>/monitor/`, eval episodes in `<log-dir>/eval_monitor/`, TensorBoard in `<log-dir>/PPO_*`.

## Play a saved policy

```bash
python -m carracing_ppo.play --model outputs/checkpoints_v2/best_model.zip --deterministic

# Headless, scored with the same shaping used in training
python -m carracing_ppo.play --render-mode None --episodes 5 --offtrack-penalty 15 --traction-threshold 0.99

# Record a GIF
python -m carracing_ppo.play --deterministic --seed 3 --gif out.gif
```
`--model` falls back to `best_model.zip` / `ppo_latest.zip` in `--checkpoint-dir`.

## Repo structure

- `carracing_neat/`: NEAT training (shared-track parallel evaluator, validation), replay, network plots.
- `neat_config/neat.cfg`: NEAT hyperparameters.
- `carracing_ppo/env.py`: shared state wrapper (used by both NEAT and PPO) (dynamics + lidar, traction, off-track penalty, multi-lap).
- `carracing_ppo/train.py`: PPO training entrypoint with eval/checkpoint callbacks.
- `carracing_ppo/play.py`: roll out a saved policy (human, headless or GIF).
- `tests/`: pytest: smoke, multi-lap, traction, lidar equivalence, NEAT evaluation.
- `scripts/`: plotting, `build_site_data.py` and `benchmark.py` (generate the project page's data and GIFs).
- `docs/`: GitHub Pages site.
- `notes/`: technical documentation and the Dec 2025 training analysis report.
- `outputs/`: checkpoints and logs (only best/latest models are tracked).

## History

The repo began as a NEAT (neuroevolution) project and switched to PPO after two days. Each step, including the bugs found when cleaning up (the off-track penalty and multi-lap mode had never worked), is described on the [project page](docs/index.html).
