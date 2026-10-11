# Technical Documentation: PPO for CarRacing-v2

This document provides a comprehensive technical overview of the `carracing-ppo` project. This project trains a Proximal Policy Optimization (PPO) agent to drive a car in the `CarRacing-v2` environment from the `gymnasium` library.

The goal of this documentation is to provide all the necessary information to understand, run, and replicate this project from scratch.

## Project Overview

The primary goal of this project is to successfully train a reinforcement learning agent to navigate a race track. Instead of using raw pixel data from the environment, which is computationally expensive to process, this project implements a custom state representation. This custom state vector, combined with a shaped reward function, allows for efficient training of a relatively simple Multi-Layer Perceptron (MLP) policy using the `stable-baselines3` implementation of PPO.

The project is structured to facilitate experimentation with different hyperparameters, environment settings, and to easily view the performance of trained agents.

## Project Structure

A clear structure separates the environment logic, training, and inference scripts.

```
├── carracing_ppo/
│   ├── __init__.py      # Makes the directory a Python package
│   ├── env.py           # Custom environment wrapper and state representation
│   ├── play.py          # Script to run a trained agent
│   └── train.py         # Main script for training the PPO agent
├── docs/                # GitHub Pages site (project story)
├── notes/               # This file + the Dec 2025 training analysis report
├── outputs/
│   ├── checkpoints*/    # Saved model checkpoints (best/latest tracked in git)
│   └── logs*/           # Monitor CSVs (train: monitor/, eval: eval_monitor/), TensorBoard
├── scripts/             # Plotting, site data, GIF recording
├── tests/               # pytest: smoke test + multi-lap behaviour
└── requirements.txt     # Python dependencies
```

## Setup and Installation

To run this project, you need Python and the dependencies listed in `requirements.txt`.

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    ```

2.  **Install dependencies:**
    Create a file named `requirements.txt` with the content below and run `pip install -r requirements.txt`.

    ```ini
    --extra-index-url https://download.pytorch.org/whl/cpu
    gymnasium[box2d]==0.29.1
    stable-baselines3==2.3.2
    torch==2.3.1+cpu
    tensorboard>=2.16.2
    tqdm>=4.67.0
    rich>=13.7.0
    numpy>=1.24
    matplotlib>=3.7
    ```

## Core Components Deep Dive

### 1. Custom Environment (`carracing_ppo/env.py`)

This project's key innovation is the `CarRacingStateWrapper`, which avoids pixel-based learning in favor of a compact and informative state vector. This is the most important part for understanding the agent's behavior.

#### State Representation

The wrapper creates a **40-dimensional state vector** on each step. This vector is composed of two parts:

*   **8 Car Dynamics Features:** These give the agent a sense of its own physical state.
    *   `speed_norm`: Normalized car speed.
    *   `local_vx`, `local_vy`: Car's velocity in its own reference frame.
    *   `cos(heading)`, `sin(heading)`: Car's orientation.
    *   `angular_velocity`: How fast the car is turning.
    *   `mean_steer`: Average steering angle of the wheels.
    *   `traction_fraction`: A measure of how well the tires are gripping the road. This is crucial for the off-track penalty.

*   **32 Lidar-like Road Sensors:** These give the agent a "view" of the track boundaries. The wrapper casts 32 rays around the car and measures the distance to the edge of the track for each ray.

The state vector is constructed in the `_build_state` method. Here is a simplified snippet showing how the base features are calculated:

```python
# From carracing_ppo/env.py

def _build_state(self, obs) -> np.ndarray:
    car = getattr(self.env.unwrapped, "car", None)
    # ... safety checks ...

    vel = car.hull.linearVelocity
    heading = float(car.hull.angle)

    # Transform velocities into the car's local frame
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    local_vx = vel.x * cos_h + vel.y * sin_h
    local_vy = -vel.x * sin_h + vel.y * cos_h
    speed = math.sqrt(vel.x**2 + vel.y**2)

    # ... calculate traction, steering ...

    base_features = np.array([...])
    lidar = self._lidar_distances(...)
    state = np.concatenate([base_features, lidar]).astype(np.float32)
    return state
```

#### Reward Shaping

To guide the agent, the wrapper modifies the reward signal:
*   **Off-track Penalty:** A penalty (`offtrack_penalty`) is subtracted from the reward if the `traction_fraction` drops below a `traction_threshold`. This effectively punishes the agent for driving on the grass.
*   **Lap Bonus / multi-lap:** CarRacing-v2 (gymnasium 0.29.1) ends a finished lap with `truncated=True`. With `allow_multiple_laps=True` (default) the wrapper pays `lap_completion_bonus`, resets the base env's `new_lap` flag, `tile_visited_count` and every tile's `road_visited`, and keeps the episode running, so the next lap earns tile rewards again. Out-of-bounds (`terminated`, -100) and the time limit are untouched. `info["laps_completed"]` is always present and is logged to the monitor CSVs.
*   **Episode length:** CarRacing stops at 1000 steps and a lap takes ~850, so a second lap needs `--max-episode-steps 2000` (or a much faster driver).

### 2. PPO Model and Training (`carracing_ppo/train.py`)

This script orchestrates the training process using `stable-baselines3`.

#### PPO Model Instantiation

The core of the script is where the `PPO` model is created. If no `--resume` path is provided, a new model is initialized with hyperparameters passed via the command line.

```python
# From carracing_ppo/train.py

# Policy network architecture is configured here
policy_kwargs = dict(net_arch=dict(pi=args.policy_hidden_sizes, vf=args.policy_hidden_sizes))

# Create a new PPO model
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.entropy_coef,
    vf_coef=args.vf_coef,
    max_grad_norm=args.max_grad_norm,
    tensorboard_log=args.log_dir,
    policy_kwargs=policy_kwargs,
    seed=args.seed,
    verbose=1,
)
```

#### Training Execution

The `model.learn()` call starts the training loop. Callbacks are used to save checkpoints periodically and to evaluate the agent's performance, saving the best-performing model.

```python
# From carracing_ppo/train.py

# Callbacks for saving models
callback = _build_callbacks(args, eval_env)

# Start training
model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

# Save the final model
latest_path = os.path.join(args.checkpoint_dir, "ppo_latest.zip")
model.save(latest_path)
```

### 3. Playing with a Trained Agent (`carracing_ppo/play.py`)

This script is used to load and visualize a trained agent.

It first loads a saved model from a `.zip` file, then creates the environment, and runs a loop where the model predicts an action and the environment is updated.

```python
# From carracing_ppo/play.py

# Load the trained PPO model
model = PPO.load(model_path, device="auto")

# Create the environment with rendering enabled
env = DummyVecEnv(
    [lambda: make_env(render_mode="human")]
)

obs = env.reset()
for _ in range(max_steps):
    # Predict the action deterministically
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, infos = env.step(action)
    if done[0]:
        break
```

## How to Use

### Training a New Agent

To train an agent, run the `train.py` script. All hyperparameters are configurable via command-line arguments.

**Basic Example:**
```bash
python -m carracing_ppo.train --total-timesteps 200000
```

**Advanced Example (more parallelism and custom policy network):**
```bash
python -m carracing_ppo.train \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --learning-rate 0.0001 \
    --policy-hidden-sizes 128,128,128 \
    --log-dir outputs/logs_8env
```

### Monitoring Training

Training progress, rewards, and other metrics are logged for TensorBoard.
```bash
tensorboard --logdir outputs/logs
```
Navigate to `http://localhost:6006/` in your browser.

### Playing with a Trained Agent

Use the `play.py` script to watch your trained agent drive.
```bash
# Play with the best model saved during training
python -m carracing_ppo.play --model outputs/checkpoints/best_model.zip

# Run 5 headless episodes with the same reward shaping used in training
python -m carracing_ppo.play --episodes 5 --render-mode None \
    --offtrack-penalty 15 --traction-threshold 0.99

# Record a GIF of one episode
python -m carracing_ppo.play --deterministic --seed 3 --gif out.gif
```
## Reward Settings Cheat Sheet

| Setting | `offtrack_penalty` | `traction_threshold` | Behaviour |
|---|---|---|---|
| Default (baseline) | 1.0 | 0.95 | Learns fast, cuts corners over grass |
| Moderate | 5–10 | 0.98 | Fewer excursions, still aggressive |
| Strict (recommended) | 15 | 0.99 | Clean laps; best results so far |

Combine strict settings with `--allow-multiple-laps --lap-completion-bonus 100 --max-episode-steps 2000` to reward fast, repeated laps.

## Operational Notes

- Vector env: `SubprocVecEnv` when `num_envs>1`, else `DummyVecEnv`; if subprocesses are not allowed it falls back to one `DummyVecEnv`.
- Eval and checkpoint frequencies are given in env steps and divided by `num_envs` internally.
- Lidar and traction read Box2D internals; if hull/wheels are missing the wrapper returns zeros instead of crashing.
- Domain randomization changes colours, not track physics; track layout is random every reset unless seeded.

## Replicating the Project

Clone the repo, follow *Setup and Installation*, run `pytest tests/`, then use the training commands in the README. The full story of how the settings were found is on the project page (`docs/index.html`).
