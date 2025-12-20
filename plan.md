Below is a practical, CPU-only plan to run **NEAT** on **CarRacing** using a **state-based observation** (i.e., a compact vector derived from physics + “lidar”-style raycasts), not pixels.

This is the core idea: **you still use Gymnasium’s CarRacing**, but you add an observation wrapper that returns a **low-dimensional state vector** that is feasible for NEAT to learn from.

CarRacing in Gymnasium is primarily described as a pixel-observation task, with continuous actions. ([gymnasium.farama.org][1])
NEAT-Python is CPU-friendly and pure Python, so it’s a good fit for this approach. ([NEAT-Python][2])

---

## Target design

### Actions (3 continuous outputs)

CarRacing uses:

* `steer ∈ [-1, 1]`
* `gas ∈ [0, 1]`
* `brake ∈ [0, 1]` ([Gym Documentation][3])

NEAT network outputs can be mapped to these ranges via simple squashing:

* `steer = tanh(o0)`
* `gas = clip((tanh(o1)+1)/2, 0, 1)`
* `brake = clip((tanh(o2)+1)/2, 0, 1)`

NEAT-Python includes tanh activation out of the box. ([GitHub][4])

### State vector (recommended: 20–40 floats)

A good CPU-friendly state vector is:

1. **Speed** (scalar)
2. **Heading and angular velocity** (2 scalars)
3. **Steering angle** (scalar)
4. **Off-road / contact cues** (few scalars if accessible)
5. **“Lidar” rays**: 16–24 distances cast around the car (the main signal)

This “lidar + velocity” approach is common in Box2D-style tasks (Gym’s Box2D docs historically mention lidar usage/changes in CarRacing variants). ([Gym Documentation][3])

Important note: Gymnasium does not provide an official “state vector” API for CarRacing as the default observation; you will be extracting a vector from the underlying simulation. That is normal for this approach, but it is more brittle across versions than pixel observations.

---

## Project structure

```
carracing-neat/
  neat_config/
    neat.cfg
  carracing_neat/
    env_state.py        # wrapper: makes state vector
    eval.py             # evaluate one genome
    train.py            # main NEAT loop
    viz.py              # plots + topology export
  outputs/
    checkpoints/
    videos/
    plots/
  pyproject.toml (or requirements.txt)
```

---

## Step-by-step setup plan (CPU-only)

### 1) Create environment & install deps

On Ubuntu:

* `pip install gymnasium[box2d] neat-python numpy matplotlib`

(If Box2D install is annoying on your system, pinning gymnasium + box2d extras to a known-good set is often worth it, but keep it simple first.)

### 2) Implement a **state observation wrapper**

Create `env_state.py` with a wrapper that:

* internally runs CarRacing normally
* returns `(state_vector, reward, terminated, truncated, info)`

Key mechanics:

* Reset: call `obs, info = env.reset()`, then compute `state = build_state(env)`
* Step: call `obs, reward, terminated, truncated, info = env.step(action)`, then compute `state = build_state(env)`

Implementation detail:

* You will use `env.unwrapped` to reach car pose/velocity and road geometry (this is standard Gym wrapper practice, but depends on internals). Gym wrapper patterns are documented generally. ([gymlibrary.dev][5])

**State construction pseudo-logic**

* speed = norm(car_velocity)
* ang_vel = car_angular_velocity
* heading = angle of car hull
* lidar distances:

  * define ray angles relative to heading (e.g., -90°..+90° plus some rear rays)
  * raycast into the world / road boundaries
  * normalize distances to [0,1] by dividing by max_range

If full world raycasting is too messy initially, start with a simpler proxy:

* sample a few points ahead/left/right in car coordinates
* estimate whether those points are “road” vs “grass” if you can access road polygons
  Then upgrade to real raycasts.

### 3) Define NEAT configuration (feed-forward first)

Create `neat.cfg` with:

* `num_inputs = len(state_vector)` (e.g., 28)
* `num_outputs = 3`
* `feed_forward = True`
* Start with modest population sizes (e.g., 100–200) and add parallel eval later.

Recommended initial NEAT settings:

* activation: `tanh` for hidden and output (simple and stable for continuous control)
* allow adding nodes/connections, but keep mutation rates moderate

NEAT-Python is documented and supports these settings in its config system. ([NEAT-Python][2])

### 4) Write genome evaluation for CarRacing

In `eval.py`:

* Run **N episodes** per genome (start with N=1, later N=3 for stability)
* Use **domain_randomize = False** at first, so learning is not slowed by extra variability (turn it on later)
* Cap episode length (e.g., 800–1000 steps)

Fitness suggestions:

* Use the environment reward as the base
* Add light shaping to reduce degenerate behaviors:

  * small penalty for excessive braking
  * penalty for leaving track (if detectable)
  * bonus for forward progress (approx via tile visitation if accessible in info)
    Keep shaping minimal; CarRacing reward already encodes “visit tiles efficiently”. ([findingtheta.com][6])

### 5) Make it CPU-fast: parallelize evaluation

NEAT evaluation is embarrassingly parallel. Use Python multiprocessing:

* Split genomes across CPU cores
* Each worker creates its own env instance (important)

This typically gives you the biggest wall-clock improvement.

### 6) Instrumentation you will want from day 1

In `train.py`:

* Save best genome each generation
* Save NEAT checkpoint every K generations
* Log:

  * best fitness
  * mean fitness
  * fitness std

In `viz.py`:

* Plot fitness curves per generation (matplotlib)
* Export network topology graph (neat-python has common examples for drawing networks; if you want, I’ll provide a clean implementation)

### 7) Validation loop (render only occasionally)

Rendering slows training. Do:

* During training: `render_mode=None`
* Every X generations: run a **demo episode** with `render_mode="human"` and optionally record video to `outputs/videos/`

---

## Key challenges and how we handle them

### Smooth control (continuous actions)

NEAT can produce jitter. Common fixes:

* **Action smoothing:** `a_t = 0.8*a_{t-1} + 0.2*a_raw`
* **Penalty for action derivatives:** small penalty on `|steer_t - steer_{t-1}|` etc.
* Use feed-forward first; if needed later, switch to recurrent for temporal stability.

### Exploration

CarRacing is sensitive to early exploration. Options:

* Add small Gaussian noise to actions for early generations only
* Use “novelty-ish” tie-breakers (e.g., maximize unique tiles visited) if you can compute it

---

## Minimal milestone plan (what “done” looks like)

1. **Milestone A (1–2 sessions):** wrapper returns a stable state vector; random policy runs without crashes
2. **Milestone B:** NEAT loop runs 20–50 generations, fitness curve moves upward
3. **Milestone C:** best genome completes meaningful track segments reliably
4. **Milestone D:** turn on domain randomization and evaluate robustness

---

## If you want, I can generate the starter code

If you tell me which Gymnasium version you’re using and whether you want **raycast-lidar** immediately or a **simpler “road-probe” state** first, I’ll provide:

* `env_state.py` wrapper
* `neat.cfg`
* `train.py` with multiprocessing
* plotting + best-genome replay script

That will be CPU-only end-to-end.

[1]: https://gymnasium.farama.org/environments/box2d/car_racing/?utm_source=chatgpt.com "Car Racing"
[2]: https://neat-python.readthedocs.io/?utm_source=chatgpt.com "Welcome to NEAT-Python's documentation! — NEAT-Python ..."
[3]: https://gym-docs.readthedocs.io/_/downloads/en/latest/pdf/?utm_source=chatgpt.com "Release 0.24.1"
[4]: https://github.com/CodeReclaimers/neat-python/blob/master/neat/activations.py?utm_source=chatgpt.com "neat-python/neat/activations.py at master"
[5]: https://www.gymlibrary.dev/api/wrappers/?utm_source=chatgpt.com "Wrappers"
[6]: https://findingtheta.com/blog/solving-gymnasiums-car-racing-with-reinforcement-learning?utm_source=chatgpt.com "Solving Gymnasium's Car Racing with Reinforcement Learning"
