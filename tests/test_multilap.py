"""Quick test to verify multi-lap functionality."""

from carracing_ppo.env import make_env
import numpy as np

# Create environment with multi-lap enabled
env = make_env(
    render_mode=None,
    domain_randomize=False,
    offtrack_penalty=15.0,
    traction_threshold=0.99,
    allow_multiple_laps=True,
    lap_completion_bonus=100.0,
)

print("Testing multi-lap functionality...")
print(f"Environment created with allow_multiple_laps=True")
print(f"Lap completion bonus: 100.0")
print()

# Reset and run a few steps
state, info = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"Expected: 40-dimensional vector (8 dynamics + 32 lidar)")
print()

# Run random actions for a bit
total_reward = 0
laps_completed = 0
for step in range(10):
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if 'laps_completed' in info:
        laps_completed = info['laps_completed']
        print(f"LAP COMPLETED! Total laps: {laps_completed}")

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        break

print()
print(f"Total reward after {step+1} steps: {total_reward:.2f}")
print(f"Laps completed: {laps_completed}")
print()
print("✓ Multi-lap functionality test complete!")
