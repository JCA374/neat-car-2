# What PPO has taught us (keep this up to date)

Each line is a lesson, followed by the steps that show it. Re-read the step's `outcome` before relying on a lesson.

## Reward shaping
- **A per-frame grass penalty makes PPO creep.** The penalty swamped the tile reward, and the policy learned to drive at walking pace. (p01, p02)
- **Ending an episode when the car is stuck breaks PPO.** It never practises getting back on the road. Episodes dropped to about 350 frames and the car spent 82–92% of the time on grass. Dropped for PPO for good. (p03, p04, p07)
- Equal rules alone (2000-frame episodes, 0.1 smoothing, raw reward) did not fix the old recipe: 0/5 on both seeds. (p10)

## Sensors
- **Forward rays help PPO as much as they help NEAT:** 16 rays over 240° took PPO from 1/5 to 3/5 tracks, with the fastest lap of the time (16.8 s). (p05 → p06)

## Luck and selection
- **A single run is luck.** p08 and p06 split apart within minutes. Every PPO step trains 2 seeds, and the lab uses 3. (p08, p10+)
- **Pick checkpoints by validation laps, not eval reward.** It's the fairer rule, but it can't rescue a policy that never learned to finish. (p09)

## Decision rate and controls (PPO lab rounds 1–2, 600k frames, 3 seeds)
- **Deciding every frame never learns to finish**, in any control mode (raw3, throttle, discrete). Deciding every 4 frames is what works. Every 2 and every 8 frames were worse. (lab round 1–2)
- `raw3_r4_g995` (raw3 controls, decide every 4 frames, gamma 0.995) was the most consistent recipe, with 5/5 on validation at its best on all 3 seeds. Its finals fall back, though. (lab round 2 → p14: 4/5 at 21.7 s on the benchmark after 13 min)
- **Calm exploration** (`log-std-init -1`, no entropy bonus) gives fast single laps but unstable runs. (lab `raw3_r4_calm`, p12 gSDE: 1/5)

## Braking (off-road analysis of p14, 60 non-benchmark tracks, `outputs/analysis/p14_raw3_r4_g995/`)
- **With raw3 controls PPO never learns to brake.** The brake is above 0.1 in 0 of 83,647 frames, and the unclipped brake output sits near −1 everywhere, even just before leaving the road. The car slows only by lifting off the gas (gas output falls from −0.44 on straights to −1.8 before corners), which is too weak for hairpins.
- So it leaves the road in sharp turns (13–20× the rate on straights), **always on the outside**. Its inputs already predict this (AUC 0.96): the missing piece is an action, not a sensor.
- The speed input's cap at 50 (38% of frames) and bang-bang steering are not the cause.
- 5 validation tracks overrate it: 5/5 there, 38/55 (69%) on fresh tracks.
- Being tested: one pedal axis (`throttle`), so lifting off becomes braking (lab round 3).

## Other results so far
- Normalising the reward: 2/5 at 17.9 s. (p11)
- RL Zoo's CarRacing settings: 0/5. (p13)
- Open questions: 32 forward rays (p15), a lap-time bonus like NEAT's (p16).
