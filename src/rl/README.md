# src/rl — Reinforcement Learning Module

This folder contains the Phase 3 RL implementation.  
Everything here builds **on top of** the Phase 2 simulation — no simulation code was changed.

---

## Files and what they do

| File | Purpose |
|---|---|
| `env_odd_exit.py` | The Gym environment. Wraps the esmini simulation. `reset()` runs the approach, `step(action)` executes the chosen MRM until episode ends. |
| `train.py` | Entry point for training. Reads the config, sets up the environment stack, runs PPO. |
| `evaluate.py` | Entry point for evaluation. Loads a trained policy and runs it on N test episodes. |
| `reward.py` | Reward function. All weights are in the YAML config — no code changes needed to tune. |
| `policy.py` | Creates/loads the SB3 PPO model. Architecture: `[256, 256]` neurons, ELU activation. |
| `ablations.py` | Reads ablation flags from config (`use_risk_features`, `use_shield`, etc.). |
| `scenario_sampler.py` | Risk-prioritised sampler: harder scenarios (more collisions, higher risk) get sampled more often as training progresses. |
| `episode_variation.py` | Per-episode speed/gap variation. Each reset draws new ego speed, front car speed, gap and rear distance so the agent doesn't overfit to one fixed scenario. |
| `callbacks.py` | SB3 training callbacks: episode logging, sampler weight updates, checkpoint saving. |

---

## How an episode works (one_shot mode)

```
env.reset()
    → approach phase runs internally (follow-lane control)
    → ODD exit fires at trigger_time_s
    → returns observation (10 features) at ODD exit moment

env.step(action)    ← called once per episode
    → agent proposes MRM 0/1/2
    → safety shield validates (may downgrade if gaps too small)
    → simulation runs MRM until: collision / full stop (MRC) / timeout
    → returns (obs, total_reward, done=True, info)
```

This is called **one-shot** because the agent makes **one decision** per episode, not one per simulation step. This matches how Phase 2 works: the baseline also picks one MRM at ODD exit and executes it.

---

## How episode variation works

Every call to `env.reset()` randomly draws new scenario parameters:

```
50% probability → baseline regime
    ego speed: 8–11 m/s  (from baseline_comparison config)
    front car: 8–11 m/s  (may be same as ego → gap stays constant)
    rear distance: 12–48 m

50% probability → short-front-gap regime
    ego speed: 8–11 m/s
    front car: 4–7 m/s  (slower → gap closes at ODD exit)
    gap at ODD exit: 4–15 m  (tight — agent must react quickly)
```

This is why training looks "varied" — each episode is a different physical situation.  
The mix ratio is set in `configs/phase3_train.yaml` under `phase3.episode_variation.mix_baseline_probability`.

**To train on different scenarios, see `TRAINING_ON_NEW_SCENARIOS.md` in the project root.**

---

## The observation (what the agent sees)

10 features, all normalised to [0, 1]:

```
Index  Feature                 How computed
  0    ego_speed_mps           esmini state / max_speed_mps
  1    front_gap_m             centre-to-centre distance to front car / 50
  2    rear_gap_m              centre-to-centre distance to rear car / 50
  3    left_gap_m              lateral distance to left neighbour / 50
  4    right_gap_m             lateral distance to right neighbour / 50
  5    front_rel_speed_mps     (ego_speed - front_speed) / 30  (+ = closing)
  6    rear_rel_speed_mps      (rear_speed - ego_speed) / 30
  7    drf                     Driving Risk Field (supervisor formula, already 0-1)
  8    dara                    TTC-based risk (already 0-1)
  9    risk_total              0.5 * drf + 0.5 * dara
```

The agent sees the scene at the **moment ODD fires** and makes one MRM decision based on it.

---

## The three MRM actions

| Action | Name | Brake | Steer | When to use |
|---|---|---|---|---|
| 0 | STRAIGHT_STOP | 100% (full) | 0° | Emergency — high risk, no time |
| 1 | IN_LANE_STOP | 77% | 0° (follows road) | No shoulder space |
| 2 | ROAD_SHOULDER_STOP | 70% | Right turn | Shoulder clear (rear gap > 30 m, right gap > 12 m) |

The safety shield automatically vetoes action 2 (pull-over) if gaps are too small.

---

## Common questions

**Q: Why does the agent always pick action 0 early in training?**  
A: PPO starts with a random policy. Action 0 (STRAIGHT_STOP) results in fewer immediate collisions since it brakes hardest, so it gets reinforced first. After enough episodes the agent learns to distinguish when 1 or 2 is better.

**Q: What is `shoulder_available` in the results?**  
A: True when `rear_gap > 30 m AND right_gap > 12 m` at ODD exit — exactly the condition under which the shield allows ROAD_SHOULDER_STOP. This ensures the reward's shoulder bonus is only reachable when the agent can actually execute it.

**Q: How long to train?**  
A: At 30–40 km/h with headless mode: ~6 episodes/min. Rough guide:
- 500 episodes: smoke test (verify nothing crashes)
- 5,000 episodes: first signs of learned behaviour
- 25,000 episodes: policy largely converged for this speed regime
