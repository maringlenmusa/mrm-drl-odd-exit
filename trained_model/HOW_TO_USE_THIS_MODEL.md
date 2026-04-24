# How to Use This Trained RL Policy

**Model file:** `policy_interrupted.zip`  
**Training status:** 25,363 / 50,000 episodes completed (early stop, still usable)  
**Algorithm:** PPO (Proximal Policy Optimization) — Stable-Baselines3

---

## What this model does

This is a Deep Reinforcement Learning (DRL) agent that selects a **Minimum Risk Manoeuvre (MRM)** for an ego vehicle when it exits its Operational Design Domain (ODD).

At ODD exit, the agent receives the current scene as a 10-feature observation and outputs one of three discrete actions:

| Action | MRM | Behaviour |
|---|---|---|
| 0 | STRAIGHT_STOP | Full brake, no steering — emergency stop in place |
| 1 | IN_LANE_STOP | Moderate brake, stays in lane — follows road geometry |
| 2 | ROAD_SHOULDER_STOP | Moderate brake + steer right — pulls over to shoulder |

---

## Requirements

Install on Python 3.10+:

```bash
pip install stable-baselines3>=2.0.0 gymnasium>=0.26.0 numpy>=1.21.0
```

PyTorch is also required (installed automatically with stable-baselines3):
```bash
pip install torch
```

---

## Quickest way to query the model (no simulator needed)

```python
from stable_baselines3 import PPO
import numpy as np

# Load the model (no environment needed for inference)
model = PPO.load("policy_interrupted.zip")

# Build an observation vector (10 features, all normalised to [0, 1])
# Feature order must be exact:
#   [0] ego_speed_mps       / 15.0   (max speed 15 m/s = 54 km/h)
#   [1] front_gap_m         / 50.0   (clipped at 50 m; no car = 1.0)
#   [2] rear_gap_m          / 50.0
#   [3] left_gap_m          / 50.0
#   [4] right_gap_m         / 50.0
#   [5] front_rel_speed_mps / 30.0   (positive = closing toward front car)
#   [6] rear_rel_speed_mps  / 30.0
#   [7] drf                          (already 0-1, Driving Risk Field)
#   [8] dara                         (already 0-1, TTC-based risk)
#   [9] risk_total                   (already 0-1, weighted DRF+DARA)

# Example: ego at 10 m/s, 19 m front gap, 35 m rear gap, 20 m right gap,
#          closing at 4 m/s, moderate risk
obs = np.array([
    10.0 / 15.0,   # ego_speed
    19.0 / 50.0,   # front_gap
    35.0 / 50.0,   # rear_gap
     1.0,          # left_gap (no left neighbor -> 1.0)
    20.0 / 50.0,   # right_gap
     4.0 / 30.0,   # front_rel_speed (closing)
     0.0 / 30.0,   # rear_rel_speed
     0.016,        # drf
     0.21,         # dara (TTC approx 4.4 s)
     0.11,         # risk_total
], dtype=np.float32)

action, _states = model.predict(obs, deterministic=True)
mrm_names = {0: "STRAIGHT_STOP", 1: "IN_LANE_STOP", 2: "ROAD_SHOULDER_STOP"}
print("Agent chose:", mrm_names[int(action)])
```

---

## Observation normalisation rules (critical — must match exactly)

| Feature | Raw value | Normalisation | Notes |
|---|---|---|---|
| `ego_speed_mps` | speed in m/s | / 15.0 | Max speed 15 m/s (54 km/h) |
| `front_gap_m` | distance to front car (m) | / 50.0, clipped | No front car = 1.0 |
| `rear_gap_m` | distance to rear car (m) | / 50.0, clipped | No rear car = 1.0 |
| `left_gap_m` | lateral distance left (m) | / 50.0, clipped | No left neighbor = 1.0 |
| `right_gap_m` | lateral distance right (m) | / 50.0, clipped | No right neighbor = 1.0 |
| `front_rel_speed_mps` | closing speed (+ = closing) | / 30.0, clipped | |
| `rear_rel_speed_mps` | rear closing speed | / 30.0, clipped | |
| `drf` | Driving Risk Field (Kiran formula) | already 0-1 | |
| `dara` | Distance-Adjusted Risk Assessment | already 0-1 | Based on TTC |
| `risk_total` | 0.5 x drf + 0.5 x dara | already 0-1 | Combined risk |

**DRF formula:**
```
risk_total = 0.5 x DRF + 0.5 x DARA
DRF = min(1.0, sum( exp(k_theta x delta_v) x exp(-k_r x distance) ) / 350.0)
    where k_theta=0.5, k_r=0.05, max_distance=50m
DARA = clip((ttc_safe - TTC) / (ttc_safe - ttc_critical), 0, 1)
    where ttc_safe=5.0s, ttc_critical=2.0s
```

---

## Evaluation results (80 episodes, mixed scenarios)

### What "valid episode" means

Only episodes where ODD fired AND the collision (if any) was **physically avoidable** are counted.

**Excluded (not the agent's fault):**
- **Before-ODD collisions** (`last_sim_time_s < trigger_time_s`): front car was already too close before ODD even fired.
- **Physically inevitable collisions** — determined by the formula below, NOT a fixed time threshold.

### How "inevitability" is determined (physics-based)

A collision is **inevitable** — impossible to avoid even with the best possible braking (STRAIGHT_STOP, 4.5 m/s²) — when:

```
(front_gap_at_ODD_exit - 5.0) ≤  v_rel²  / (2 × 4.5)
```

Where:
- `front_gap_at_ODD_exit` = centre-to-centre distance to front car when ODD fires (m)
- `5.0 m` = esmini bounding-box collision distance (fires when centres are ~5 m apart; calibrated from data)
- `v_rel` = actual closing speed at ODD exit (m/s), measured from gap change between first two ODD-active simulation steps
- `4.5 m/s²` = maximum deceleration (STRAIGHT_STOP, full brake)

**Why a fixed time threshold is wrong:** two episodes at 1.15 s and 2.0 s after ODD had physically avoidable collisions (the agent chose a lighter MRM instead of full braking). A 2 s cutoff would incorrectly exclude them. Only the physics formula is correct.

In practice, **30 of 32 collisions** in the RL evaluation were physically inevitable. 2 were genuine agent failures.

---

### Correct results (physics-based filter applied)

**Both systems evaluated on 80 episodes:**
- 50% baseline regime: ego 8–11 m/s, front gap ~20 m at ODD exit
- 50% short-front-gap regime: ego 8–11 m/s, front gap 4–15 m at ODD exit
- 30% of episodes include static shoulder obstacles (right_gap ~4 m → shoulder blocked)

| Metric | RL Policy (this model) | Phase 2 Rule-Based Baseline |
|---|---|---|
| **MRC rate (valid episodes)** | **94.9%** | 82.6% |
| Collision rate (valid) | **5.1%** | 17.4% |
| Easy scenarios MRC (gap > 20 m) | **100%** | 100% |
| Hard scenarios MRC (gap ≤ 20 m) | **66.7%** | 63.2% |
| ROAD_SHOULDER_STOP used | **51%** | 26% |
| Shoulder used correctly | **80%** | 34% |
| IN_LANE_STOP used | 33% | 32% |
| STRAIGHT_STOP used | 15% | 41% |
| Avg time ODD exit → MRC | 5.2 s | 5.0 s |

Valid episodes counted: **39** for RL, **121** for baseline (baseline ran more episodes with wider controlled scenarios).

---


---


## Training configuration

| Parameter | Value |
|---|---|
| Algorithm | PPO (Stable-Baselines3) |
| Network | 256 x 256 neurons, ELU activation |
| Episodes trained | 25,363 (of 50,000 requested) |
| Ego speed | 30-40 km/h (8-11 m/s) |
| Road | 500 m curved road, 20 degree right curve |
| ODD trigger | 7.0 s into simulation |
| Deceleration | 4.5 m/s2 (city braking) |
| Shoulder obstacles | 30% of episodes (right_gap ~4 m) |
| MRM decision mode | One-shot (agent decides once at ODD exit) |
