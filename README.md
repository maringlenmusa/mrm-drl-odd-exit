# DRL-Based Minimum Risk Manoeuvre Selection at ODD Exit

**Master Thesis Project** — Deep Reinforcement Learning for autonomous vehicle safety decisions.

---

## Overview

When an automated vehicle exits its Operational Design Domain (ODD) — e.g. a sensor failure, road hazard, or unexpected situation — it must execute a **Minimum Risk Manoeuvre (MRM)**: a controlled action to bring the vehicle to a safe state.

This project implements and evaluates a Deep RL agent that selects the optimal MRM at ODD exit using scene context (gaps, speeds, risk scores) as input.

**Three MRM types (from ISO 23793):**

| ID | MRM | What it does |
|---|---|---|
| 0 | STRAIGHT_STOP | Full brake, no steer — stops in place |
| 1 | IN_LANE_STOP | Moderate brake, follows lane geometry |
| 2 | ROAD_SHOULDER_STOP | Moderate brake + steer right → pulls over |

---

## Project structure

```
src/
  phase2/          Phase 2: rule-based baseline (risk + shield + MRM rules)
  rl/              Phase 3: RL training, evaluation, environment
    env_odd_exit.py      Gym-style environment wrapping esmini simulation
    train.py             PPO training entrypoint
    evaluate.py          Evaluation entrypoint
    reward.py            Reward function design
    policy.py            SB3 PPO model factory
    ablations.py         Ablation flag system
    scenario_sampler.py  Risk-prioritised scenario sampling
    episode_variation.py Per-episode speed/gap variation (Phase 2 distributions)
  sim/             esmini interface (esminiLib, UDP bridge, state parser)
  risk/            DRF + DARA + combined risk
  decision/        MRM catalog, baseline rules, safety shield, executor
  features/        Observation builder, neighbour selector
  metrics/         MRC checker, episode tracker

configs/
  phase2.yaml            Phase 2 baseline + simulation parameters
  phase3_train.yaml      RL training configuration
  phase3_eval.yaml       Evaluation configuration

scripts/
  inspect_policy.py        Quick policy inspection (no simulator)
  compare_rl_vs_baseline.py RL vs Phase 2 comparison table

trained_model/
  policy_interrupted.zip     Trained PPO policy (25,363 episodes)
  HOW_TO_USE_THIS_MODEL.md   Full usage guide + evaluation results

docs/
  phase3_how_to_run.md     Step-by-step training/evaluation guide
  mrm_controller_outputs.md MRM inputs, outputs, thresholds explained
```

---

## Setup

**Requirements:** Python 3.10+, esmini (for full simulation)

```bash
pip install -r requirements.txt
```

For full simulation runs, esmini must be installed separately (see [esmini releases](https://github.com/esmini/esmini/releases)). Place `esmini/bin/` in the project root.

---

## Quick start: query the trained model (no simulator needed)

```python
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("trained_model/policy_interrupted.zip")

# Observation: 10 features, normalised [0,1]
# [ego_speed/15, front_gap/50, rear_gap/50, left_gap/50, right_gap/50,
#  front_rel_speed/30, rear_rel_speed/30, drf, dara, risk_total]
obs = np.array([
    10.0/15.0,  # ego speed 10 m/s (36 km/h)
    19.0/50.0,  # front gap 19 m
    35.0/50.0,  # rear gap 35 m (shoulder available)
    1.0,        # left gap (no car)
    20.0/50.0,  # right gap 20 m (clear shoulder)
    4.0/30.0,   # closing at 4 m/s
    0.0,
    0.016, 0.21, 0.11,  # risk scores
], dtype=np.float32)

action, _ = model.predict(obs, deterministic=True)
print({0: "STRAIGHT_STOP", 1: "IN_LANE_STOP", 2: "ROAD_SHOULDER_STOP"}[int(action)])
```

---

## Running training

```bash
# Full training (overnight, ~14 hours at 30-40 km/h)
py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml

# With esmini UI (watch episodes live)
# Set sim.headless: false + sim.esmini_lib_show_ui: true in config first
```

See `docs/phase3_how_to_run.md` for full details.

---

## Running evaluation (RL vs baseline)

```bash
# Evaluate trained RL policy
py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml

# Run Phase 2 rule-based baseline
py -3 -u -m src.main --phase phase2_baseline --config configs/phase2.yaml --headless

# Compare results (physics-based filter for inevitable collisions)
py -3 scripts/compare_rl_vs_baseline.py
```

---

## Key results

Evaluated on 80 episodes (50% easy scenarios, 50% short-front-gap).  
Inevitable collisions (physically impossible to avoid with any MRM) excluded using physics formula.

| Metric | RL Policy (25k episodes) | Phase 2 Rule-Based |
|---|---|---|
| **MRC rate** | **94.9%** | 82.6% |
| Collision rate | **5.1%** | 17.4% |
| Shoulder used correctly | **80%** | 34% |
| Unnecessary STRAIGHT_STOP | 15% | 41% |

See `trained_model/HOW_TO_USE_THIS_MODEL.md` for full methodology and results.

---

## System design

```
Observation (10 features)           Safety shield (veto unsafe pull-over)
  → PPO [256×256 ELU] network   →       → esmini physics simulation
  → MRM action (0/1/2)          →       → MRC check (full stop)
                                          → Reward: collision/MRC/shoulder/risk
```

- **Simulation:** esmini (OpenSCENARIO/OpenDRIVE) via esminiLib in-process
- **RL algorithm:** PPO, Stable-Baselines3, `net_arch=[256,256]`, ELU activation
- **Speed regime:** 30–40 km/h (8–11 m/s) city/sub-urban
- **Risk metrics:** DRF (Driving Risk Field, supervisor formula) + DARA (TTC-based)
- **Decision mode:** one-shot — agent decides MRM once at ODD exit

---

## Citation / Contact

Master Thesis, Technische Hochschule Ostwestfalen-Lippe  
Supervisor: Kiran Bhaskar Sajikumar, AG Intelligent Systems, inIT
