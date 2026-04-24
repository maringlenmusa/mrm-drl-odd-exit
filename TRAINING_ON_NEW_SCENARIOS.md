# Training on New Scenarios

This guide explains how to train the RL agent on different driving scenarios.  
You do not need to touch any Python code for most changes — everything is controlled through the YAML config file.

---

## Quick map: what controls what

```
configs/phase3_train.yaml
  │
  ├── state_bridge.*          → ego speed, front/rear car positions and speeds
  ├── baseline_comparison.*   → speed/distance ranges for "easy" training episodes
  ├── short_front_gap_coverage.* → tight-gap ranges for "hard" training episodes
  ├── phase3.episode_variation.mix_baseline_probability → ratio easy:hard per episode
  ├── shoulder_obstacles.*    → whether/how often shoulder is blocked by obstacles
  ├── odd_exit.trigger_time_s → when the ODD exit fires (seconds into scenario)
  ├── sim.decel_mps2          → braking strength (m/s²)
  └── reward.*                → collision penalty, MRC bonus, shoulder bonus
```

---

## 1. Change the speed regime

The current setup runs at 30–40 km/h (8–11 m/s).  
To change to a different speed, edit these four blocks together:

```yaml
# --- ego and actors ---
state_bridge:
  initial_speed_mps: 9.0       # ego start speed (m/s). 9 m/s = 32 km/h
  ego_decel_mps2: 4.5          # MUST match sim.decel_mps2 below
  actors:
    - speed_mps: 6.0           # front car speed (m/s). Keep slower than ego.

# --- simulation physics ---
sim:
  decel_mps2: 4.5              # max braking deceleration (m/s²)

# --- training episode ranges ---
baseline_comparison:
  speed_min_mps: 8.0           # min ego speed for easy episodes
  speed_max_mps: 11.0          # max ego speed for easy episodes

short_front_gap_coverage:
  ego_speed_min_mps: 8.0
  ego_speed_max_mps: 11.0
  lead_speed_min_mps: 4.0      # front car speed range
  lead_speed_max_mps: 7.0

# --- normalisation (important: set to ~max expected speed) ---
risk:
  max_speed_mps: 15.0
```

**Rule of thumb:** `ego_decel_mps2` = `sim.decel_mps2` always.  
`risk.max_speed_mps` should be slightly above the maximum ego speed you expect.

**Example: 60–80 km/h (highway) regime:**
```yaml
state_bridge:
  initial_speed_mps: 20.0
  ego_decel_mps2: 7.0
  actors:
    - speed_mps: 15.0
sim:
  decel_mps2: 7.0
baseline_comparison:
  speed_min_mps: 17.0    # 60 km/h
  speed_max_mps: 22.0    # 80 km/h
short_front_gap_coverage:
  ego_speed_min_mps: 17.0
  ego_speed_max_mps: 22.0
  lead_speed_min_mps: 10.0
  lead_speed_max_mps: 15.0
risk:
  max_speed_mps: 30.0
```

---

## 2. Change the gap at ODD exit (tight vs safe scenarios)

The `short_front_gap_coverage` section controls how close the ego is to the lead car when ODD fires.

```yaml
short_front_gap_coverage:
  gap_at_trigger_min_m: 4.0    # smallest gap at ODD exit (m). < 3m → almost always collision
  gap_at_trigger_max_m: 15.0   # largest gap for "short gap" episodes
```

**Why it matters:**  
Larger gap → easier to stop, more MRC episodes → agent learns reward signal faster.  
Smaller gap → more physical pressure, tests emergency braking.

The `gap_at_trigger_min_m` must be large enough that stopping is *physically possible*:
```
min stoppable gap = v_rel² / (2 × decel) + 5m (bounding box)
```
At v_rel = 4 m/s, decel = 4.5 m/s²: min = 1.8 + 5 = **6.8 m** (use ≥ 7 m to be safe).

---

## 3. Change the mix of easy vs hard episodes

```yaml
phase3:
  episode_variation:
    mix_baseline_probability: 0.5   # 0.0 = all hard, 1.0 = all easy, 0.5 = 50/50
```

**Practical guidance:**
- Start with `0.7` (more easy) if the agent is not learning (too many collisions)
- Use `0.5` for balanced training
- Use `0.3` (more hard) for stress-testing a policy that already converges

---

## 4. Add shoulder obstacles (blocked shoulder)

When `shoulder_obstacles.enabled: true`, 30% of episodes have static barriers on the right shoulder, making the shoulder unavailable. This teaches the agent to check the right gap before pulling over.

```yaml
shoulder_obstacles:
  enabled: true
  probability: 0.3        # fraction of episodes with obstacles (0.0 to 1.0)
  right_offset_m: 4.0     # how far right of lane centre the barriers are
  count: 4                # number of barriers
  spacing_m: 40.0         # metres between barriers
  start_ahead_m: 20.0     # first barrier starts this far ahead of ego
```

Set `probability: 0.0` to train without any obstacles.  
Set `probability: 0.5` if you want shoulder blocking to be more common.

---

## 5. Change the ODD exit timing

```yaml
odd_exit:
  trigger_time_s: 7.0    # ODD fires at this simulation time (seconds)
```

Increasing this gives the ego more approach time (wider gaps by the time ODD fires).  
Decreasing it makes ODD fire earlier (smaller gaps, harder scenarios).

Also update `scenario_sampler.trigger_time_s_variations` to match:
```yaml
scenario_sampler:
  trigger_time_s_variations: [7.0, 8.0]   # can list multiple values
```

---

## 6. Use a different road network

Two roads are available out of the box:

```yaml
road_network: "straight"       # flat 500 m straight road
road_network: "curved_20deg"   # 500 m with 20° right curve (current default)
road_curvature_rad_m: -6.9813170079773183e-04   # only used for curved road
```

For a straight road, also set `road_curvature_rad_m: 0.0`.

---

## 7. Tune the reward

All reward weights are in the config — no code changes needed:

```yaml
reward:
  collision_penalty: 100.0    # negative reward for any collision
  success_bonus: 10.0         # positive reward for reaching full stop (MRC)
  shoulder_bonus: 5.0         # extra bonus for shoulder stop when shoulder was available
  lane_block_penalty: 2.0     # small penalty for STRAIGHT_STOP when shoulder was free
  risk_step_penalty_weight: 0.5  # weight on per-step risk penalty (0 = disable)
  step_cost: 0.01             # fixed cost per step (encourages faster stopping)
```

**To remove the risk signal from reward** (ablation experiment):
```yaml
ablations:
  use_risk_penalty_in_reward: false
```

**To remove risk from the observation entirely** (another ablation):
```yaml
ablations:
  use_risk_features: false
```

**To disable the safety shield** (test how unsafe the agent becomes):
```yaml
ablations:
  use_shield: false
```

---

## 8. Full training command

After editing the config:

```bash
py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
```

To watch episodes live (much slower, useful for debugging):
```yaml
# Set in config:
sim:
  headless: false
  esmini_lib_show_ui: true
```

Checkpoints are saved periodically if you set:
```yaml
training:
  checkpoint_every_steps: 1000
```

Stop any time with **Ctrl+C** — the current policy is saved automatically as `policy_interrupted.zip`.

---

## 9. Evaluate and compare after training

```bash
# Update configs/phase3_eval.yaml to point to the new policy, then:
py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml

# Compare RL vs rule-based baseline (physics-based filter for inevitable collisions):
py -3 scripts/compare_rl_vs_baseline.py
```

Results land in `logs/runs/<timestamp>_phase3_eval/eval_summary.json`.

---

## File reference: what to edit for common goals

| Goal | File to edit | Key field |
|---|---|---|
| Change speed | `configs/phase3_train.yaml` | `state_bridge.initial_speed_mps`, `sim.decel_mps2` |
| Change gap difficulty | `configs/phase3_train.yaml` | `short_front_gap_coverage.gap_at_trigger_*` |
| Change easy/hard ratio | `configs/phase3_train.yaml` | `phase3.episode_variation.mix_baseline_probability` |
| Add/remove obstacles | `configs/phase3_train.yaml` | `shoulder_obstacles.*` |
| Change reward weights | `configs/phase3_train.yaml` | `reward.*` |
| Run ablation (no risk) | `configs/phase3_train.yaml` | `ablations.use_risk_features: false` |
| Longer training | `configs/phase3_train.yaml` | `training.total_timesteps` |
| Different road | `configs/phase3_train.yaml` | `road_network` |
| Point eval to new model | `configs/phase3_eval.yaml` | `evaluation.policy_path` |
