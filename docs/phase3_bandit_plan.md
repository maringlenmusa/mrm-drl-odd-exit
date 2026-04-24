# Phase 3 plan (your chosen design)

This document captures your concrete choices for reward, episode data, and scenario coverage so Phase 3 does not fail. It replaces the generic “bandit advice” with your specific design.

---

## 1. Reward design (your choices)

**Safety**

- **Collision**: large negative reward (e.g. -1000).
- **Near-miss**: penalty based on **min gap or min TTC during the stop** (so the agent is penalized for getting too close even without collision).

**Lane blocking**

- When the car does **straight stop** (or in-lane stop), if it **blocks the lane** (e.g. leaves ego in the travel lane), apply a **lane-blocking penalty** (e.g. `-C_lane * time_stopped_in_lane` or equivalent). This makes straight/in-lane stop distinguishable and discourages blocking traffic when avoidable.

**Shoulder**

- **Bonus** for successful shoulder stop when feasible (shoulder was available and no collision).
- **Penalty** when shoulder is chosen but **fails**: e.g. collision, or **shoulder not available**.
- **Shoulder availability rule**: shoulder is **not available** when ego is **20 m or less from the rear car** (configurable value, e.g. `min_rear_gap_for_shoulder_m: 20` in config). So: `shoulder_available = (rear_gap_m > threshold)`. The shoulder is also considered "blocked" in that case (rear too close to safely pull over).

**Concrete implementation**

- In **To_Do** Step 3.2 and in `src/rl/reward.py`: implement the above terms; read `min_rear_gap_for_shoulder_m` from config for shoulder_available.
- Document the full formula in `docs/phase3_reward_design.md` (or phase2_verification).

---

## 2. Data stored per episode (your list)

**Episode-level fields to add**

| Field | Meaning |
|-------|--------|
| **shoulder_available** | True when rear_gap_m > config threshold (e.g. 20 m) at trigger / when deciding. Encodes "was it safe to use shoulder in this situation?" |
| **min_gap** | Minimum gap (e.g. min of front_gap_m) over the episode during the stop — for near-miss penalty. |
| **time_to_stop** | Time from ODD exit (or first brake) until vehicle stopped (e.g. speed below stop threshold). |
| **time_to_mrc** | Time from ODD exit until MRC reached (full stop for required duration). |
| **max_decel** | Maximum deceleration over the episode (for comfort / analysis). |

**Implementation**

- **Config**: add e.g. `min_rear_gap_for_shoulder_m: 20` (or under `baseline` / `shield`). Use it for `shoulder_available` and in reward.
- **Episode tracker / Phase 2 runner**: compute and pass to logger:
  - `shoulder_available` = (rear_gap_m at trigger or at ODD exit > min_rear_gap_for_shoulder_m).
  - `min_gap` = min over steps (e.g. min of front_gap_m after ODD exit).
  - `time_to_stop` = first time speed drops below stop threshold after ODD exit (relative to ODD exit time).
  - `time_to_mrc` = time from ODD exit until MRC checker says reached (or from steps).
  - `max_decel` = max of |deceleration| over steps (need acceleration from state or derive from speed deltas).
- **Logger**: add these to **EPISODE_CSV_FIELDS** in `src/logger.py` and write them in `episodes.csv` (and episode.json if used). Document in phase2_verification or phase3 docs.

---

## 3. Scenario coverage: short front gap (straight stop / stop in lane)

**Goal:** Generate or run many scenarios with **short front gap** so you get plenty of situations where **straight stop** or **stop in lane** is the right choice (shoulder not preferred or not available).

**Deliverable: Python script for scenario coverage**

- **Purpose:** Produce a list of scenario parameters (or run Phase 2) with **short front gap** (e.g. small initial distance to lead vehicle, and/or high speed so gap closes fast), so that you hit a lot of **straight stop** and/or **stop in lane** decisions and balance the distribution (not only "shoulder best" cases).
- **Suggested behaviour:**
  - Script (e.g. `scripts/run_short_front_gap_coverage.py` or under `src/phase2/`) that:
    - Takes config (or CLI) for: front gap range (e.g. 15–40 m), ego/front speeds, trigger times, number of episodes.
    - Builds param list with **short front gap** (e.g. grid or sample over short gaps).
    - Runs Phase 2 (or baseline comparison) for each, writing to a dedicated run dir (e.g. `logs/runs/..._short_front_gap_coverage/`).
    - Optionally writes a small summary CSV: episode_id, front_gap_m, rear_gap_m, shoulder_available, primary_mrm, success_mrc, collision, etc.
  - Can reuse the same pattern as `src/phase2/run_baseline_comparison.py` (build_param_list, run_phase2 per episode) but with parameter ranges focused on **short front gap**.
- **Config:** Use or extend `scenario_sampler` / state_bridge so that "initial front gap" (distance to lead) is set per episode from the script (e.g. 20 m, 25 m, 30 m for coverage).

**To_Do / docs**

- Add a task in To_Do (Phase 3 or Phase 2.8): "Python script for scenario coverage: short front gap to get many straight stop / in-lane stop episodes."
- Mention in `docs/phase2_verification.md` or `docs/phase3_scenarios.md` that the short-front-gap script is used for coverage and stress-testing.

---

## 4. Counterfactuals (run all 3 actions per context)

Left as optional: per episode context, run the sim for all 3 MRMs and store (x_i, r_straight, r_lane, r_shoulder). Not required for your current plan; document in thesis if you use single-action rollouts only.

---

## 5. Suggested order of work

1. **Config**: Add `min_rear_gap_for_shoulder_m: 20` (or similar) and use it for shoulder_available.
2. **Episode logging**: Add shoulder_available, min_gap, time_to_stop, time_to_mrc, max_decel to episode tracker and EPISODE_CSV_FIELDS; implement in Phase 2 runner.
3. **Reward**: Implement in `src/rl/reward.py` (Phase 3) using the formula above and the new episode fields.
4. **Python script**: Implement `scripts/run_short_front_gap_coverage.py` (or equivalent) for short-front-gap scenario coverage.
5. **Document**: phase3_reward_design.md and phase2_verification / phase3_scenarios for the new fields and script.
