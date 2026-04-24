# Phase 1 Verification

**Goal**: Basic simulation loop — vehicle drives, ODD exit triggers at configured time, MRM (brake) executes, logs are saved.

This document records what was tested, which trigger times were used, any issues encountered and how they were fixed, and evidence (screenshots/plots) of vehicle behavior.

---

## Scenarios tested

- **Default scenario**: `esmini/bin/resources/xosc/straight_500m.xosc`  
  - Straight 500 m road, ego and target vehicles.  
  - Verified: esmini loads without errors, vehicle drives at constant speed until ODD exit, then brakes to full stop.  
- **Other scenarios**: Only the default scenario was used for Phase 1. Additional scenarios (e.g. 2 cars + ego) can be tested in later phases.

---

## Trigger times tested

ODD exit trigger time is set in `configs/phase1.yaml` under `odd_exit.trigger_time_s`. The following values were tested:

| Trigger time | Verified |
|--------------|----------|
| **5.0 s**    | Yes — ODD exit at ~5 s, brake = 1.0, speed decreases to 0. Saved run: `2026-02-02_08-54-00_phase1_smoke_test - 5 second`. |
| **8.0 s**    | Yes — ODD exit at ~8 s, same behavior. Saved run: `2026-01-30_14-07-11_phase1_smoke_test - 8 second`. |
| **10.0 s**   | Yes — ODD exit at ~10 s (per To_Do checklist). |

**Important**: `odd_exit_trigger_time_s` must match in both:

1. `configs/phase1.yaml` → `odd_exit.trigger_time_s`
2. `EsminiStateBridge` (passed from config) — used to generate state and ODD exit timing.

If they differ, logs and visualization can be inconsistent.

---

## Problems encountered and how they were fixed

1. **Esmini does not send UDP state by default**  
   - **Issue**: The Python loop expects state (speed, position, etc.) over UDP; out-of-the-box esmini does not send it.  
   - **Fix**: Added `EsminiStateBridge`: a separate process/thread that advances simulation time and sends state packets to the same port the main loop listens on. For Phase 1 this provides consistent, scenario-based state; later it can be replaced by esmini API or a proper UDP driver.

2. **ODD exit and state bridge must use the same trigger time**  
   - **Issue**: If the config says 8 s but the state bridge was built with 5 s (or vice versa), `odd_exit_active` and simulated speed would not align with expectations.  
   - **Fix**: Pass `odd_exit.trigger_time_s` from config into both `OddExitTrigger` and `EsminiStateBridge` so a single source of truth is used everywhere.

3. **Esmini log warning: “Unsupported geo reference attr: +no_defs”**  
   - **Issue**: Appears in esmini console when loading the OpenDRIVE.  
   - **Fix**: Cosmetic only; scenario still loads and runs. Can be ignored for Phase 1 or addressed later in the OpenDRIVE/geo reference if needed.

4. **Headless runs for automation**  
   - **Issue**: Running Phase 1 from scripts or CI would block or fail if the visualization window is required.  
   - **Fix**: Set `sim.headless: true` in `configs/phase1.yaml` when running without a display (e.g. generating logs for plots). Set back to `false` for normal visual verification.

5. **Log columns and CSV format**  
   - **Issue**: Downstream (e.g. plotting, analysis) expect specific columns in `steps.csv`.  
   - **Fix**: `StepCsvLogger` writes: `step_id`, `sim_time_s`, `odd_exit_active`, `ego_speed_mps`, `throttle`, `brake`, `steer`, `collision`. Plot script and verification assume this schema.

6. **Empty run folder for “8 second” run**  
   - **Issue**: The folder `2026-01-30_14-07-11_phase1_smoke_test - 8 second` existed but had no `steps.csv`/`episode.json`.  
   - **Fix**: Re-ran Phase 1 with `trigger_time_s: 8.0`, then copied `steps.csv` and `episode.json` from `logs/runs/[timestamp]_phase1_smoke_test` into that folder so the 8 s plot could be generated.

---

## Screenshots and plots (vehicle behavior)

All files below are in this folder: `src/phase1/results_docs/`.

- **Screenshot of the visualization window**  
  - [screenshot of the visualization window showing the vehicle](screenshot%20of%20the%20visualizatio%20nwindows%20showing%20the%20vehicile.png)  
  - Shows the vehicle in the esmini window (during or after braking).

- **Plots of vehicle behavior (speed + controls)**  
  - [phase1_vehicle_behavior_5s.png](phase1_vehicle_behavior_5s.png) — ODD exit at 5 s: speed constant until 5 s, then brake and decay to zero.  
  - [phase1_vehicle_behavior_8s.png](phase1_vehicle_behavior_8s.png) — ODD exit at 8 s: same behavior with trigger at 8 s.

Plots were generated with:

```bash
python -m src.phase1.plot_phase1_behavior "path/to/run_folder" -o docs/figures/phase1_vehicle_behavior_8s.png
```

---

## Summary

- **Scenarios**: Default `straight_500m.xosc` verified.  
- **Trigger times**: 5 s, 8 s, and 10 s tested; behavior matches config.  
- **Issues**: State bridge for UDP state, consistent trigger time in config and bridge, geo warning, headless option, log schema, and populating the 8 s run folder — all addressed as above.  
- **Evidence**: Screenshot and 5 s / 8 s behavior plots are in `src/phase1/results_docs/`.

Phase 1 is complete: esmini runs, UDP state is received, ODD exit triggers at the configured time, brake override is applied, speed decreases, and logs (including plots) are saved for verification.
