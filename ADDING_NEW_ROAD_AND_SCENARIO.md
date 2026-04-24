# Adding an Entirely New Road and Scenario

This guide explains how to plug in a new road network and driving scenario so the RL agent can be trained in a completely different environment.

There are **two paths** depending on how much control you need:

| Path | Effort | Best for |
|---|---|---|
| **A – Custom .xosc file** | Low | You already have an esmini scenario |
| **B – New road network + parametric scenario** | Medium | You want to keep the parametric speed/gap variation on a new road |

---

## Path A: Plug in a custom `.xosc` file directly

This is the fastest path. You write (or have) a custom OpenSCENARIO file and point the config at it.

### Requirements for your `.xosc`

The ego vehicle **must** be controlled via `UDPDriverController` (so the Python code can send throttle/brake/steer):

```xml
<ObjectController>
   <CatalogReference catalogName="ControllerCatalog" entryName="UDPDriverController"/>
</ObjectController>
```

The ControllerCatalog must be at `../xosc/Catalogs/Controllers/` relative to the xosc file (this is the standard esmini path — already present in `esmini/bin/resources/xosc/Catalogs/`).

### Step 1 — Put your scenario file in the esmini resources folder

```
esmini/bin/resources/xosc/my_new_scenario.xosc
```

### Step 2 — Update `configs/phase3_train.yaml`

```yaml
sim:
  scenario_path: "esmini/bin/resources/xosc/my_new_scenario.xosc"

  # Disable the auto-generated scenario — use your file directly
  use_generated_scenario_from_state_bridge: false
```

Also set `sim.state_source: "esmini"` (already the default).

### Step 3 — Run training

```bash
py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
```

### Limitations of Path A

- The `state_bridge` speed/gap variation **will not apply** — your xosc controls the scenario exactly as written.
- Each episode uses the same scenario. The RL agent will not see varied speeds/gaps.
- If you want variation, use Path B.

---

## Path B: New road + parametric scenario generation

This keeps the per-episode speed/gap variation (different speeds and gaps each episode) but runs it on your new road geometry.

The parametric scenario is auto-generated from the `state_bridge` config every episode. You need to provide:
1. A `.xodr` road file (OpenDRIVE)
2. Register it as a named road in `src/phase2/scenario_from_config.py`

### Step 1 — Add your road file

Put your `.xodr` in:
```
esmini/bin/resources/xodr/my_road.xodr
```

Optionally, a 3D model:
```
esmini/bin/resources/models/my_road.osgb  (optional, for visual rendering)
```

### Step 2 — Register the road in `scenario_from_config.py`

Open `src/phase2/scenario_from_config.py` and add an entry to the `ROAD_NETWORKS` dictionary near the top of the file:

```python
ROAD_NETWORKS = {
    "straight": {
        "logic_file": "../xodr/straight_500m.xodr",
        "scene_graph": "../models/straight_500m.osgb",
    },
    "curved_20deg": {
        "logic_file": "../xodr/curve_20deg_500m.xodr",
        "scene_graph": "../models/straight_500m.osgb",
    },

    # --- Add your new road here ---
    "my_road": {
        "logic_file": "../xodr/my_road.xodr",
        "scene_graph": "../models/my_road.osgb",  # or reuse straight_500m.osgb if no model
    },
}
```

### Step 3 — Update `configs/phase3_train.yaml`

```yaml
road_network: "my_road"     # must match the key you added above
road_curvature_rad_m: 0.0   # set to actual curvature if your road curves

sim:
  use_generated_scenario_from_state_bridge: true   # keep this on for parametric variation
```

### Step 4 — Adjust the curvature value

If your road is straight: `road_curvature_rad_m: 0.0`

If your road curves, the curvature in rad/m = turn_angle_radians / road_length_m.  
Example: 20° turn over 500 m = `(20 × π/180) / 500 = 0.000698 rad/m`  
For a right turn, use a negative value: `road_curvature_rad_m: -0.000698`

### Step 5 — Check road length and adjust positions

The auto-generated scenario places actors along the road using `s` coordinates (distance along road).  
In `state_bridge`:
```yaml
state_bridge:
  initial_x_m: 50.0     # ego starts 50 m along the road
  actors:
    - x_m: 100.0         # front car at 100 m along the road
```

Make sure these values fit within your road length. The default scenario stops at 30 s, well within 500 m. If your road is shorter, you may need to reduce `initial_x_m` and the front car position.

---

## What happens under the hood

When `use_generated_scenario_from_state_bridge: true`, the code in `src/phase2/scenario_from_config.py` builds a fresh `.xosc` for every episode with:
- The ego vehicle at the configured initial position and speed
- All actors (front car, rear car) at their parametrically-sampled positions
- The UDPDriverController so Python controls the ego
- A 30-second stop trigger

The generated file is written to:
```
esmini/bin/resources/xosc/generated_phase2.xosc
```
(overwritten each episode — do not use this file as your custom scenario)

---

## Checklist for a new road / scenario

- [ ] `.xodr` road file placed in `esmini/bin/resources/xodr/`
- [ ] Road registered in `ROAD_NETWORKS` dict in `src/phase2/scenario_from_config.py`
- [ ] `road_network:` key in `configs/phase3_train.yaml` matches the dict key
- [ ] `road_curvature_rad_m:` updated to match actual road geometry
- [ ] `state_bridge.initial_x_m` and actor `x_m` values fit within road length
- [ ] `use_generated_scenario_from_state_bridge: true` (for parametric) or `false` (for fixed xosc)
- [ ] Run a quick visual test first:
  ```bash
  # Set total_timesteps: 5, headless: false, esmini_lib_show_ui: true
  py -3 -u -m src.main --phase phase3_train --config configs/phase3_train.yaml
  ```
  Check that the scene looks correct before running full training.

---

## FAQ

**Q: My road has multiple lanes — which lane does the ego use?**  
A: The generated scenario places ego in `laneId = -1` (rightmost driving lane in esmini convention). If your road has different lane numbering, you may need to adjust the `TeleportAction` in the generated scenario (see `generate_xosc_from_state_bridge()` in `src/phase2/scenario_from_config.py`).

**Q: Do I need to retrain from scratch for a new road?**  
A: Yes, if the road geometry is significantly different (different curves, different gaps). The observation features are road-agnostic (gaps, speeds, risk) so in principle a policy trained on one road can transfer, but results will be better with road-specific training.

**Q: Can I add more than two actors (front + rear)?**  
A: Yes. Add entries to `state_bridge.actors` in the config. Each actor gets its own `SpeedAction` and `TeleportAction` in the generated scenario. Actors with `rear_follower: true` also get a delayed brake action.

**Q: My xodr uses a different coordinate system — road starts at a different position.**  
A: Adjust `state_bridge.initial_x_m` to set where along the road (`s` coordinate) the ego starts. The actor `x_m` values are also `s` coordinates, not world X coordinates.
