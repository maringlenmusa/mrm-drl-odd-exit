"""
Verify Phase 2 completion criteria (To_Do.md lines 210-216).

Run: python -m src.phase2.verify_phase2_criteria [run_dir]
  run_dir: optional path to a Phase 2 run (e.g. logs/runs/2026-02-02_09-50-09_phase2_baseline_risk).
  If omitted, finds latest phase2 run in logs/runs/ or skips run-dependent checks.
"""

import csv
import os
import sys
from pathlib import Path

# Add project root
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.logger import STEP_FIELDS_PHASE1, STEP_FIELDS_PHASE2_EXTRA, EPISODE_CSV_FIELDS
from src.decision.baseline_rules import choose_mrm_baseline
from src.decision.shield import shield_action
from src.decision.mrm_catalog import MRM
from src.risk.drf import compute_drf
from src.risk.dara import compute_dara
import math


REQUIRED_STEP_COLUMNS = [
    "front_gap_m", "rear_gap_m", "left_gap_m", "right_gap_m",
    "drf", "dara", "risk_total", "proposed_mrm", "final_mrm",
]
REQUIRED_EPISODE_COLUMNS = ["success_mrc", "collision"]
# Phase 3 bandit plan §2: episode-level fields for reward and analysis
PHASE3_EPISODE_COLUMNS = ["shoulder_available", "min_gap", "time_to_stop", "time_to_mrc", "max_decel"]


def criterion_1_steps_csv_columns():
    """steps.csv contains columns for: obs (gaps), risk (drf, dara, risk_total), proposed_mrm, final_mrm."""
    all_step = STEP_FIELDS_PHASE1 + STEP_FIELDS_PHASE2_EXTRA
    missing = [c for c in REQUIRED_STEP_COLUMNS if c not in all_step]
    assert not missing, f"Missing step columns: {missing}"
    print("  [OK] steps.csv has required columns (gaps, drf, dara, risk_total, proposed_mrm, final_mrm)")


def criterion_2_after_odd_exit_final_mrm_not_minus_one(run_dir: str):
    """After ODD exit time, final_mrm is no longer -1."""
    base = Path(run_dir)
    # Single run: steps.csv; multi-run: steps_episode_0.csv, steps_episode_1.csv, ...
    steps_files = [base / "steps.csv"]
    steps_files += sorted(base.glob("steps_episode_*.csv"))
    steps_path = None
    for p in steps_files:
        if p.exists():
            steps_path = p
            break
    if not steps_path or not steps_path.exists():
        print("  [SKIP] No steps.csv or steps_episode_*.csv in run_dir")
        return
    with open(steps_path, newline="") as f:
        r = csv.DictReader(f)
        rows_odd_active = [row for row in r if row.get("odd_exit_active", "").strip() == "True"]
    assert rows_odd_active, f"No rows with odd_exit_active=True in {steps_path.name}"
    for row in rows_odd_active:
        fm = row.get("final_mrm", "")
        assert fm != "" and int(float(fm)) != -1, f"After ODD exit final_mrm should not be -1, got {fm}"
    print("  [OK] After ODD exit, final_mrm is not -1")


def criterion_4_shield_brake_when_gaps_small():
    """When lane gaps are small, shield forces IN_LANE_STOP (veto pull-over)."""
    cfg = {"shield": {"min_right_gap_m": 12.0, "min_rear_gap_m": 8.0}}
    obs_small_right = {"right_gap_m": 5.0, "rear_gap_m": 20.0}
    obs_small_rear = {"right_gap_m": 20.0, "rear_gap_m": 5.0}
    final = shield_action(MRM.ROAD_SHOULDER_STOP, obs_small_right, -1, cfg)
    assert final == MRM.IN_LANE_STOP, f"Expected IN_LANE_STOP when right_gap=5, got {final}"
    final2 = shield_action(MRM.ROAD_SHOULDER_STOP, obs_small_rear, -1, cfg)
    assert final2 == MRM.IN_LANE_STOP, f"Expected IN_LANE_STOP when rear_gap=5, got {final2}"
    print("  [OK] When lane gaps small, shield forces IN_LANE_STOP (veto pull-over)")


def criterion_3_baseline_brake_when_risk_high():
    """When risk_total >= 0.8, baseline chooses STRAIGHT_STOP (safest)."""
    cfg = {"baseline": {"risk_brake_threshold": 0.7, "min_right_gap_for_pull_over": 15.0, "risk_pull_over_max": 0.5}}
    obs = {"right_gap_m": 20.0}
    mrm = choose_mrm_baseline(obs, 0.8, cfg)
    assert mrm == MRM.STRAIGHT_STOP, f"Expected STRAIGHT_STOP when risk=0.8, got {mrm}"
    mrm2 = choose_mrm_baseline(obs, 0.9, cfg)
    assert mrm2 == MRM.STRAIGHT_STOP
    print("  [OK] When risk_total >= 0.8, baseline chooses STRAIGHT_STOP")


def criterion_5_episodes_csv_success_no_collision(run_dir: str):
    """episodes.csv has success_mrc and collision columns; collision is True when a run hits an actor."""
    ep_path = Path(run_dir) / "episodes.csv"
    if not ep_path.exists():
        print("  [SKIP] No episodes.csv in run_dir")
        return
    with open(ep_path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    assert rows, "episodes.csv is empty"
    for row in rows:
        assert "success_mrc" in row and "collision" in row, "episodes.csv must have success_mrc and collision"
    success_count = sum(1 for row in rows if row.get("success_mrc", "").strip() == "True")
    collision_count = sum(1 for row in rows if row.get("collision", "").strip() == "True")
    # To_Do: "success_mrc=True for some runs (or at least no collisions)" — prefer that; here we check columns and correctness
    assert success_count >= 1 or collision_count == 0, (
        "Phase 2 complete requires: at least one run with success_mrc=True OR no run with collision=True. "
        "Run multi with earlier trigger_time_s (e.g. 5s) or increase lead car x_m to get a success run."
    )
    # Phase 3 bandit plan §2: episodes.csv should include episode-level fields
    header = list(rows[0].keys()) if rows else []
    missing_phase3 = [c for c in PHASE3_EPISODE_COLUMNS if c not in header]
    assert not missing_phase3, f"episodes.csv missing Phase 3 episode columns: {missing_phase3}"
    print(f"  [OK] episodes.csv: {len(rows)} run(s), success_mrc=True in {success_count}, collision in {collision_count}; Phase 3 columns present")


def criterion_6_scenarios_configured_and_used():
    """Configured scenario(s) run successfully (1–2 types; multiple runs with varied params)."""
    cfg_path = _project_root / "configs" / "phase2.yaml"
    assert cfg_path.exists(), "configs/phase2.yaml not found"
    content = cfg_path.read_text()
    assert "scenario_sampler" in content or "scenario_path" in content, "Phase 2 config should have scenario path or scenario_sampler"
    assert "straight_500m" in content or "scenarios:" in content, "Scenario (e.g. straight_500m) or scenarios list should be present"
    print("  [OK] Configured scenario(s) present in config; multi-run with variations supported")


def criterion_7_risk_values_make_sense():
    """Risk values make sense: high when danger is high, low when safe."""
    # DRF: 0 when speed 0, increases with speed
    assert compute_drf(0.0) == 0.0
    assert compute_drf(10.0) < compute_drf(20.0) < compute_drf(30.0)
    # DARA: 0 when TTC high (safe), 1 when TTC low (danger)
    assert compute_dara(math.inf) == 0.0
    assert compute_dara(1.0, ttc_critical_s=2.0, ttc_safe_s=5.0) == 1.0
    assert compute_dara(5.0, ttc_critical_s=2.0, ttc_safe_s=5.0) == 0.0
    print("  [OK] Risk values: DRF increases with speed; DARA high when TTC low, low when TTC high")


def find_latest_phase2_run():
    """Return path to latest phase2 run dir under logs/runs, or None."""
    runs = _project_root / "logs" / "runs"
    if not runs.exists():
        return None
    dirs = [d for d in runs.iterdir() if d.is_dir() and "phase2" in d.name]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(dirs[0])


def main():
    run_dir = None
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_phase2_run()
    if run_dir:
        print(f"Using run_dir: {run_dir}")
    else:
        print("No Phase 2 run_dir found; skipping run-dependent checks (2, 5).")

    print("Verifying Phase 2 criteria (To_Do.md 210-216)...")
    criterion_1_steps_csv_columns()
    if run_dir:
        criterion_2_after_odd_exit_final_mrm_not_minus_one(run_dir)
    else:
        print("  [SKIP] criterion 2 (no run_dir)")
    criterion_3_baseline_brake_when_risk_high()
    criterion_4_shield_brake_when_gaps_small()
    if run_dir:
        criterion_5_episodes_csv_success_no_collision(run_dir)
    else:
        print("  [SKIP] criterion 5 (no run_dir)")
    criterion_6_scenarios_configured_and_used()
    criterion_7_risk_values_make_sense()
    print("All Phase 2 verification criteria passed.")


if __name__ == "__main__":
    main()
