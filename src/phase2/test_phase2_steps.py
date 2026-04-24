"""
Tests for Phase 2 Steps 2.3 (risk), 2.4 (baseline + shield), 2.5 (metrics).

Run: python -m src.phase2.test_phase2_steps
"""

import math
from src.types import EgoState, ActorState
from src.risk.ttc import estimate_ttc
from src.risk.drf import compute_drf, compute_drf_from_scene, compute_drf_total_ego_centric
from src.risk.dara import compute_dara
from src.risk.risk_combiner import compute_risk
from src.decision.mrm_catalog import MRM, mrm_name, MRM_ORDER_SAFE_TO_RISKY
from src.decision.baseline_rules import choose_mrm_baseline
from src.decision.shield import shield_action
from src.decision.executor import get_control_for_mrm
from src.metrics.mrc_checker import MrcChecker
from src.metrics.episode_tracker import EpisodeTracker


def test_ttc():
    """TTC: infinity when closing_speed <= 0, correct value otherwise."""
    assert estimate_ttc(10.0, 0.0) == math.inf
    assert estimate_ttc(10.0, -1.0) == math.inf
    assert estimate_ttc(0.0, 5.0) == math.inf
    assert estimate_ttc(20.0, 10.0) == 2.0
    print("  [OK] estimate_ttc()")


def test_drf():
    """DRF: 0 when speed 0, increases with speed (fallback when no actors)."""
    assert compute_drf(0.0) == 0.0
    assert compute_drf(15.0, max_speed_mps=30.0) == 0.25
    assert compute_drf(30.0, max_speed_mps=30.0) == 1.0
    assert compute_drf(40.0, max_speed_mps=30.0) == 1.0
    # Supervisor-aligned: total ego-centric risk from scene (stationary lead = included)
    ego = EgoState(x=0.0, y=0.0, speed_mps=10.0, yaw_rad=0.0, lane_id=0)
    other = ActorState(actor_id=1, x=30.0, y=0.0, speed_mps=0.0, yaw_rad=0.0, lane_id=0)
    raw = compute_drf_total_ego_centric(0, 0, 10.0, 0.0, [(30.0, 0.0, 0.0, 0.0)])
    assert raw > 0.0  # ego approaching stationary vehicle must yield positive DRF
    drf_norm = compute_drf_from_scene(ego, [other])
    assert 0.0 < drf_norm <= 1.0
    drf_empty = compute_drf_from_scene(ego, [])
    assert drf_empty == 0.0
    print("  [OK] compute_drf() and supervisor-aligned compute_drf_from_scene()")


def test_dara():
    """DARA: 1.0 when TTC very low, 0.0 when TTC high."""
    assert compute_dara(1.0, ttc_critical_s=2.0, ttc_safe_s=5.0) == 1.0
    assert compute_dara(2.0, ttc_critical_s=2.0, ttc_safe_s=5.0) == 1.0
    assert compute_dara(5.0, ttc_critical_s=2.0, ttc_safe_s=5.0) == 0.0
    assert compute_dara(math.inf) == 0.0
    assert 0.0 < compute_dara(3.5, ttc_critical_s=2.0, ttc_safe_s=5.0) < 1.0
    print("  [OK] compute_dara()")


def test_risk_combiner():
    """Risk total in [0,1], increases when danger increases."""
    r1 = compute_risk(0.0, 0.0)
    r2 = compute_risk(1.0, 0.0, weight_drf=1.0, weight_dara=0.0)
    r3 = compute_risk(0.0, 1.0, weight_drf=0.0, weight_dara=1.0)
    r4 = compute_risk(0.5, 0.5)
    assert 0.0 <= r1 <= 1.0 and r1 == 0.0
    assert 0.0 <= r2 <= 1.0 and r2 == 1.0
    assert 0.0 <= r3 <= 1.0 and r3 == 1.0
    assert 0.0 <= r4 <= 1.0 and r4 == 0.5
    assert compute_risk(0.8, 0.2) > compute_risk(0.2, 0.2)
    print("  [OK] compute_risk()")


def test_mrm_catalog_and_name():
    """MRM IDs and mrm_name()."""
    assert MRM.STRAIGHT_STOP == 0 and MRM.IN_LANE_STOP == 1 and MRM.ROAD_SHOULDER_STOP == 2
    assert mrm_name(MRM.STRAIGHT_STOP) == "STRAIGHT_STOP"
    assert mrm_name(MRM.ROAD_SHOULDER_STOP) == "ROAD_SHOULDER_STOP"
    assert len(MRM_ORDER_SAFE_TO_RISKY) == 3
    print("  [OK] MRM catalog and mrm_name()")


def test_baseline_rules():
    """Baseline: STRAIGHT_STOP when risk high; ROAD_SHOULDER when gap large and risk low; else IN_LANE_STOP."""
    cfg = {"baseline": {"risk_brake_threshold": 0.7, "min_right_gap_for_pull_over": 15.0, "risk_pull_over_max": 0.5}}
    obs_high_risk = {"right_gap_m": 20.0}
    obs_low_risk_large_gap = {"right_gap_m": 20.0}
    obs_low_risk_small_gap = {"right_gap_m": 5.0}
    assert choose_mrm_baseline(obs_high_risk, 0.8, cfg) == MRM.STRAIGHT_STOP
    assert choose_mrm_baseline(obs_low_risk_large_gap, 0.3, cfg) == MRM.ROAD_SHOULDER_STOP
    assert choose_mrm_baseline(obs_low_risk_small_gap, 0.3, cfg) == MRM.IN_LANE_STOP
    print("  [OK] choose_mrm_baseline()")


def test_shield():
    """Shield: veto pull-over when gaps small; downgrade-only (no riskier than current)."""
    cfg = {"shield": {"min_right_gap_m": 12.0, "min_rear_gap_m": 8.0}}
    obs_small_gap = {"right_gap_m": 5.0, "rear_gap_m": 20.0}
    obs_large_gap = {"right_gap_m": 15.0, "rear_gap_m": 10.0}
    # Veto: proposed pull-over but right gap too small -> IN_LANE_STOP
    assert shield_action(MRM.ROAD_SHOULDER_STOP, obs_small_gap, -1, cfg) == MRM.IN_LANE_STOP
    # Allow pull-over when gaps OK
    assert shield_action(MRM.ROAD_SHOULDER_STOP, obs_large_gap, -1, cfg) == MRM.ROAD_SHOULDER_STOP
    # Downgrade-only: current IN_LANE_STOP, proposed PULL_OVER -> keep IN_LANE_STOP (safer)
    assert shield_action(MRM.ROAD_SHOULDER_STOP, obs_large_gap, MRM.IN_LANE_STOP, cfg) == MRM.IN_LANE_STOP
    print("  [OK] shield_action()")


def test_executor_three_mrms():
    """Executor: all 3 MRM types produce correct control commands."""
    cfg = {
        "mrm": {
            "brake": {"throttle_value": 0.0, "brake_value": 1.0, "steer_value": 0.0},
            "pull_over_right": {"throttle_value": 0.0, "brake_value": 0.7, "steer_value": -0.3},
        }
    }
    c0 = get_control_for_mrm(MRM.BRAKE_IN_LANE, cfg)
    assert c0.override and c0.brake == 1.0 and c0.steer == 0.0
    c1 = get_control_for_mrm(MRM.IN_LANE_STOP, cfg)
    assert c1.brake == 1.0 and c1.steer == 0.0
    c2 = get_control_for_mrm(MRM.ROAD_SHOULDER_STOP, cfg)
    assert c2.brake == 0.7 and c2.steer == -0.3
    print("  [OK] get_control_for_mrm() for all 3 MRM types")


def test_mrc_checker():
    """MrcChecker: detects stopped for required time."""
    checker = MrcChecker(stop_speed_mps=0.5, stopped_for_s=2.0)
    assert not checker.is_mrc_reached()
    checker.update(0.0, 1.0)
    assert not checker.is_mrc_reached()
    checker.update(0.0, 1.0)
    assert checker.is_mrc_reached()
    checker.reset()
    assert not checker.is_mrc_reached()
    checker.update(10.0, 1.0)
    checker.update(0.0, 1.0)
    assert not checker.is_mrc_reached()
    print("  [OK] MrcChecker")


def test_episode_tracker():
    """EpisodeTracker: max_risk, avg_risk, mrm_switches."""
    t = EpisodeTracker()
    t.update(0.2, -1)
    t.update(0.8, MRM.BRAKE_IN_LANE)
    t.update(0.5, MRM.BRAKE_IN_LANE)
    t.update(0.3, MRM.ROAD_SHOULDER_STOP)
    s = t.get_summary()
    assert s["max_risk"] == 0.8
    assert s["avg_risk"] == (0.2 + 0.8 + 0.5 + 0.3) / 4
    assert s["mrm_switches"] == 1
    assert s["step_count"] == 4
    t.reset()
    assert t.get_summary()["step_count"] == 0
    print("  [OK] EpisodeTracker")


def main():
    print("Phase 2 Steps 2.3, 2.4, 2.5 tests")
    test_ttc()
    test_drf()
    test_dara()
    test_risk_combiner()
    test_mrm_catalog_and_name()
    test_baseline_rules()
    test_shield()
    test_executor_three_mrms()
    test_mrc_checker()
    test_episode_tracker()
    print("All Phase 2 (2.3, 2.4, 2.5) tests passed.")


if __name__ == "__main__":
    main()
