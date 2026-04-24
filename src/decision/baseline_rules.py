"""
Baseline MRM selection: rule-based choice from observation and risk.

When to issue each MRM:
- STRAIGHT_STOP (safest): Emergency — risk high (e.g. >= 0.7). Brake only, no steer.
- IN_LANE_STOP (ideal for "stop in lane"): Risk moderate; pull-over not chosen (gap/risk).
  Brake while staying in lane (on curved road, follows lane). Issued when risk < risk_brake_threshold
  and either right gap is small or risk is above risk_pull_over_max.
- ROAD_SHOULDER_STOP: Risk low and right gap large — pull over to clear the lane.
"""

from typing import Any, Dict

from src.decision.mrm_catalog import MRM


def choose_mrm_baseline(
    obs: Dict[str, Any],
    risk_total: float,
    cfg: dict,
) -> int:
    """
    Choose MRM by baseline rules.

    - risk_total >= risk_brake_threshold → STRAIGHT_STOP (emergency, safest).
    - risk_total < risk_pull_over_max and right_gap_m >= min_right_gap → ROAD_SHOULDER_STOP.
    - Else → IN_LANE_STOP (ideal stop-in-lane: brake in lane, follows curve if curved).
    """
    baseline = cfg.get("baseline", {})
    risk_brake = float(baseline.get("risk_brake_threshold", 0.7))
    min_right_gap = float(baseline.get("min_right_gap_for_pull_over", 15.0))
    risk_pull_max = float(baseline.get("risk_pull_over_max", 0.5))

    # Emergency: high risk → safest (straight stop, no steering)
    if risk_total >= risk_brake:
        return MRM.STRAIGHT_STOP

    # Low risk and enough space to the right → pull over
    right_gap_m = obs.get("right_gap_m", 0.0)
    if right_gap_m >= min_right_gap and risk_total < risk_pull_max:
        return MRM.ROAD_SHOULDER_STOP

    # Else: ideal for stop in lane — moderate risk, or no room to pull over. Brake in lane (follows curve).
    return MRM.IN_LANE_STOP
