"""
Safety shield: veto or downgrade proposed MRM based on gaps and current MRM.

- Vetoes lane-change (pull over) when lateral gaps are too small.
- Downgrade-only: never allow switching from a safer MRM to a riskier one
  (e.g. from BRAKE_IN_LANE to PULL_OVER_RIGHT).
"""

from typing import Any, Dict

from src.decision.mrm_catalog import MRM, MRM_ORDER_SAFE_TO_RISKY, mrm_name


def _mrm_safety_index(mrm_id: int) -> int:
    """Lower index = safer. BRAKE_IN_LANE=0, ROAD_SHOULDER_STOP=2."""
    for i, m in enumerate(MRM_ORDER_SAFE_TO_RISKY):
        if m == mrm_id:
            return i
    return 999


def shield_action(
    proposed_mrm: int,
    obs: Dict[str, Any],
    current_mrm: int,
    cfg: dict,
) -> int:
    """
    Apply safety shield: possibly downgrade or veto proposed MRM.

    Rules:
    1. Downgrade-only: if proposed is riskier than current, keep current (or safer).
    2. If proposed is ROAD_SHOULDER_STOP (pull over), require min right_gap_m and
       optionally min rear/left gaps; else force BRAKE_IN_LANE.

    Args:
        proposed_mrm: MRM chosen by baseline (or policy).
        obs: Observation dict (right_gap_m, left_gap_m, rear_gap_m, etc.).
        current_mrm: Currently executed MRM (-1 before ODD exit).
        cfg: Config with shield.min_right_gap_m, shield.min_rear_gap_m (optional).

    Returns:
        Final MRM id to execute (may be downgraded from proposed).
    """
    shield_cfg = cfg.get("shield", {})
    min_right_gap = float(shield_cfg.get("min_right_gap_m", 12.0))
    min_rear_gap = float(shield_cfg.get("min_rear_gap_m", 8.0))

    final = proposed_mrm

    # Require gaps for pull-over (ROAD_SHOULDER_STOP)
    if proposed_mrm == MRM.ROAD_SHOULDER_STOP:
        right_gap = obs.get("right_gap_m", 0.0)
        rear_gap = obs.get("rear_gap_m", 0.0)
        if right_gap < min_right_gap or rear_gap < min_rear_gap:
            final = MRM.IN_LANE_STOP  # brake in lane when pull-over not safe

    # Downgrade-only: never go from safer to riskier
    if current_mrm >= 0:
        current_idx = _mrm_safety_index(current_mrm)
        proposed_idx = _mrm_safety_index(final)
        if proposed_idx > current_idx:
            # Proposed is riskier; keep current (safer)
            final = current_mrm

    return final
