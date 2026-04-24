"""
MRM (Minimum Risk Maneuver) catalog.

Defines available MRM types. Aligned with ISO 23793 (see docs/supervisor_answers.md).
Order of safety (safest first): Straight stop → In-lane stop → Road shoulder stop.

  Type 1 Straight stop (safest): Stop in a straight line — brake only, no steering.
  Type 2 In-lane stop: Stop while staying in the lane (brake in lane; on curved road, same control as straight for now).
  Type 3 Road shoulder stop: Pull over to road shoulder and stop (brake + steer to shoulder).
"""


class MRM:
    """Minimum Risk Maneuver types (ISO 23793)."""
    STRAIGHT_STOP = 0       # Type 1: stop straight — brake only, no steer (safest)
    IN_LANE_STOP = 1        # Type 2: stop in lane — brake in lane, no steer to shoulder
    ROAD_SHOULDER_STOP = 2  # Type 3: pull over to road shoulder and stop (brake + steer)

    # Aliases for backward compatibility and readability
    BRAKE_IN_LANE = 0       # Type 1/2: brake in lane (no steer)
    PULL_OVER_RIGHT = 2     # Type 3: road shoulder stop


# Ordered by "riskiness" for shield: 0 = safest (straight stop), 2 = pull over (requires space)
MRM_ORDER_SAFE_TO_RISKY = (MRM.STRAIGHT_STOP, MRM.IN_LANE_STOP, MRM.ROAD_SHOULDER_STOP)


def mrm_name(mrm_id: int) -> str:
    """Convert MRM ID to string for logging and config."""
    names = {
        MRM.STRAIGHT_STOP: "STRAIGHT_STOP",
        MRM.IN_LANE_STOP: "IN_LANE_STOP",
        MRM.ROAD_SHOULDER_STOP: "ROAD_SHOULDER_STOP",
    }
    return names.get(mrm_id, f"MRM_{mrm_id}")
