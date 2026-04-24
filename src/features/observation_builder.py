"""
Observation builder: build a flat observation dict from SimState.

Gaps are in ego frame (using ego heading): front/rear = longitudinal distance,
left/right = lateral distance.
"""

from typing import Dict, Any

from src.types import SimState
from src.features.neighbor_selector import select_neighbors, _to_ego_frame


# Sentinel for "no neighbor" gaps (use large value so policies see "safe")
NO_GAP_M = 999.0


def build_observation(
    state: SimState,
    *,
    lane_half_width_m: float = 1.75,
    max_lateral_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Build observation dict from SimState. Front/rear/left/right gaps are
    straight-line relative to the vehicle (using ego heading).
    """
    ego = state.ego
    neighbors = select_neighbors(
        ego,
        state.actors,
        lane_half_width_m=lane_half_width_m,
        max_lateral_m=max_lateral_m,
    )

    h = getattr(ego, "yaw_rad", 0.0)

    def gap_front(a):
        longi, _ = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        return longi

    def gap_rear(a):
        longi, _ = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        return -longi

    def gap_left(a):
        _, lat = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        return lat

    def gap_right(a):
        _, lat = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        return -lat

    # Gaps in ego frame: front/rear = longitudinal, left/right = lateral
    front_gap_m = gap_front(neighbors.front) if neighbors.front else NO_GAP_M
    rear_gap_m = gap_rear(neighbors.rear) if neighbors.rear else NO_GAP_M
    left_gap_m = gap_left(neighbors.left) if neighbors.left else NO_GAP_M
    right_gap_m = gap_right(neighbors.right) if neighbors.right else NO_GAP_M

    # Clamp to non-negative
    front_gap_m = max(0.0, front_gap_m)
    rear_gap_m = max(0.0, rear_gap_m)
    left_gap_m = max(0.0, left_gap_m)
    right_gap_m = max(0.0, right_gap_m)

    # Relative speeds (m/s): positive = closing. Front: ego faster => closing; rear: rear faster => closing
    front_rel_speed_mps = (ego.speed_mps - neighbors.front.speed_mps) if neighbors.front else 0.0
    rear_rel_speed_mps = (neighbors.rear.speed_mps - ego.speed_mps) if neighbors.rear else 0.0
    left_rel_speed_mps = 0.0   # lateral; optional to add longitudinal component later
    right_rel_speed_mps = 0.0

    return {
        "ego_speed_mps": ego.speed_mps,
        "ego_lane_id": ego.lane_id,
        "front_gap_m": front_gap_m,
        "rear_gap_m": rear_gap_m,
        "left_gap_m": left_gap_m,
        "right_gap_m": right_gap_m,
        "front_rel_speed_mps": front_rel_speed_mps,
        "rear_rel_speed_mps": rear_rel_speed_mps,
        "left_rel_speed_mps": left_rel_speed_mps,
        "right_rel_speed_mps": right_rel_speed_mps,
    }
