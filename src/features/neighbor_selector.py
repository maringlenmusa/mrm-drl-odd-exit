"""
Neighbor selector: find front, rear, left, right neighbors around ego.

Uses ego heading (yaw_rad) so directions are straight-line relative to the vehicle:
- Front = ahead of ego (in direction of heading)
- Rear = behind ego
- Left = to the left of ego (perpendicular)
- Right = to the right of ego
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from src.types import EgoState, ActorState


@dataclass
class Neighbors:
    """Nearest actors in each direction. None if no actor in that direction."""
    front: Optional[ActorState] = None
    rear: Optional[ActorState] = None
    left: Optional[ActorState] = None
    right: Optional[ActorState] = None


def _to_ego_frame(dx: float, dy: float, yaw_rad: float) -> tuple:
    """Project (dx, dy) world offset into ego frame: (longitudinal, lateral). Ahead = +long, left = +lat."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    longitudinal = dx * c + dy * s
    lateral = -dx * s + dy * c
    return (longitudinal, lateral)


def select_neighbors(
    ego: EgoState,
    actors: List[ActorState],
    *,
    lane_half_width_m: float = 1.75,
    max_lateral_m: float = 10.0,
) -> Neighbors:
    """
    Select the nearest neighbor in each direction using ego's heading (straight-line relative to vehicle).

    - Front: actor ahead (longitudinal > 0), nearest.
    - Rear: actor behind (longitudinal < 0), nearest.
    - Left: actor to the left (lateral > 0), within max_lateral_m, nearest.
    - Right: actor to the right (lateral < 0), within max_lateral_m, nearest.
    """
    out = Neighbors()
    if not actors:
        return out

    h = getattr(ego, "yaw_rad", 0.0)

    # Front: longitudinal > 0, smallest longitudinal
    front_candidates = []
    for a in actors:
        longi, _ = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        if longi > 0:
            front_candidates.append((a, longi))
    if front_candidates:
        out.front = min(front_candidates, key=lambda t: t[1])[0]

    # Rear: longitudinal < 0, smallest |longitudinal|
    rear_candidates = []
    for a in actors:
        longi, _ = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        if longi < 0:
            rear_candidates.append((a, -longi))
    if rear_candidates:
        out.rear = min(rear_candidates, key=lambda t: t[1])[0]

    # Left/right: only actors that are NOT front or rear (so same-lane ahead/behind don't count as "side").
    # On a curved road the car ahead can be slightly right in ego frame; we want it as front only, right_gap = 999.
    exclude = {id(out.front)} if out.front else set()
    if out.rear:
        exclude.add(id(out.rear))

    # Left: lateral > 0, within max_lateral_m, not front/rear, smallest lateral
    left_candidates = []
    for a in actors:
        if id(a) in exclude:
            continue
        _, lat = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        if 0 < lat <= max_lateral_m:
            left_candidates.append((a, lat))
    if left_candidates:
        out.left = min(left_candidates, key=lambda t: t[1])[0]

    # Right: lateral < 0, within max_lateral_m, not front/rear, smallest |lateral|
    right_candidates = []
    for a in actors:
        if id(a) in exclude:
            continue
        _, lat = _to_ego_frame(a.x - ego.x, a.y - ego.y, h)
        if -max_lateral_m <= lat < 0:
            right_candidates.append((a, -lat))
    if right_candidates:
        out.right = min(right_candidates, key=lambda t: t[1])[0]

    return out
