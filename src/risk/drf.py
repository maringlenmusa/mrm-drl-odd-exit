"""
Driving Risk Field (DRF): total ego-centric risk.

Aligned with supervisor's UDP controller script (testUDPDriver-print-osi-info.py)
and paper "Driving Risk Field Model and Its Application in Trajectory Planning".

- When ego + other actors are available: use supervisor's formula (relative motion
  + distance decay) summed over all other vehicles within max_distance_m.
- Fallback when no other actors: speed-based risk in [0, 1] for compatibility.
"""

import math
from typing import List

from src.types import EgoState, ActorState


# Supervisor's constants (same as testUDPDriver-print-osi-info.py)
DEFAULT_K_THETA = 0.5   # relative motion sensitivity
DEFAULT_K_R = 0.05     # distance decay
DEFAULT_MAX_DISTANCE_M = 50.0


def _velocity_xy(speed_mps: float, yaw_rad: float) -> tuple:
    """Return (vx, vy) from speed and yaw."""
    return (speed_mps * math.cos(yaw_rad), speed_mps * math.sin(yaw_rad))


def compute_drf_total_ego_centric(
    ego_x: float,
    ego_y: float,
    ego_vx: float,
    ego_vy: float,
    other_actors: List[tuple],
    *,
    k_theta: float = DEFAULT_K_THETA,
    k_r: float = DEFAULT_K_R,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
) -> float:
    """
    Total ego-centric risk from all other vehicles (supervisor's formula).

    For each other vehicle: contribution = xi_RM * xi_D where
    - xi_RM = exp(k_theta * delta_v), delta_v = ve*cos_theta_e - vo*cos_theta_o
    - xi_D = exp(-k_r * d)
    with cos_theta_e/o = velocity projected onto relative position direction.

    Args:
        ego_x, ego_y: Ego position (m).
        ego_vx, ego_vy: Ego velocity (m/s).
        other_actors: List of (x, y, vx, vy) for each other vehicle.
        k_theta: Relative motion sensitivity (default 0.5).
        k_r: Distance decay (default 0.05).
        max_distance_m: Max distance to consider (default 50 m).

    Returns:
        Raw total risk (sum of contributions; unbounded non-negative).
    """
    ve = math.hypot(ego_vx, ego_vy)
    if ve < 1e-3:
        return 0.0

    total_risk = 0.0
    for (xo, yo, vox, voy) in other_actors:
        vo = math.hypot(vox, voy)
        dx = xo - ego_x
        dy = yo - ego_y
        d = math.hypot(dx, dy)

        if d > max_distance_m or d < 1e-3:
            continue

        cos_theta_e = (ego_vx * dx + ego_vy * dy) / (ve * d)
        cos_theta_e = max(-1.0, min(1.0, cos_theta_e))
        # Stationary or nearly stationary other vehicle: treat cos_theta_o = 0 so
        # delta_v = ve*cos_theta_e (ego approaching = positive risk). Supervisor script
        # skips vo < 1e-3; we include stationary obstacles so DRF > 0 when closing on a stopped vehicle.
        if vo < 1e-3:
            cos_theta_o = 0.0
        else:
            cos_theta_o = (vox * dx + voy * dy) / (vo * d)
            cos_theta_o = max(-1.0, min(1.0, cos_theta_o))

        delta_v = ve * cos_theta_e - vo * cos_theta_o
        xi_RM = math.exp(k_theta * delta_v)
        xi_D = math.exp(-k_r * d)
        total_risk += xi_RM * xi_D

    return total_risk


# Fixed scale to map raw DRF into [0,1] for risk_combiner; not configurable (supervisor formula has no scale).
_DRF_NORMALIZE_SCALE = 350.0


def compute_drf_from_scene(
    ego: EgoState,
    actors: List[ActorState],
    *,
    k_theta: float = DEFAULT_K_THETA,
    k_r: float = DEFAULT_K_R,
    max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
) -> float:
    """
    DRF (total ego-centric risk) from scene state, aligned with supervisor.

    Uses same formula as supervisor's calculate_total_risk(): relative motion
    (xi_RM) and distance decay (xi_D) per other vehicle, then sum. Result is
    scaled by a fixed factor so it fits in [0, 1] for the risk combiner.

    Args:
        ego: EgoState (x, y, speed_mps, yaw_rad).
        actors: Other actors (ActorState with x, y, speed_mps, yaw_rad).
        k_theta, k_r, max_distance_m: Same as supervisor script.

    Returns:
        DRF value in [0, 1].
    """
    vx, vy = _velocity_xy(ego.speed_mps, ego.yaw_rad)
    other_list = []
    for a in actors:
        ax, ay = _velocity_xy(a.speed_mps, a.yaw_rad)
        other_list.append((a.x, a.y, ax, ay))
    raw = compute_drf_total_ego_centric(
        ego.x, ego.y, vx, vy, other_list,
        k_theta=k_theta, k_r=k_r, max_distance_m=max_distance_m,
    )
    return min(1.0, raw / _DRF_NORMALIZE_SCALE) if _DRF_NORMALIZE_SCALE > 0 else raw


def compute_drf(
    speed_mps: float,
    *,
    max_speed_mps: float = 30.0,
) -> float:
    """
    Compute DRF from speed only (fallback when no other actors).

    Simple model: risk = (speed / max_speed)^2 in [0, 1].
    Returns 0.0 when speed is 0. Use compute_drf_from_scene() when
    ego + actors are available to match supervisor's total ego-centric risk.

    Args:
        speed_mps: Ego speed (m/s).
        max_speed_mps: Speed at which risk saturates (default 30 m/s).

    Returns:
        DRF value in [0.0, 1.0].
    """
    if speed_mps <= 0:
        return 0.0
    if max_speed_mps <= 0:
        return 1.0
    ratio = min(1.0, speed_mps / max_speed_mps)
    return ratio * ratio
