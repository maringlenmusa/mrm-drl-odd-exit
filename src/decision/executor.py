"""
MRM executor: Converts MRM decisions into control commands.

Translates high-level MRM decisions into low-level control (throttle, brake, steer).

  STRAIGHT_STOP (safest): Brake only, no steer — vehicle stops in a straight line.
  IN_LANE_STOP: Brake only, no steer — lower speed and keep same direction as when driving (no steering).
  ROAD_SHOULDER_STOP: Brake + steer right — vehicle moves to shoulder and stops.

Config: mrm.brake for STRAIGHT_STOP and IN_LANE_STOP (brake, steer 0); mrm.pull_over_right for Type 3.
"""

from src.types import ControlCommand
from src.decision.mrm_catalog import MRM


def get_control_for_mrm(mrm_id: int, cfg: dict) -> ControlCommand:
    """
    Get control command for a specific MRM.

    Args:
        mrm_id: MRM identifier from MRM catalog (STRAIGHT_STOP, IN_LANE_STOP, ROAD_SHOULDER_STOP).
        cfg: Configuration dictionary containing mrm, optional road_curvature_rad_m and mrm.in_lane_stop_steer_gain.

    Returns:
        ControlCommand with appropriate values for the MRM.
    """
    mrm = cfg.get("mrm", {})

    if mrm_id == MRM.STRAIGHT_STOP or mrm_id == MRM.BRAKE_IN_LANE:
        # Safest: brake only, no steering — 30% harder braking than IN_LANE_STOP (full brake)
        straight_cfg = mrm.get("straight_stop_brake", mrm.get("brake", {}))
        return ControlCommand(
            override=True,
            throttle=float(straight_cfg.get("throttle_value", 0.0)),
            brake=float(straight_cfg.get("brake_value", 1.0)),
            steer=0.0,
            follow_road_while_braking=False,
        )

    if mrm_id == MRM.IN_LANE_STOP:
        # Brake while following the lane (road); softer than straight stop (1/1.3 ≈ 0.77 so straight is 30% harder)
        brake_cfg = mrm.get("brake", {})
        default_in_lane_brake = 1.0 / 1.3  # ~0.769
        return ControlCommand(
            override=True,
            throttle=float(brake_cfg.get("throttle_value", 0.0)),
            brake=float(brake_cfg.get("brake_value", default_in_lane_brake)),
            steer=0.0,
            follow_road_while_braking=True,
        )

    if mrm_id == MRM.ROAD_SHOULDER_STOP or mrm_id == MRM.PULL_OVER_RIGHT:
        po = mrm.get("pull_over_right", mrm.get("brake", {}))
        return ControlCommand(
            override=True,
            throttle=float(po.get("throttle_value", 0.0)),
            brake=float(po.get("brake_value", 0.7)),
            steer=float(po.get("steer_value", -0.3)),
            follow_road_while_braking=False,
        )

    # Fallback: safe default (full brake, no steering)
    return ControlCommand(override=True, throttle=0.0, brake=1.0, steer=0.0, follow_road_while_braking=False)


def get_control_follow_lane(cfg: dict) -> ControlCommand:
    """
    Control for normal driving: follow the lane. Used before ODD exit.
    Throttle so the ego moves; brake 0. The state provider (esminiLib) uses
    SE_ReportObjectRoadPos when brake~0 and throttle>0 so the ego follows the road.
    """
    mrm = cfg.get("mrm", {})
    normal = mrm.get("normal_driving", {})
    throttle = float(normal.get("throttle_value", 0.4))
    brake = 0.0
    return ControlCommand(override=True, throttle=throttle, brake=brake, steer=0.0, follow_road_while_braking=False)


def no_override_command() -> ControlCommand:
    """No override (e.g. when not using follow_lane)."""
    return ControlCommand(override=False, throttle=0.0, brake=0.0, steer=0.0, follow_road_while_braking=False)
