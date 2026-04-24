"""
Type definitions for the ODD exit RL project.

This module defines structured data types to keep the codebase clean and type-safe.
All simulation state, control commands, and logging structures are defined here.
Phase 2+: ActorState and actors list for multi-actor observation building.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EgoState:
    """Ego vehicle state information."""
    x: float
    y: float
    speed_mps: float
    yaw_rad: float
    lane_id: Optional[int] = None


@dataclass
class ActorState:
    """State of a non-ego actor (other vehicle or obstacle). Used for neighbor selection and observation building."""
    actor_id: int
    x: float
    y: float
    speed_mps: float
    yaw_rad: float
    lane_id: Optional[int] = None


@dataclass
class SimState:
    """Complete simulation state at a given time step."""
    sim_time_s: float
    ego: EgoState
    collision: bool = False   # optional if sim provides
    actors: List[ActorState] = field(default_factory=list)  # Phase 2+: other actors (empty for Phase 1)
    raw: dict = None          # keep raw payload for debugging


@dataclass
class ControlCommand:
    """Control command to send to the simulator."""
    override: bool            # True => override controller
    throttle: float           # 0..1
    brake: float              # 0..1
    steer: float              # -1..1 (or rad), depending on your interface
    follow_road_while_braking: bool = False  # If True, brake but stay on road (IN_LANE_STOP)


@dataclass
class Phase1StepLog:
    """Log entry for a single simulation step in Phase 1."""
    step_id: int
    sim_time_s: float
    odd_exit_active: bool
    ego_speed_mps: float
    throttle: float
    brake: float
    steer: float
    collision: bool
