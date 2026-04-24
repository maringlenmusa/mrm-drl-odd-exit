"""
Esmini State Bridge - Generates state packets from esmini simulation.

Since esmini doesn't automatically send UDP state, this bridge:
1. Tracks simulation time
2. Generates ego + optional other actors (synthetic) so gaps and collision work
3. Sends state via UDP

With state_bridge.actors in config, at least 2 cars (ego + others) are simulated;
gaps change as ego approaches others; collision=True when distance < collision_threshold_m.
"""

import math
import time
import socket
import json
import threading
from typing import Optional, List, Dict, Any


def _default_actors_config() -> List[Dict[str, Any]]:
    """Default: lead car (same lane) + optional left-lane car so we have at least 2 cars."""
    return [
        {"actor_id": 1, "x_m": 100.0, "y_m": 1.75, "speed_mps": 0.0, "lane_id": -1},   # lead car, same lane
        {"actor_id": 2, "x_m": 50.0, "y_m": 5.25, "speed_mps": 12.0, "lane_id": -2},   # left lane
    ]


def _arc_st_to_xy_yaw(s: float, t: float, curvature_rad_m: float) -> tuple:
    """
    Convert road (s, t) to world (x, y) and yaw (rad) for a constant-curvature arc.
    Reference line: start (0,0), hdg=0. Curvature < 0 = turn right.
    t positive = left of reference.
    """
    if abs(curvature_rad_m) < 1e-9:
        return (s, t, 0.0)
    k = curvature_rad_m
    hdg = k * s
    # Reference point at s
    x_ref = math.sin(k * s) / k
    y_ref = (1.0 - math.cos(k * s)) / k
    # Lateral offset: right of ref = -t in our convention (t left = positive)
    x_w = x_ref - t * math.sin(hdg)
    y_w = y_ref + t * math.cos(hdg)
    return (x_w, y_w, hdg)


class EsminiStateBridge:
    """
    Generates and sends simulation state packets via UDP.
    Supports synthetic other vehicles so gaps and collision reflect scenario.
    """

    def __init__(
        self,
        send_ip: str,
        send_port: int,
        dt: float = 0.05,
        odd_exit_trigger_time_s: float = 8.0,
        initial_speed_mps: float = 15.0,
        initial_x_m: float = 0.0,
        initial_y_m: float = 1.75,
        actors_config: Optional[List[Dict[str, Any]]] = None,
        collision_threshold_m: float = 3.0,
        ego_decel_mps2: float = 5.0,
        curvature_rad_m: float = 0.0,
    ):
        """
        Initialize the state bridge.

        Args:
            send_ip: IP to send state packets to
            send_port: Port to send to
            dt: Simulation timestep (s)
            odd_exit_trigger_time_s: Time (s) at which ODD exit triggers; after this ego decelerates
            initial_speed_mps: Ego initial speed (m/s)
            initial_x_m: Ego initial s (longitudinal) in m; e.g. 50 for straight_500m
            initial_y_m: Ego lateral offset t (m); e.g. 1.75
            actors_config: List of dicts with actor_id, x_m, y_m, speed_mps, lane_id (optional).
                          If None, uses default (lead car + left-lane car). If [], no other actors.
            collision_threshold_m: Collision when distance ego–actor < this (m)
            ego_decel_mps2: Ego deceleration when braking (m/s²). Rear follower uses ego_decel_mps2 × decel_factor.
            curvature_rad_m: Road curvature (rad/m). 0 = straight; negative = turn right (e.g. -6.98e-4 for 20° over 500 m).
        """
        self.send_ip = send_ip
        self.send_port = send_port
        self.dt = dt
        self.odd_exit_trigger_time_s = odd_exit_trigger_time_s
        self.initial_speed_mps = initial_speed_mps
        self.initial_x = float(initial_x_m)
        self.initial_y = float(initial_y_m)
        self.actors_config = actors_config if actors_config is not None else _default_actors_config()
        self.collision_threshold_m = collision_threshold_m
        self.ego_decel_mps2 = float(ego_decel_mps2)
        self.curvature_rad_m = float(curvature_rad_m)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._collision_occurred = False  # latch: once True, stay True

        self.road_length = 500.0
        self.ego_lane_id = -1

    def _generate_state(self, sim_time_s: float) -> dict:
        """
        Generate state packet for given simulation time.
        
        Args:
            sim_time_s: Current simulation time in seconds
            
        Returns:
            Dictionary matching the expected JSON state format
        """
        odd_exit_time = self.odd_exit_trigger_time_s
        ego_decel = self.ego_decel_mps2

        if sim_time_s < odd_exit_time:
            speed = self.initial_speed_mps
        else:
            time_since_exit = sim_time_s - odd_exit_time
            speed = max(0.0, self.initial_speed_mps - ego_decel * time_since_exit)

        # Ego longitudinal s (same formula as before; we treat "x" as s along road)
        if sim_time_s < odd_exit_time:
            s_ego = self.initial_x + self.initial_speed_mps * sim_time_s
        else:
            t_exit = odd_exit_time
            s_at_exit = self.initial_x + self.initial_speed_mps * t_exit
            t_decel = sim_time_s - t_exit
            s_ego = s_at_exit + self.initial_speed_mps * t_decel - 0.5 * ego_decel * t_decel * t_decel
        t_ego = self.initial_y
        if abs(self.curvature_rad_m) >= 1e-9:
            x, y, yaw_rad = _arc_st_to_xy_yaw(s_ego, t_ego, self.curvature_rad_m)
        else:
            x, y, yaw_rad = s_ego, t_ego, 0.0

        # Other actors: s (longitudinal), t (lateral); then convert to (x,y,yaw) if curved
        actors = []
        for ac in self.actors_config:
            if ac.get("rear_follower"):
                distance_behind = float(ac.get("distance_behind_ego_m", 20.0))
                reaction_delay = float(ac.get("reaction_delay_s", 2.0))
                factor = float(ac.get("decel_factor", 1.0))
                rear_decel = self.ego_decel_mps2 * factor if factor > 0 else float(ac.get("decel_mps2", ego_decel))
                s_rear_0 = self.initial_x - distance_behind
                t_brake_start = odd_exit_time + reaction_delay
                if sim_time_s < t_brake_start:
                    a_s = s_rear_0 + self.initial_speed_mps * sim_time_s
                    aspeed = self.initial_speed_mps
                else:
                    t_decel_rear = sim_time_s - t_brake_start
                    aspeed = max(0.0, self.initial_speed_mps - rear_decel * t_decel_rear)
                    s_at_brake = s_rear_0 + self.initial_speed_mps * t_brake_start
                    a_s = s_at_brake + self.initial_speed_mps * t_decel_rear - 0.5 * rear_decel * t_decel_rear * t_decel_rear
                a_t = float(ac["y_m"])
            else:
                a_s = float(ac["x_m"]) + float(ac.get("speed_mps", 0)) * sim_time_s
                a_t = float(ac["y_m"])
                aspeed = float(ac.get("speed_mps", 0))
            if abs(self.curvature_rad_m) >= 1e-9:
                ax, ay, ayaw = _arc_st_to_xy_yaw(a_s, a_t, self.curvature_rad_m)
            else:
                ax, ay, ayaw = a_s, a_t, 0.0
            actors.append({
                "actor_id": int(ac["actor_id"]),
                "x": ax,
                "y": ay,
                "speed_mps": aspeed,
                "yaw_rad": ayaw,
                "lane_id": ac.get("lane_id"),
            })

        # Collision: True if distance from ego to any actor < threshold; latch once True
        for a in actors:
            dist = math.hypot(x - a["x"], y - a["y"])
            if dist < self.collision_threshold_m:
                self._collision_occurred = True
                break
        collision = self._collision_occurred

        state = {
            "sim_time_s": sim_time_s,
            "ego": {
                "x": float(x),
                "y": float(y),
                "speed_mps": float(speed),
                "yaw_rad": float(yaw_rad),
                "lane_id": self.ego_lane_id,
            },
            "collision": collision,
            "actors": actors,
        }
        return state
    
    def _send_loop(self):
        """Main loop that generates and sends state packets."""
        sim_time = 0.0
        start_wall = time.time()
        
        while self.running:
            # Generate state
            state = self._generate_state(sim_time)
            
            # Encode as JSON
            payload = json.dumps(state).encode("utf-8")
            
            # Send via UDP
            try:
                self.sock.sendto(payload, (self.send_ip, self.send_port))
            except Exception as e:
                print(f"[StateBridge] Error sending packet: {e}")
                break
            
            # Wait for next timestep
            time.sleep(self.dt)
            
            # Update simulation time
            sim_time += self.dt
            
            # Stop after reasonable time (30s max)
            if sim_time > 30.0:
                break
    
    def start(self):
        """Start the state bridge in a background thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
        # Give it a moment to start
        time.sleep(0.1)
    
    def stop(self):
        """Stop the state bridge."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.sock.close()
