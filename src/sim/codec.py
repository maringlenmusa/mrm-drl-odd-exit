"""
Codec for encoding/decoding simulation state and control commands.

This module separates packet format concerns from business logic.
- JsonCodec: JSON state/control (used with state_bridge).
- EsminiDriverCodec: esmini UDPDriverController binary DRIVER_INPUT format
  (port 49950) so the UI ego applies our throttle/brake/steer.
"""

import json
import math
import struct
from src.types import SimState, EgoState, ActorState, ControlCommand

# esmini ControllerUDPDriver.hpp: UDP_DRIVER_MESSAGE_VERSION=1, DRIVER_INPUT=1
UDP_DRIVER_MESSAGE_VERSION = 1
UDP_DRIVER_INPUT_MODE_DRIVER_INPUT = 1


class Codec:
    """Base class for state/control codecs."""
    
    def decode_state(self, packet: bytes) -> SimState:
        """
        Decode a UDP packet into a SimState object.
        
        Args:
            packet: Raw UDP packet bytes
            
        Returns:
            Parsed SimState object
        """
        raise NotImplementedError

    def encode_control(self, cmd: ControlCommand) -> bytes:
        """
        Encode a ControlCommand into bytes for UDP transmission.
        
        Args:
            cmd: ControlCommand to encode
            
        Returns:
            Encoded bytes
        """
        raise NotImplementedError


class JsonCodec(Codec):
    """JSON-based codec for state and control messages."""
    
    def decode_state(self, packet: bytes) -> SimState:
        """
        Decode JSON packet into SimState.
        
        Expected JSON format:
        {
            "sim_time_s": float,
            "ego": {
                "x": float,
                "y": float,
                "speed_mps": float,
                "yaw_rad": float,
                "lane_id": int (optional)
            },
            "collision": bool (optional),
            "actors": [ {"actor_id", "x", "y", "speed_mps", "yaw_rad", "lane_id"}, ... ] (optional, Phase 2+)
        }
        """
        msg = json.loads(packet.decode("utf-8"))

        ego = EgoState(
            x=msg["ego"]["x"],
            y=msg["ego"]["y"],
            speed_mps=msg["ego"]["speed_mps"],
            yaw_rad=msg["ego"]["yaw_rad"],
            lane_id=msg["ego"].get("lane_id")
        )

        actors = []
        for a in msg.get("actors", []):
            actors.append(ActorState(
                actor_id=int(a["actor_id"]),
                x=float(a["x"]),
                y=float(a["y"]),
                speed_mps=float(a["speed_mps"]),
                yaw_rad=float(a["yaw_rad"]),
                lane_id=a.get("lane_id")
            ))

        return SimState(
            sim_time_s=msg["sim_time_s"],
            ego=ego,
            collision=bool(msg.get("collision", False)),
            actors=actors,
            raw=msg
        )

    def encode_control(self, cmd: ControlCommand) -> bytes:
        """
        Encode ControlCommand into JSON bytes.
        
        Output format:
        {
            "override": bool,
            "throttle": float,
            "brake": float,
            "steer": float
        }
        """
        msg = {
            "override": cmd.override,
            "throttle": cmd.throttle,
            "brake": cmd.brake,
            "steer": cmd.steer
        }
        return json.dumps(msg).encode("utf-8")


class EsminiDriverCodec(Codec):
    """
    Codec for esmini UDPDriverController DRIVER_INPUT mode (port 49950).

    Packet format matches ControllerUDPDriver.hpp DMMessage:
    - DMHeader: version (uint32), inputMode (uint32), objectId (uint32), frameNumber (uint32)
    - DMMSGDriverInput: throttle (float64 [0,1]), brake (float64 [0,1]), steeringAngle (float64 [-pi/2, pi/2])
    Little-endian. State is still from state_bridge (decode_state not used for esmini state).
    """

    def __init__(self, object_id: int = 0):
        self._object_id = object_id
        self._frame_number = 0

    def decode_state(self, packet: bytes) -> SimState:
        """State comes from state_bridge; esmini state input not used here."""
        raise NotImplementedError("EsminiDriverCodec does not decode state; use state_bridge + JsonCodec for state")

    def encode_control(self, cmd: ControlCommand) -> bytes:
        """
        Encode ControlCommand as esmini DMMessage (DRIVER_INPUT).
        steer in ControlCommand is [-1, 1] -> steeringAngle = steer * (pi/2) in radians.
        """
        self._frame_number += 1
        # Clamp to esmini ranges: throttle/brake [0,1], steeringAngle [-pi/2, pi/2]
        throttle = max(0.0, min(1.0, float(cmd.throttle)))
        brake = max(0.0, min(1.0, float(cmd.brake)))
        steer_norm = max(-1.0, min(1.0, float(cmd.steer)))
        steering_angle_rad = steer_norm * (math.pi / 2.0)
        header = struct.pack(
            "<IIII",
            UDP_DRIVER_MESSAGE_VERSION,
            UDP_DRIVER_INPUT_MODE_DRIVER_INPUT,
            self._object_id,
            self._frame_number,
        )
        body = struct.pack("<ddd", throttle, brake, steering_angle_rad)
        return header + body
