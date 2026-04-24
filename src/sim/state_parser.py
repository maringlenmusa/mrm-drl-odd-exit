"""
State parser wrapper for codec-based state decoding.

Provides a clean interface for parsing simulation state packets.
"""

from src.sim.codec import Codec
from src.types import SimState


def parse_state(codec: Codec, packet: bytes) -> SimState:
    """
    Parse a UDP packet into a SimState using the provided codec.
    
    Args:
        codec: Codec instance to use for decoding
        packet: Raw UDP packet bytes
        
    Returns:
        Parsed SimState object
    """
    return codec.decode_state(packet)
