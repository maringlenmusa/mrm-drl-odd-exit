"""
Time-to-Collision (TTC) estimation.

Used for DARA (scene risk). Returns infinity when not closing (closing_speed <= 0).
"""

import math
from typing import Union


def estimate_ttc(gap_m: float, closing_speed_mps: float) -> Union[float, type(math.inf)]:
    """
    Estimate time-to-collision (seconds) from gap and closing speed.

    Args:
        gap_m: Distance to obstacle (m), must be >= 0.
        closing_speed_mps: Approach speed (m/s). Positive = closing.

    Returns:
        TTC in seconds if closing_speed_mps > 0 and gap_m > 0; else math.inf.
    """
    if closing_speed_mps <= 0 or gap_m <= 0:
        return math.inf
    return gap_m / closing_speed_mps
