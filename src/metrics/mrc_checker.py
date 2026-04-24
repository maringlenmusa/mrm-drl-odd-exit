"""
MRC (Minimum Risk Condition) checker.

Supervisor: MRC = full stop. Optionally "stopped for N seconds" (configurable).
"""

from typing import Optional


class MrcChecker:
    """
    Detects when the vehicle has reached MRC (full stop, optionally for N seconds).
    """

    def __init__(
        self,
        stop_speed_mps: float = 0.5,
        stopped_for_s: float = 2.0,
    ):
        """
        Args:
            stop_speed_mps: Speed below which vehicle is considered stopped (default 0.5).
            stopped_for_s: Required duration at or below stop_speed_mps to declare MRC (default 2.0).
        """
        self.stop_speed_mps = stop_speed_mps
        self.stopped_for_s = stopped_for_s
        self._time_stopped_s: float = 0.0
        self._mrc_reached: bool = False

    def update(self, speed_mps: float, dt_s: float) -> None:
        """
        Update with current speed and elapsed time.

        Args:
            speed_mps: Current ego speed (m/s).
            dt_s: Elapsed time since last update (s).
        """
        if self._mrc_reached:
            return
        if speed_mps <= self.stop_speed_mps:
            self._time_stopped_s += dt_s
            if self._time_stopped_s >= self.stopped_for_s:
                self._mrc_reached = True
        else:
            self._time_stopped_s = 0.0

    def is_mrc_reached(self) -> bool:
        """Return True if MRC has been reached (stopped for required time)."""
        return self._mrc_reached

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._time_stopped_s = 0.0
        self._mrc_reached = False
