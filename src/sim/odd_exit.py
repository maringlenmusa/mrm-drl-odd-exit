"""
ODD (Operational Design Domain) exit trigger logic.

Phase 1: Simple time-based trigger.
Future phases: Can add condition-based triggers (e.g., sensor failure,
unexpected obstacle, road condition changes, etc.).
"""


class OddExitTrigger:
    """Determines when ODD exit conditions are met."""
    
    def __init__(self, trigger_time_s: float):
        """
        Initialize time-based ODD exit trigger.
        
        Args:
            trigger_time_s: Simulation time (seconds) at which ODD exit is triggered
        """
        self.trigger_time_s = trigger_time_s

    def is_active(self, sim_time_s: float) -> bool:
        """
        Check if ODD exit is currently active.
        
        Args:
            sim_time_s: Current simulation time in seconds
            
        Returns:
            True if ODD exit is active, False otherwise
        """
        return sim_time_s >= self.trigger_time_s
