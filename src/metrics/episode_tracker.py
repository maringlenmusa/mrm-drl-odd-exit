"""
Episode tracker: aggregate risk and MRM switches over an episode.

max_risk and avg_risk are computed over steps where only_accumulate_risk_when=True.
- whole_episode: caller passes True every step → max/avg risk over entire run (peak risk during approach).
- after_odd_exit: caller passes True only when ODD active → max/avg risk during MRM/stop phase only.
"""

from typing import List, Optional


class EpisodeTracker:
    """
    Tracks max_risk, avg_risk (during ODD phase only by default), mrm_switches, primary_mrm, step_count.
    mrm_switches = number of times the chosen MRM changed (0 = stayed on one MRM). primary_mrm = last MRM id used (0=straight/brake, 1=in-lane, etc.; -1 if none).
    """

    def __init__(self):
        self._max_risk: float = 0.0
        self._risk_sum: float = 0.0
        self._step_count: int = 0
        self._risk_step_count: int = 0  # steps where risk was accumulated (e.g. after ODD exit)
        self._mrm_switches: int = 0
        self._prev_mrm: Optional[int] = -1
        self._primary_mrm: int = -1  # last MRM id when ODD active (which maneuver was used)

    def update(
        self,
        risk_total: float,
        mrm_id: int,
        *,
        only_accumulate_risk_when: bool = True,
    ) -> None:
        """
        Update with step risk and chosen MRM.

        Args:
            risk_total: Combined risk for this step in [0, 1].
            mrm_id: MRM chosen this step (-1 if before ODD exit).
            only_accumulate_risk_when: If True, include this step in max_risk and avg_risk
                (e.g. pass odd_active so risk is only over the MRM/stop phase).
        """
        if only_accumulate_risk_when:
            self._max_risk = max(self._max_risk, risk_total)
            self._risk_sum += risk_total
            self._risk_step_count += 1
        self._step_count += 1
        if mrm_id >= 0 and self._prev_mrm >= 0 and mrm_id != self._prev_mrm:
            self._mrm_switches += 1
        if mrm_id >= 0:
            self._prev_mrm = mrm_id
            self._primary_mrm = mrm_id

    def get_summary(self) -> dict:
        """
        Return episode summary dict.
        max_risk and avg_risk are over steps where risk was accumulated (e.g. after ODD exit only).
        """
        avg_risk = (
            self._risk_sum / self._risk_step_count
            if self._risk_step_count > 0
            else 0.0
        )
        return {
            "max_risk": self._max_risk,
            "avg_risk": avg_risk,
            "mrm_switches": self._mrm_switches,
            "primary_mrm": self._primary_mrm,
            "step_count": self._step_count,
        }

    def reset(self) -> None:
        """Reset for a new episode."""
        self._max_risk = 0.0
        self._risk_sum = 0.0
        self._step_count = 0
        self._risk_step_count = 0
        self._mrm_switches = 0
        self._prev_mrm = -1
        self._primary_mrm = -1
