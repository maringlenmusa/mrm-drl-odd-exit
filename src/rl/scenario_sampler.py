"""
Risk-prioritized scenario sampler for Phase 3 training.

The sampler starts with uniform weights and updates them from episode outcomes:
- collisions increase difficulty score
- high max_risk increases difficulty score
- failure to reach MRC increases difficulty score

Harder scenarios get larger sampling probability in later episodes.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional


class ScenarioSampler:
    """Weighted scenario sampler with risk-prioritized weight updates."""

    def __init__(
        self,
        scenarios: List[str],
        *,
        seed: Optional[int] = None,
        collision_weight: float = 2.0,
        risk_weight: float = 1.0,
        failure_weight: float = 1.0,
        min_weight: float = 0.1,
    ):
        if not scenarios:
            raise ValueError("ScenarioSampler requires at least one scenario path")
        self.scenarios: List[str] = [str(s) for s in scenarios]
        self._weights: List[float] = [1.0 for _ in self.scenarios]
        self._rng = random.Random(seed)

        self.collision_weight = float(collision_weight)
        self.risk_weight = float(risk_weight)
        self.failure_weight = float(failure_weight)
        self.min_weight = float(min_weight)

        # Fast index lookup for updates.
        self._idx: Dict[str, int] = {s: i for i, s in enumerate(self.scenarios)}

    @property
    def weights(self) -> List[float]:
        """Current scenario weights (copy)."""
        return list(self._weights)

    def get_distribution(self) -> Dict[str, float]:
        """Return normalized scenario probabilities."""
        total = sum(self._weights)
        if total <= 0:
            p = 1.0 / len(self.scenarios)
            return {s: p for s in self.scenarios}
        return {
            s: (self._weights[i] / total)
            for i, s in enumerate(self.scenarios)
        }

    def sample(self) -> str:
        """Sample one scenario path according to current weights."""
        return self._rng.choices(self.scenarios, weights=self._weights, k=1)[0]

    def update_weights(self, episode_summaries: List[dict]) -> None:
        """
        Update sampling weights using episode outcomes.

        Each episode summary should include:
          - scenario_path (preferred) or scenario
          - max_risk (0..1 ideally; values are clamped)
          - collision (bool)
          - success_mrc (bool)
        """
        if not episode_summaries:
            return

        score_sum = [0.0 for _ in self.scenarios]
        count = [0 for _ in self.scenarios]

        for ep in episode_summaries:
            scenario = ep.get("scenario_path", ep.get("scenario"))
            if scenario not in self._idx:
                continue
            i = self._idx[scenario]

            max_risk = float(ep.get("max_risk", 0.0))
            if max_risk < 0.0:
                max_risk = 0.0
            if max_risk > 1.0:
                max_risk = 1.0

            collision = bool(ep.get("collision", False))
            success_mrc = bool(ep.get("success_mrc", False))

            # Difficulty score (higher = harder/more dangerous).
            difficulty = 0.0
            if collision:
                difficulty += self.collision_weight
            difficulty += self.risk_weight * max_risk
            if not success_mrc:
                difficulty += self.failure_weight

            score_sum[i] += difficulty
            count[i] += 1

        # Update each scenario's weight to 1 + avg_difficulty (or keep unchanged if unseen).
        new_weights = list(self._weights)
        for i in range(len(self.scenarios)):
            if count[i] == 0:
                continue
            avg_difficulty = score_sum[i] / count[i]
            new_weights[i] = max(self.min_weight, 1.0 + avg_difficulty)

        self._weights = new_weights

