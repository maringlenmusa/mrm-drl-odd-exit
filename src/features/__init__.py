"""
Features module: observation building for Phase 2+.

Provides neighbor selection and observation dict construction from SimState.
"""

from src.features.neighbor_selector import select_neighbors, Neighbors
from src.features.observation_builder import build_observation

__all__ = ["select_neighbors", "Neighbors", "build_observation"]
