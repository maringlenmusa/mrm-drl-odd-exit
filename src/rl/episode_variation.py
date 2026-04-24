"""
Per-episode scenario variation for Phase 3 RL training.

Every time the environment resets, this module draws fresh scenario parameters
so the agent trains on a varied distribution rather than one fixed scenario.

Two regimes are sampled (controlled by phase3.episode_variation.mix_baseline_probability):

  Baseline regime (easy):
      ego and front car at similar speeds, large initial gap.
      Range from configs/phase3_train.yaml → baseline_comparison.*

  Short-front-gap regime (hard):
      front car slower than ego, gap at ODD exit is tight (4–15 m).
      Range from configs/phase3_train.yaml → short_front_gap_coverage.*

To train on a different scenario distribution, edit those two config sections.
See TRAINING_ON_NEW_SCENARIOS.md for a full guide.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional

from src.phase2.run_baseline_comparison import (
    build_param_list as build_baseline_params,
    get_baseline_defaults,
)
from src.phase2.run_short_front_gap_coverage import (
    build_param_list as build_sfg_params,
    get_short_front_gap_defaults,
)


def sample_reset_options(cfg: dict, rng: random.Random) -> Dict[str, Any]:
    """
    Build one `options` dict for `OddExitEnv.reset(options=...)`.

    Reads:
      - phase3.episode_variation.mix_baseline_probability (default 0.5)
      - baseline_comparison / short_front_gap_coverage (same keys as phase2.yaml)

    Returns:
      dict with trigger_time_s and state_bridge overrides (no scenario_path).
    """
    ev = (cfg.get("phase3") or {}).get("episode_variation") or {}
    if not ev.get("enabled", True):
        return {}

    p_bl = float(ev.get("mix_baseline_probability", 0.5))
    use_baseline = rng.random() < p_bl

    if use_baseline:
        return _sample_baseline_options(cfg, rng)

    return _sample_short_front_gap_options(cfg, rng)


def _sample_baseline_options(cfg: dict, rng: random.Random) -> Dict[str, Any]:
    """
    Draw one 'easy' episode: similar ego/front speeds, comfortable gap.

    Reads speed and distance ranges from config:
        baseline_comparison.speed_min_mps  →  baseline_comparison.speed_max_mps
        baseline_comparison.distance_min_m →  baseline_comparison.distance_max_m

    To change what 'easy' looks like, edit those ranges in the YAML config.
    """
    bl = get_baseline_defaults(cfg)
    # One fresh sample (same as build with n_episodes=1)
    params = build_baseline_params(
        mode="sample",
        speed_min=bl["speed_min"],
        speed_max=bl["speed_max"],
        distance_min=bl["distance_min"],
        distance_max=bl["distance_max"],
        n_episodes=1,
        seed=rng.randint(0, 2**31 - 1),
    )
    ego_spd, front_spd, dist_rear = params[0]
    bl_trigger = bl["trigger_time_s"]

    actors = list(copy.deepcopy(cfg.get("state_bridge", {}).get("actors", [])))
    if len(actors) >= 1:
        actors[0] = dict(actors[0])
        actors[0]["speed_mps"] = front_spd
    if len(actors) >= 2:
        actors[1] = dict(actors[1])
        actors[1]["distance_behind_ego_m"] = dist_rear
        actors[1]["speed_mps"] = round(
            ego_spd * (1.0 + rng.uniform(-0.05, 0.05)), 2
        )

    return {
        "trigger_time_s": bl_trigger,
        "state_bridge": {
            "initial_speed_mps": ego_spd,
            "actors": actors,
        },
    }


def _sample_short_front_gap_options(cfg: dict, rng: random.Random) -> Dict[str, Any]:
    """
    Draw one 'hard' episode: front car slower than ego, tight gap at ODD exit.

    The gap at ODD exit is drawn from:
        short_front_gap_coverage.gap_at_trigger_min_m
        short_front_gap_coverage.gap_at_trigger_max_m

    The initial ego and front car positions are then back-calculated so that
    by the time ODD fires (trigger_time_s), the gap has closed to that value.

    To change what 'hard' looks like, edit those gap and speed ranges in the config.
    Important: gap_at_trigger_min_m must be large enough that stopping is physically
    possible. Rule of thumb: ≥ v_rel²/(2×decel) + 5  (e.g., ≥ 7 m at 30–40 km/h).
    """
    sfg = get_short_front_gap_defaults(cfg)
    params = build_sfg_params(
        mode="gap_at_trigger",
        front_gap_min_m=sfg["front_gap_min_m"],
        front_gap_max_m=sfg["front_gap_max_m"],
        speed_min_mps=sfg["speed_min_mps"],
        speed_max_mps=sfg["speed_max_mps"],
        trigger_times_s=sfg["trigger_times_s"],
        n_episodes=1,
        seed=rng.randint(0, 2**31 - 1),
        gap_at_trigger_min_m=sfg["gap_at_trigger_min_m"],
        gap_at_trigger_max_m=sfg["gap_at_trigger_max_m"],
        ego_speed_min_mps=sfg["ego_speed_min_mps"],
        ego_speed_max_mps=sfg["ego_speed_max_mps"],
        lead_speed_min_mps=sfg["lead_speed_min_mps"],
        lead_speed_max_mps=sfg["lead_speed_max_mps"],
        min_initial_gap_m=sfg["min_initial_gap_m"],
        distance_behind_ego_min_m=sfg["distance_behind_ego_min_m"],
        distance_behind_ego_max_m=sfg["distance_behind_ego_max_m"],
    )
    initial_gap, ego_spd, front_spd, trigger_s, dist_rear = params[0]
    initial_x_m = sfg["initial_x_m"]

    actors = list(copy.deepcopy(cfg.get("state_bridge", {}).get("actors", [])))
    if len(actors) >= 1:
        actors[0] = dict(actors[0])
        actors[0]["x_m"] = initial_x_m + initial_gap
        actors[0]["speed_mps"] = front_spd
    if len(actors) >= 2:
        actors[1] = dict(actors[1])
        actors[1]["distance_behind_ego_m"] = dist_rear
        actors[1]["speed_mps"] = round(
            ego_spd * (1.0 + rng.uniform(-0.05, 0.05)), 2
        )

    return {
        "trigger_time_s": trigger_s,
        "state_bridge": {
            "initial_speed_mps": ego_spd,
            "initial_x_m": initial_x_m,
            "actors": actors,
        },
    }


try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym


class Phase2StyleVariationEnv(gym.Wrapper):
    """
    On each reset, merges `sample_reset_options(cfg, rng)` into reset options.

    Outer wrappers (e.g. SampledScenarioEnv) can still pass scenario_path;
    merge order: variation first, then caller options (caller wins on keys).
    """

    def __init__(self, env, cfg: dict, seed: Optional[int] = None):
        super().__init__(env)
        self._cfg = cfg
        self._rng = random.Random(seed if seed is not None else 0)

    def reset(self, *, seed=None, options=None):
        base = sample_reset_options(self._cfg, self._rng)
        merged = {**base, **dict(options or {})}
        return self.env.reset(seed=seed, options=merged)
