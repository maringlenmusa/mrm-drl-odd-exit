"""
Phase 2 baseline comparison: run many episodes with varied ego speed, front car speed,
and distance behind ego (fixed ODD trigger time, rear speed = ego speed). Writes
baseline_results.csv for Phase 3 comparison and thesis.

Can be invoked via scripts/run_baseline_comparison.py (CLI) or src.main --phase phase2_baseline.
"""

import copy
import csv
import os
import random
from typing import List, Tuple

from src.logger import make_run_dir, write_episode_summary, EpisodeCsvLogger
from src.phase2.run_phase2 import run_phase2


# Default ranges (plan: ego/front 20–35 m/s, distance_behind_ego 12–48 m)
DEFAULT_SPEED_MIN = 20.0
DEFAULT_SPEED_MAX = 35.0
DEFAULT_DISTANCE_MIN = 12.0
DEFAULT_DISTANCE_MAX = 48.0
DEFAULT_TRIGGER_TIME_S = 8.0
DEFAULT_N_EPISODES = 50

BASELINE_RESULTS_FIELDS = [
    "episode_id",
    "initial_speed_mps",
    "front_speed_mps",
    "distance_behind_ego_m",
    "trigger_time_s",
    "success_mrc",
    "collision",
    "steps",
    "max_risk",
    "avg_risk",
    "mrm_switches",
    "primary_mrm",
    "last_sim_time_s",
]


def build_param_list(
    mode: str,
    speed_min: float,
    speed_max: float,
    distance_min: float,
    distance_max: float,
    n_episodes: int,
    speed_steps: int = 4,
    distance_steps: int = 4,
    seed: int = None,
) -> List[Tuple[float, float, float]]:
    """
    Build list of (initial_speed_mps, front_speed_mps, distance_behind_ego_m).
    mode: "grid" or "sample".
    """
    if mode == "grid":
        return _grid_combinations(
            speed_min, speed_max, speed_steps,
            distance_min, distance_max, distance_steps,
            n_cap=n_episodes if n_episodes else None,
            seed=seed,
        )
    return _sample_combinations(
        speed_min, speed_max, distance_min, distance_max,
        n_episodes, seed=seed,
    )


def _grid_combinations(
    speed_min: float,
    speed_max: float,
    speed_steps: int,
    distance_min: float,
    distance_max: float,
    distance_steps: int,
    n_cap: int = None,
    seed: int = None,
) -> List[Tuple[float, float, float]]:
    """Build list of (ego_speed, front_speed, distance_behind_ego_m). All combinations; optionally cap at n_cap."""
    if speed_steps < 1:
        speed_steps = 1
    if distance_steps < 1:
        distance_steps = 1
    speed_step = (speed_max - speed_min) / max(1, speed_steps - 1) if speed_steps > 1 else 0
    dist_step = (distance_max - distance_min) / max(1, distance_steps - 1) if distance_steps > 1 else 0
    speeds = [speed_min + i * speed_step for i in range(speed_steps)] if speed_steps > 1 else [speed_min]
    if speed_steps > 1 and abs(speeds[-1] - speed_max) > 1e-6:
        speeds.append(speed_max)
    dists = [distance_min + i * dist_step for i in range(distance_steps)] if distance_steps > 1 else [distance_min]
    if distance_steps > 1 and abs(dists[-1] - distance_max) > 1e-6:
        dists.append(distance_max)
    out = []
    for ego in speeds:
        for front in speeds:
            for d in dists:
                out.append((round(ego, 2), round(front, 2), round(d, 2)))
    if n_cap is not None and len(out) > n_cap:
        if seed is not None:
            random.seed(seed)
        random.shuffle(out)
        out = out[:n_cap]
    return out


def _sample_combinations(
    speed_min: float,
    speed_max: float,
    distance_min: float,
    distance_max: float,
    n: int,
    seed: int = None,
) -> List[Tuple[float, float, float]]:
    """Random sample of n (ego_speed, front_speed, distance_behind_ego_m) in ranges."""
    if seed is not None:
        random.seed(seed)
    out = []
    for _ in range(n):
        ego = round(random.uniform(speed_min, speed_max), 2)
        front = round(random.uniform(speed_min, speed_max), 2)
        dist = round(random.uniform(distance_min, distance_max), 2)
        out.append((ego, front, dist))
    return out


def _primary_mrm_from_steps(run_dir: str, episode_id: int) -> int:
    """Read steps_episode_{episode_id}.csv and return first final_mrm after ODD exit (or -1)."""
    path = os.path.join(run_dir, f"steps_episode_{episode_id}.csv")
    if not os.path.isfile(path):
        return -1
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    odd = row.get("odd_exit_active", "").strip().lower()
                    if odd in ("true", "1"):
                        fm = row.get("final_mrm", "")
                        if fm != "":
                            v = int(float(fm))
                            if v >= 0:
                                return v
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return -1


def run_baseline_comparison(
    cfg: dict,
    run_dir: str,
    param_list: List[Tuple[float, float, float]],
    trigger_time_s: float,
    headless: bool = False,
    write_primary_mrm: bool = True,
) -> None:
    """
    Run one Phase 2 episode per (initial_speed_mps, front_speed_mps, distance_behind_ego_m);
    write baseline_results.csv with params and outcomes.
    """
    cfg = copy.deepcopy(cfg)
    if headless:
        cfg.setdefault("sim", {})["headless"] = True
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"]["name"] = "phase2_baseline_comparison"

    episode_csv_logger = EpisodeCsvLogger(run_dir, mode="w")
    episode_csv_logger._episode_id = 0

    baseline_path = os.path.join(run_dir, "baseline_results.csv")
    with open(baseline_path, "w", newline="") as bf:
        writer = csv.DictWriter(bf, fieldnames=BASELINE_RESULTS_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for ep_idx, (initial_speed_mps, front_speed_mps, distance_behind_ego_m) in enumerate(param_list):
            cfg_ep = copy.deepcopy(cfg)
            cfg_ep["odd_exit"] = dict(cfg["odd_exit"])
            cfg_ep["odd_exit"]["trigger_time_s"] = float(trigger_time_s)
            cfg_ep["state_bridge"] = copy.deepcopy(cfg.get("state_bridge", {}))
            cfg_ep["state_bridge"]["initial_speed_mps"] = initial_speed_mps
            actors = list(cfg_ep["state_bridge"].get("actors", []))
            if len(actors) >= 1:
                actors[0] = dict(actors[0])
                actors[0]["speed_mps"] = front_speed_mps
            if len(actors) >= 2:
                actors[1] = dict(actors[1])
                actors[1]["distance_behind_ego_m"] = distance_behind_ego_m
                rear_speed_factor = 1.0 + random.uniform(-0.05, 0.05)
                actors[1]["speed_mps"] = round(initial_speed_mps * rear_speed_factor, 2)
            cfg_ep["state_bridge"]["actors"] = actors

            ep_row = run_phase2(
                cfg_ep,
                run_dir,
                episode_id=ep_idx,
                episode_csv_logger=episode_csv_logger,
            )

            primary_mrm = _primary_mrm_from_steps(run_dir, ep_idx) if write_primary_mrm else -1
            row = {
                "episode_id": ep_idx + 1,
                "initial_speed_mps": initial_speed_mps,
                "front_speed_mps": front_speed_mps,
                "distance_behind_ego_m": distance_behind_ego_m,
                "trigger_time_s": trigger_time_s,
                "success_mrc": ep_row.get("success_mrc", False),
                "collision": ep_row.get("collision", False),
                "steps": ep_row.get("steps", 0),
                "max_risk": ep_row.get("max_risk"),
                "avg_risk": ep_row.get("avg_risk"),
                "mrm_switches": ep_row.get("mrm_switches", 0),
                "primary_mrm": primary_mrm,
                "last_sim_time_s": ep_row.get("last_sim_time_s"),
            }
            writer.writerow(row)
            bf.flush()

    episode_csv_logger.close()
    write_episode_summary(run_dir, {
        "num_episodes": len(param_list),
        "trigger_time_s": trigger_time_s,
        "baseline_results_csv": "baseline_results.csv",
    })


def get_baseline_defaults(cfg: dict) -> dict:
    """Read baseline_comparison and odd_exit from config; return dict of defaults."""
    bc = cfg.get("baseline_comparison", {})
    odd = cfg.get("odd_exit", {})
    return {
        "n_episodes": int(bc.get("n_episodes", DEFAULT_N_EPISODES)),
        "trigger_time_s": float(bc.get("trigger_time_s", odd.get("trigger_time_s", DEFAULT_TRIGGER_TIME_S))),
        "speed_min": float(bc.get("speed_min_mps", DEFAULT_SPEED_MIN)),
        "speed_max": float(bc.get("speed_max_mps", DEFAULT_SPEED_MAX)),
        "distance_min": float(bc.get("distance_min_m", DEFAULT_DISTANCE_MIN)),
        "distance_max": float(bc.get("distance_max_m", DEFAULT_DISTANCE_MAX)),
        "speed_steps": int(bc.get("speed_steps", 4)),
        "distance_steps": int(bc.get("distance_steps", 4)),
    }


def run_baseline_from_config(
    cfg: dict,
    run_dir: str = None,
    headless: bool = False,
    n_episodes: int = None,
    trigger_time_s: float = None,
    mode: str = "grid",
    seed: int = None,
) -> str:
    """
    Run baseline comparison using config's baseline_comparison section (with optional overrides).
    Returns run_dir path.
    """
    cfg = copy.deepcopy(cfg)
    defaults = get_baseline_defaults(cfg)
    n = n_episodes if n_episodes is not None else defaults["n_episodes"]
    trigger = trigger_time_s if trigger_time_s is not None else defaults["trigger_time_s"]
    param_list = build_param_list(
        mode=mode,
        speed_min=defaults["speed_min"],
        speed_max=defaults["speed_max"],
        distance_min=defaults["distance_min"],
        distance_max=defaults["distance_max"],
        n_episodes=n,
        speed_steps=defaults["speed_steps"],
        distance_steps=defaults["distance_steps"],
        seed=seed,
    )
    if run_dir is None:
        cfg["run"] = dict(cfg.get("run", {}))
        cfg["run"]["name"] = "phase2_baseline_comparison"
        if headless:
            cfg.setdefault("sim", {})["headless"] = True
        run_dir = make_run_dir(cfg)
    run_baseline_comparison(cfg, run_dir, param_list, trigger, headless=headless, write_primary_mrm=True)
    return run_dir
