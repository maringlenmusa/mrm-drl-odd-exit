"""
Short front gap scenario coverage (phase3_bandit_plan.md §3).

Runs many Phase 2 episodes so that **when ODD exit hits**, the ego is **close to the lead**
(gap_at_trigger ~ 10–25 m), giving high risk and more straight/in-lane stop decisions.

Physics: gap(t) = initial_gap + t*(v_lead - v_ego). So at trigger_s we want
  gap_at_trigger = initial_gap + trigger_s*(v_lead - v_ego)
  => initial_gap = gap_at_trigger + trigger_s*(v_ego - v_lead).
We require v_ego > v_lead (closing) so initial_gap is positive and shrinks to gap_at_trigger.

Usage:
  python -m src.main --phase phase2_short_front_gap --config configs/phase2.yaml [--headless]
  or: python -m src.phase2.run_short_front_gap_coverage [--config configs/phase2.yaml] [--headless]
"""

import copy
import csv
import os
import random
from typing import List, Tuple, Optional, Union

from src.logger import make_run_dir, write_episode_summary, EpisodeCsvLogger
from src.phase2.run_phase2 import run_phase2


# Legacy: initial front gap range (used when mode is "grid" or "sample")
DEFAULT_FRONT_GAP_MIN_M = 15.0
DEFAULT_FRONT_GAP_MAX_M = 40.0
DEFAULT_SPEED_MIN_MPS = 20.0
DEFAULT_SPEED_MAX_MPS = 35.0
# Gap-at-trigger mode: desired gap when ODD exits (ego close to lead → high risk → straight/in-lane stop)
DEFAULT_GAP_AT_TRIGGER_MIN_M = 10.0
DEFAULT_GAP_AT_TRIGGER_MAX_M = 25.0
DEFAULT_EGO_SPEED_MIN_MPS = 22.0
DEFAULT_EGO_SPEED_MAX_MPS = 28.0
DEFAULT_LEAD_SPEED_MIN_MPS = 10.0
DEFAULT_LEAD_SPEED_MAX_MPS = 18.0
DEFAULT_MIN_INITIAL_GAP_M = 20.0
DEFAULT_TRIGGER_TIMES_S = [7.0]
DEFAULT_N_EPISODES = 54
# Rear car: same speed as ego (as in other sims); distance behind ego varied per episode
DEFAULT_DISTANCE_BEHIND_EGO_MIN_M = 12.0
DEFAULT_DISTANCE_BEHIND_EGO_MAX_M = 48.0

SUMMARY_CSV_FIELDS = [
    "episode_id",
    "front_gap_m",
    "initial_speed_mps",
    "front_speed_mps",
    "trigger_time_s",
    "distance_behind_ego_m",
    "rear_speed_mps",
    "shoulder_available",
    "primary_mrm",
    "success_mrc",
    "collision",
    "min_gap",
    "time_to_stop",
    "time_to_mrc",
    "max_decel",
    "steps",
    "max_risk",
    "avg_risk",
]


def build_param_list(
    mode: str,
    front_gap_min_m: float,
    front_gap_max_m: float,
    speed_min_mps: float,
    speed_max_mps: float,
    trigger_times_s: List[float],
    n_episodes: int,
    front_gap_steps: int = 4,
    speed_steps: int = 3,
    seed: Optional[int] = None,
    # gap_at_trigger mode: desired gap when ODD exits (ego close to lead)
    gap_at_trigger_min_m: Optional[float] = None,
    gap_at_trigger_max_m: Optional[float] = None,
    ego_speed_min_mps: Optional[float] = None,
    ego_speed_max_mps: Optional[float] = None,
    lead_speed_min_mps: Optional[float] = None,
    lead_speed_max_mps: Optional[float] = None,
    min_initial_gap_m: float = DEFAULT_MIN_INITIAL_GAP_M,
    distance_behind_ego_min_m: Optional[float] = None,
    distance_behind_ego_max_m: Optional[float] = None,
) -> Union[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float, float]]]:
    """
    Build list of params. Mode "gap_at_trigger": 5-tuples (initial_gap, v_ego, v_lead, trigger_s, distance_behind_ego_m)
    with rear distance varied per episode (rear speed = ego speed, as in other sims). Grid/sample: 4-tuples.
    """
    if mode == "gap_at_trigger":
        return _build_gap_at_trigger_param_list(
            gap_at_trigger_min_m=gap_at_trigger_min_m or DEFAULT_GAP_AT_TRIGGER_MIN_M,
            gap_at_trigger_max_m=gap_at_trigger_max_m or DEFAULT_GAP_AT_TRIGGER_MAX_M,
            ego_speed_min_mps=ego_speed_min_mps or DEFAULT_EGO_SPEED_MIN_MPS,
            ego_speed_max_mps=ego_speed_max_mps or DEFAULT_EGO_SPEED_MAX_MPS,
            lead_speed_min_mps=lead_speed_min_mps or DEFAULT_LEAD_SPEED_MIN_MPS,
            lead_speed_max_mps=lead_speed_max_mps or DEFAULT_LEAD_SPEED_MAX_MPS,
            trigger_times_s=trigger_times_s,
            n_episodes=n_episodes,
            min_initial_gap_m=min_initial_gap_m,
            seed=seed,
            distance_behind_ego_min_m=distance_behind_ego_min_m or DEFAULT_DISTANCE_BEHIND_EGO_MIN_M,
            distance_behind_ego_max_m=distance_behind_ego_max_m or DEFAULT_DISTANCE_BEHIND_EGO_MAX_M,
        )
    if mode == "grid":
        return _grid_short_front_gap(
            front_gap_min_m,
            front_gap_max_m,
            front_gap_steps,
            speed_min_mps,
            speed_max_mps,
            speed_steps,
            trigger_times_s,
            n_cap=n_episodes,
            seed=seed,
        )
    return _sample_short_front_gap(
        front_gap_min_m,
        front_gap_max_m,
        speed_min_mps,
        speed_max_mps,
        trigger_times_s,
        n_episodes,
        seed=seed,
    )


def _build_gap_at_trigger_param_list(
    gap_at_trigger_min_m: float,
    gap_at_trigger_max_m: float,
    ego_speed_min_mps: float,
    ego_speed_max_mps: float,
    lead_speed_min_mps: float,
    lead_speed_max_mps: float,
    trigger_times_s: List[float],
    n_episodes: int,
    min_initial_gap_m: float,
    seed: Optional[int] = None,
    distance_behind_ego_min_m: float = DEFAULT_DISTANCE_BEHIND_EGO_MIN_M,
    distance_behind_ego_max_m: float = DEFAULT_DISTANCE_BEHIND_EGO_MAX_M,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Build (initial_front_gap_m, v_ego, v_lead, trigger_time_s, distance_behind_ego_m).
    When ODD exits at trigger_time_s, gap = gap_at_trigger. Rear car: speed = ego (in scenario), distance random.
    """
    if seed is not None:
        random.seed(seed)
    out: List[Tuple[float, float, float, float, float]] = []
    attempts = 0
    max_attempts = n_episodes * 20
    while len(out) < n_episodes and attempts < max_attempts:
        attempts += 1
        gap_at_trigger = random.uniform(gap_at_trigger_min_m, gap_at_trigger_max_m)
        v_ego = round(random.uniform(ego_speed_min_mps, ego_speed_max_mps), 2)
        v_lead = round(random.uniform(lead_speed_min_mps, lead_speed_max_mps), 2)
        if v_ego <= v_lead:
            continue
        tt = random.choice(trigger_times_s)
        # initial_gap so that at tt: gap(tt) = gap_at_trigger
        initial_gap = gap_at_trigger + tt * (v_ego - v_lead)
        if initial_gap < min_initial_gap_m:
            continue
        initial_gap = round(initial_gap, 2)
        dist_rear = round(random.uniform(distance_behind_ego_min_m, distance_behind_ego_max_m), 2)
        out.append((initial_gap, v_ego, v_lead, tt, dist_rear))
    # If we didn't get enough, fill from a small grid
    while len(out) < n_episodes:
        for gap_at in [gap_at_trigger_min_m, (gap_at_trigger_min_m + gap_at_trigger_max_m) / 2, gap_at_trigger_max_m]:
            for v_ego in [ego_speed_min_mps, (ego_speed_min_mps + ego_speed_max_mps) / 2, ego_speed_max_mps]:
                for v_lead in [lead_speed_min_mps, (lead_speed_min_mps + lead_speed_max_mps) / 2, lead_speed_max_mps]:
                    if v_ego <= v_lead:
                        continue
                    for tt in trigger_times_s:
                        initial_gap = gap_at + tt * (v_ego - v_lead)
                        if initial_gap >= min_initial_gap_m:
                            dist_rear = round(random.uniform(distance_behind_ego_min_m, distance_behind_ego_max_m), 2)
                            out.append((round(initial_gap, 2), round(v_ego, 2), round(v_lead, 2), tt, dist_rear))
                        if len(out) >= n_episodes:
                            break
                    if len(out) >= n_episodes:
                        break
                if len(out) >= n_episodes:
                    break
            if len(out) >= n_episodes:
                break
        if len(out) >= n_episodes:
            break
    return out[:n_episodes]


def _grid_short_front_gap(
    front_gap_min: float,
    front_gap_max: float,
    front_gap_steps: int,
    speed_min: float,
    speed_max: float,
    speed_steps: int,
    trigger_times: List[float],
    n_cap: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float, float]]:
    """Grid over (front_gap, ego_speed, front_speed, trigger_time)."""
    if front_gap_steps < 1:
        front_gap_steps = 1
    if speed_steps < 1:
        speed_steps = 1
    gap_step = (front_gap_max - front_gap_min) / max(1, front_gap_steps - 1) if front_gap_steps > 1 else 0
    speed_step = (speed_max - speed_min) / max(1, speed_steps - 1) if speed_steps > 1 else 0
    gaps = [front_gap_min + i * gap_step for i in range(front_gap_steps)]
    if front_gap_steps > 1 and abs(gaps[-1] - front_gap_max) > 1e-6:
        gaps.append(front_gap_max)
    speeds = [speed_min + i * speed_step for i in range(speed_steps)]
    if speed_steps > 1 and abs(speeds[-1] - speed_max) > 1e-6:
        speeds.append(speed_max)
    out = []
    for gap in gaps:
        for ego_s in speeds:
            for front_s in speeds:
                for tt in trigger_times:
                    out.append((round(gap, 2), round(ego_s, 2), round(front_s, 2), float(tt)))
    if n_cap is not None and len(out) > n_cap:
        if seed is not None:
            random.seed(seed)
        random.shuffle(out)
        out = out[:n_cap]
    return out


def _sample_short_front_gap(
    front_gap_min: float,
    front_gap_max: float,
    speed_min: float,
    speed_max: float,
    trigger_times: List[float],
    n: int,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float, float]]:
    """Random sample of n (front_gap_m, initial_speed_mps, front_speed_mps, trigger_time_s)."""
    if seed is not None:
        random.seed(seed)
    out = []
    for _ in range(n):
        gap = round(random.uniform(front_gap_min, front_gap_max), 2)
        ego_s = round(random.uniform(speed_min, speed_max), 2)
        front_s = round(random.uniform(speed_min, speed_max), 2)
        tt = random.choice(trigger_times)
        out.append((gap, ego_s, front_s, tt))
    return out


def _primary_mrm_from_steps(run_dir: str, episode_id: int) -> int:
    """First final_mrm after ODD exit in steps_episode_{id}.csv, or -1."""
    path = os.path.join(run_dir, f"steps_episode_{episode_id}.csv")
    if not os.path.isfile(path):
        return -1
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if row.get("odd_exit_active", "").strip().lower() in ("true", "1"):
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


def run_short_front_gap_coverage(
    cfg: dict,
    run_dir: str,
    param_list: Union[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float, float]]],
    headless: bool = False,
    distance_behind_ego_m: float = 30.0,
    initial_x_m: float = 50.0,
) -> None:
    """
    Run one Phase 2 episode per param tuple. Params: (front_gap_m, initial_speed_mps, front_speed_mps, trigger_time_s)
    or 5-tuple with (..., distance_behind_ego_m). Rear car speed = ego (set in scenario); distance per episode when 5-tuple.
    """
    cfg = copy.deepcopy(cfg)
    if headless:
        cfg.setdefault("sim", {})["headless"] = True
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"]["name"] = "phase2_short_front_gap_coverage"

    episode_csv_logger = EpisodeCsvLogger(run_dir, mode="w")
    episode_csv_logger._episode_id = 0

    summary_path = os.path.join(run_dir, "short_front_gap_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=SUMMARY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for ep_idx, param in enumerate(param_list):
            if len(param) == 5:
                front_gap_m, initial_speed_mps, front_speed_mps, trigger_time_s, dist_rear = param
            else:
                front_gap_m, initial_speed_mps, front_speed_mps, trigger_time_s = param
                dist_rear = distance_behind_ego_m
            cfg_ep = copy.deepcopy(cfg)
            cfg_ep["odd_exit"] = dict(cfg["odd_exit"])
            cfg_ep["odd_exit"]["trigger_time_s"] = float(trigger_time_s)
            cfg_ep["state_bridge"] = copy.deepcopy(cfg.get("state_bridge", {}))
            cfg_ep["state_bridge"]["initial_speed_mps"] = initial_speed_mps
            cfg_ep["state_bridge"]["initial_x_m"] = initial_x_m
            actors = list(cfg_ep["state_bridge"].get("actors", []))
            # Lead car: place at ego_x + front_gap_m so initial front gap = front_gap_m
            if len(actors) >= 1:
                actors[0] = dict(actors[0])
                actors[0]["x_m"] = initial_x_m + front_gap_m
                actors[0]["speed_mps"] = front_speed_mps
            # Rear car: speed = ego ±1–5% at random per episode (so gap stays similar or closes/opens slightly)
            if len(actors) >= 2:
                actors[1] = dict(actors[1])
                actors[1]["distance_behind_ego_m"] = dist_rear
                rear_speed_factor = 1.0 + random.uniform(-0.05, 0.05)  # 1–5% faster or slower
                rear_speed_mps = round(initial_speed_mps * rear_speed_factor, 2)
                actors[1]["speed_mps"] = rear_speed_mps
            else:
                rear_speed_mps = None
            cfg_ep["state_bridge"]["actors"] = actors

            ep_row = run_phase2(
                cfg_ep,
                run_dir,
                episode_id=ep_idx,
                episode_csv_logger=episode_csv_logger,
            )

            primary_mrm = _primary_mrm_from_steps(run_dir, ep_idx)
            row = {
                "episode_id": ep_idx + 1,
                "front_gap_m": front_gap_m,
                "initial_speed_mps": initial_speed_mps,
                "front_speed_mps": front_speed_mps,
                "trigger_time_s": trigger_time_s,
                "distance_behind_ego_m": dist_rear,
                "rear_speed_mps": rear_speed_mps,
                "shoulder_available": ep_row.get("shoulder_available", ""),
                "primary_mrm": primary_mrm,
                "success_mrc": ep_row.get("success_mrc", False),
                "collision": ep_row.get("collision", False),
                "min_gap": ep_row.get("min_gap"),
                "time_to_stop": ep_row.get("time_to_stop"),
                "time_to_mrc": ep_row.get("time_to_mrc"),
                "max_decel": ep_row.get("max_decel"),
                "steps": ep_row.get("steps", 0),
                "max_risk": ep_row.get("max_risk"),
                "avg_risk": ep_row.get("avg_risk"),
            }
            writer.writerow(row)
            sf.flush()

    episode_csv_logger.close()
    write_episode_summary(run_dir, {
        "num_episodes": len(param_list),
        "short_front_gap_summary_csv": "short_front_gap_summary.csv",
        "purpose": "Short front gap coverage for straight/in-lane stop episodes (phase3_bandit_plan §3)",
    })


def get_short_front_gap_defaults(cfg: dict) -> dict:
    """Read short_front_gap_coverage and related config; return defaults for param list and run."""
    sfg = cfg.get("short_front_gap_coverage", {})
    odd = cfg.get("odd_exit", {})
    sb = cfg.get("state_bridge", {})
    actors = sb.get("actors") or []
    rear_dist = 30.0
    if len(actors) >= 2 and isinstance(actors[1], dict):
        rear_dist = float(actors[1].get("distance_behind_ego_m", 30.0))
    trigger_list = sfg.get("trigger_time_s_variations")
    if trigger_list is None:
        tt = float(odd.get("trigger_time_s", 7.0))
        trigger_list = [tt]
    if isinstance(trigger_list, (int, float)):
        trigger_list = [float(trigger_list)]
    return {
        "front_gap_min_m": float(sfg.get("front_gap_min_m", DEFAULT_FRONT_GAP_MIN_M)),
        "front_gap_max_m": float(sfg.get("front_gap_max_m", DEFAULT_FRONT_GAP_MAX_M)),
        "speed_min_mps": float(sfg.get("speed_min_mps", DEFAULT_SPEED_MIN_MPS)),
        "speed_max_mps": float(sfg.get("speed_max_mps", DEFAULT_SPEED_MAX_MPS)),
        "trigger_times_s": [float(t) for t in trigger_list],
        "n_episodes": int(sfg.get("n_episodes", DEFAULT_N_EPISODES)),
        "front_gap_steps": int(sfg.get("front_gap_steps", 4)),
        "speed_steps": int(sfg.get("speed_steps", 3)),
        "mode": str(sfg.get("mode", "gap_at_trigger")),
        "seed": cfg.get("run", {}).get("seed"),
        "distance_behind_ego_m": float(sfg.get("distance_behind_ego_m", rear_dist)),
        "initial_x_m": float(sb.get("initial_x_m", 50.0)),
        # gap_at_trigger mode
        "gap_at_trigger_min_m": float(sfg.get("gap_at_trigger_min_m", DEFAULT_GAP_AT_TRIGGER_MIN_M)),
        "gap_at_trigger_max_m": float(sfg.get("gap_at_trigger_max_m", DEFAULT_GAP_AT_TRIGGER_MAX_M)),
        "ego_speed_min_mps": float(sfg.get("ego_speed_min_mps", DEFAULT_EGO_SPEED_MIN_MPS)),
        "ego_speed_max_mps": float(sfg.get("ego_speed_max_mps", DEFAULT_EGO_SPEED_MAX_MPS)),
        "lead_speed_min_mps": float(sfg.get("lead_speed_min_mps", DEFAULT_LEAD_SPEED_MIN_MPS)),
        "lead_speed_max_mps": float(sfg.get("lead_speed_max_mps", DEFAULT_LEAD_SPEED_MAX_MPS)),
        "min_initial_gap_m": float(sfg.get("min_initial_gap_m", DEFAULT_MIN_INITIAL_GAP_M)),
        "distance_behind_ego_min_m": float(sfg.get("distance_behind_ego_min_m", DEFAULT_DISTANCE_BEHIND_EGO_MIN_M)),
        "distance_behind_ego_max_m": float(sfg.get("distance_behind_ego_max_m", DEFAULT_DISTANCE_BEHIND_EGO_MAX_M)),
    }


def run_from_config(
    cfg: dict,
    run_dir: Optional[str] = None,
    headless: bool = False,
    n_episodes: Optional[int] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Run short-front-gap coverage using config's short_front_gap_coverage section.
    Returns run_dir path.
    """
    cfg = copy.deepcopy(cfg)
    defaults = get_short_front_gap_defaults(cfg)
    n = n_episodes if n_episodes is not None else defaults["n_episodes"]
    mode = mode if mode is not None else defaults["mode"]
    param_list = build_param_list(
        mode=mode,
        front_gap_min_m=defaults["front_gap_min_m"],
        front_gap_max_m=defaults["front_gap_max_m"],
        speed_min_mps=defaults["speed_min_mps"],
        speed_max_mps=defaults["speed_max_mps"],
        trigger_times_s=defaults["trigger_times_s"],
        n_episodes=n,
        front_gap_steps=defaults["front_gap_steps"],
        speed_steps=defaults["speed_steps"],
        seed=seed or defaults.get("seed"),
        gap_at_trigger_min_m=defaults.get("gap_at_trigger_min_m"),
        gap_at_trigger_max_m=defaults.get("gap_at_trigger_max_m"),
        ego_speed_min_mps=defaults.get("ego_speed_min_mps"),
        ego_speed_max_mps=defaults.get("ego_speed_max_mps"),
        lead_speed_min_mps=defaults.get("lead_speed_min_mps"),
        lead_speed_max_mps=defaults.get("lead_speed_max_mps"),
        min_initial_gap_m=defaults.get("min_initial_gap_m", DEFAULT_MIN_INITIAL_GAP_M),
        distance_behind_ego_min_m=defaults.get("distance_behind_ego_min_m"),
        distance_behind_ego_max_m=defaults.get("distance_behind_ego_max_m"),
    )
    if run_dir is None:
        cfg["run"] = dict(cfg.get("run", {}))
        cfg["run"]["name"] = "phase2_short_front_gap_coverage"
        if headless:
            cfg.setdefault("sim", {})["headless"] = True
        run_dir = make_run_dir(cfg)
    run_short_front_gap_coverage(
        cfg,
        run_dir,
        param_list,
        headless=headless,
        distance_behind_ego_m=defaults["distance_behind_ego_m"],
        initial_x_m=defaults["initial_x_m"],
    )
    return run_dir


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Short front gap coverage (phase3_bandit_plan §3)")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Config YAML")
    parser.add_argument("--headless", action="store_true", help="No UI")
    parser.add_argument("--n_episodes", type=int, default=None, help="Cap or number of episodes")
    parser.add_argument("--mode", choices=["gap_at_trigger", "grid", "sample"], default=None,
                        help="gap_at_trigger = ego close to lead when ODD hits (recommended)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = run_from_config(
        cfg,
        headless=args.headless,
        n_episodes=args.n_episodes,
        mode=args.mode,
        seed=args.seed,
    )
    print("Done. Logs and short_front_gap_summary.csv saved in:", run_dir)
