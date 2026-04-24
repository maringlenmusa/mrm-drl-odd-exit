"""
Phase 2 baseline comparison CLI: vary ego speed, front speed, rear distance; fixed ODD trigger.
Writes baseline_results.csv for Phase 3 comparison.

Usage:
  python scripts/run_baseline_comparison.py [--config configs/phase2.yaml] [--n 50] [--headless]
  python -m src.main --phase phase2_baseline --config configs/phase2.yaml [--headless]
"""

import os
import sys

# Project root for imports when run as scripts/run_baseline_comparison.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.logger import make_run_dir
from src.phase2.run_baseline_comparison import (
    build_param_list,
    get_baseline_defaults,
    run_baseline_comparison,
    DEFAULT_N_EPISODES,
    DEFAULT_SPEED_MIN,
    DEFAULT_SPEED_MAX,
    DEFAULT_DISTANCE_MIN,
    DEFAULT_DISTANCE_MAX,
    DEFAULT_TRIGGER_TIME_S,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 2 baseline comparison: vary ego speed, front speed, rear distance; fixed ODD trigger."
    )
    parser.add_argument(
        "--config",
        default="configs/phase2.yaml",
        help="Path to Phase 2 config YAML",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Number of episodes (default: from config baseline_comparison.n_episodes or {DEFAULT_N_EPISODES})",
    )
    parser.add_argument(
        "--mode",
        choices=("grid", "sample"),
        default="grid",
        help="grid = discrete steps per dimension; sample = random n combinations",
    )
    parser.add_argument(
        "--speed-min",
        type=float,
        default=None,
        help=f"Ego/front speed min (m/s), default {DEFAULT_SPEED_MIN}",
    )
    parser.add_argument(
        "--speed-max",
        type=float,
        default=None,
        help=f"Ego/front speed max (m/s), default {DEFAULT_SPEED_MAX}",
    )
    parser.add_argument(
        "--distance-min",
        type=float,
        default=None,
        help=f"Distance behind ego min (m), default {DEFAULT_DISTANCE_MIN}",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=None,
        help=f"Distance behind ego max (m), default {DEFAULT_DISTANCE_MAX}",
    )
    parser.add_argument(
        "--trigger-time",
        type=float,
        default=None,
        help=f"ODD exit trigger time (s), fixed for all episodes. Default from config or {DEFAULT_TRIGGER_TIME_S}",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --mode sample (or grid with n_cap)",
    )
    parser.add_argument(
        "--speed-steps",
        type=int,
        default=4,
        help="Number of speed values per dimension in grid (default 4)",
    )
    parser.add_argument(
        "--distance-steps",
        type=int,
        default=4,
        help="Number of distance_behind_ego values in grid (default 4)",
    )
    parser.add_argument(
        "--no-primary-mrm",
        action="store_true",
        help="Do not compute primary_mrm from steps CSV",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = get_baseline_defaults(cfg)

    speed_min = args.speed_min if args.speed_min is not None else defaults["speed_min"]
    speed_max = args.speed_max if args.speed_max is not None else defaults["speed_max"]
    distance_min = args.distance_min if args.distance_min is not None else defaults["distance_min"]
    distance_max = args.distance_max if args.distance_max is not None else defaults["distance_max"]
    trigger_time_s = args.trigger_time if args.trigger_time is not None else defaults["trigger_time_s"]
    n_episodes = args.n if args.n is not None else defaults["n_episodes"]
    speed_steps = args.speed_steps
    distance_steps = args.distance_steps

    param_list = build_param_list(
        mode=args.mode,
        speed_min=speed_min,
        speed_max=speed_max,
        distance_min=distance_min,
        distance_max=distance_max,
        n_episodes=n_episodes,
        speed_steps=speed_steps,
        distance_steps=distance_steps,
        seed=args.seed,
    )

    cfg = __import__("copy").deepcopy(cfg)
    cfg["run"] = dict(cfg.get("run", {}))
    cfg["run"]["name"] = "phase2_baseline_comparison"
    if args.headless:
        cfg.setdefault("sim", {})["headless"] = True

    run_dir = make_run_dir(cfg)
    print(f"Baseline comparison: {len(param_list)} episodes (trigger_time_s={trigger_time_s}, headless={args.headless})")
    print(f"Speed range: {speed_min}–{speed_max} m/s, distance behind ego: {distance_min}–{distance_max} m")
    print(f"Logs and baseline_results.csv -> {run_dir}")

    run_baseline_comparison(
        cfg,
        run_dir,
        param_list,
        trigger_time_s,
        headless=args.headless,
        write_primary_mrm=not args.no_primary_mrm,
    )
    print("Done. Baseline table:", os.path.join(run_dir, "baseline_results.csv"))


if __name__ == "__main__":
    main()
