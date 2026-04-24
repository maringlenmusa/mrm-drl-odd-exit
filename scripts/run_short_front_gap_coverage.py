"""
Short front gap coverage CLI (phase3_bandit_plan.md §3).
Runs many Phase 2 episodes with small initial distance to lead so straight/in-lane stop is exercised.

Usage:
  python scripts/run_short_front_gap_coverage.py [--config configs/phase2.yaml] [--n_episodes 20] [--headless]
  python -m src.main --phase phase2_short_front_gap --config configs/phase2.yaml [--headless]
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.phase2.run_short_front_gap_coverage import run_from_config


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Short front gap coverage: many episodes with small front gap (straight/in-lane stop)."
    )
    parser.add_argument("--config", default="configs/phase2.yaml", help="Config YAML")
    parser.add_argument("--headless", action="store_true", help="No UI")
    parser.add_argument("--n_episodes", type=int, default=None, help="Number of episodes (or cap for grid)")
    parser.add_argument("--mode", choices=["grid", "sample"], default=None, help="Param list mode")
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


if __name__ == "__main__":
    main()
