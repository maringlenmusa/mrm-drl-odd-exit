"""
Logging utilities for the ODD exit RL project.

Phase 1: steps.csv (basic), episode.json.
Phase 2+: steps.csv with obs/risk/MRM columns, episodes.csv (one row per episode), episode.json.
"""

import csv
import json
import os
from datetime import datetime

# Phase 1 step columns (backward compatible)
STEP_FIELDS_PHASE1 = [
    "step_id",
    "sim_time_s",
    "odd_exit_active",
    "ego_speed_mps",
    "throttle",
    "brake",
    "steer",
    "collision",
]

# Phase 2 additional step columns (gaps, risk, proposed/final MRM, provenance)
STEP_FIELDS_PHASE2_EXTRA = [
    "state_source",  # "esmini" = logged from esmini UDP; "state_bridge" = synthetic (our code)
    "front_gap_m",
    "rear_gap_m",
    "left_gap_m",
    "right_gap_m",
    "drf",
    "dara",
    "risk_total",
    "proposed_mrm",
    "final_mrm",
    # Phase 3 RL: per-step reward breakdown
    "step_reward",              # total reward this step
    "step_reward_risk_penalty", # dense risk component (-risk_weight * risk_total)
    "step_reward_step_cost",    # fixed step cost (-step_cost)
    "step_reward_terminal",     # terminal component (non-zero only at episode end)
]

EPISODE_CSV_FIELDS = [
    "episode_id",
    "state_source",  # "esmini" or "state_bridge" — whether step state was from esmini or synthetic
    "success_mrc",
    "collision",
    "steps",
    "max_risk",
    "avg_risk",
    "mrm_switches",   # number of times MRM changed during episode (0 = stayed on one)
    "primary_mrm",    # MRM id used: 0=straight/brake, 1=in-lane stop, etc. (-1 if none)
    "last_sim_time_s",
    # Phase 3 bandit plan §2: episode-level fields for reward and analysis
    "shoulder_available",  # True when rear_gap_m > min_rear_gap_for_shoulder_m at ODD exit
    "min_gap",            # min front_gap_m during stop (for near-miss penalty)
    "time_to_stop",       # time from ODD exit until speed below stop threshold (s)
    "time_to_mrc",        # time from ODD exit until MRC reached (s)
    "max_decel",          # max deceleration over episode (m/s²)
    # Phase 3 RL: episode-level reward breakdown (so you can see what drove each outcome)
    "total_reward",                    # sum of all step rewards for this episode
    "reward_dense",                    # total dense penalty (risk + step costs, all steps)
    "reward_terminal",                 # terminal component only (collision/MRC/shoulder/lane)
    "reward_collision_penalty",        # 0 or -collision_penalty
    "reward_success_bonus",            # 0 or +success_bonus
    "reward_shoulder_bonus",           # 0 or +shoulder_bonus
    "reward_lane_block_penalty",       # 0 or -lane_block_penalty
]


def make_run_dir(cfg: dict) -> str:
    """
    Create a unique run directory for logging.

    Args:
        cfg: Configuration dictionary containing run name

    Returns:
        Path to the created run directory
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg["run"]["name"]
    run_dir = f"logs/runs/{ts}_{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class StepCsvLogger:
    """Logger for writing step-by-step simulation data to CSV. Phase 1 or Phase 2 columns."""

    def __init__(self, run_dir: str, use_phase2: bool = False, filename_suffix: str = ""):
        """
        Initialize the CSV logger.

        Args:
            run_dir: Directory where logs will be written
            use_phase2: If True, include Phase 2 columns (gaps, drf, dara, risk_total, proposed_mrm, final_mrm)
            filename_suffix: Optional suffix for filename, e.g. "_episode_0" -> steps_episode_0.csv
        """
        base = "steps" + (filename_suffix if filename_suffix else "")
        self.path = f"{run_dir}/{base}.csv"
        self.file = open(self.path, "w", newline="")
        fieldnames = STEP_FIELDS_PHASE1 + (STEP_FIELDS_PHASE2_EXTRA if use_phase2 else [])
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def log_step(self, row: dict):
        """
        Write a single step to the CSV file.

        Args:
            row: Dictionary containing step data (extra keys ignored for Phase 1)
        """
        self.writer.writerow(row)

    def close(self):
        """Close the CSV file."""
        self.file.close()


class EpisodeCsvLogger:
    """Logger for writing one row per episode to episodes.csv (Phase 2+)."""

    def __init__(self, run_dir: str, mode: str = "w"):
        """
        Initialize the episode CSV logger.

        Args:
            run_dir: Directory where episodes.csv will be written
            mode: "w" = write (new file, header); "a" = append (no header, for multi-episode runs)
        """
        self.path = f"{run_dir}/episodes.csv"
        self.file = open(self.path, mode, newline="")
        self._mode = mode
        self.writer = csv.DictWriter(self.file, fieldnames=EPISODE_CSV_FIELDS, extrasaction="ignore")
        if mode == "w":
            self.writer.writeheader()
        self._episode_id = 0

    def log_episode(self, row: dict):
        """
        Write one episode row. Adds episode_id if missing.

        Args:
            row: Dict with success_mrc, collision, steps, max_risk, avg_risk, mrm_switches, last_sim_time_s, etc.
        """
        self._episode_id += 1
        out = dict(row)
        if "episode_id" not in out:
            out["episode_id"] = self._episode_id
        self.writer.writerow(out)
        self.file.flush()  # flush immediately so episodes.csv is visible during training

    def close(self):
        """Close the CSV file."""
        self.file.close()


def write_episode_summary(run_dir: str, summary: dict):
    """
    Write episode summary to JSON file.
    
    Args:
        run_dir: Directory where logs are written
        summary: Dictionary containing episode summary data
    """
    path = f"{run_dir}/episode.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
