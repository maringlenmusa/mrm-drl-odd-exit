"""
Phase 1 vehicle behavior plot.

Reads steps.csv from a Phase 1 run and generates a plot for documentation:
- Ego speed vs time (with ODD exit / MRM region shaded)
- Throttle, brake, steer vs time

Use for docs/phase1_verification.md and result proof (To_Do.md Phase 1).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_steps(run_dir: Path) -> pd.DataFrame:
    """Load steps.csv from run directory."""
    path = run_dir / "steps.csv" if run_dir.is_dir() else Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"steps.csv not found: {path}")
    df = pd.read_csv(path)
    for col in ("sim_time_s", "odd_exit_active", "ego_speed_mps", "throttle", "brake", "steer"):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {path.name}")
    df["odd_exit_active"] = df["odd_exit_active"].astype(bool)
    return df


def plot_phase1_behavior(
    df: pd.DataFrame,
    out_path: Path,
    title: str | None = None,
    run_name: str | None = None,
) -> None:
    """Generate Phase 1 vehicle behavior figure."""
    t = df["sim_time_s"].values
    speed = df["ego_speed_mps"].values
    odd = df["odd_exit_active"].values
    throttle = df["throttle"].values
    brake = df["brake"].values
    steer = df["steer"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5), height_ratios=[1.2, 1])
    fig.subplots_adjust(hspace=0.08)

    # ---- Speed ----
    ax1.plot(t, speed, color="C0", linewidth=1.5, label="Ego speed (m/s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Shade ODD exit / MRM region
    if odd.any():
        idx = np.where(odd)[0]
        t_start, t_end = t[idx[0]], t[idx[-1]]
        ax1.axvspan(t_start, t_end, alpha=0.2, color="orange", label="ODD exit / MRM")
    ax1.set_title(title or "Phase 1: Vehicle behavior")

    # ---- Controls ----
    ax2.plot(t, throttle, color="green", linewidth=1, alpha=0.9, label="Throttle")
    ax2.plot(t, brake, color="red", linewidth=1, alpha=0.9, label="Brake")
    ax2.plot(t, steer, color="gray", linewidth=0.8, alpha=0.7, label="Steer")
    ax2.set_ylabel("Control")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", ncol=3)

    if run_name:
        fig.suptitle(run_name, fontsize=9, y=1.02, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 1 vehicle behavior from steps.csv")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Path to run directory (contains steps.csv) or to steps.csv. "
        "Default: src/phase1/results_docs/2026-02-02_08-54-00_phase1_smoke_test - 5 second",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for figure. Default: docs/figures/phase1_vehicle_behavior.png",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title override.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.run_dir is None:
        run_dir = root / "src" / "phase1" / "results_docs" / "2026-02-02_08-54-00_phase1_smoke_test - 5 second"
    else:
        run_dir = Path(args.run_dir)
    if run_dir.suffix == ".csv":
        run_dir = run_dir.parent
        steps_path = run_dir / "steps.csv"
    else:
        steps_path = run_dir / "steps.csv"

    if not steps_path.exists():
        print(f"Error: {steps_path} not found.")
        return

    df = load_steps(run_dir)
    run_name = run_dir.name if run_dir.is_dir() else run_dir.parent.name

    out_path = Path(args.output) if args.output else root / "docs" / "figures" / "phase1_vehicle_behavior.png"
    out_path = out_path if out_path.is_absolute() else root / out_path

    plot_phase1_behavior(df, out_path, title=args.title, run_name=run_name)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
