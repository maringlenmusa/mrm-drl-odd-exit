"""
Fair comparison: RL vs Phase 2 baseline on SAME scenario regimes.

Phase 2 is run separately on:
  1. baseline_comparison  (large gap, easy — same as phase2_baseline)
  2. short_front_gap      (tight gap, hard — same as phase2_short_front_gap)

RL eval is run on a 50/50 mix of both.
We then compare:
  - RL (full mix) vs Phase2 (full mix = baseline + short_gap combined)
  - RL vs Phase2 within the easy regime only
  - RL vs Phase2 within the hard regime only

Episode regime identification:
  - Easy (baseline): min_gap at ODD > 20 m  OR  last_sim_time_s >> trigger  AND gap is large
  - Hard (short-gap): min_gap at ODD <= 20 m at ODD exit
  We use min_gap column (min front_gap_m after ODD exit) as the discriminator.
  Episodes with min_gap < 20 m = short-front-gap regime.
"""

import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRIGGER_S = 7.0
SHORT_GAP_THRESHOLD_M = 20.0   # min_gap < this = short-front-gap regime episode

# Physics constants for inevitability check
MAX_DECEL_MS2 = 4.5   # STRAIGHT_STOP: full brake × 4.5 m/s²
BB_DIST_M     = 5.0   # esmini collision distance (bounding box centre-to-centre ≈ 5 m)
                       # calibrated: ego 5 m long (centre+1.4 m front), front car ~4.5 m


def _is_inevitable(row, run_dir: str) -> bool:
    """
    Return True if the collision was physically inevitable — i.e. even with
    STRAIGHT_STOP applied from the very first ODD-active step, the ego cannot
    stop before the front car's bounding box (effective gap = gap - BB_DIST_M).

    Formula:  effective_gap < v_rel² / (2 × MAX_DECEL)
    where effective_gap = front_gap_at_ODD - BB_DIST_M
          v_rel         = actual closing speed at ODD exit (from gap change, dt=0.05 s)
    """
    # episodes with no step file or no ODD step logged are treated as inevitable
    # (collision fired before any ODD-active state was received)
    ep_id = row.get("episode_id", "")
    step_file = os.path.join(run_dir, "steps_episode_%s.csv" % ep_id)
    if not os.path.isfile(step_file):
        return True
    try:
        steps = list(csv.DictReader(open(step_file)))
        odd_steps = [s for s in steps if s.get("odd_exit_active", "").lower() in ("true", "1")]
        if len(odd_steps) < 2:
            return True   # no gap data → inevitable
        gap0  = float(odd_steps[0]["front_gap_m"])
        gap1  = float(odd_steps[1]["front_gap_m"])
        v_rel = (gap0 - gap1) / 0.05    # closing speed in m/s (positive = closing)
        if v_rel <= 0:
            return False  # not closing → avoidable
        eff_gap   = gap0 - BB_DIST_M
        stop_dist = (v_rel ** 2) / (2 * MAX_DECEL_MS2)
        return eff_gap <= stop_dist      # True = inevitable
    except Exception:
        return True


def valid_episodes(rows, run_dir: str = ""):
    """
    Only episodes where ODD fired AND the agent had a realistic chance to act.

    Excluded:
      1. Episode ended before ODD trigger (sim_time < trigger_s) — before-ODD crash.
      2. Collision that was physically inevitable: effective_gap ≤ stop_dist at ODD exit
         (gap_to_front_car - 5 m bounding box ≤ v_rel²/(2×4.5)).
         These episodes end in a crash regardless of which MRM the agent picks.
    """
    result = []
    for r in rows:
        t = float(r.get("last_sim_time_s") or 0)
        if t < TRIGGER_S:
            continue   # before ODD
        if r.get("collision") in ("True", True, "true"):
            if _is_inevitable(r, run_dir):
                continue   # physically unavoidable — not the agent's fault
        result.append(r)
    return result


def regime(row):
    """'hard' if min_gap was tight at ODD exit; 'easy' otherwise."""
    v = row.get("min_gap")
    if v in (None, "", "None"):
        # min_gap=None means episode ended before any ODD-active step was logged.
        # These are the hardest episodes (front car was already too close).
        return "hard"
    try:
        return "hard" if float(v) < SHORT_GAP_THRESHOLD_M else "easy"
    except ValueError:
        return "hard"


def pct(n, d):
    return 100 * n / d if d else 0.0


def stats(rows, label, show_regime=True, run_dir=""):
    total = len(rows)
    valid = valid_episodes(rows, run_dir)
    n_v = len(valid)
    before_odd = total - n_v

    mrc  = sum(1 for r in valid if r.get("success_mrc") in ("True", True, "true"))
    coll = sum(1 for r in valid if r.get("collision") in ("True", True, "true"))
    mrm0 = sum(1 for r in valid if str(r.get("primary_mrm")) == "0")
    mrm1 = sum(1 for r in valid if str(r.get("primary_mrm")) == "1")
    mrm2 = sum(1 for r in valid if str(r.get("primary_mrm")) == "2")

    sh_av   = [r for r in valid if r.get("shoulder_available") in ("True", True, "true")]
    sh_used = [r for r in sh_av if str(r.get("primary_mrm")) == "2"]

    easy_v = [r for r in valid if regime(r) == "easy"]
    hard_v = [r for r in valid if regime(r) == "hard"]
    easy_mrc = sum(1 for r in easy_v if r.get("success_mrc") in ("True", True, "true"))
    hard_mrc = sum(1 for r in hard_v if r.get("success_mrc") in ("True", True, "true"))

    print("=" * 62)
    print("  %s" % label)
    print("=" * 62)
    print("  Total episodes             : %d" % total)
    print("  Before-ODD collisions      : %d (%.0f%%) -- not agent's fault, excluded" % (
        before_odd, pct(before_odd, total)))
    print("  Valid (ODD fired)          : %d" % n_v)
    print()
    print("  MRC rate (all valid)       : %d/%d = %.1f%%   KEY METRIC" % (mrc, n_v, pct(mrc, n_v)))
    print("  Collision (all valid)      : %d/%d = %.1f%%" % (coll, n_v, pct(coll, n_v)))

    if show_regime and (easy_v or hard_v):
        print()
        print("  -- By scenario regime (after ODD) --")
        print("  Easy (min_gap > 20m)  : %d ep  MRC=%d (%.0f%%)" % (
            len(easy_v), easy_mrc, pct(easy_mrc, len(easy_v))))
        print("  Hard (min_gap <= 20m) : %d ep  MRC=%d (%.0f%%)" % (
            len(hard_v), hard_mrc, pct(hard_mrc, len(hard_v))))

    print()
    print("  MRM choices (valid):")
    print("    STRAIGHT_STOP  (0) : %d (%.0f%%)" % (mrm0, pct(mrm0, n_v)))
    print("    IN_LANE_STOP   (1) : %d (%.0f%%)" % (mrm1, pct(mrm1, n_v)))
    print("    ROAD_SHOULDER  (2) : %d (%.0f%%)" % (mrm2, pct(mrm2, n_v)))

    if sh_av:
        print("  Shoulder correct       : %d/%d = %.0f%%" % (
            len(sh_used), len(sh_av), pct(len(sh_used), len(sh_av))))
    print()

    return {
        "n_total": total, "n_valid": n_v, "n_before_odd": before_odd,
        "mrc_all": pct(mrc, n_v), "coll_all": pct(coll, n_v),
        "n_easy": len(easy_v), "mrc_easy": pct(easy_mrc, len(easy_v)),
        "n_hard": len(hard_v), "mrc_hard": pct(hard_mrc, len(hard_v)),
        "mrm0": pct(mrm0, n_v), "mrm1": pct(mrm1, n_v), "mrm2": pct(mrm2, n_v),
        "sh_correct": pct(len(sh_used), len(sh_av)) if sh_av else None,
    }


def find_latest(pattern):
    runs = glob.glob("logs/runs/" + pattern)
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def load_csv(run_dir):
    if not run_dir:
        return []
    p = os.path.join(run_dir, "episodes.csv")
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p)))


def main():
    rl_run       = find_latest("*_phase3_eval")
    bl_easy_run  = find_latest("*_phase2_baseline_comparison")
    bl_hard_run  = find_latest("*_phase2_short_front_gap_coverage")

    rl_rows      = load_csv(rl_run)
    bl_easy_rows = load_csv(bl_easy_run)
    bl_hard_rows = load_csv(bl_hard_run)
    bl_all_rows  = bl_easy_rows + bl_hard_rows

    print("\n[RL eval run     ]:", rl_run or "NOT FOUND")
    print("[Phase2 easy run ]:", bl_easy_run or "NOT FOUND")
    print("[Phase2 hard run ]:", bl_hard_run or "NOT FOUND")
    print()

    r_rl = stats(rl_rows,      "RL Policy (PPO, 25k eps)  — mixed distribution",
                 run_dir=rl_run or "") if rl_rows else None
    r_p2 = stats(bl_all_rows,  "Phase 2 Baseline (rule-based) — mixed distribution",
                 run_dir=bl_easy_run or "") if bl_all_rows else None

    if r_rl and r_p2:
        print("=" * 62)
        print("  FINAL COMPARISON TABLE (valid episodes, before-ODD excluded)")
        print("=" * 62)
        fmt = "  %-36s  %7s  %7s"
        print(fmt % ("Metric", "RL", "Baseline"))
        print("  " + "-" * 54)
        rows = [
            ("MRC rate — ALL scenarios (%)",   "mrc_all"),
            ("  of which: easy scenarios (%)", "mrc_easy"),
            ("  of which: hard scenarios (%)", "mrc_hard"),
            ("Collision rate (%)",             "coll_all"),
            ("STRAIGHT_STOP usage (%)",        "mrm0"),
            ("IN_LANE_STOP usage (%)",         "mrm1"),
            ("ROAD_SHOULDER usage (%)",        "mrm2"),
            ("Shoulder correct (%)",           "sh_correct"),
        ]
        for label, key in rows:
            v1 = r_rl.get(key)
            v2 = r_p2.get(key)
            s1 = "%.1f" % v1 if v1 is not None else " N/A"
            s2 = "%.1f" % v2 if v2 is not None else " N/A"
            print(fmt % (label, s1, s2))
        print()
        print("  Easy = min_gap > 20 m at ODD exit (large gap, comfortable)")
        print("  Hard = min_gap <= 20 m at ODD exit (tight gap, high risk)")


if __name__ == "__main__":
    main()
