"""
Phase 3 RL evaluation.

Runs a trained policy on N test episodes and produces:
  - logs/runs/<timestamp>_phase3_eval/episodes.csv   (one row per episode)
  - logs/runs/<timestamp>_phase3_eval/steps_episode_N.csv  (sim steps)
  - logs/runs/<timestamp>_phase3_eval/eval_summary.json    (structured metrics)
  - logs/runs/<timestamp>_phase3_eval/eval_summary.csv     (one-row metrics table)

Usage via main:
    py -3 -u -m src.main --phase phase3_eval --config configs/phase3_eval.yaml

Usage directly:
    from src.rl.evaluate import evaluate
    evaluate(cfg, run_dir)
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from src.logger import make_run_dir, EpisodeCsvLogger
from src.rl.ablations import apply_ablations
from src.rl.env_odd_exit import OddExitEnv
from src.rl.episode_variation import Phase2StyleVariationEnv
from src.rl.policy import load_model
from src.decision.mrm_catalog import mrm_name


# Fields written to eval_summary.csv (one row = one eval run).
EVAL_SUMMARY_FIELDS = [
    "policy_path",
    "n_episodes",
    # --- overall (including before-ODD collisions) ---
    "mrc_rate_overall",
    "collision_rate_overall",
    "before_odd_collision_rate",
    # --- valid episodes only (ODD fired, agent had a chance to act) ---
    "n_valid",
    "mrc_rate_valid",
    "collision_rate_valid",
    "timeout_rate_valid",
    "avg_reward_valid",
    "avg_reward_overall",
    # --- MRM distribution (valid episodes) ---
    "mrm0_rate_valid",   # STRAIGHT_STOP
    "mrm1_rate_valid",   # IN_LANE_STOP
    "mrm2_rate_valid",   # ROAD_SHOULDER_STOP
    # --- shoulder decision quality ---
    "shoulder_correct_rate",  # when shoulder=True → agent picked MRM2
    "shoulder_wrong_rate",    # when shoulder=True → agent did NOT pick MRM2
    # --- risk stats ---
    "avg_max_risk",
    "avg_avg_risk",
    # --- timing ---
    "avg_time_to_mrc",
]


def evaluate(cfg: dict, run_dir: Optional[str] = None) -> str:
    """
    Evaluate a trained RL policy on N test episodes.

    Reads:
      - cfg["evaluation"]["policy_path"]   path to policy.zip
      - cfg["evaluation"]["n_episodes"]    number of test episodes
      - cfg["evaluation"]["deterministic"] True = always pick best action (default True)
      - all other sim/env config from cfg (same as training)

    Returns:
      run_dir where results were saved.
    """
    if run_dir is None:
        cfg = dict(cfg)
        cfg.setdefault("run", {})
        cfg["run"]["name"] = cfg["run"].get("name", "phase3_eval")
        run_dir = make_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)

    ev = cfg.get("evaluation", {})
    policy_path = str(ev.get("policy_path", ""))
    n_episodes = int(ev.get("n_episodes", 100))
    deterministic = bool(ev.get("deterministic", True))
    trigger_s = float(cfg.get("odd_exit", {}).get("trigger_time_s", 7.0))

    if not policy_path or not os.path.isfile(policy_path):
        raise FileNotFoundError(
            f"policy_path not found: '{policy_path}'. "
            "Set evaluation.policy_path in the eval config."
        )

    # headless by default for evaluation (set esmini_lib_show_ui: true to watch)
    show_ui = bool(cfg.get("sim", {}).get("esmini_lib_show_ui", False))
    if not show_ui:
        cfg.setdefault("sim", {})["headless"] = True
        cfg["sim"]["esmini_lib_show_ui"] = False

    flags = apply_ablations(cfg)
    seed = cfg.get("run", {}).get("seed")

    # Shared episode logger so all episodes go into one episodes.csv
    episode_logger = EpisodeCsvLogger(run_dir, mode="w")

    base_env = OddExitEnv(
        cfg,
        run_dir=run_dir,
        episode_csv_logger=episode_logger,
        ablations=flags,
    )
    env = Phase2StyleVariationEnv(base_env, cfg=cfg, seed=seed)
    model = load_model(policy_path, env=env)

    print(f"\n[Phase3 Eval] Policy  : {policy_path}")
    print(f"[Phase3 Eval] Episodes: {n_episodes}  deterministic={deterministic}")
    print(f"[Phase3 Eval] Logs    : {run_dir}\n")

    rows: List[Dict] = []

    for ep in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        action, _ = model.predict(obs, deterministic=deterministic)

        step_out = env.step(int(action))
        if len(step_out) == 5:
            _, reward, terminated, truncated, info = step_out
        else:
            _, reward, terminated, truncated, info = *step_out, False

        sim_time = float(info.get("sim_time_s", 0.0))
        mrc      = bool(info.get("mrc_reached", False))
        coll     = bool(info.get("collision", False))
        timeout  = bool(info.get("timeout", False))
        final_mrm = int(info.get("final_mrm", -1))
        shoulder = bool(info.get("shoulder_available", False))
        max_risk = float(info.get("max_risk", 0.0))
        avg_risk = float(info.get("avg_risk", 0.0))

        before_odd = coll and sim_time < trigger_s + 1.0
        valid = not before_odd

        outcome = "MRC" if mrc else ("BEFORE-ODD-COLL" if before_odd else "COLLISION" if coll else "timeout")
        print(
            "  ep %3d/%d  shoulder=%-5s  -> %-22s  [%-14s]  r=%7.2f" % (
                ep+1, n_episodes, str(shoulder), mrm_name(final_mrm), outcome, reward
            )
        )

        rows.append({
            "ep": ep + 1,
            "sim_time_s": sim_time,
            "mrc": mrc,
            "collision": coll,
            "before_odd_collision": before_odd,
            "valid": valid,
            "timeout": timeout,
            "final_mrm": final_mrm,
            "shoulder": shoulder,
            "reward": round(reward, 4),
            "max_risk": round(max_risk, 4),
            "avg_risk": round(avg_risk, 4),
        })

    episode_logger.close()
    env.close()

    # ------------------------------------------------------------------
    # Compute summary metrics
    # ------------------------------------------------------------------
    summary = _compute_summary(rows, n_episodes, policy_path, trigger_s)

    # Write eval_summary.json
    json_path = os.path.join(run_dir, "eval_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write eval_summary.csv (single row — easy to paste into thesis table)
    csv_path = os.path.join(run_dir, "eval_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in EVAL_SUMMARY_FIELDS})

    _print_summary(summary)
    print(f"\n[Phase3 Eval] eval_summary.json : {json_path}")
    print(f"[Phase3 Eval] eval_summary.csv  : {csv_path}")
    print(f"[Phase3 Eval] episodes.csv       : {os.path.join(run_dir, 'episodes.csv')}")
    return run_dir


def _compute_summary(rows: List[Dict], n_episodes: int, policy_path: str,
                     trigger_s: float) -> Dict:
    valid    = [r for r in rows if r["valid"]]
    before   = [r for r in rows if r["before_odd_collision"]]
    mrc_all  = [r for r in rows if r["mrc"]]
    coll_all = [r for r in rows if r["collision"]]

    n_valid  = len(valid)
    v_mrc    = [r for r in valid if r["mrc"]]
    v_coll   = [r for r in valid if r["collision"]]
    v_to     = [r for r in valid if r["timeout"]]
    v_reward = [r["reward"] for r in valid]

    mrm_cnt = Counter(r["final_mrm"] for r in valid)

    sh_true      = [r for r in valid if r["shoulder"]]
    sh_correct   = [r for r in sh_true if r["final_mrm"] == 2]   # picked shoulder
    sh_wrong     = [r for r in sh_true if r["final_mrm"] != 2]

    time_to_mrc  = []
    for ep_id, row in enumerate(rows):
        if row["mrc"] and row["sim_time_s"] > trigger_s:
            time_to_mrc.append(row["sim_time_s"] - trigger_s)

    def pct(num, denom): return round(100 * num / denom, 1) if denom else 0.0
    def mean(lst):       return round(float(np.mean(lst)), 4) if lst else None

    return {
        "policy_path": policy_path,
        "n_episodes": n_episodes,
        "n_valid": n_valid,

        # overall rates (includes before-ODD collisions)
        "mrc_rate_overall":            pct(len(mrc_all),  n_episodes),
        "collision_rate_overall":      pct(len(coll_all), n_episodes),
        "before_odd_collision_rate":   pct(len(before),   n_episodes),

        # valid-episode rates (the real evaluation metric)
        "mrc_rate_valid":              pct(len(v_mrc),  n_valid),
        "collision_rate_valid":        pct(len(v_coll), n_valid),
        "timeout_rate_valid":          pct(len(v_to),   n_valid),
        "avg_reward_valid":            mean(v_reward),
        "avg_reward_overall":          mean([r["reward"] for r in rows]),

        # MRM distribution
        "mrm0_rate_valid": pct(mrm_cnt.get(0, 0), n_valid),
        "mrm1_rate_valid": pct(mrm_cnt.get(1, 0), n_valid),
        "mrm2_rate_valid": pct(mrm_cnt.get(2, 0), n_valid),

        # shoulder decision quality
        "n_shoulder_available": len(sh_true),
        "shoulder_correct_rate": pct(len(sh_correct), len(sh_true)) if sh_true else None,
        "shoulder_wrong_rate":   pct(len(sh_wrong),   len(sh_true)) if sh_true else None,

        # risk
        "avg_max_risk": mean([r["max_risk"] for r in valid]),
        "avg_avg_risk": mean([r["avg_risk"] for r in valid]),

        # timing
        "avg_time_to_mrc": mean(time_to_mrc),
    }


def _print_summary(s: Dict) -> None:
    n = s["n_episodes"]
    v = s["n_valid"]
    line = "=" * 62
    sep  = "-" * 54
    print("\n" + line)
    print("  EVALUATION SUMMARY   (%d episodes total, %d valid)" % (n, v))
    print(line)
    print("  Before-ODD collisions  : %5.1f%%  (agent had no chance)" % s["before_odd_collision_rate"])
    print("  %s" % sep)
    print("  Valid episodes (%d):" % v)
    print("  MRC rate    (REAL)     : %5.1f%%  <-- main metric" % s["mrc_rate_valid"])
    print("  Post-ODD collision     : %5.1f%%" % s["collision_rate_valid"])
    print("  Timeout                : %5.1f%%" % s["timeout_rate_valid"])
    print("  Avg reward (valid)     : %s" % s["avg_reward_valid"])
    print()
    print("  MRM distribution (valid episodes):")
    print("    STRAIGHT_STOP  (0)   : %5.1f%%" % s["mrm0_rate_valid"])
    print("    IN_LANE_STOP   (1)   : %5.1f%%" % s["mrm1_rate_valid"])
    print("    ROAD_SHOULDER  (2)   : %5.1f%%" % s["mrm2_rate_valid"])
    if s["shoulder_correct_rate"] is not None:
        print()
        print("  Shoulder decision (%d ep where shoulder was available):" % s["n_shoulder_available"])
        print("    Correct (used shoulder)    : %5.1f%%" % s["shoulder_correct_rate"])
        print("    Wrong   (ignored shoulder) : %5.1f%%" % s["shoulder_wrong_rate"])
    if s["avg_time_to_mrc"] is not None:
        print()
        print("  Avg time from ODD exit to MRC  : %.2f s" % s["avg_time_to_mrc"])
    print("  Avg max_risk (valid)           : %s" % s["avg_max_risk"])
    print(line + "\n")
