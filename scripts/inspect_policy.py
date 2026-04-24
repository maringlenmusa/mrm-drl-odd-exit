"""
Inspect a trained Phase 3 RL policy.

Usage:
    py -3 scripts/inspect_policy.py --policy logs/runs/.../artifacts/policy.zip
                                    --config configs/phase3_train.yaml
                                    [--n_episodes 20] [--ui]

What it shows:
  - Which MRM the agent picks at each ODD exit observation
  - MRC rate, collision rate, reward breakdown
  - Decision pattern: does the agent prefer shoulder when available?
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import load_config
from src.decision.mrm_catalog import mrm_name
from src.rl.env_odd_exit import OddExitEnv, OBS_FEATURES
from src.rl.ablations import apply_ablations
from src.rl.policy import load_model
from src.rl.episode_variation import Phase2StyleVariationEnv


def run_inspection(policy_path: str, cfg: dict, n_episodes: int, show_ui: bool):
    if show_ui:
        cfg["sim"]["esmini_lib_show_ui"] = True
        cfg["sim"]["headless"] = False
    else:
        cfg["sim"]["esmini_lib_show_ui"] = False
        cfg["sim"]["headless"] = True

    flags = apply_ablations(cfg)
    env = Phase2StyleVariationEnv(
        OddExitEnv(cfg, ablations=flags),
        cfg=cfg,
        seed=cfg.get("run", {}).get("seed"),
    )
    model = load_model(policy_path, env=env)

    print(f"\n[Inspect] Policy: {policy_path}")
    print(f"[Inspect] Running {n_episodes} episodes ...\n")

    results = []
    for ep in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs_at_odd_exit = obs.copy()

        # Ask policy what it would do at this ODD exit state
        action, _ = model.predict(obs, deterministic=True)
        proposed = int(action)

        # Step env (one_shot: runs to completion)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs_final, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs_final, reward, done, info = step_out

        final_mrm = info.get("final_mrm", proposed)
        mrc       = info.get("mrc_reached", False)
        collision = info.get("collision", False)
        shoulder  = info.get("shoulder_available", False)
        risk      = info.get("risk_total", 0.0)

        # Decode obs features at ODD exit
        obs_dict = {OBS_FEATURES[i]: float(obs_at_odd_exit[i]) for i in range(len(OBS_FEATURES))}

        # Denormalise key features for readability
        ego_speed  = obs_dict["ego_speed_mps"]  * cfg.get("risk",{}).get("max_speed_mps", 40.0)
        front_gap  = obs_dict["front_gap_m"]    * 50.0
        rear_gap   = obs_dict["rear_gap_m"]     * 50.0
        right_gap  = obs_dict["right_gap_m"]    * 50.0
        risk_total = obs_dict["risk_total"]

        results.append({
            "ep": ep + 1,
            "ego_speed": round(ego_speed, 1),
            "front_gap": round(front_gap, 1),
            "rear_gap":  round(rear_gap, 1),
            "right_gap": round(right_gap, 1),
            "risk_total": round(risk_total, 3),
            "shoulder": shoulder,
            "proposed": proposed,
            "final_mrm": final_mrm,
            "mrc": mrc,
            "collision": collision,
            "reward": round(reward, 2),
            "sim_time_s": float(info.get("sim_time_s", 0.0)),
        })

        status = "MRC    " if mrc else "COLLISION" if collision else "timeout"
        shield_note = ("→" + mrm_name(final_mrm)) if final_mrm != proposed else ""
        print(
            f"  ep {ep+1:3d}/{n_episodes}  "
            f"v={ego_speed:4.1f}m/s  fg={front_gap:5.1f}m  rg={rear_gap:5.1f}m  "
            f"risk={risk_total:.2f}  "
            f"shoulder={str(shoulder):5s}  "
            f"→ {mrm_name(proposed):20s}{shield_note:25s}  "
            f"[{status}]  r={reward:.2f}"
        )

    # --- Summary ---
    print("\n" + "="*70)
    print(f"POLICY INSPECTION SUMMARY  ({n_episodes} episodes)")
    print("="*70)

    n_mrc   = sum(1 for r in results if r["mrc"])
    n_coll  = sum(1 for r in results if r["collision"])
    n_to    = sum(1 for r in results if not r["mrc"] and not r["collision"])
    avg_r   = np.mean([r["reward"] for r in results])
    # Separate before-ODD collisions (agent had no chance to brake) from post-ODD.
    # In one_shot mode: if a collision happened from the approach, the inner loop
    # ends immediately → sim_time barely moved past trigger_time.
    # Threshold: if episode sim_time < trigger + 1.0s, collision was from approach.
    trigger_s = cfg.get("odd_exit", {}).get("trigger_time_s", 7.0)
    before_odd_coll = [r for r in results
                       if r["collision"] and r.get("sim_time_s", 99) < trigger_s + 1.0]
    post_odd_coll   = [r for r in results
                       if r["collision"] and r.get("sim_time_s", 99) >= trigger_s + 1.0]
    valid = [r for r in results if not (r["collision"] and r.get("sim_time_s", 99) < trigger_s + 1.0)]

    print(f"  Total episodes       : {n_episodes}")
    print(f"  Before-ODD collision : {len(before_odd_coll)} ({100*len(before_odd_coll)//n_episodes}%)"
          "  ← approach too tight, agent had no chance")
    print(f"  ─────── episodes where agent actually acted ───────────────")
    v = len(valid)
    v_mrc  = sum(1 for r in valid if r["mrc"])
    v_coll = sum(1 for r in valid if r["collision"])
    v_to   = sum(1 for r in valid if not r["mrc"] and not r["collision"])
    v_avg  = np.mean([r["reward"] for r in valid]) if valid else 0
    print(f"  Valid episodes       : {v}")
    print(f"  MRC rate (REAL)      : {v_mrc}/{v} ({100*v_mrc//v if v else 0}%)  ← use this for 'did it learn?'")
    print(f"  Post-ODD collision   : {v_coll}/{v} ({100*v_coll//v if v else 0}%)")
    print(f"  Timeout              : {v_to}/{v} ({100*v_to//v if v else 0}%)")
    print(f"  Avg reward (valid)   : {v_avg:.2f}")
    print(f"  Avg reward (all)     : {avg_r:.2f}")

    print("\n  MRM decision breakdown:")
    mrm_cnt = Counter(mrm_name(r["final_mrm"]) for r in results)
    for k, v in sorted(mrm_cnt.items(), key=lambda x: -x[1]):
        print(f"    {k:25s}: {v:3d} ({100*v//n_episodes}%)")

    print("\n  When shoulder_available=True:")
    sh_true = [r for r in results if r["shoulder"]]
    if sh_true:
        cnt = Counter(mrm_name(r["final_mrm"]) for r in sh_true)
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"    {k:25s}: {v:3d}/{len(sh_true)} ({100*v//len(sh_true)}%)")
    else:
        print("    (no shoulder-available episodes)")

    print("\n  When shoulder_available=False:")
    sh_false = [r for r in results if not r["shoulder"]]
    if sh_false:
        cnt = Counter(mrm_name(r["final_mrm"]) for r in sh_false)
        for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"    {k:25s}: {v:3d}/{len(sh_false)} ({100*v//len(sh_false)}%)")
    else:
        print("    (no shoulder-unavailable episodes)")

    print("\n  The ideal policy would:")
    print("    shoulder=True  → ROAD_SHOULDER_STOP  (best reward: +15)")
    print("    shoulder=False → IN_LANE_STOP or STRAIGHT_STOP  (reward: +8–10)")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect trained Phase 3 RL policy")
    parser.add_argument("--policy", required=True, help="Path to policy.zip")
    parser.add_argument("--config", default="configs/phase3_train.yaml",
                        help="Config YAML (default: configs/phase3_train.yaml)")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of episodes to run (default: 20)")
    parser.add_argument("--ui", action="store_true",
                        help="Show esmini UI while running (headless by default)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inspection(args.policy, cfg, args.n_episodes, show_ui=args.ui)


if __name__ == "__main__":
    main()
