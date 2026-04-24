"""
Main entry point for the ODD exit RL project.

Usage:
    python -m src.main --config configs/phase1.yaml
    python -m src.main --phase phase2 --config configs/phase2.yaml
    python -m src.main --phase phase2_baseline --config configs/phase2.yaml [--headless]
    python -m src.main --phase phase3_random --config configs/phase2.yaml
"""

import argparse
import csv
import os
import numpy as np
from src.config import load_config
from src.logger import make_run_dir
from src.phase1.run_phase1 import run_phase1
from src.phase2.run_phase2 import run_phase2, run_phase2_multi
from src.phase2.run_baseline_comparison import run_baseline_from_config
from src.phase2.run_short_front_gap_coverage import run_from_config as run_short_front_gap_from_config
from src.rl.train import train as run_phase3_train
from src.rl.evaluate import evaluate as run_phase3_eval


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ODD Exit RL - Phase 1, Phase 2 simulation, or Phase 3 RL"
    )
    parser.add_argument(
        "--phase",
        default="phase1",
        choices=[
            "phase1",
            "phase2",
            "phase2_baseline",
            "phase2_short_front_gap",
            "phase3_random",  # Phase 3: run N episodes with random policy to verify env
            "phase3_train",   # Phase 3: PPO training
            "phase3_eval",    # Phase 3: evaluate a trained policy
        ],
        help=(
            "Phase to run: phase1, phase2, phase2_baseline (comparison table), "
            "phase2_short_front_gap (short front gap coverage), "
            "phase3_random (RL env smoke-test with random actions), "
            "phase3_train (PPO training), "
            "phase3_eval (evaluate trained policy)"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/phase1.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Phase 2 only: run multiple episodes (scenario_sampler.scenarios × trigger_time_s_variations)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI (no esmini window). Overrides config sim.headless. Use for logs-only / bulk data."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.phase not in ("phase2_baseline", "phase2_short_front_gap"):
        if args.headless:
            cfg.setdefault("sim", {})["headless"] = True
        run_dir = make_run_dir(cfg)
    else:
        run_dir = None

    print(f"Starting {args.phase} run: {cfg['run']['name']}")
    if args.headless:
        print("Headless mode: no UI, logs only.")
    if args.phase == "phase2" and args.multi:
        print("Multi-episode mode: scenario_sampler.scenarios × trigger_time_s_variations")
    if args.phase == "phase2" and not args.headless:
        print("Design: Python = ODD/MRM + logging; esmini = scenario + physics. Same scenario runs in UI.")
        print("With state_source=esmini, state & collision come from esmini; with state_bridge, state is synthetic.")
    if args.phase == "phase2_baseline":
        run_dir = run_baseline_from_config(cfg, headless=args.headless)
        print("Done. Logs and baseline_results.csv saved in:", run_dir)
    elif args.phase == "phase2_short_front_gap":
        run_dir = run_short_front_gap_from_config(cfg, headless=args.headless)
        print("Done. Logs and short_front_gap_summary.csv saved in:", run_dir)
    elif args.phase == "phase2":
        if not run_dir:
            run_dir = make_run_dir(cfg)
        print(f"Logs will be saved to: {run_dir}")
        if args.multi:
            run_phase2_multi(cfg, run_dir)
        else:
            run_phase2(cfg, run_dir)
        print("Done. Logs saved in:", run_dir)
    elif args.phase == "phase3_random":
        _run_phase3_random(cfg, run_dir, headless=args.headless)
    elif args.phase == "phase3_train":
        run_phase3_train(cfg, run_dir=run_dir)
    elif args.phase == "phase3_eval":
        run_phase3_eval(cfg, run_dir=run_dir)
        print("Done. Eval results saved in:", run_dir)
    else:
        print(f"Logs will be saved to: {run_dir}")
        run_phase1(cfg, run_dir)
        print("Done. Logs saved in:", run_dir)


def _run_phase3_random(cfg: dict, run_dir: str, *, headless: bool = False):
    """
    Phase 3 smoke-test: run N episodes with a RANDOM policy using the same
    scenario-parameter variation as the Phase 2 baseline-comparison and
    short-front-gap-coverage scripts.

    Half the episodes are drawn from the "baseline" regime
    (ego 20–35 m/s, front 20–35 m/s, rear 12–48 m, trigger 7–8 s)
    and half from the "short front gap" regime
    (gap at ODD exit 15–30 m, ego 20–25 m/s, front 10–18 m/s, trigger 7 s)
    so we get a balanced mix of MRC, collision, and timeout outcomes.

    State always comes from esmini physics (state_source=esmini + generated xosc).
    """
    import copy
    import csv
    import random
    import numpy as np

    from src.rl.env_odd_exit import OddExitEnv
    from src.phase2.run_baseline_comparison import (
        build_param_list as build_baseline_params,
        get_baseline_defaults,
    )
    from src.phase2.run_short_front_gap_coverage import (
        build_param_list as build_sfg_params,
        get_short_front_gap_defaults,
    )

    # ------------------------------------------------------------------
    # Config prep
    # ------------------------------------------------------------------
    cfg = copy.deepcopy(cfg)
    if headless:
        cfg.setdefault("sim", {})["headless"] = True
        cfg["sim"]["esmini_lib_show_ui"] = False

    n_episodes = cfg.get("phase3", {}).get("random_episodes", 10)
    seed = cfg.get("run", {}).get("seed", 42)
    rng = random.Random(seed)

    print(f"\n[Phase 3] Random-policy smoke-test: {n_episodes} episodes")
    print(f"[Phase 3] Params: baseline + short-front-gap mix  |  seed={seed}")
    print(f"[Phase 3] Logs → {run_dir}")

    # ------------------------------------------------------------------
    # Generate parameter lists (same logic as Phase 2 scripts)
    # ------------------------------------------------------------------
    n_half = max(1, n_episodes // 2)
    n_rest = n_episodes - n_half          # may differ by 1 for odd counts

    # --- Baseline episodes: varied speeds + rear distance ---
    bl = get_baseline_defaults(cfg)
    bl_params = build_baseline_params(
        mode="sample",
        speed_min=bl["speed_min"],
        speed_max=bl["speed_max"],
        distance_min=bl["distance_min"],
        distance_max=bl["distance_max"],
        n_episodes=n_half,
        seed=seed,
    )
    # Each bl_params entry: (ego_speed, front_speed, distance_behind_ego_m)
    # Map to options dict
    bl_trigger = bl["trigger_time_s"]
    baseline_options = []
    for ego_spd, front_spd, dist_rear in bl_params:
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
        baseline_options.append(
            {
                "trigger_time_s": bl_trigger,
                "state_bridge": {
                    "initial_speed_mps": ego_spd,
                    "actors": actors,
                },
                "_meta": {
                    "regime": "baseline",
                    "initial_speed_mps": ego_spd,
                    "front_speed_mps": front_spd,
                    "distance_behind_ego_m": dist_rear,
                    "trigger_time_s": bl_trigger,
                },
            }
        )

    # --- Short-front-gap episodes: ego close to lead at ODD exit ---
    sfg = get_short_front_gap_defaults(cfg)
    sfg_params = build_sfg_params(
        mode="gap_at_trigger",
        front_gap_min_m=sfg["front_gap_min_m"],
        front_gap_max_m=sfg["front_gap_max_m"],
        speed_min_mps=sfg["speed_min_mps"],
        speed_max_mps=sfg["speed_max_mps"],
        trigger_times_s=sfg["trigger_times_s"],
        n_episodes=n_rest,
        seed=seed + 1 if seed is not None else None,
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
    # Each sfg_params entry: (initial_front_gap_m, v_ego, v_lead, trigger_s, dist_rear)
    initial_x_m = sfg["initial_x_m"]
    sfg_options = []
    for initial_gap, ego_spd, front_spd, trigger_s, dist_rear in sfg_params:
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
        sfg_options.append(
            {
                "trigger_time_s": trigger_s,
                "state_bridge": {
                    "initial_speed_mps": ego_spd,
                    "initial_x_m": initial_x_m,
                    "actors": actors,
                },
                "_meta": {
                    "regime": "short_front_gap",
                    "initial_speed_mps": ego_spd,
                    "front_speed_mps": front_spd,
                    "initial_front_gap_m": initial_gap,
                    "distance_behind_ego_m": dist_rear,
                    "trigger_time_s": trigger_s,
                },
            }
        )

    # Interleave baseline and short-front-gap episodes for variety
    all_options = []
    for i in range(max(len(baseline_options), len(sfg_options))):
        if i < len(baseline_options):
            all_options.append(baseline_options[i])
        if i < len(sfg_options):
            all_options.append(sfg_options[i])
    all_options = all_options[:n_episodes]

    # ------------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------------
    env = OddExitEnv(cfg, run_dir=run_dir)

    # Summary CSV (extends episodes.csv with scenario params)
    summary_fields = [
        "episode", "regime", "initial_speed_mps", "front_speed_mps",
        "initial_front_gap_m", "distance_behind_ego_m", "trigger_time_s",
        "steps", "total_reward", "collision", "mrc_reached", "timeout",
        "primary_mrm", "shoulder_available",
    ]
    summary_path = os.path.join(run_dir, "random_policy_summary.csv")
    os.makedirs(run_dir, exist_ok=True)
    summary_file = open(summary_path, "w", newline="")
    summary_writer = csv.DictWriter(
        summary_file, fieldnames=summary_fields, extrasaction="ignore"
    )
    summary_writer.writeheader()

    episode_results = []
    for ep_idx, ep_opts in enumerate(all_options):
        meta = ep_opts.pop("_meta", {})  # remove before passing to reset()

        result = env.reset(options=ep_opts)
        obs = result[0] if isinstance(result, tuple) else result

        total_reward = 0.0
        steps = 0
        done = False
        info = {}
        while not done:
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            total_reward += reward
            steps += 1

        status = (
            "COLLISION" if info.get("collision")
            else "MRC    " if info.get("mrc_reached")
            else "timeout"
        )
        print(
            f"  ep {ep_idx + 1:2d}/{n_episodes}  "
            f"[{meta.get('regime', '?'):16s}]  "
            f"v_ego={meta.get('initial_speed_mps', 0):.1f}m/s  "
            f"steps={steps:4d}  reward={total_reward:7.2f}  [{status}]"
        )

        row = {
            "episode": ep_idx + 1,
            "regime": meta.get("regime", ""),
            "initial_speed_mps": meta.get("initial_speed_mps", ""),
            "front_speed_mps": meta.get("front_speed_mps", ""),
            "initial_front_gap_m": meta.get("initial_front_gap_m", ""),
            "distance_behind_ego_m": meta.get("distance_behind_ego_m", ""),
            "trigger_time_s": meta.get("trigger_time_s", ""),
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "collision": info.get("collision", False),
            "mrc_reached": info.get("mrc_reached", False),
            "timeout": info.get("timeout", False),
            "primary_mrm": info.get("final_mrm", -1),
            "shoulder_available": info.get("shoulder_available", False),
        }
        summary_writer.writerow(row)
        summary_file.flush()
        episode_results.append(row)

    summary_file.close()
    env.close()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    n_mrc = sum(1 for r in episode_results if r["mrc_reached"])
    n_col = sum(1 for r in episode_results if r["collision"])
    n_to  = sum(1 for r in episode_results if r["timeout"])
    avg_r = np.mean([r["total_reward"] for r in episode_results])
    n_bl  = sum(1 for r in episode_results if r.get("regime") == "baseline")
    n_sfg = sum(1 for r in episode_results if r.get("regime") == "short_front_gap")
    print(
        f"\n[Phase 3] Random policy — {n_episodes} episodes complete:\n"
        f"  Episodes        : {n_bl} baseline  +  {n_sfg} short-front-gap\n"
        f"  MRC success     : {n_mrc}/{n_episodes}\n"
        f"  Collision       : {n_col}/{n_episodes}\n"
        f"  Timeout         : {n_to}/{n_episodes}\n"
        f"  Avg reward      : {avg_r:.3f}\n"
        f"  Summary CSV     : {summary_path}\n"
        f"Done. Logs saved in: {run_dir}"
    )


if __name__ == "__main__":
    main()
