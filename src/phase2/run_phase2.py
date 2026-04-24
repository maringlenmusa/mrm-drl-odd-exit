"""
Phase 2 main execution loop.

Same structure as Phase 1, plus:
1. Build observation from state (gaps, speeds)
2. Compute risk (TTC, DRF, DARA, risk_total)
3. Baseline MRM choice + safety shield -> final MRM
4. MRC checker and episode tracker
5. Step log with obs/risk/proposed_mrm/final_mrm; episodes.csv + episode.json
"""

import os
import time
import socket
import math
from src.sim.esmini_runner import EsminiRunner
from src.sim.udp_bridge import UdpBridge
from src.sim.esmini_state_bridge import EsminiStateBridge
from src.sim.esmini_lib_state_provider import EsminiLibStateProvider
from src.sim.codec import JsonCodec, EsminiDriverCodec
from src.sim.state_parser import parse_state
from src.sim.odd_exit import OddExitTrigger
from src.features.observation_builder import build_observation
from src.risk.ttc import estimate_ttc
from src.risk.drf import compute_drf, compute_drf_from_scene
from src.risk.dara import compute_dara
from src.risk.risk_combiner import compute_risk
from src.decision.mrm_catalog import MRM, mrm_name as _mrm_name
from src.decision.baseline_rules import choose_mrm_baseline
from src.decision.shield import shield_action
from src.decision.executor import get_control_for_mrm, get_control_follow_lane, no_override_command
from src.metrics.mrc_checker import MrcChecker
from src.metrics.episode_tracker import EpisodeTracker
from src.logger import StepCsvLogger, EpisodeCsvLogger, write_episode_summary


def _event_log(msg: str, run_dir: str, log_file_handle=None):
    """Print to console and optionally append to run_dir/run.log."""
    print(msg)
    if log_file_handle is not None:
        try:
            log_file_handle.write(msg + "\n")
            log_file_handle.flush()
        except Exception:
            pass


def run_phase2(
    cfg: dict,
    run_dir: str,
    *,
    episode_id: int = None,
    episode_csv_logger: "EpisodeCsvLogger | None" = None,
):
    """
    Execute Phase 2 simulation run: risk, baseline MRM, shield, metrics, Phase 2 logging.

    Args:
        cfg: Configuration dictionary (sim, udp, odd_exit, mrm, risk, baseline, shield, stop_conditions)
        run_dir: Directory for writing logs (steps.csv or steps_episode_N.csv, episodes.csv, episode.json)
        episode_id: If set, steps are written to steps_episode_{episode_id}.csv (for multi-episode runs)
        episode_csv_logger: If set, log one episode row to this logger and do not close it (caller owns it)
    """
    steps_suffix = f"_episode_{episode_id}" if episode_id is not None else ""
    use_shared_episode_logger = episode_csv_logger is not None

    # Resolve scenario path: when using esmini (lib or exe) we want one source of truth from config.
    # Generate .xosc from state_bridge so scenario (Ego/Target positions, speeds) matches config and logs.
    headless = cfg["sim"].get("headless", False)
    state_source = cfg["sim"].get("state_source", "state_bridge")
    default_scenario_path = cfg["sim"].get("scenario_path", "esmini/bin/resources/xosc/straight_500m.xosc")
    scenario_path = default_scenario_path
    use_generated = cfg["sim"].get("use_generated_scenario_from_state_bridge", False)
    # Generate when: UI will show it (not headless) OR we run esminiLib (so logged state matches config)
    if (not headless and use_generated) or (state_source == "esmini" and use_generated):
        try:
            from src.phase2.scenario_from_config import generate_xosc_from_state_bridge
            sb_cfg = cfg.get("state_bridge", {})
            esmini_exe = cfg["sim"]["esmini_exe"]
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            exe_abs = os.path.normpath(os.path.join(project_root, esmini_exe)) if not os.path.isabs(esmini_exe) else esmini_exe
            esmini_bin_dir = os.path.dirname(exe_abs)
            generated_path = generate_xosc_from_state_bridge(
                sb_cfg,
                output_path=None,
                esmini_bin_dir=esmini_bin_dir,
                road_network_key=cfg.get("road_network", "straight"),
                odd_exit_trigger_time_s=cfg.get("odd_exit", {}).get("trigger_time_s"),
            )
            if generated_path and os.path.isfile(generated_path):
                scenario_path = generated_path
                print("[Phase2] Scenario from state_bridge config (Ego/Target positions match config and logs)")
            else:
                print("[Phase2] Generated scenario path invalid, using config scenario_path")
        except Exception as e:
            print(f"[Phase2] Could not generate scenario from config: {e}. Using config scenario_path.")
    if not scenario_path or not str(scenario_path).strip():
        scenario_path = default_scenario_path

    # 1) Simulator: single source of truth.
    # - state_source=esmini: run ONLY esminiLib in-process. All logged state comes from esminiLib. No esmini.exe (no UI).
    # - state_source=state_bridge: run esmini.exe (UI) + state_bridge (synthetic state). Logs = synthetic.
    if state_source == "esmini":
        show_ui = cfg["sim"].get("esmini_lib_show_ui", False)
        if show_ui:
            print("[Phase2] State source = esmini — single sim (esminiLib) with viewer. Logs and UI = same run.")
        else:
            print("[Phase2] State source = esmini — single sim (esminiLib). All logged state from esmini. No UI (set sim.esmini_lib_show_ui: true to show).")
    else:
        print("[Phase2] WARNING: state_source = state_bridge — logs are SYNTHETIC (our code), NOT from esmini.")
        print("[Phase2]         Set sim.state_source: esmini for real esmini state (no UI in that mode).")
    sim = EsminiRunner(cfg["sim"]["esmini_exe"], scenario_path)
    window_size = cfg["sim"].get("window_size", None)
    run_esmini_exe = state_source != "esmini"  # only start esmini.exe when using state_bridge (so UI matches synthetic)
    if run_esmini_exe:
        sim.start(headless=headless, window_size=window_size)
        if not headless:
            time.sleep(2.5)  # give esmini time to open and render the window
    else:
        # state_source=esmini: no esmini.exe — one source of truth (esminiLib only)
        print("[Phase2] esmini.exe not started (state_source=esmini). Logs = esminiLib only.")

    # 2) UDP; state from esmini or from state_bridge
    udp = UdpBridge(
        listen_ip=cfg["udp"]["listen_ip"],
        listen_port=cfg["udp"]["listen_port"],
        send_ip=cfg["udp"]["send_ip"],
        send_port=cfg["udp"]["send_port"],
        timeout_s=cfg["udp"]["recv_timeout_s"],
    )
    state_bridge = None
    esmini_lib_provider = None
    if state_source == "state_bridge":
        sb_cfg = cfg.get("state_bridge", {})
        ego_decel = float(sb_cfg.get("ego_decel_mps2", cfg["sim"].get("decel_mps2", 5.0)))
        curvature = float(cfg.get("road_curvature_rad_m", 0.0))
        state_bridge = EsminiStateBridge(
            send_ip=cfg["udp"]["listen_ip"],
            send_port=cfg["udp"]["listen_port"],
            dt=cfg["sim"]["dt"],
            odd_exit_trigger_time_s=cfg["odd_exit"]["trigger_time_s"],
            initial_speed_mps=float(sb_cfg.get("initial_speed_mps", cfg["sim"].get("initial_speed_mps", 15.0))),
            initial_x_m=float(sb_cfg.get("initial_x_m", 0.0)),
            initial_y_m=float(sb_cfg.get("initial_y_m", 1.75)),
            actors_config=sb_cfg.get("actors"),
            collision_threshold_m=float(sb_cfg.get("collision_threshold_m", 3.0)),
            ego_decel_mps2=ego_decel,
            curvature_rad_m=curvature,
        )
        state_bridge.start()
        time.sleep(0.5)  # let state_bridge send a few packets before first recv
    else:
        # state_source == "esmini": use esminiLib in-process (same physics as esmini) so we get state and can log receive/send
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        esmini_exe = cfg["sim"]["esmini_exe"]
        exe_abs = os.path.normpath(os.path.join(project_root, esmini_exe)) if not os.path.isabs(esmini_exe) else esmini_exe
        esmini_bin_dir = os.path.dirname(exe_abs)
        show_ui = cfg["sim"].get("esmini_lib_show_ui", False)
        window_size = cfg["sim"].get("window_size", None)
        decel_mps2 = float(cfg["sim"].get("decel_mps2", 5.0))
        esmini_lib_provider = EsminiLibStateProvider(
            scenario_path=scenario_path,
            esmini_bin_dir=esmini_bin_dir,
            dt=cfg["sim"]["dt"],
            decel_mps2=decel_mps2,
            show_ui=show_ui,
            window_size=window_size,
        )
        esmini_lib_provider.start()
        time.sleep(0.5)
        if show_ui:
            time.sleep(1.5)  # give esminiLib viewer time to create and show the window

    # State: JSON (from state_bridge or from esmini if it sends same format). Control: binary when send_port 49950.
    state_codec = JsonCodec()
    send_port = cfg["udp"]["send_port"]
    control_codec = EsminiDriverCodec(object_id=0) if send_port == 49950 else JsonCodec()
    odd = OddExitTrigger(cfg["odd_exit"]["trigger_time_s"])

    # Risk config (DRF aligned with supervisor's script: k_theta, k_r, max_distance_m)
    risk_cfg = cfg.get("risk", {})
    max_speed_mps = float(risk_cfg.get("max_speed_mps", 30.0))
    ttc_safe_s = float(risk_cfg.get("ttc_safe_s", 5.0))
    ttc_critical_s = float(risk_cfg.get("ttc_critical_s", 2.0))
    weight_drf = float(risk_cfg.get("weight_drf", 0.5))
    weight_dara = float(risk_cfg.get("weight_dara", 0.5))
    drf_k_theta = float(risk_cfg.get("k_theta", 0.5))
    drf_k_r = float(risk_cfg.get("k_r", 0.05))
    drf_max_distance_m = float(risk_cfg.get("max_distance_m", 50.0))

    stop_cfg = cfg.get("stop_conditions", {})
    mrc_checker = MrcChecker(
        stop_speed_mps=float(stop_cfg.get("mrc_stop_speed_mps", 0.5)),
        stopped_for_s=float(stop_cfg.get("stopped_for_s", 2.0)),
    )
    episode_tracker = EpisodeTracker()
    min_rear_gap_for_shoulder_m = float(cfg.get("baseline", {}).get("min_rear_gap_for_shoulder_m", 20.0))

    step_logger = StepCsvLogger(run_dir, use_phase2=True, filename_suffix=steps_suffix)
    if not use_shared_episode_logger:
        episode_csv_logger = EpisodeCsvLogger(run_dir)

    step_id = 0
    odd_exit_ever_triggered = False
    last_sim_time = None
    state = None
    current_mrm = -1
    start_wall = time.time()
    trigger_time_s = cfg["odd_exit"]["trigger_time_s"]
    # Episode-level fields (Phase 3 bandit plan §2): shoulder_available, min_gap, time_to_stop, time_to_mrc, max_decel
    odd_exit_sim_time_s = None
    shoulder_available = False
    min_gap = 999.0
    time_to_stop_s = None
    time_to_mrc_s = None
    max_decel = 0.0
    prev_speed = None

    # Event log: console + run_dir/run.log (collision, ODD exit, MRM, MRC)
    run_log_path = os.path.join(run_dir, "run.log")
    run_log_handle = open(run_log_path, "a", encoding="utf-8") if os.path.isdir(run_dir) else None
    if run_log_handle:
        if state_source == "esmini":
            run_log_handle.write("state_source=esmini - Single sim (esminiLib). All step data from esmini. No esmini.exe (no UI).\n")
        else:
            run_log_handle.write("state_source=state_bridge — All step data was SYNTHETIC (generated by our code), NOT from esmini.\n")
        if episode_id is not None:
            run_log_handle.write(f"\n--- Episode {episode_id + 1} (trigger_time_s={trigger_time_s}) ---\n")

    consecutive_timeouts = 0
    max_consecutive_timeouts = 15  # don't exit on first timeout; allow slow startup

    while True:
        if (time.time() - start_wall) > stop_cfg.get("hard_timeout_s", 30.0):
            break

        if state_source == "esmini" and esmini_lib_provider is not None:
            packet = esmini_lib_provider.get_state(timeout_s=cfg["udp"]["recv_timeout_s"])
            if packet is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= max_consecutive_timeouts:
                    break
                continue
            consecutive_timeouts = 0
        else:
            try:
                packet = udp.recv_packet()
                consecutive_timeouts = 0
            except socket.timeout:
                consecutive_timeouts += 1
                if consecutive_timeouts >= max_consecutive_timeouts:
                    break
                continue

        # State is only from received packet (esminiLib or state_bridge UDP); we never overwrite or hardcode it
        state = parse_state(state_codec, packet)
        dt = cfg["sim"]["dt"] if last_sim_time is None else max(0.0, state.sim_time_s - last_sim_time)
        last_sim_time = state.sim_time_s

        odd_active = odd.is_active(state.sim_time_s)
        if odd_active and not odd_exit_ever_triggered:
            odd_exit_ever_triggered = True
            _event_log(
                f"[ODD EXIT] Triggered at sim_time_s={state.sim_time_s:.2f} (config trigger_time_s={trigger_time_s})",
                run_dir, run_log_handle
            )
        # Observation and risk
        obs = build_observation(state)
        # Episode-level fields: capture at first ODD exit, then update over episode
        if odd_active and odd_exit_sim_time_s is None:
            odd_exit_sim_time_s = state.sim_time_s
            shoulder_available = (obs["rear_gap_m"] > min_rear_gap_for_shoulder_m)
        if odd_active:
            min_gap = min(min_gap, obs["front_gap_m"])
        if odd_exit_sim_time_s is not None and state.ego.speed_mps <= mrc_checker.stop_speed_mps and time_to_stop_s is None:
            time_to_stop_s = state.sim_time_s - odd_exit_sim_time_s
        if prev_speed is not None and dt > 0 and state.ego.speed_mps < prev_speed:
            decel_step = (prev_speed - state.ego.speed_mps) / dt
            max_decel = max(max_decel, decel_step)
        prev_speed = state.ego.speed_mps
        min_ttc = min(
            estimate_ttc(obs["front_gap_m"], obs["front_rel_speed_mps"]),
            estimate_ttc(obs["rear_gap_m"], obs["rear_rel_speed_mps"]),
        )
        if obs["front_gap_m"] >= 999.0 and obs["rear_gap_m"] >= 999.0:
            min_ttc = math.inf
        # Total ego-centric risk (DRF): supervisor's formula when we have other actors
        if state.actors:
            drf = compute_drf_from_scene(
                state.ego,
                state.actors,
                k_theta=drf_k_theta,
                k_r=drf_k_r,
                max_distance_m=drf_max_distance_m,
            )
        else:
            drf = compute_drf(state.ego.speed_mps, max_speed_mps=max_speed_mps)
        dara = compute_dara(min_ttc, ttc_safe_s=ttc_safe_s, ttc_critical_s=ttc_critical_s)
        risk_total = compute_risk(drf, dara, weight_drf=weight_drf, weight_dara=weight_dara)

        proposed_mrm = -1
        final_mrm = -1
        if odd_active:
            proposed_mrm = choose_mrm_baseline(obs, risk_total, cfg)
            final_mrm = shield_action(proposed_mrm, obs, current_mrm, cfg)
            if final_mrm != current_mrm:
                _event_log(
                    f"[MRM] sim_time_s={state.sim_time_s:.2f} -> {_mrm_name(final_mrm)} (proposed: {_mrm_name(proposed_mrm)})",
                    run_dir, run_log_handle
                )
            current_mrm = final_mrm

        if odd_active:
            cmd = get_control_for_mrm(final_mrm, cfg)
        else:
            cmd = get_control_follow_lane(cfg)

        # Send control to in-process esminiLib so brake/throttle apply there (collision & speed from physics)
        if state_source == "esmini" and esmini_lib_provider is not None:
            esmini_lib_provider.send_control(
                cmd.throttle, cmd.brake, cmd.steer,
                follow_road_while_braking=cmd.follow_road_while_braking,
            )

        payload = control_codec.encode_control(cmd)
        udp.send_packet(payload)

        mrc_checker.update(state.ego.speed_mps, dt)
        # max_risk/avg_risk: whole_episode = over all steps; after_odd_exit = only when ODD active
        risk_phase = (cfg.get("risk", {}) or {}).get("accumulate_risk_phase", "whole_episode")
        only_after_odd = risk_phase == "after_odd_exit"
        episode_tracker.update(
            risk_total,
            final_mrm if odd_active else -1,
            only_accumulate_risk_when=(odd_active if only_after_odd else True),
        )

        done_stopped = mrc_checker.is_mrc_reached()

        step_logger.log_step({
            "step_id": step_id,
            "sim_time_s": state.sim_time_s,
            "odd_exit_active": odd_active,
            "ego_speed_mps": state.ego.speed_mps,
            "throttle": cmd.throttle,
            "brake": cmd.brake,
            "steer": cmd.steer,
            "collision": state.collision,
            "state_source": state_source,  # "esmini" = what we received from UDP; "state_bridge" = synthetic
            "front_gap_m": obs["front_gap_m"],
            "rear_gap_m": obs["rear_gap_m"],
            "left_gap_m": obs["left_gap_m"],
            "right_gap_m": obs["right_gap_m"],
            "drf": drf,
            "dara": dara,
            "risk_total": risk_total,
            "proposed_mrm": proposed_mrm,
            "final_mrm": final_mrm,
        })

        if state.collision:
            _event_log(f"[COLLISION] at sim_time_s={state.sim_time_s:.2f}", run_dir, run_log_handle)
            break
        if done_stopped:
            if odd_exit_sim_time_s is not None:
                time_to_mrc_s = state.sim_time_s - odd_exit_sim_time_s
            _event_log(f"[MRC REACHED] at sim_time_s={state.sim_time_s:.2f} (full stop)", run_dir, run_log_handle)
            break
        if step_id >= cfg["run"].get("max_steps", 5000):
            break

        step_id += 1

    if state_bridge is not None:
        state_bridge.stop()
    if esmini_lib_provider is not None:
        esmini_lib_provider.stop()
    step_logger.close()
    if run_log_handle is not None:
        run_log_handle.close()

    if run_esmini_exe:
        sim.stop()
    udp.close()

    ep_summary = episode_tracker.get_summary()
    ep_row = {
        "episode_id": (episode_id + 1) if episode_id is not None else 1,
        "state_source": state_source,  # "esmini" = state from esmini; "state_bridge" = synthetic
        "success_mrc": mrc_checker.is_mrc_reached(),
        "collision": bool(state.collision) if state is not None else False,
        "steps": step_id,
        "max_risk": ep_summary["max_risk"],
        "avg_risk": ep_summary["avg_risk"],
        "mrm_switches": ep_summary["mrm_switches"],
        "primary_mrm": ep_summary.get("primary_mrm", -1),
        "last_sim_time_s": float(last_sim_time) if last_sim_time is not None else None,
        "shoulder_available": shoulder_available,
        "min_gap": min_gap if min_gap < 999.0 else None,
        "time_to_stop": time_to_stop_s,
        "time_to_mrc": time_to_mrc_s,
        "max_decel": max_decel if max_decel > 0 else None,
    }
    episode_csv_logger.log_episode(ep_row)
    if not use_shared_episode_logger:
        episode_csv_logger.close()

    summary = {
        "odd_exit_triggered": odd_exit_ever_triggered,
        "steps": step_id,
        "success_mrc": mrc_checker.is_mrc_reached(),
        "ended_by_collision": bool(state.collision) if state is not None else None,
        "ended_by_stopped": mrc_checker.is_mrc_reached(),
        "last_sim_time_s": float(last_sim_time) if last_sim_time is not None else None,
        "max_risk": ep_summary["max_risk"],
        "avg_risk": ep_summary["avg_risk"],
        "mrm_switches": ep_summary["mrm_switches"],
        "shoulder_available": shoulder_available,
        "min_gap": min_gap if min_gap < 999.0 else None,
        "time_to_stop": time_to_stop_s,
        "time_to_mrc": time_to_mrc_s,
        "max_decel": max_decel if max_decel > 0 else None,
    }
    write_episode_summary(run_dir, summary)
    return ep_row


def run_phase2_multi(cfg: dict, run_dir: str):
    """
    Run multiple Phase 2 episodes (scenario and/or trigger_time variations) and log to one episodes.csv.

    Reads scenario_sampler from config:
      scenario_sampler.scenarios: list of scenario paths (or single path)
      scenario_sampler.trigger_time_s_variations: list of ODD exit trigger times (e.g. [5.0, 8.0, 10.0])

    Runs one episode per (scenario, trigger_time_s) and appends one row per episode to episodes.csv.
    Steps are written to steps_episode_0.csv, steps_episode_1.csv, ...
    """
    import copy
    sampler = cfg.get("scenario_sampler", {})
    scenarios = sampler.get("scenarios", [cfg["sim"]["scenario_path"]])
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    trigger_times = sampler.get("trigger_time_s_variations", [cfg["odd_exit"]["trigger_time_s"]])
    if isinstance(trigger_times, (int, float)):
        trigger_times = [float(trigger_times)]

    episode_csv_logger = EpisodeCsvLogger(run_dir, mode="w")
    episode_csv_logger._episode_id = 0

    for ep_idx, (scenario_path, trigger_time_s) in enumerate(
        (s, t) for s in scenarios for t in trigger_times
    ):
        cfg_ep = copy.deepcopy(cfg)
        cfg_ep["sim"] = dict(cfg["sim"])
        cfg_ep["sim"]["scenario_path"] = scenario_path
        cfg_ep["odd_exit"] = dict(cfg["odd_exit"])
        cfg_ep["odd_exit"]["trigger_time_s"] = float(trigger_time_s)
        run_phase2(cfg_ep, run_dir, episode_id=ep_idx, episode_csv_logger=episode_csv_logger)

    episode_csv_logger.close()
    # Write a combined summary (last episode's summary; full table is in episodes.csv)
    write_episode_summary(run_dir, {
        "num_episodes": (ep_idx + 1),
        "scenarios": scenarios,
        "trigger_time_s_variations": trigger_times,
    })