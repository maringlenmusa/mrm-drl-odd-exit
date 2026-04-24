"""
Phase 1 main execution loop.

This is the core simulation loop that:
1. Starts esmini with a scenario
2. Receives simulation state continuously (UDP)
3. Triggers ODD exit (time-based)
4. Sends MRM control override (UDP)
5. Writes clean logs (steps.csv + episode.json)
"""

import time
import socket
from src.sim.esmini_runner import EsminiRunner
from src.sim.udp_bridge import UdpBridge
from src.sim.esmini_state_bridge import EsminiStateBridge
from src.sim.codec import JsonCodec
from src.sim.state_parser import parse_state
from src.sim.odd_exit import OddExitTrigger
from src.decision.mrm_catalog import MRM
from src.decision.executor import get_control_for_mrm, no_override_command
from src.logger import StepCsvLogger, write_episode_summary


def run_phase1(cfg: dict, run_dir: str):
    """
    Execute Phase 1 simulation run.
    
    Args:
        cfg: Configuration dictionary
        run_dir: Directory for writing logs
    """
    # 1) Start simulator (same scenario runs in UI)
    sim = EsminiRunner(cfg["sim"]["esmini_exe"], cfg["sim"]["scenario_path"])
    headless = cfg["sim"].get("headless", False)
    window_size = cfg["sim"].get("window_size", None)
    sim.start(headless=headless, window_size=window_size)

    # 2) Setup UDP (state from esmini or state_bridge; control to esmini)
    udp = UdpBridge(
        listen_ip=cfg["udp"]["listen_ip"],
        listen_port=cfg["udp"]["listen_port"],
        send_ip=cfg["udp"]["send_ip"],
        send_port=cfg["udp"]["send_port"],
        timeout_s=cfg["udp"]["recv_timeout_s"]
    )

    state_source = cfg["sim"].get("state_source", "state_bridge")
    state_bridge = None
    if state_source == "state_bridge":
        # 3) Synthetic state: bridge sends state to our listen port (esmini doesn't send UDP by default)
        sb_cfg = cfg.get("state_bridge", {})
        state_bridge = EsminiStateBridge(
            send_ip=cfg["udp"]["listen_ip"],
            send_port=cfg["udp"]["listen_port"],
            dt=cfg["sim"]["dt"],
            odd_exit_trigger_time_s=cfg["odd_exit"]["trigger_time_s"],
            initial_speed_mps=float(sb_cfg.get("initial_speed_mps", 15.0)),
            actors_config=sb_cfg.get("actors", []),
            collision_threshold_m=float(sb_cfg.get("collision_threshold_m", 3.0)),
        )
        state_bridge.start()
        time.sleep(0.2)  # let first packets arrive
    else:
        time.sleep(0.3)  # state_source == "esmini": receive state from esmini on listen_port

    codec = JsonCodec()
    odd = OddExitTrigger(cfg["odd_exit"]["trigger_time_s"])

    logger = StepCsvLogger(run_dir)

    # 4) Loop variables
    step_id = 0
    odd_exit_ever_triggered = False

    # Stop detection
    stopped_time_s = 0.0
    last_sim_time = None
    state = None  # Initialize for potential early exit

    # 5) Main simulation loop
    start_wall = time.time()
    while True:
        # Hard timeout (wall time or sim time)
        if (time.time() - start_wall) > cfg["stop_conditions"]["hard_timeout_s"]:
            break

        # Receive state
        try:
            packet = udp.recv_packet()
        except socket.timeout:
            # No data -> stop gracefully (or retry)
            break

        state = parse_state(codec, packet)

        # Compute dt from sim_time (more robust than fixed dt)
        if last_sim_time is None:
            dt = cfg["sim"]["dt"]
        else:
            dt = max(0.0, state.sim_time_s - last_sim_time)
        last_sim_time = state.sim_time_s

        # Check ODD exit
        odd_active = odd.is_active(state.sim_time_s)
        odd_exit_ever_triggered = odd_exit_ever_triggered or odd_active

        # Decide control
        if odd_active:
            cmd = get_control_for_mrm(MRM.BRAKE_IN_LANE, cfg)
        else:
            cmd = no_override_command()

        # Send control
        payload = codec.encode_control(cmd)
        udp.send_packet(payload)

        # Stop condition: if after odd_exit and speed low for N seconds
        if odd_active and state.ego.speed_mps <= cfg["stop_conditions"]["after_odd_exit_stop_speed_mps"]:
            stopped_time_s += dt
        else:
            stopped_time_s = 0.0

        done_stopped = stopped_time_s >= cfg["stop_conditions"]["stopped_for_s"]

        # Log step
        logger.log_step({
            "step_id": step_id,
            "sim_time_s": state.sim_time_s,
            "odd_exit_active": odd_active,
            "ego_speed_mps": state.ego.speed_mps,
            "throttle": cmd.throttle,
            "brake": cmd.brake,
            "steer": cmd.steer,
            "collision": state.collision
        })

        # Break conditions
        if state.collision:
            break
        if done_stopped:
            break
        if step_id >= cfg["run"]["max_steps"]:
            break

        step_id += 1

    # 6) Cleanup
    if state_bridge is not None:
        state_bridge.stop()
    logger.close()
    udp.close()
    sim.stop()

    # 7) Episode summary
    summary = {
        "odd_exit_triggered": odd_exit_ever_triggered,
        "steps": step_id,
        "ended_by_collision": bool(state.collision) if state is not None else None,
        "ended_by_stopped": bool(done_stopped) if state is not None else None,
        "last_sim_time_s": float(last_sim_time) if last_sim_time is not None else None
    }
    write_episode_summary(run_dir, summary)
