"""
Simple test: receive state, send control to esmini, verify esmini physics is running.

Run from project root:
  python scripts/test_esmini_send_receive.py

What it does:
  1. Starts esmini with a scenario (UI opens – you see the scenario and physics).
  2. Starts a state source (state_bridge) so we receive state on UDP (same format as if esmini sent it).
  3. Listens on listen_port, sends control on send_port (49950 = esmini UDPDriverController).
  4. Prints received state (sim_time, ego x, speed, collision) and sent control (throttle, brake, steer).
  5. After 5 s, sends brake=1 so the ego in esmini should brake (if scenario uses UDPDriverController).

You verify:
  - We receive data (state packets).
  - We send data (control to esmini).
  - Esmini window is running the same scenario we use in code.

Note: State in this test comes from the state_bridge (synthetic). To see the ego
brake in the esmini window, use a scenario that has UDPDriverController (e.g.
Phase 2 generated scenario); default straight_500m.xosc may use a different controller.
"""

import os
import sys
import time
import socket

# Run from project root; ensure imports work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.sim.esmini_runner import EsminiRunner
from src.sim.udp_bridge import UdpBridge
from src.sim.esmini_state_bridge import EsminiStateBridge
from src.sim.codec import JsonCodec, EsminiDriverCodec
from src.sim.state_parser import parse_state
from src.types import ControlCommand


def load_config():
    """Load phase2 config or use defaults."""
    try:
        from src.config import load_config as _load
        return _load("configs/phase2.yaml")
    except Exception:
        pass
    return {
        "sim": {
            "esmini_exe": "esmini/bin/esmini.exe",
            "scenario_path": "esmini/bin/resources/xosc/straight_500m.xosc",
            "dt": 0.05,
        },
        "udp": {
            "listen_ip": "127.0.0.1",
            "listen_port": 48100,
            "send_ip": "127.0.0.1",
            "send_port": 49950,
            "recv_timeout_s": 0.5,
        },
        "state_bridge": {
            "initial_speed_mps": 20.0,
            "initial_x_m": 50.0,
            "initial_y_m": 1.75,
            "actors": [],
            "collision_threshold_m": 3.0,
        },
        "odd_exit": {"trigger_time_s": 8.0},
    }


def main(headless: bool = False):
    cfg = load_config()
    sim_cfg = cfg["sim"]
    udp_cfg = cfg["udp"]
    sb_cfg = cfg.get("state_bridge", {})

    scenario_path = sim_cfg["scenario_path"]
    if not os.path.isabs(scenario_path):
        scenario_path = os.path.normpath(os.path.join(_project_root, scenario_path))

    print("=" * 60)
    print("Test: receive state & send control to esmini")
    print("=" * 60)
    print(f"Scenario: {scenario_path}")
    print(f"Listen:   {udp_cfg['listen_ip']}:{udp_cfg['listen_port']} (state)")
    print(f"Send:     {udp_cfg['send_ip']}:{udp_cfg['send_port']} (control -> esmini)")
    if headless:
        print("Headless: no UI (quick UDP test only).")
    else:
        print("Esmini window will open; after ~5 s we send brake=1 (ego should brake).")
    print("=" * 60)

    # 1) Start esmini (physics runs here; same scenario we use in code)
    sim = EsminiRunner(sim_cfg["esmini_exe"], scenario_path)
    sim.start(headless=headless)
    time.sleep(2.5 if not headless else 1.0)

    # 2) UDP and state source (so we have something to receive)
    udp = UdpBridge(
        listen_ip=udp_cfg["listen_ip"],
        listen_port=udp_cfg["listen_port"],
        send_ip=udp_cfg["send_ip"],
        send_port=udp_cfg["send_port"],
        timeout_s=udp_cfg["recv_timeout_s"],
    )
    state_bridge = EsminiStateBridge(
        send_ip=udp_cfg["listen_ip"],
        send_port=udp_cfg["listen_port"],
        dt=sim_cfg["dt"],
        odd_exit_trigger_time_s=cfg["odd_exit"]["trigger_time_s"],
        initial_speed_mps=float(sb_cfg.get("initial_speed_mps", 20.0)),
        initial_x_m=float(sb_cfg.get("initial_x_m", 50.0)),
        initial_y_m=float(sb_cfg.get("initial_y_m", 1.75)),
        actors_config=sb_cfg.get("actors", []),
        collision_threshold_m=float(sb_cfg.get("collision_threshold_m", 3.0)),
    )
    state_bridge.start()
    time.sleep(0.5)

    state_codec = JsonCodec()
    send_port = udp_cfg["send_port"]
    control_codec = EsminiDriverCodec(object_id=0) if send_port == 49950 else JsonCodec()

    recv_count = 0
    send_count = 0
    start_wall = time.time()
    brake_start_time = 5.0  # after 5 s, send brake
    run_duration = 15.0     # run for 15 s

    print("\nReceiving state (every 20th) and sending control after 5 s...\n")

    while (time.time() - start_wall) < run_duration:
        try:
            packet = udp.recv_packet()
        except socket.timeout:
            continue

        recv_count += 1
        state = parse_state(state_codec, packet)

        if recv_count % 20 == 1:
            print(f"  [RECV] sim_time_s={state.sim_time_s:.2f}  ego x={state.ego.x:.1f}  speed={state.ego.speed_mps:.2f} m/s  collision={state.collision}")

        # After brake_start_time, send full brake to esmini (so UI ego should brake)
        now = time.time() - start_wall
        if now >= brake_start_time:
            cmd = ControlCommand(override=True, throttle=0.0, brake=1.0, steer=0.0)
            if send_count == 0:
                print(f"\n  [SEND] throttle=0 brake=1 steer=0  (esmini ego should brake now)\n")
            send_count += 1
        else:
            cmd = ControlCommand(override=False, throttle=0.0, brake=0.0, steer=0.0)

        payload = control_codec.encode_control(cmd)
        udp.send_packet(payload)

    state_bridge.stop()
    udp.close()
    sim.stop()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Received state packets: {recv_count}")
    print(f"  Sent control packets:   {send_count}")
    print(f"  Esmini was running:     {scenario_path}")
    print()
    print("If you saw the esmini window and received/sent counts > 0, the loop works.")
    print("If the ego braked in the window after ~5 s, control is reaching esmini.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test receive/send to esmini")
    parser.add_argument("--headless", action="store_true", help="Run esmini without UI (quick check)")
    args = parser.parse_args()
    main(headless=args.headless)
