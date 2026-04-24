"""
State provider using esminiLib (in-process): runs the same scenario as esmini and
sends state (position, speed, collision) so we log what comes from esmini's physics.

esmini.exe does not send UDP state by default. This module uses esminiLib (same
physics as esmini) so when state_source=esmini we still get real simulation state
and can log receive/send correctly.
"""

import os
import sys
import time
import math
import json
import threading
import queue
from typing import Optional, Dict, Any

# ctypes for esminiLib
import ctypes
from ctypes import c_int, c_float, c_uint32, c_void_p, POINTER, Structure

# id_t in esmini is uint32
id_t = c_uint32


class SE_ScenarioObjectState(Structure):
    _fields_ = [
        ("id", c_int),
        ("model_id", c_int),
        ("ctrl_type", c_int),
        ("timestamp", c_float),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("h", c_float),
        ("p", c_float),
        ("r", c_float),
        ("roadId", id_t),
        ("junctionId", id_t),
        ("t", c_float),
        ("laneId", c_int),
        ("laneOffset", c_float),
        ("s", c_float),
        ("speed", c_float),
        ("centerOffsetX", c_float),
        ("centerOffsetY", c_float),
        ("centerOffsetZ", c_float),
        ("width", c_float),
        ("length", c_float),
        ("height", c_float),
        ("objectType", c_int),
        ("objectCategory", c_int),
        ("wheel_angle", c_float),
        ("wheel_rot", c_float),
        ("visibilityMask", c_int),
    ]


def _load_esmini_lib(esmini_bin_dir: str):
    """Load esminiLib.dll (Windows) or libesminiLib.so (Linux)."""
    if sys.platform == "win32":
        lib_name = "esminiLib.dll"
    else:
        lib_name = "libesminiLib.so"
    lib_path = os.path.join(esmini_bin_dir, lib_name)
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"esminiLib not found: {lib_path}")
    # Add bin dir to PATH so DLL can load dependencies (e.g. esminiRMLib)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.path.abspath(esmini_bin_dir) + os.pathsep + old_path
    try:
        lib = ctypes.CDLL(lib_path)
    finally:
        os.environ["PATH"] = old_path
    return lib


def _bind_esmini_api(lib):
    """Bind SE_* functions used by the state provider."""
    lib.SE_AddPath.argtypes = [ctypes.c_char_p]
    lib.SE_AddPath.restype = c_int

    lib.SE_Init.argtypes = [ctypes.c_char_p, c_int, c_int, c_int, c_int]
    lib.SE_Init.restype = c_int

    lib.SE_StepDT.argtypes = [c_float]
    lib.SE_StepDT.restype = c_int

    lib.SE_Close.restype = None

    lib.SE_GetNumberOfObjects.restype = c_int

    lib.SE_GetId.argtypes = [c_int]
    lib.SE_GetId.restype = c_int

    lib.SE_GetObjectState.argtypes = [c_int, POINTER(SE_ScenarioObjectState)]
    lib.SE_GetObjectState.restype = c_int

    lib.SE_GetObjectNumberOfCollisions.argtypes = [c_int]
    lib.SE_GetObjectNumberOfCollisions.restype = c_int

    lib.SE_ReportObjectSpeed.argtypes = [c_int, c_float]
    lib.SE_ReportObjectSpeed.restype = c_int

    lib.SE_ReportObjectPosXYH.argtypes = [c_int, c_float, c_float, c_float, c_float]
    lib.SE_ReportObjectPosXYH.restype = c_int

    lib.SE_ReportObjectRoadPos.argtypes = [c_int, c_float, id_t, c_int, c_float, c_float]
    lib.SE_ReportObjectRoadPos.restype = c_int

    # Enable collision detection (off by default)
    lib.SE_CollisionDetection.argtypes = [ctypes.c_int]  # bool as int
    lib.SE_CollisionDetection.restype = None

    # Viewer window (call before SE_Init when use_viewer=1)
    lib.SE_SetWindowPosAndSize.argtypes = [c_int, c_int, c_int, c_int]  # x, y, w, h
    lib.SE_SetWindowPosAndSize.restype = None

    return lib


def _state_to_json(sim_time_s: float, objects: list, ego_id: int, collision: bool) -> bytes:
    """Build our JSON state format from esmini object states (matches JsonCodec.decode_state)."""
    ego = None
    actors = []
    for obj in objects:
        o = {
            "actor_id": obj["id"],
            "x": obj["x"],
            "y": obj["y"],
            "speed_mps": obj["speed"],
            "yaw_rad": obj["h"],
            "lane_id": obj["laneId"],
        }
        if obj["id"] == ego_id:
            ego = {"x": obj["x"], "y": obj["y"], "speed_mps": obj["speed"], "yaw_rad": obj["h"], "lane_id": obj["laneId"]}
        else:
            actors.append(o)
    if ego is None and objects:
        ego = {
            "x": objects[0]["x"],
            "y": objects[0]["y"],
            "speed_mps": objects[0]["speed"],
            "yaw_rad": objects[0]["h"],
            "lane_id": objects[0]["laneId"],
        }
    if ego is None:
        ego = {"x": 0.0, "y": 0.0, "speed_mps": 0.0, "yaw_rad": 0.0, "lane_id": None}
    msg = {
        "sim_time_s": sim_time_s,
        "ego": ego,
        "collision": collision,
        "actors": actors,
    }
    return json.dumps(msg).encode("utf-8")


class EsminiLibStateProvider:
    """
    Runs scenario in-process via esminiLib; produces state (our JSON format) and
    accepts control (throttle/brake/steer) so we log receive/send from esmini physics.
    """

    def __init__(
        self,
        scenario_path: str,
        esmini_bin_dir: str,
        dt: float = 0.05,
        ego_id: int = 0,
        decel_mps2: float = 5.0,
        accel_mps2: float = 2.0,
        *,
        show_ui: bool = False,
        window_size: Optional[str] = None,
        fast_mode: bool = False,
    ):
        self.scenario_path = os.path.abspath(scenario_path)
        self.esmini_bin_dir = os.path.abspath(esmini_bin_dir)
        self.dt = dt
        self.ego_id = ego_id
        self.decel_mps2 = decel_mps2
        self.accel_mps2 = accel_mps2
        self.show_ui = show_ui
        # fast_mode: skip the real-time sleep in the sim loop (headless training only).
        # When True the sim runs as fast as esmini can step, not at real-time pace.
        # Never use with show_ui=True (viewer can't keep up).
        self.fast_mode = fast_mode and not show_ui
        # window_size: "x y w h" e.g. "60 60 1280 720"
        self.window_size = window_size or "60 60 1280 720"
        self._lib = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_queue: queue.Queue = queue.Queue(maxsize=1)
        self._control_queue: queue.Queue = queue.Queue(maxsize=1)
        self._sim_time = 0.0
        self._last_control: Optional[Dict[str, float]] = None
        # Pull-over (ROAD_SHOULDER_STOP): cap total heading change so we do a gentle move to shoulder then brake straight
        self._pull_over_heading_applied_rad: float = 0.0
        self._PULL_OVER_MAX_HEADING_RAD: float = 0.12  # ~20° total turn so ego can move onto shoulder
        # Cache for road position when esmini reports roadId=0 (so we still follow road). Store Python int/float only.
        self._last_road_id: Optional[int] = None
        self._last_lane_id: Optional[int] = None
        self._last_lane_offset: Optional[float] = None
        self._last_s: Optional[float] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.3)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._lib:
            try:
                self._lib.SE_Close()
            except Exception:
                pass
        self._lib = None

    def get_state(self, timeout_s: float = 0.5) -> Optional[bytes]:
        """Get one state packet (JSON bytes) or None on timeout."""
        try:
            return self._state_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def send_control(
        self,
        throttle: float,
        brake: float,
        steer: float,
        *,
        follow_road_while_braking: bool = False,
    ) -> None:
        """Send control to the esminiLib loop (non-blocking)."""
        try:
            self._control_queue.put_nowait({
                "throttle": throttle,
                "brake": brake,
                "steer": steer,
                "follow_road_while_braking": follow_road_while_braking,
            })
        except queue.Full:
            pass

    def _run_loop(self) -> None:
        try:
            self._lib = _load_esmini_lib(self.esmini_bin_dir)
            _bind_esmini_api(self._lib)
            # Add path so scenario can find xodr etc.
            self._lib.SE_AddPath(self.esmini_bin_dir.encode("utf-8"))
            osc_path = self.scenario_path.encode("utf-8")
            # disable_ctrls=1 so we control ego via Report
            # use_viewer: 1=window on screen (esmini bitmask). threads: 0=single thread (viewer+step same thread), 1=viewer in separate thread
            # On some platforms the window only appears if created on main thread; we run in a worker thread so try threads=1 first.
            use_viewer = 1 if self.show_ui else 0
            threads = 1 if self.show_ui else 0
            if self.show_ui:
                parts = self.window_size.split()
                x = int(parts[0]) if len(parts) >= 1 else 60
                y = int(parts[1]) if len(parts) >= 2 else 60
                w = int(parts[2]) if len(parts) >= 3 else 1280
                h = int(parts[3]) if len(parts) >= 4 else 720
                self._lib.SE_SetWindowPosAndSize(x, y, w, h)
            ret = self._lib.SE_Init(osc_path, 1, use_viewer, threads, 0)
            if ret != 0:
                print(f"[EsminiLibStateProvider] SE_Init failed (return {ret}). Check scenario path and esmini resources.")
                return
            if self.show_ui:
                print("[EsminiLibStateProvider] Viewer enabled; window should appear (if not, try running from main thread or check display).")
            # Enable collision detection so SE_GetObjectNumberOfCollisions returns hits
            self._lib.SE_CollisionDetection(1)
            n = self._lib.SE_GetNumberOfObjects()
            if n <= 0:
                return
            ego_id = self._lib.SE_GetId(0) if n > 0 else 0
            self._sim_time = 0.0
            while self._running:
                # Get control (non-blocking)
                try:
                    ctrl = self._control_queue.get_nowait()
                    self._last_control = ctrl
                except queue.Empty:
                    ctrl = self._last_control
                if ctrl is None:
                    ctrl = {"throttle": 0.0, "brake": 0.0, "steer": 0.0, "follow_road_while_braking": False}
                ctrl.setdefault("follow_road_while_braking", False)
                # Current state
                st = SE_ScenarioObjectState()
                self._lib.SE_GetObjectState(ego_id, ctypes.byref(st))
                speed = max(0.0, st.speed)
                # Apply control: speed change
                acc = ctrl["throttle"] * self.accel_mps2 - ctrl["brake"] * self.decel_mps2
                new_speed = max(0.0, speed + acc * self.dt)
                self._lib.SE_ReportObjectSpeed(ego_id, ctypes.c_float(new_speed))

                # Follow lane: (1) normal driving (throttle, no brake) or (2) IN_LANE_STOP (brake but follow road).
                follow_road = (
                    (ctrl["brake"] < 0.01 and ctrl["throttle"] > 0.01)
                    or ctrl.get("follow_road_while_braking", False)
                )
                road_length = 500.0
                did_road_pos = False
                if follow_road:
                    # Use esmini's reported road/lane when available; else use cache (when roadId is 0). Store Python int/float only to avoid ctypes issues.
                    if st.roadId != 0:
                        rid = int(st.roadId)
                        lid = int(st.laneId)
                        loff = float(st.laneOffset)
                        s_val = max(0.0, min(road_length, float(st.s) + new_speed * self.dt))
                        self._last_road_id, self._last_lane_id, self._last_lane_offset, self._last_s = rid, lid, loff, s_val
                    elif self._last_road_id is not None and self._last_lane_id is not None and self._last_lane_offset is not None and self._last_s is not None:
                        rid = self._last_road_id
                        lid = self._last_lane_id
                        loff = self._last_lane_offset
                        s_val = max(0.0, min(road_length, self._last_s + new_speed * self.dt))
                        self._last_s = s_val
                    else:
                        rid = lid = loff = s_val = None
                    if rid is not None and lid is not None and loff is not None and s_val is not None:
                        self._lib.SE_ReportObjectRoadPos(
                            ego_id,
                            ctypes.c_float(self._sim_time),
                            id_t(rid),
                            c_int(lid),
                            ctypes.c_float(loff),
                            ctypes.c_float(s_val),
                        )
                        did_road_pos = True
                if not did_road_pos:
                    # MRM or no throttle: world position update (brake straight, brake in lane with steer, or pull to shoulder).
                    h = st.h
                    dx = new_speed * math.cos(h) * self.dt
                    dy = new_speed * math.sin(h) * self.dt
                    steer_input = ctrl["steer"]
                    is_pull_over = ctrl["brake"] > 0.5 and abs(steer_input) > 0.1
                    if is_pull_over and self._pull_over_heading_applied_rad >= self._PULL_OVER_MAX_HEADING_RAD:
                        steer_input = 0.0
                    steer_rad = steer_input * (math.pi / 2.0)
                    delta_h = (new_speed * math.tan(steer_rad) / 3.5) * self.dt if abs(new_speed) > 0.01 else 0.0
                    if is_pull_over:
                        remaining = self._PULL_OVER_MAX_HEADING_RAD - self._pull_over_heading_applied_rad
                        delta_h = max(-remaining, min(remaining, delta_h))
                        self._pull_over_heading_applied_rad += abs(delta_h)
                    else:
                        self._pull_over_heading_applied_rad = 0.0
                    new_h = h + delta_h
                    self._lib.SE_ReportObjectPosXYH(
                        ego_id,
                        ctypes.c_float(self._sim_time),
                        ctypes.c_float(st.x + dx),
                        ctypes.c_float(st.y + dy),
                        ctypes.c_float(new_h),
                    )
                self._lib.SE_StepDT(ctypes.c_float(self.dt))
                self._sim_time += self.dt
                # Collect all object states
                n = self._lib.SE_GetNumberOfObjects()
                objects = []
                for i in range(n):
                    oid = self._lib.SE_GetId(i)
                    st = SE_ScenarioObjectState()
                    if self._lib.SE_GetObjectState(oid, ctypes.byref(st)) == 0:
                        objects.append({
                            "id": st.id,
                            "x": st.x,
                            "y": st.y,
                            "speed": st.speed,
                            "h": st.h,
                            "laneId": st.laneId,
                        })
                # Collision for ego
                n_coll = self._lib.SE_GetObjectNumberOfCollisions(ego_id)
                collision = n_coll > 0
                payload = _state_to_json(self._sim_time, objects, ego_id, collision)
                if self.fast_mode:
                    # Blocking put: producer waits for the consumer to read each state.
                    # This keeps sim_time in sync with the env's step counter (no skipping),
                    # and is still faster than real-time because there is no sleep.
                    self._state_queue.put(payload)
                else:
                    # Real-time mode: non-blocking put; overwrite if consumer is slow.
                    try:
                        self._state_queue.put_nowait(payload)
                    except queue.Full:
                        try:
                            self._state_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._state_queue.put_nowait(payload)
                    time.sleep(self.dt)
        except Exception as e:
            if self._running:
                print(f"[EsminiLibStateProvider] Error: {e}")
        finally:
            if self._lib:
                try:
                    self._lib.SE_Close()
                except Exception:
                    pass
                self._lib = None
