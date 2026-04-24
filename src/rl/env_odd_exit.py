"""
Phase 3: Gym-style RL environment for MRM decision-making at ODD exit.

Wraps the Phase 2 simulation (same sim, risk, shield, logging) without
modifying any Phase 2 code.  The RL agent proposes one of three MRM
actions each step; the safety shield may downgrade the choice.

Action space:
    Discrete(3)
        0 = STRAIGHT_STOP   (safest  – brake only, no steer)
        1 = IN_LANE_STOP    (brake in lane – follows curve)
        2 = ROAD_SHOULDER_STOP (pull over to shoulder + brake)

Observation space:
    Box(N_OBS,) with values normalised to [0, 1]:
        [ego_speed, front_gap, rear_gap, left_gap, right_gap,
         front_rel_speed, rear_rel_speed, drf, dara, risk_total]

Episode lifecycle:
    reset()  → starts sim, runs "approach phase" with follow-lane until
               ODD exit triggers, returns obs at ODD exit moment.
    step(a)  → applies MRM action (after shield), advances one sim step,
               returns (obs, reward, done, info).
    close()  → tears down all sim resources.

Design note:
    All simulation components (EsminiRunner, EsminiLibStateProvider,
    EsminiStateBridge, UdpBridge) are the exact same objects used in
    run_phase2.py.  No physics or scenario logic was changed.
"""

import copy
import math
import os
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Gym / Gymnasium compatibility shim
# --------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    _GYM_VERSION = "gym"

# --------------------------------------------------------------------------
# Phase 2 simulation imports (unchanged)
# --------------------------------------------------------------------------
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
from src.decision.shield import shield_action
from src.decision.executor import get_control_for_mrm, get_control_follow_lane
from src.metrics.mrc_checker import MrcChecker
from src.metrics.episode_tracker import EpisodeTracker
from src.logger import StepCsvLogger, EpisodeCsvLogger, write_episode_summary
from src.rl.reward import compute_reward_detail

# --------------------------------------------------------------------------
# Observation feature names  (index must match _build_obs_array order)
# --------------------------------------------------------------------------
OBS_FEATURES = [
    "ego_speed_mps",        # 0
    "front_gap_m",          # 1
    "rear_gap_m",           # 2
    "left_gap_m",           # 3
    "right_gap_m",          # 4
    "front_rel_speed_mps",  # 5
    "rear_rel_speed_mps",   # 6
    "drf",                  # 7
    "dara",                 # 8
    "risk_total",           # 9
]
N_OBS = len(OBS_FEATURES)


class OddExitEnv(gym.Env):
    """
    Gym environment for MRM decision-making at ODD exit.

    Wraps Phase 2 simulation exactly as-is; only the decision layer is
    replaced: instead of choose_mrm_baseline(), the RL agent picks the MRM.
    The safety shield (shield_action) is still applied.

    Parameters
    ----------
    cfg : dict
        Full Phase 2 / Phase 3 config dict (from load_config).
    run_dir : str, optional
        Directory for step CSVs, episodes.csv, run.log.
        If None, no logging is written.
    episode_csv_logger : EpisodeCsvLogger, optional
        Shared logger owned by the caller (e.g. train.py).
        If None and run_dir is set, the env creates and owns one.
    ablations : dict, optional
        Ablation flags, e.g. {"no_risk_features": True, "no_risk_penalty": True}.
        Passed through to observation building and reward computation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: dict,
        run_dir: Optional[str] = None,
        *,
        episode_csv_logger: Optional[EpisodeCsvLogger] = None,
        ablations: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.run_dir = run_dir
        self._ablations: Dict[str, bool] = ablations or {}

        # ------------------------------------------------------------------
        # Gym spaces
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(N_OBS, dtype=np.float32),
            high=np.ones(N_OBS, dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Normalisation constants
        # ------------------------------------------------------------------
        risk_cfg = cfg.get("risk", {})
        self._max_speed_mps: float = float(risk_cfg.get("max_speed_mps", 40.0))
        self._max_gap_m: float = 50.0        # clip + normalise gaps at 50 m
        self._max_rel_speed_mps: float = 30.0

        # ------------------------------------------------------------------
        # Risk parameters (mirrors run_phase2.py exactly)
        # ------------------------------------------------------------------
        self._ttc_safe_s = float(risk_cfg.get("ttc_safe_s", 5.0))
        self._ttc_critical_s = float(risk_cfg.get("ttc_critical_s", 2.0))
        self._weight_drf = float(risk_cfg.get("weight_drf", 0.5))
        self._weight_dara = float(risk_cfg.get("weight_dara", 0.5))
        self._drf_k_theta = float(risk_cfg.get("k_theta", 0.5))
        self._drf_k_r = float(risk_cfg.get("k_r", 0.05))
        self._drf_max_distance_m = float(risk_cfg.get("max_distance_m", 50.0))

        stop_cfg = cfg.get("stop_conditions", {})
        self._mrc_stop_speed_mps = float(stop_cfg.get("mrc_stop_speed_mps", 0.5))
        self._stopped_for_s = float(stop_cfg.get("stopped_for_s", 2.0))
        self._hard_timeout_s = float(stop_cfg.get("hard_timeout_s", 30.0))
        # Post-ODD step budget only (step() after ODD); approach uses _run_until_odd_exit.
        # Phase 2 run_phase2 uses run.max_steps for the full loop including approach — keep RL cap high.
        self._max_steps = int(cfg.get("run", {}).get("max_steps", 5000))

        self._min_rear_gap_for_shoulder_m = float(
            cfg.get("baseline", {}).get("min_rear_gap_for_shoulder_m", 20.0)
        )
        # shoulder_available must use the SAME thresholds as the shield so the
        # reward's +5 bonus is only "in reach" when the shield would actually allow
        # ROAD_SHOULDER_STOP.  Using looser thresholds (e.g. rear_gap > 20 while
        # shield requires rear_gap > 30) creates states where shoulder_available=True
        # but the agent can never earn the shoulder bonus — confusing credit assignment.
        shield_cfg = cfg.get("shield", {})
        self._shoulder_min_right_gap_m = float(shield_cfg.get("min_right_gap_m", 12.0))
        self._shoulder_min_rear_gap_m  = float(shield_cfg.get("min_rear_gap_m", 30.0))
        self._state_source: str = cfg["sim"].get("state_source", "state_bridge")

        # MRM decision mode:
        #   "one_shot"      — agent picks MRM once at ODD exit; sim runs to completion
        #                     (correct: matches how Phase 2 works; trains all 3 MRMs fairly)
        #   "per_step_free" — agent picks every step but shield only checks gaps, no
        #                     downgrade-only constraint (also trains all 3 MRMs)
        self._mrm_decision_mode: str = str(
            cfg.get("training", {}).get("mrm_decision_mode", "one_shot")
        )

        # ------------------------------------------------------------------
        # Codecs (same as run_phase2.py)
        # ------------------------------------------------------------------
        self._state_codec = JsonCodec()
        send_port = cfg["udp"]["send_port"]
        self._control_codec = (
            EsminiDriverCodec(object_id=0) if send_port == 49950 else JsonCodec()
        )

        # ------------------------------------------------------------------
        # Episode CSV logger ownership
        # ------------------------------------------------------------------
        self._shared_episode_logger: Optional[EpisodeCsvLogger] = episode_csv_logger
        self._owned_episode_logger: Optional[EpisodeCsvLogger] = None
        if episode_csv_logger is None and run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            self._owned_episode_logger = EpisodeCsvLogger(run_dir, mode="w")

        # ------------------------------------------------------------------
        # Running sim handles (reset each episode)
        # ------------------------------------------------------------------
        self._sim: Optional[EsminiRunner] = None
        self._esmini_lib_provider: Optional[EsminiLibStateProvider] = None
        self._state_bridge: Optional[EsminiStateBridge] = None
        self._udp: Optional[UdpBridge] = None

        # Episode-level runtime state
        self._odd: Optional[OddExitTrigger] = None
        self._mrc_checker: Optional[MrcChecker] = None
        self._episode_tracker: Optional[EpisodeTracker] = None
        self._step_logger: Optional[StepCsvLogger] = None
        self._run_log_handle = None

        self._state = None
        self._obs_dict: Optional[Dict[str, Any]] = None
        self._drf: float = 0.0
        self._dara: float = 0.0
        self._risk_total: float = 0.0
        self._current_mrm: int = -1
        self._step_id: int = 0
        self._last_sim_time: Optional[float] = None
        self._start_wall: float = 0.0
        self._odd_exit_ever_triggered: bool = False

        # Episode-level fields tracked for reward and logging
        self._odd_exit_sim_time_s: Optional[float] = None
        self._shoulder_available: bool = False
        self._min_gap: float = 999.0
        self._time_to_stop_s: Optional[float] = None
        self._time_to_mrc_s: Optional[float] = None
        self._max_decel: float = 0.0
        self._prev_speed: Optional[float] = None

        # Reward accumulators (reset each episode)
        self._total_reward: float = 0.0
        self._dense_reward_sum: float = 0.0
        self._terminal_reward: float = 0.0
        self._terminal_collision_penalty: float = 0.0
        self._terminal_success_bonus: float = 0.0
        self._terminal_shoulder_bonus: float = 0.0
        self._terminal_lane_block_penalty: float = 0.0

        self._episode_id: int = 0
        self._consecutive_timeouts: int = 0
        self._max_consecutive_timeouts: int = 15
        self._current_scenario_path: str = str(
            cfg.get("sim", {}).get("scenario_path", "")
        )

    # ======================================================================
    # gym.Env interface
    # ======================================================================

    def reset(self, *, seed=None, options=None):
        """
        Start a new episode.

        Tears down the previous simulation, starts a fresh one, runs the
        "approach phase" with follow-lane control until the ODD exit
        triggers, then returns the observation at that moment.

        Parameters
        ----------
        seed : int, optional
            RNG seed (passed to super; no stochastic components in env itself).
        options : dict, optional
            Per-episode scenario overrides (same parameters as Phase 2 scripts):

            "trigger_time_s"  → float  override ODD trigger time
            "scenario_path"   → str    explicit .xosc path (skips generation)
            "state_bridge"    → dict   per-episode state_bridge overrides, e.g.:
                {
                  "initial_speed_mps": 25.0,        # ego speed
                  "initial_x_m": 50.0,              # ego start x
                  "actors": [
                    {"speed_mps": 15.0, "x_m": 120.0},   # lead car (index 0)
                    {"distance_behind_ego_m": 30.0},       # rear car (index 1)
                  ]
                }
            The actors list is merged index-by-index onto the base config actors.
            When state_source=esmini and use_generated_scenario_from_state_bridge=True,
            the env auto-regenerates the .xosc from the updated state_bridge config so
            esmini physics matches the episode parameters exactly.

        Returns
        -------
        obs : np.ndarray  shape (N_OBS,), dtype float32
        info : dict  (gymnasium API; gym users can ignore second return value)
        """
        super().reset(seed=seed)

        # Clean up any lingering sim from the previous episode
        self._teardown_sim()
        self._episode_id += 1

        # ------------------------------------------------------------------
        # Build per-episode config: start from base config then apply options
        # ------------------------------------------------------------------
        ep_cfg = self.cfg
        trigger_time_s = self.cfg["odd_exit"]["trigger_time_s"]
        scenario_path = self.cfg["sim"].get(
            "scenario_path",
            "esmini/bin/resources/xosc/straight_500m.xosc",
        )

        if options:
            if "trigger_time_s" in options:
                trigger_time_s = float(options["trigger_time_s"])
            if "scenario_path" in options:
                scenario_path = str(options["scenario_path"])

            # state_bridge overrides: ego speed, actors, initial position, etc.
            if "state_bridge" in options:
                ep_cfg = copy.deepcopy(self.cfg)
                sb_overrides = options["state_bridge"]
                ep_cfg["state_bridge"] = copy.deepcopy(
                    ep_cfg.get("state_bridge", {})
                )
                # Apply top-level keys (initial_speed_mps, initial_x_m, …)
                for k, v in sb_overrides.items():
                    if k != "actors":
                        ep_cfg["state_bridge"][k] = v
                # Merge actors index-by-index
                if "actors" in sb_overrides:
                    base_actors = [
                        dict(a)
                        for a in ep_cfg["state_bridge"].get("actors", [])
                    ]
                    for i, actor_patch in enumerate(sb_overrides["actors"]):
                        if actor_patch is None:
                            continue
                        if i < len(base_actors):
                            base_actors[i].update(actor_patch)
                        else:
                            base_actors.append(dict(actor_patch))
                    ep_cfg["state_bridge"]["actors"] = base_actors

        # Apply trigger_time_s and scenario_path to ep_cfg
        if ep_cfg is self.cfg:
            ep_cfg = copy.deepcopy(self.cfg)
        ep_cfg["odd_exit"]["trigger_time_s"] = trigger_time_s
        ep_cfg["sim"]["scenario_path"] = scenario_path

        # ------------------------------------------------------------------
        # Resolve scenario path: regenerate .xosc from state_bridge config
        # so esmini physics matches episode parameters (mirrors run_phase2.py)
        # ------------------------------------------------------------------
        scenario_path = self._resolve_scenario_path(ep_cfg, scenario_path)
        ep_cfg["sim"]["scenario_path"] = scenario_path
        self._current_scenario_path = str(scenario_path)

        # Reset all per-episode state
        self._current_mrm = -1
        self._step_id = 0
        self._last_sim_time = None
        self._start_wall = time.time()
        self._odd_exit_ever_triggered = False
        self._odd_exit_sim_time_s = None
        self._shoulder_available = False
        self._min_gap = 999.0
        self._time_to_stop_s = None
        self._time_to_mrc_s = None
        self._max_decel = 0.0
        self._prev_speed = None
        self._state = None
        self._obs_dict = None
        self._total_reward = 0.0
        self._dense_reward_sum = 0.0
        self._terminal_reward = 0.0
        self._terminal_collision_penalty = 0.0
        self._terminal_success_bonus = 0.0
        self._terminal_shoulder_bonus = 0.0
        self._terminal_lane_block_penalty = 0.0
        self._drf = 0.0
        self._dara = 0.0
        self._risk_total = 0.0
        self._consecutive_timeouts = 0

        self._odd = OddExitTrigger(trigger_time_s)
        self._mrc_checker = MrcChecker(
            stop_speed_mps=self._mrc_stop_speed_mps,
            stopped_for_s=self._stopped_for_s,
        )
        self._episode_tracker = EpisodeTracker()

        # Step + run logging
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            steps_suffix = f"_episode_{self._episode_id}"
            self._step_logger = StepCsvLogger(
                self.run_dir, use_phase2=True, filename_suffix=steps_suffix
            )
            run_log_path = os.path.join(self.run_dir, "run.log")
            self._run_log_handle = open(run_log_path, "a", encoding="utf-8")
            self._run_log_handle.write(
                f"\n--- Episode {self._episode_id} "
                f"(trigger_time_s={trigger_time_s}) ---\n"
            )

        # Start simulation components
        self._start_sim(ep_cfg, scenario_path)

        # Run until ODD exit; return initial observation
        obs = self._run_until_odd_exit(ep_cfg)

        # gymnasium API returns (obs, info); gym API returns just obs
        if _GYM_VERSION == "gymnasium":
            return obs, {}
        return obs

    def step(self, action: int):
        """
        Apply one MRM action.

        mrm_decision_mode (from cfg training.mrm_decision_mode):

        "one_shot" (default / recommended):
            Agent decides the MRM once, right here.  The env then runs the
            full braking manoeuvre internally until collision / MRC / timeout
            and returns done=True.  One gym step = one episode.  This is the
            correct design for the problem (agent picks MRM at ODD exit, same
            as Phase 2 baseline) and ensures PPO trains on all three MRMs.

        "per_step_free":
            One sim step per gym step (finer-grained but slower to collect
            n_steps).  Shield still vetoes unsafe pull-overs but the
            downgrade-only constraint is disabled so the agent can switch
            MRMs freely between steps.

        Parameters
        ----------
        action : int  — 0=STRAIGHT_STOP, 1=IN_LANE_STOP, 2=ROAD_SHOULDER_STOP
        """
        if self._mrm_decision_mode == "one_shot":
            return self._step_one_shot(action)
        return self._step_per_step(action, free_switch=True)

    # ------------------------------------------------------------------
    # one_shot: decide MRM once → execute to episode end
    # ------------------------------------------------------------------
    def _step_one_shot(self, action: int):
        """
        Agent picks MRM once; env runs full manoeuvre and returns done=True.
        Shield only checks gap constraints (no downgrade-only lock).
        """
        proposed_mrm = int(action)
        use_shield = not self._ablations.get("no_shield", False)
        if use_shield:
            # Pass current_mrm=-1 so downgrade-only never fires:
            # the agent is free to pick any MRM; shield only blocks
            # pull-over when gaps are too small.
            final_mrm = shield_action(
                proposed_mrm, self._obs_dict, -1, self.cfg
            )
        else:
            final_mrm = proposed_mrm
        self._current_mrm = final_mrm

        # Guard: if approach phase returned before any packet arrived, state is None.
        # Treat as immediate timeout so training continues on the next episode.
        if self._state is None:
            self._log_event("[MRM DECISION] state=None after approach timeout — skipping episode")
            obs  = self._build_obs_array()
            info = self._build_info(timeout=True)
            self._finalize_episode(collision=False, timeout=True)
            if _GYM_VERSION == "gymnasium":
                return obs, 0.0, True, False, info
            return obs, 0.0, True, info

        sim_t_str = "%.2fs" % self._state.sim_time_s
        self._log_event(
            "[MRM DECISION] %s (proposed: %s) at sim_t=%s" % (
                _mrm_name(final_mrm), _mrm_name(proposed_mrm), sim_t_str
            )
        )

        # Run sim loop until episode ends, accumulate reward
        collision = False
        timeout = False
        while True:
            if (time.time() - self._start_wall) > self._hard_timeout_s:
                timeout = True
                break

            cmd = get_control_for_mrm(final_mrm, self.cfg)
            self._send_control(cmd)

            packet = self._get_packet()
            if packet is None:
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                    timeout = True
                    break
                continue
            self._consecutive_timeouts = 0

            self._state = parse_state(self._state_codec, packet)
            dt = (
                self.cfg["sim"]["dt"]
                if self._last_sim_time is None
                else max(0.0, self._state.sim_time_s - self._last_sim_time)
            )
            self._last_sim_time = self._state.sim_time_s

            self._obs_dict = build_observation(self._state)
            self._drf, self._dara, self._risk_total = self._compute_risk()
            self._update_episode_fields(dt)
            self._mrc_checker.update(self._state.ego.speed_mps, dt)
            self._episode_tracker.update(
                self._risk_total, final_mrm, only_accumulate_risk_when=True
            )
            self._step_id += 1

            collision = bool(self._state.collision)
            mrc_reached = self._mrc_checker.is_mrc_reached()
            step_timeout = self._step_id >= self._max_steps
            timeout = step_timeout or (time.time() - self._start_wall) > self._hard_timeout_s
            done = collision or mrc_reached or timeout

            rwd = compute_reward_detail(
                collision=collision,
                mrc_reached=mrc_reached,
                risk_total=self._risk_total,
                final_mrm=final_mrm,
                obs_dict=self._obs_dict,
                shoulder_available=self._shoulder_available,
                cfg=self.cfg,
                done=done,
                ablations=self._ablations,
            )
            self._total_reward       += rwd["total"]
            self._dense_reward_sum   += rwd["dense_risk_penalty"] + rwd["dense_step_cost"]
            self._terminal_reward    += (
                rwd["terminal_collision_penalty"] + rwd["terminal_success_bonus"]
                + rwd["terminal_shoulder_bonus"]  + rwd["terminal_lane_block_penalty"]
            )
            self._terminal_collision_penalty  += rwd["terminal_collision_penalty"]
            self._terminal_success_bonus      += rwd["terminal_success_bonus"]
            self._terminal_shoulder_bonus     += rwd["terminal_shoulder_bonus"]
            self._terminal_lane_block_penalty += rwd["terminal_lane_block_penalty"]

            self._log_step(proposed_mrm, final_mrm, cmd, rwd)

            if collision:
                self._log_event(
                    f"[COLLISION] at sim_time_s={self._state.sim_time_s:.2f}"
                )
            if mrc_reached:
                if self._odd_exit_sim_time_s is not None:
                    self._time_to_mrc_s = (
                        self._state.sim_time_s - self._odd_exit_sim_time_s
                    )
                self._log_event(
                    f"[MRC REACHED] at sim_time_s={self._state.sim_time_s:.2f}"
                )
            if done:
                break

        obs  = self._build_obs_array()
        info = self._build_info(timeout=timeout)
        self._finalize_episode(collision=collision, timeout=timeout)

        total_reward = self._total_reward  # captured before finalize clears nothing (finalize is read-only)
        if _GYM_VERSION == "gymnasium":
            return obs, total_reward, True, False, info
        return obs, total_reward, True, info

    # ------------------------------------------------------------------
    # per_step_free: one sim step per gym step; no downgrade-only lock
    # ------------------------------------------------------------------
    def _step_per_step(self, action: int, free_switch: bool = True):
        """
        One sim step per gym step.  When free_switch=True the downgrade-only
        shield constraint is disabled so the agent can freely switch MRMs.
        """
        proposed_mrm = int(action)
        use_shield = not self._ablations.get("no_shield", False)
        if use_shield:
            # free_switch: pass -1 so no downgrade-only; shield still checks gaps
            shield_history = -1 if free_switch else self._current_mrm
            final_mrm = shield_action(
                proposed_mrm, self._obs_dict, shield_history, self.cfg
            )
        else:
            final_mrm = proposed_mrm
        if final_mrm != self._current_mrm:
            self._log_event(
                f"[MRM] step={self._step_id} "
                f"sim_t={self._state.sim_time_s:.2f}s "
                f"-> {_mrm_name(final_mrm)} (proposed: {_mrm_name(proposed_mrm)})"
            )
        self._current_mrm = final_mrm

        cmd = get_control_for_mrm(final_mrm, self.cfg)
        self._send_control(cmd)

        packet = self._get_packet()
        if packet is None:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                obs  = self._build_obs_array()
                info = self._build_info(timeout=True)
                self._finalize_episode(collision=False, timeout=True)
                if _GYM_VERSION == "gymnasium":
                    return obs, 0.0, True, False, info
                return obs, 0.0, True, info
            obs = self._build_obs_array()
            if _GYM_VERSION == "gymnasium":
                return obs, 0.0, False, False, {}
            return obs, 0.0, False, {}
        self._consecutive_timeouts = 0

        self._state = parse_state(self._state_codec, packet)
        dt = (
            self.cfg["sim"]["dt"]
            if self._last_sim_time is None
            else max(0.0, self._state.sim_time_s - self._last_sim_time)
        )
        self._last_sim_time = self._state.sim_time_s

        self._obs_dict = build_observation(self._state)
        self._drf, self._dara, self._risk_total = self._compute_risk()
        self._update_episode_fields(dt)
        self._mrc_checker.update(self._state.ego.speed_mps, dt)
        self._episode_tracker.update(
            self._risk_total, final_mrm, only_accumulate_risk_when=True
        )
        self._step_id += 1

        collision  = bool(self._state.collision)
        mrc_reached = self._mrc_checker.is_mrc_reached()
        step_timeout = self._step_id >= self._max_steps
        timeout    = step_timeout or (time.time() - self._start_wall) > self._hard_timeout_s
        done       = collision or mrc_reached or timeout

        if step_timeout and not collision and not mrc_reached:
            self._log_event(
                f"[EPISODE END] step limit ({self._max_steps}) — "
                "increase run.max_steps if episodes end before MRC"
            )
        if collision:
            self._log_event(
                f"[COLLISION] at sim_time_s={self._state.sim_time_s:.2f}"
            )
        if mrc_reached:
            if self._odd_exit_sim_time_s is not None:
                self._time_to_mrc_s = (
                    self._state.sim_time_s - self._odd_exit_sim_time_s
                )
            self._log_event(
                f"[MRC REACHED] at sim_time_s={self._state.sim_time_s:.2f}"
            )

        rwd = compute_reward_detail(
            collision=collision, mrc_reached=mrc_reached,
            risk_total=self._risk_total, final_mrm=final_mrm,
            obs_dict=self._obs_dict, shoulder_available=self._shoulder_available,
            cfg=self.cfg, done=done, ablations=self._ablations,
        )
        reward = rwd["total"]
        self._total_reward       += reward
        self._dense_reward_sum   += rwd["dense_risk_penalty"] + rwd["dense_step_cost"]
        self._terminal_reward    += (
            rwd["terminal_collision_penalty"] + rwd["terminal_success_bonus"]
            + rwd["terminal_shoulder_bonus"]  + rwd["terminal_lane_block_penalty"]
        )
        self._terminal_collision_penalty  += rwd["terminal_collision_penalty"]
        self._terminal_success_bonus      += rwd["terminal_success_bonus"]
        self._terminal_shoulder_bonus     += rwd["terminal_shoulder_bonus"]
        self._terminal_lane_block_penalty += rwd["terminal_lane_block_penalty"]

        self._log_step(proposed_mrm, final_mrm, cmd, rwd)

        obs  = self._build_obs_array()
        info = self._build_info(timeout=timeout)
        if done:
            self._finalize_episode(collision=collision, timeout=timeout)

        if _GYM_VERSION == "gymnasium":
            return obs, reward, done, False, info
        return obs, reward, done, info

    def close(self):
        """Tear down simulation and close all file handles."""
        self._teardown_sim()
        if self._owned_episode_logger is not None:
            self._owned_episode_logger.close()
            self._owned_episode_logger = None

    # ======================================================================
    # Sim lifecycle helpers
    # ======================================================================

    def _resolve_scenario_path(self, cfg: dict, default_path: str) -> str:
        """
        Generate .xosc from state_bridge config when configured to do so.

        Mirrors the scenario-generation block in run_phase2.py exactly.
        When state_source=esmini + use_generated_scenario_from_state_bridge=True,
        the generated file embeds the episode's exact ego/actor parameters so
        esmini physics is consistent with what the env logged.

        Falls back to default_path on any error.
        """
        state_source = cfg["sim"].get("state_source", "state_bridge")
        use_generated = cfg["sim"].get(
            "use_generated_scenario_from_state_bridge", False
        )
        headless = cfg["sim"].get("headless", False)

        should_generate = (
            (not headless and use_generated)
            or (state_source == "esmini" and use_generated)
        )
        if not should_generate:
            return default_path

        try:
            from src.phase2.scenario_from_config import generate_xosc_from_state_bridge

            sb_cfg = cfg.get("state_bridge", {})
            esmini_exe = cfg["sim"]["esmini_exe"]
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            exe_abs = (
                os.path.normpath(os.path.join(project_root, esmini_exe))
                if not os.path.isabs(esmini_exe)
                else esmini_exe
            )
            esmini_bin_dir = os.path.dirname(exe_abs)
            generated = generate_xosc_from_state_bridge(
                sb_cfg,
                output_path=None,
                esmini_bin_dir=esmini_bin_dir,
                road_network_key=cfg.get("road_network", "straight"),
                odd_exit_trigger_time_s=cfg.get("odd_exit", {}).get(
                    "trigger_time_s"
                ),
            )
            if generated and os.path.isfile(generated):
                return generated
        except Exception as exc:
            self._log_event(
                f"[OddExitEnv] Could not generate scenario: {exc}. "
                "Using default scenario_path."
            )
        return default_path

    def _start_sim(self, cfg: dict, scenario_path: str) -> None:
        """
        Start all simulation components for a new episode.
        Mirrors the setup block in run_phase2.py exactly.
        """
        state_source = self._state_source

        # esmini.exe — only when using state_bridge (UI matches synthetic state)
        run_esmini_exe = state_source != "esmini"
        if run_esmini_exe:
            headless = cfg["sim"].get("headless", True)  # default headless for RL
            self._sim = EsminiRunner(cfg["sim"]["esmini_exe"], scenario_path)
            self._sim.start(headless=headless)
            if not headless:
                time.sleep(2.5)
        else:
            self._log_event(
                "[OddExitEnv] state_source=esmini — esminiLib in-process. "
                "No esmini.exe."
            )

        # UDP bridge (needed in both modes for state_bridge; also used for
        # sending EsminiDriverCodec packets when send_port==49950)
        self._udp = UdpBridge(
            listen_ip=cfg["udp"]["listen_ip"],
            listen_port=cfg["udp"]["listen_port"],
            send_ip=cfg["udp"]["send_ip"],
            send_port=cfg["udp"]["send_port"],
            timeout_s=cfg["udp"]["recv_timeout_s"],
        )

        if state_source == "state_bridge":
            sb_cfg = cfg.get("state_bridge", {})
            ego_decel = float(
                sb_cfg.get("ego_decel_mps2", cfg["sim"].get("decel_mps2", 5.0))
            )
            self._state_bridge = EsminiStateBridge(
                send_ip=cfg["udp"]["listen_ip"],
                send_port=cfg["udp"]["listen_port"],
                dt=cfg["sim"]["dt"],
                odd_exit_trigger_time_s=cfg["odd_exit"]["trigger_time_s"],
                initial_speed_mps=float(
                    sb_cfg.get(
                        "initial_speed_mps",
                        cfg["sim"].get("initial_speed_mps", 15.0),
                    )
                ),
                initial_x_m=float(sb_cfg.get("initial_x_m", 0.0)),
                initial_y_m=float(sb_cfg.get("initial_y_m", 1.75)),
                actors_config=sb_cfg.get("actors"),
                collision_threshold_m=float(
                    sb_cfg.get("collision_threshold_m", 3.0)
                ),
                ego_decel_mps2=ego_decel,
                curvature_rad_m=float(cfg.get("road_curvature_rad_m", 0.0)),
            )
            self._state_bridge.start()
            time.sleep(0.5)
        else:
            # EsminiLib (in-process) — mirrors run_phase2.py
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            esmini_exe = cfg["sim"]["esmini_exe"]
            exe_abs = (
                os.path.normpath(os.path.join(project_root, esmini_exe))
                if not os.path.isabs(esmini_exe)
                else esmini_exe
            )
            esmini_bin_dir = os.path.dirname(exe_abs)
            show_ui = cfg["sim"].get("esmini_lib_show_ui", False)
            window_size = cfg["sim"].get("window_size", None)
            decel_mps2 = float(cfg["sim"].get("decel_mps2", 5.0))
            headless = cfg["sim"].get("headless", False)
            # fast_mode: run sim without real-time sleep — only safe when headless.
            # Reads from sim.fast_mode config; defaults True when headless, False otherwise.
            fast_mode = bool(cfg["sim"].get("fast_mode", headless and not show_ui))
            self._esmini_lib_provider = EsminiLibStateProvider(
                scenario_path=os.path.abspath(scenario_path),
                esmini_bin_dir=esmini_bin_dir,
                dt=cfg["sim"]["dt"],
                decel_mps2=decel_mps2,
                show_ui=show_ui,
                window_size=window_size,
                fast_mode=fast_mode,
            )
            self._esmini_lib_provider.start()
            time.sleep(0.5)
            if show_ui:
                time.sleep(1.5)

    def _teardown_sim(self) -> None:
        """Stop simulation and close all per-episode handles."""
        # Step logger
        if self._step_logger is not None:
            try:
                self._step_logger.close()
            except Exception:
                pass
            self._step_logger = None

        # Run log
        if self._run_log_handle is not None:
            try:
                self._run_log_handle.close()
            except Exception:
                pass
            self._run_log_handle = None

        # State bridge
        if self._state_bridge is not None:
            try:
                self._state_bridge.stop()
            except Exception:
                pass
            self._state_bridge = None

        # EsminiLib provider
        if self._esmini_lib_provider is not None:
            try:
                self._esmini_lib_provider.stop()
            except Exception:
                pass
            self._esmini_lib_provider = None

        # esmini.exe
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None

        # UDP bridge
        if self._udp is not None:
            try:
                self._udp.close()
            except Exception:
                pass
            self._udp = None

    # ======================================================================
    # State / packet helpers
    # ======================================================================

    def _get_packet(self) -> Optional[bytes]:
        """Receive one state packet from esminiLib or UDP (blocking, with timeout)."""
        if (
            self._state_source == "esmini"
            and self._esmini_lib_provider is not None
        ):
            return self._esmini_lib_provider.get_state(
                timeout_s=self.cfg["udp"]["recv_timeout_s"]
            )
        try:
            return self._udp.recv_packet()
        except socket.timeout:
            return None

    def _send_control(self, cmd) -> None:
        """Send control to esminiLib and/or UDP (mirrors run_phase2.py)."""
        if (
            self._state_source == "esmini"
            and self._esmini_lib_provider is not None
        ):
            self._esmini_lib_provider.send_control(
                cmd.throttle,
                cmd.brake,
                cmd.steer,
                follow_road_while_braking=cmd.follow_road_while_braking,
            )
        payload = self._control_codec.encode_control(cmd)
        self._udp.send_packet(payload)

    # ======================================================================
    # Risk / observation helpers
    # ======================================================================

    def _compute_risk(self) -> Tuple[float, float, float]:
        """
        Compute (drf, dara, risk_total) from current state.
        Exact copy of the risk block in run_phase2.py.
        """
        obs = self._obs_dict
        state = self._state

        min_ttc = min(
            estimate_ttc(obs["front_gap_m"], obs["front_rel_speed_mps"]),
            estimate_ttc(obs["rear_gap_m"], obs["rear_rel_speed_mps"]),
        )
        if obs["front_gap_m"] >= 999.0 and obs["rear_gap_m"] >= 999.0:
            min_ttc = math.inf

        if state.actors:
            drf = compute_drf_from_scene(
                state.ego,
                state.actors,
                k_theta=self._drf_k_theta,
                k_r=self._drf_k_r,
                max_distance_m=self._drf_max_distance_m,
            )
        else:
            drf = compute_drf(state.ego.speed_mps, max_speed_mps=self._max_speed_mps)

        dara = compute_dara(
            min_ttc,
            ttc_safe_s=self._ttc_safe_s,
            ttc_critical_s=self._ttc_critical_s,
        )
        risk_total = compute_risk(
            drf, dara,
            weight_drf=self._weight_drf,
            weight_dara=self._weight_dara,
        )
        return drf, dara, risk_total

    def _build_obs_array(self) -> np.ndarray:
        """
        Build the normalised [0,1] observation array for the agent.
        When obs_dict is None (e.g. before first state arrives) returns zeros.
        """
        if self._obs_dict is None:
            return np.zeros(N_OBS, dtype=np.float32)

        obs = self._obs_dict
        ms = self._max_speed_mps
        mg = self._max_gap_m
        mr = self._max_rel_speed_mps

        arr = np.array(
            [
                obs["ego_speed_mps"] / ms,
                min(obs["front_gap_m"], mg) / mg,
                min(obs["rear_gap_m"], mg) / mg,
                min(obs["left_gap_m"], mg) / mg,
                min(obs["right_gap_m"], mg) / mg,
                max(0.0, obs["front_rel_speed_mps"]) / mr,
                max(0.0, obs["rear_rel_speed_mps"]) / mr,
                float(self._drf),
                float(self._dara),
                float(self._risk_total),
            ],
            dtype=np.float32,
        )

        # Ablation: zero out risk features so agent cannot use them
        if self._ablations.get("no_risk_features", False):
            arr[7:] = 0.0  # indices 7,8,9 = drf, dara, risk_total

        return np.clip(arr, 0.0, 1.0)

    # ======================================================================
    # Episode-level tracking (mirrors run_phase2.py field tracking)
    # ======================================================================

    def _update_episode_fields(self, dt: float) -> None:
        """
        Update shoulder_available, min_gap, time_to_stop, max_decel.
        Mirrors the tracking block inside the run_phase2.py while loop.
        """
        state = self._state
        obs = self._obs_dict
        odd_active = self._odd.is_active(state.sim_time_s)

        if odd_active and self._odd_exit_sim_time_s is None:
            self._odd_exit_sim_time_s = state.sim_time_s
            # shoulder_available = True only when BOTH gaps satisfy the shield thresholds,
            # i.e. shield would actually allow ROAD_SHOULDER_STOP here.
            self._shoulder_available = (
                obs["rear_gap_m"]  > self._shoulder_min_rear_gap_m
                and obs["right_gap_m"] > self._shoulder_min_right_gap_m
            )

        if odd_active:
            self._min_gap = min(self._min_gap, obs["front_gap_m"])

        if (
            self._odd_exit_sim_time_s is not None
            and state.ego.speed_mps <= self._mrc_stop_speed_mps
            and self._time_to_stop_s is None
        ):
            self._time_to_stop_s = state.sim_time_s - self._odd_exit_sim_time_s

        if (
            self._prev_speed is not None
            and dt > 0
            and state.ego.speed_mps < self._prev_speed
        ):
            decel_step = (self._prev_speed - state.ego.speed_mps) / dt
            self._max_decel = max(self._max_decel, decel_step)
        self._prev_speed = state.ego.speed_mps

    # ======================================================================
    # Approach phase: run sim until ODD exit, then return
    # ======================================================================

    def _run_until_odd_exit(self, cfg: dict) -> np.ndarray:
        """
        Run follow-lane control until ODD exit triggers.

        Returns the normalised observation array at the first ODD-active step.
        The sim is left running; the caller's next step() call will send the
        first MRM command and receive the next state.
        """
        consecutive_timeouts = 0
        approach_step = 0

        while True:
            # Hard wall-clock guard (prevents infinite loop on sim failure)
            if (time.time() - self._start_wall) > self._hard_timeout_s:
                self._log_event(
                    "[OddExitEnv] Hard timeout during approach — returning zero obs."
                )
                return np.zeros(N_OBS, dtype=np.float32)

            packet = self._get_packet()
            if packet is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= self._max_consecutive_timeouts:
                    self._log_event(
                        "[OddExitEnv] Too many timeouts during approach — "
                        "returning zero obs."
                    )
                    return np.zeros(N_OBS, dtype=np.float32)
                continue
            consecutive_timeouts = 0

            state = parse_state(self._state_codec, packet)
            dt = (
                cfg["sim"]["dt"]
                if self._last_sim_time is None
                else max(0.0, state.sim_time_s - self._last_sim_time)
            )
            self._last_sim_time = state.sim_time_s
            self._state = state

            self._obs_dict = build_observation(state)
            self._drf, self._dara, self._risk_total = self._compute_risk()

            odd_active = self._odd.is_active(state.sim_time_s)

            if odd_active and not self._odd_exit_ever_triggered:
                # ---- ODD exit just fired — set up episode tracking ----
                self._odd_exit_ever_triggered = True
                self._odd_exit_sim_time_s = state.sim_time_s
                self._shoulder_available = (
                    self._obs_dict["rear_gap_m"]  > self._shoulder_min_rear_gap_m
                    and self._obs_dict["right_gap_m"] > self._shoulder_min_right_gap_m
                )
                self._log_event(
                    f"[ODD EXIT] Triggered at sim_time_s={state.sim_time_s:.2f} "
                    f"(trigger_time_s={self._odd.trigger_time_s})"
                )

                # Log this approach step (no MRM yet)
                follow_cmd = get_control_follow_lane(cfg)
                self._log_step(-1, -1, follow_cmd)

                # Send a neutral "coast" command so the sim stops accelerating
                # while the agent decides its first action in step()
                self._send_control(follow_cmd)

                return self._build_obs_array()

            # Approach: follow lane
            cmd = get_control_follow_lane(cfg)
            self._send_control(cmd)
            self._log_step(-1, -1, cmd)
            approach_step += 1

            # Collision before ODD exit is unusual but handle it cleanly
            if state.collision:
                self._log_event(
                    f"[COLLISION before ODD exit] "
                    f"at sim_time_s={state.sim_time_s:.2f}"
                )
                return self._build_obs_array()

    # ======================================================================
    # Logging helpers
    # ======================================================================

    def _log_event(self, msg: str) -> None:
        """Print to console and append to run.log."""
        print(msg)
        if self._run_log_handle is not None:
            try:
                self._run_log_handle.write(msg + "\n")
                self._run_log_handle.flush()
            except Exception:
                pass

    def _log_step(self, proposed_mrm: int, final_mrm: int, cmd, rwd: dict = None) -> None:
        """Write one row to the step CSV (same columns as run_phase2.py + reward detail)."""
        if self._step_logger is None or self._state is None or self._obs_dict is None:
            return
        rwd = rwd or {}
        terminal_sum = (
            rwd.get("terminal_collision_penalty", 0.0)
            + rwd.get("terminal_success_bonus", 0.0)
            + rwd.get("terminal_shoulder_bonus", 0.0)
            + rwd.get("terminal_lane_block_penalty", 0.0)
        )
        self._step_logger.log_step(
            {
                "step_id": self._step_id,
                "sim_time_s": self._state.sim_time_s,
                "odd_exit_active": (
                    self._odd.is_active(self._state.sim_time_s)
                    if self._odd is not None
                    else False
                ),
                "ego_speed_mps": self._state.ego.speed_mps,
                "throttle": cmd.throttle,
                "brake": cmd.brake,
                "steer": cmd.steer,
                "collision": self._state.collision,
                "state_source": self._state_source,
                "front_gap_m": self._obs_dict["front_gap_m"],
                "rear_gap_m": self._obs_dict["rear_gap_m"],
                "left_gap_m": self._obs_dict["left_gap_m"],
                "right_gap_m": self._obs_dict["right_gap_m"],
                "drf": self._drf,
                "dara": self._dara,
                "risk_total": self._risk_total,
                "proposed_mrm": proposed_mrm,
                "final_mrm": final_mrm,
                # Reward columns
                "step_reward": rwd.get("total", 0.0),
                "step_reward_risk_penalty": rwd.get("dense_risk_penalty", 0.0),
                "step_reward_step_cost": rwd.get("dense_step_cost", 0.0),
                "step_reward_terminal": terminal_sum,
            }
        )

    def _build_info(self, timeout: bool = False) -> dict:
        """Build the info dict returned by step()."""
        ep_sum = (
            self._episode_tracker.get_summary()
            if self._episode_tracker is not None
            else {}
        )
        return {
            "episode_id": self._episode_id,
            "sim_time_s": (
                float(self._last_sim_time)
                if self._last_sim_time is not None
                else 0.0
            ),
            "step_id": self._step_id,
            "collision": (
                bool(self._state.collision) if self._state is not None else False
            ),
            "mrc_reached": (
                self._mrc_checker.is_mrc_reached()
                if self._mrc_checker is not None
                else False
            ),
            "timeout": timeout,
            "final_mrm": self._current_mrm,
            "risk_total": float(self._risk_total),
            "shoulder_available": self._shoulder_available,
            "max_risk": ep_sum.get("max_risk", 0.0),
            "avg_risk": ep_sum.get("avg_risk", 0.0),
            "scenario_path": self._current_scenario_path,
        }

    def _finalize_episode(self, collision: bool, timeout: bool) -> None:
        """
        Write episode row to episodes.csv and episode.json.
        Mirrors the end-of-loop block in run_phase2.py.
        """
        ep_sum = (
            self._episode_tracker.get_summary()
            if self._episode_tracker is not None
            else {}
        )
        ep_row = {
            "episode_id": self._episode_id,
            "state_source": self._state_source,
            "success_mrc": (
                self._mrc_checker.is_mrc_reached()
                if self._mrc_checker is not None
                else False
            ),
            "collision": collision,
            "steps": self._step_id,
            "max_risk": ep_sum.get("max_risk", 0.0),
            "avg_risk": ep_sum.get("avg_risk", 0.0),
            "mrm_switches": ep_sum.get("mrm_switches", 0),
            "primary_mrm": ep_sum.get("primary_mrm", -1),
            "last_sim_time_s": (
                float(self._last_sim_time)
                if self._last_sim_time is not None
                else None
            ),
            "shoulder_available": self._shoulder_available,
            "min_gap": self._min_gap if self._min_gap < 999.0 else None,
            "time_to_stop": self._time_to_stop_s,
            "time_to_mrc": self._time_to_mrc_s,
            "max_decel": self._max_decel if self._max_decel > 0 else None,
            # Reward breakdown
            "total_reward":               round(self._total_reward, 4),
            "reward_dense":               round(self._dense_reward_sum, 4),
            "reward_terminal":            round(self._terminal_reward, 4),
            "reward_collision_penalty":   round(self._terminal_collision_penalty, 4),
            "reward_success_bonus":       round(self._terminal_success_bonus, 4),
            "reward_shoulder_bonus":      round(self._terminal_shoulder_bonus, 4),
            "reward_lane_block_penalty":  round(self._terminal_lane_block_penalty, 4),
        }

        # Shared logger (owned by caller, e.g. train.py)
        if self._shared_episode_logger is not None:
            self._shared_episode_logger.log_episode(ep_row)
        # Owned logger (created in __init__ when run_dir given and no shared logger)
        elif self._owned_episode_logger is not None:
            self._owned_episode_logger.log_episode(ep_row)

        # episode.json summary
        if self.run_dir:
            write_episode_summary(
                self.run_dir,
                {
                    "episode_id": self._episode_id,
                    "success_mrc": ep_row["success_mrc"],
                    "collision": collision,
                    "timeout": timeout,
                    "steps": self._step_id,
                    "max_risk": ep_row["max_risk"],
                    "avg_risk": ep_row["avg_risk"],
                    "shoulder_available": self._shoulder_available,
                    "time_to_mrc": self._time_to_mrc_s,
                    "primary_mrm": ep_row["primary_mrm"],
                    "total_reward": ep_row["total_reward"],
                    "reward_dense": ep_row["reward_dense"],
                    "reward_terminal": ep_row["reward_terminal"],
                    "reward_collision_penalty": ep_row["reward_collision_penalty"],
                    "reward_success_bonus": ep_row["reward_success_bonus"],
                    "reward_shoulder_bonus": ep_row["reward_shoulder_bonus"],
                    "reward_lane_block_penalty": ep_row["reward_lane_block_penalty"],
                },
            )
