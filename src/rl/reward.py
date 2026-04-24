"""
Phase 3 RL reward function.

Design from docs/phase3_bandit_plan.md:

Dense per-step signal (when ODD is active):
  - Risk penalty  : -risk_weight * risk_total   (penalises high-risk situations)
  - Step cost     : -step_cost                  (encourages faster stopping)

Terminal signal (only at the final step of each episode):
  - Collision         : -collision_penalty       (large negative — safety critical)
  - MRC success       : +success_bonus           (reached full stop safely)
  - Shoulder bonus    : +shoulder_bonus          (shoulder used when it was the right choice)
  - Lane-block penalty: -lane_block_penalty      (ego blocked lane when shoulder was available)

All weights are read from cfg["reward"] so they can be tuned per experiment.

Ablation flags (from ablations dict):
  - no_risk_penalty: set risk_weight = 0 (keep risk in obs but don't penalise it)
"""

from typing import Dict, Any, Optional

from src.decision.mrm_catalog import MRM


def compute_reward(
    *,
    collision: bool,
    mrc_reached: bool,
    risk_total: float,
    final_mrm: int,
    obs_dict: Dict[str, Any],
    shoulder_available: bool,
    cfg: dict,
    done: bool,
    ablations: Optional[Dict[str, bool]] = None,
) -> float:
    """
    Compute step reward for the RL agent.

    Returns:
        Float reward for this step.
    """
    return compute_reward_detail(
        collision=collision,
        mrc_reached=mrc_reached,
        risk_total=risk_total,
        final_mrm=final_mrm,
        obs_dict=obs_dict,
        shoulder_available=shoulder_available,
        cfg=cfg,
        done=done,
        ablations=ablations,
    )["total"]


def compute_reward_detail(
    *,
    collision: bool,
    mrc_reached: bool,
    risk_total: float,
    final_mrm: int,
    obs_dict: Dict[str, Any],
    shoulder_available: bool,
    cfg: dict,
    done: bool,
    ablations: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    """
    Same calculation as compute_reward() but returns all components so the
    caller can log each part individually.

    Returns a dict with:
      total                   — combined reward this step
      dense_risk_penalty      — -risk_weight * risk_total  (0 when ablated)
      dense_step_cost         — fixed -step_cost per step
      terminal_collision_penalty — applied at done when collision
      terminal_success_bonus     — applied at done when MRC reached
      terminal_shoulder_bonus    — applied at done when shoulder used correctly
      terminal_lane_block_penalty— applied at done when lane blocked despite available shoulder
    """
    ablations = ablations or {}
    reward_cfg = cfg.get("reward", {})

    risk_weight = float(reward_cfg.get("risk_step_penalty_weight", 0.5))
    step_cost   = float(reward_cfg.get("step_cost", 0.1))
    if ablations.get("no_risk_penalty", False):
        risk_weight = 0.0

    d_risk  = -risk_weight * risk_total
    d_step  = -step_cost
    t_coll  = 0.0
    t_succ  = 0.0
    t_shldr = 0.0
    t_lane  = 0.0

    if done:
        collision_penalty = float(reward_cfg.get("collision_penalty", 100.0))
        success_bonus     = float(reward_cfg.get("success_bonus", 10.0))
        shoulder_bonus    = float(reward_cfg.get("shoulder_bonus", 5.0))
        lane_block_pen    = float(reward_cfg.get("lane_block_penalty", 2.0))

        if collision:
            t_coll = -collision_penalty
        elif mrc_reached:
            t_succ = success_bonus
            if shoulder_available:
                if final_mrm == MRM.ROAD_SHOULDER_STOP:
                    # Shoulder was available and agent used it — best outcome
                    t_shldr = shoulder_bonus
                elif final_mrm == MRM.STRAIGHT_STOP:
                    # Shoulder available but agent stopped straight in the lane — blocks traffic
                    t_lane = -lane_block_pen
                # IN_LANE_STOP when shoulder available: no bonus, no penalty
                # (agent chose a reasonable option, just not the ideal one)

    total = d_risk + d_step + t_coll + t_succ + t_shldr + t_lane
    return {
        "total":                      float(total),
        "dense_risk_penalty":         float(d_risk),
        "dense_step_cost":            float(d_step),
        "terminal_collision_penalty": float(t_coll),
        "terminal_success_bonus":     float(t_succ),
        "terminal_shoulder_bonus":    float(t_shldr),
        "terminal_lane_block_penalty":float(t_lane),
    }
