"""
Simple tests for Phase 2 features: neighbor_selector and observation_builder.

Run: python -m src.phase2.test_features

Verifies:
- select_neighbors() finds front/rear/left/right correctly
- build_observation() returns dict with ego_speed_mps, ego_lane_id, *_gap_m, *_rel_speed_mps
"""

from src.types import EgoState, ActorState, SimState
from src.features.neighbor_selector import select_neighbors, Neighbors
from src.features.observation_builder import build_observation, NO_GAP_M


def test_neighbor_selector():
    """Verify front/rear/left/right neighbors are selected correctly."""
    ego = EgoState(x=100.0, y=1.75, speed_mps=15.0, yaw_rad=0.0, lane_id=-1)

    # Actors: one ahead (front), one behind (rear), one left, one right. Same lane = |dy| <= 1.75.
    actors = [
        ActorState(actor_id=1, x=130.0, y=1.75, speed_mps=10.0, yaw_rad=0.0, lane_id=-1),   # front (same lane)
        ActorState(actor_id=2, x=70.0, y=1.75, speed_mps=20.0, yaw_rad=0.0, lane_id=-1),   # rear (same lane)
        ActorState(actor_id=3, x=100.0, y=5.5, speed_mps=15.0, yaw_rad=0.0, lane_id=-2),   # left (y > ego.y)
        ActorState(actor_id=4, x=100.0, y=-1.8, speed_mps=15.0, yaw_rad=0.0, lane_id=0),   # right (y < ego.y)
    ]

    neighbors = select_neighbors(ego, actors, lane_half_width_m=1.75)

    assert neighbors.front is not None and neighbors.front.actor_id == 1
    assert neighbors.rear is not None and neighbors.rear.actor_id == 2
    assert neighbors.left is not None and neighbors.left.actor_id == 3
    assert neighbors.right is not None and neighbors.right.actor_id == 4

    # No actors: all None
    empty = select_neighbors(ego, [])
    assert empty.front is None and empty.rear is None and empty.left is None and empty.right is None

    print("  [OK] select_neighbors(): front/rear/left/right correct")


def test_observation_builder():
    """Verify build_observation() returns dict with required keys and sensible values."""
    ego = EgoState(x=100.0, y=1.75, speed_mps=15.0, yaw_rad=0.0, lane_id=-1)
    # Same lane (same y): only front and rear, no left/right neighbors
    actors = [
        ActorState(actor_id=1, x=120.0, y=1.75, speed_mps=10.0, yaw_rad=0.0, lane_id=-1),  # front 20m
        ActorState(actor_id=2, x=80.0, y=1.75, speed_mps=20.0, yaw_rad=0.0, lane_id=-1),  # rear 20m
    ]
    state = SimState(sim_time_s=5.0, ego=ego, collision=False, actors=actors)

    obs = build_observation(state)

    required = [
        "ego_speed_mps", "ego_lane_id",
        "front_gap_m", "rear_gap_m", "left_gap_m", "right_gap_m",
        "front_rel_speed_mps", "rear_rel_speed_mps", "left_rel_speed_mps", "right_rel_speed_mps",
    ]
    for key in required:
        assert key in obs, f"Missing key: {key}"

    assert obs["ego_speed_mps"] == 15.0
    assert obs["ego_lane_id"] == -1
    assert obs["front_gap_m"] == 20.0
    assert obs["rear_gap_m"] == 20.0
    assert obs["left_gap_m"] == NO_GAP_M  # no left neighbor (same y only)
    assert obs["right_gap_m"] == NO_GAP_M  # no right neighbor
    assert obs["front_rel_speed_mps"] == 5.0   # ego 15, front 10 => closing +5
    assert obs["rear_rel_speed_mps"] == 5.0    # rear 20, ego 15 => closing +5

    # No actors: all gaps NO_GAP_M, rel speeds 0
    state_empty = SimState(sim_time_s=0.0, ego=ego, collision=False, actors=[])
    obs_empty = build_observation(state_empty)
    assert obs_empty["front_gap_m"] == NO_GAP_M and obs_empty["rear_gap_m"] == NO_GAP_M
    assert obs_empty["front_rel_speed_mps"] == 0.0 and obs_empty["rear_rel_speed_mps"] == 0.0

    print("  [OK] build_observation(): dict with ego_speed_mps, ego_lane_id, *_gap_m, *_rel_speed_mps")


def main():
    print("Phase 2 feature tests")
    test_neighbor_selector()
    test_observation_builder()
    print("All Phase 2 feature tests passed.")


if __name__ == "__main__":
    main()
