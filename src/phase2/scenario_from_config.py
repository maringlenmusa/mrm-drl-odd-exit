"""
Generate OpenSCENARIO (.xosc) from state_bridge config so the UI (esmini) shows
exactly what the simulation uses: same ego position/speed and same other actors.

This keeps config as the single source of truth: state_bridge drives logic and
logs; the generated scenario drives what you see in the UI.
"""

import os
import tempfile
from typing import Dict, List, Any, Optional


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# Known road networks: straight 500 m or 20° right curve over 500 m (lane layout matches straight_500m).
ROAD_NETWORKS = {
    "straight": {
        "logic_file": "../xodr/straight_500m.xodr",
        "scene_graph": "../models/straight_500m.osgb",
    },
    "curved_20deg": {
        "logic_file": "../xodr/curve_20deg_500m.xodr",
        "scene_graph": "../models/straight_500m.osgb",  # reuse; esmini uses xodr for logic, vehicles follow curve
    },
}


def generate_xosc_from_state_bridge(
    state_bridge_config: Dict[str, Any],
    output_path: Optional[str] = None,
    *,
    road_network: Optional[Dict[str, str]] = None,
    road_network_key: Optional[str] = None,
    esmini_bin_dir: Optional[str] = None,
    odd_exit_trigger_time_s: Optional[float] = None,
) -> str:
    """
    Generate OpenSCENARIO (.xosc) from state_bridge config.

    Uses straight_500m-style lane layout (road id 1, lane -1). Road can be straight or curved_20deg.
    Ego and each actor get TeleportAction + SpeedAction from config.
    If an actor has rear_follower=true and odd_exit_trigger_time_s is set, adds an Act
    that makes that vehicle brake (to 0) starting at odd_exit_trigger_time_s + reaction_delay_s.

    Args:
        state_bridge_config: Dict with initial_speed_mps, initial_x_m, initial_y_m,
            and actors: [ {actor_id, x_m, y_m, speed_mps, lane_id?}, or rear_follower, ... }, ... ]
        output_path: Where to write the .xosc file. If None and esmini_bin_dir is set,
            writes to esmini_bin_dir/resources/xosc/generated_phase2.xosc.
        road_network: Optional {"logic_file": "...", "scene_graph": "..."}; overridden by road_network_key if set.
        road_network_key: "straight" | "curved_20deg" to pick road (20° curve to the right over 500 m).
        esmini_bin_dir: If set and output_path is None, write under this dir so esmini resolves paths.
        odd_exit_trigger_time_s: When ODD exit triggers (s). Used with rear_follower for delayed brake.

    Returns:
        Absolute path to the written .xosc file.
    """
    if road_network is None:
        road_network = ROAD_NETWORKS.get(
            road_network_key if road_network_key else "straight",
            ROAD_NETWORKS["straight"],
        )

    ego_speed = float(state_bridge_config.get("initial_speed_mps", 20.0))
    ego_s = float(state_bridge_config.get("initial_x_m", 50.0))
    ego_y = float(state_bridge_config.get("initial_y_m", 1.75))
    actors_cfg = state_bridge_config.get("actors") or []

    # Lane: for straight_500m, s is longitudinal; laneId -1 is one lane. We map y_m to lane
    # heuristically: same lane as ego -> -1; different y -> use a different lane (e.g. -2 for left).
    def s_from_x(x_m: float) -> float:
        return max(0.0, min(500.0, x_m))

    def lane_from_y(y_m: float) -> int:
        # Simple: if y close to ego_y, same lane (-1); else -2 (other lane)
        if abs(y_m - ego_y) < 1.0:
            return -1
        return -2

    ego_lane = -1

    entities_xml = []
    init_actors = []

    # Ego: UDPDriverController so esmini receives our throttle/brake/steer on port 49950
    entities_xml.append(
        """      <ScenarioObject name="Ego">
         <Vehicle name="car_white" vehicleCategory="car" model3d="../models/car_white.osgb">
            <BoundingBox>
               <Center x="1.4" y="0.0" z="0.9"/>
               <Dimensions width="2.0" length="5.0" height="1.8"/>
            </BoundingBox>
            <Performance maxSpeed="69" maxDeceleration="30" maxAcceleration="10"/>
            <Axles>
               <FrontAxle maxSteering="30" wheelDiameter="0.8" trackWidth="1.68" positionX="2.98" positionZ="0.4"/>
               <RearAxle maxSteering="30" wheelDiameter="0.8" trackWidth="1.68" positionX="0" positionZ="0.4"/>
            </Axles>
            <Properties>
                <Property name="model_id" value="0"/>
                <Property name="scaleMode" value="ModelToBB"/>
            </Properties>
            <ObjectController>
               <CatalogReference catalogName="ControllerCatalog" entryName="UDPDriverController"/>
            </ObjectController>
         </Vehicle>
      </ScenarioObject>"""
    )

    # Rear-follower info for delayed brake Act (same speed as ego, brake after reaction_delay_s, decel_factor × ego)
    rear_entity_name: Optional[str] = None
    rear_brake_trigger_time: Optional[float] = None
    rear_brake_duration_s: float = 5.0  # time to reach 0 in xosc (shorter = stronger decel)
    ego_decel = float(state_bridge_config.get("ego_decel_mps2", 5.0))

    # Other actors (Target1, Target2, ...)
    for i, ac in enumerate(actors_cfg):
        name = f"Target{i + 1}"
        entry_name = "car_red"  # from VehicleCatalog
        entities_xml.append(
            f"""      <ScenarioObject name="{name}">
         <CatalogReference catalogName="VehicleCatalog" entryName="{entry_name}"/>
      </ScenarioObject>"""
        )
        if ac.get("rear_follower"):
            # Rear vehicle: behind ego, same speed as ego; brake triggered later in an Act; decel = ego_decel × decel_factor
            distance_behind = float(ac.get("distance_behind_ego_m", 20.0))
            reaction_delay = float(ac.get("reaction_delay_s", 2.0))
            factor = float(ac.get("decel_factor", 1.0))
            x_m = ego_s - distance_behind
            speed_mps = float(ac.get("speed_mps", ego_speed))  # default same as ego; override for ±% per episode
            if odd_exit_trigger_time_s is not None:
                rear_entity_name = name
                rear_brake_trigger_time = odd_exit_trigger_time_s + reaction_delay
                # Duration so linear ramp to 0 matches rear decel = ego_decel × factor
                if factor > 0 and ego_decel > 0:
                    rear_decel = ego_decel * factor
                    rear_brake_duration_s = round(speed_mps / rear_decel, 2)
                    rear_brake_duration_s = max(1.0, min(10.0, rear_brake_duration_s))
        else:
            speed_mps = float(ac.get("speed_mps", 0.0))
            x_m = float(ac.get("x_m", 100.0))
        y_m = float(ac.get("y_m", ego_y))
        lane_id = int(ac.get("lane_id", lane_from_y(y_m)))
        s = s_from_x(x_m)
        init_actors.append(
            f"""            <Private entityRef="{name}">
               <PrivateAction>
                  <LongitudinalAction>
                     <SpeedAction>
                        <SpeedActionDynamics dynamicsShape="step" dynamicsDimension="time" value="0.0"/>
                        <SpeedActionTarget>
                           <AbsoluteTargetSpeed value="{speed_mps}"/>
                        </SpeedActionTarget>
                     </SpeedAction>
                  </LongitudinalAction>
               </PrivateAction>
               <PrivateAction>
                  <TeleportAction>
                     <Position>
                        <LanePosition roadId="1" laneId="{lane_id}" offset="0" s="{s}"/>
                     </Position>
                  </TeleportAction>
               </PrivateAction>
            </Private>"""
        )

    logic_file = road_network.get("logic_file", "../xodr/straight_500m.xodr")
    scene_graph = road_network.get("scene_graph", "../models/straight_500m.osgb")

    # Build Ego Init: Teleport and Speed first (esmini warns if ActivateController before position), then ActivateController
    ego_init = f"""            <Private entityRef="Ego">
               <PrivateAction>
                  <TeleportAction>
                     <Position>
                        <LanePosition roadId="1" laneId="{ego_lane}" offset="0" s="{s_from_x(ego_s)}"/>
                     </Position>
                  </TeleportAction>
               </PrivateAction>
               <PrivateAction>
                  <LongitudinalAction>
                     <SpeedAction>
                        <SpeedActionDynamics dynamicsShape="step" dynamicsDimension="time" value="0.0"/>
                        <SpeedActionTarget>
                           <AbsoluteTargetSpeed value="{ego_speed}"/>
                        </SpeedActionTarget>
                     </SpeedAction>
                  </LongitudinalAction>
               </PrivateAction>
               <PrivateAction>
                  <ControllerAction>
                     <ActivateControllerAction longitudinal="true" lateral="true"/>
                  </ControllerAction>
               </PrivateAction>
            </Private>"""
    init_actions = [ego_init]
    init_actions.extend(init_actors)

    # Optional Act: rear vehicle brakes at odd_exit_trigger_time_s + reaction_delay_s (e.g. 8+2=10 s)
    rear_brake_act_xml = ""
    if rear_entity_name and rear_brake_trigger_time is not None:
        t_trigger = rear_brake_trigger_time
        rear_brake_act_xml = f"""
         <Act name="RearBrakeAct">
            <ManeuverGroup maximumExecutionCount="1" name="RearBrakeGroup">
               <Actors selectTriggeringEntities="false">
                  <EntityRef entityRef="{_escape_xml(rear_entity_name)}"/>
               </Actors>
               <Maneuver name="RearBrake">
                  <Event name="RearBrakeEvent" priority="overwrite">
                     <Action name="RearBrakeAction">
                        <PrivateAction>
                           <LongitudinalAction>
                              <SpeedAction>
                                 <SpeedActionDynamics dynamicsShape="linear" dynamicsDimension="time" value="{rear_brake_duration_s}"/>
                                 <SpeedActionTarget>
                                    <AbsoluteTargetSpeed value="0"/>
                                 </SpeedActionTarget>
                              </SpeedAction>
                           </LongitudinalAction>
                        </PrivateAction>
                     </Action>
                     <StartTrigger>
                        <ConditionGroup>
                           <Condition name="RearBrakeTrigger" delay="0" conditionEdge="rising">
                              <ByValueCondition>
                                 <SimulationTimeCondition value="{t_trigger}" rule="greaterThan"/>
                              </ByValueCondition>
                           </Condition>
                        </ConditionGroup>
                     </StartTrigger>
                  </Event>
               </Maneuver>
            </ManeuverGroup>
            <StartTrigger>
               <ConditionGroup>
                  <Condition name="ActStart" delay="0" conditionEdge="none">
                     <ByValueCondition>
                        <SimulationTimeCondition value="0" rule="greaterThan"/>
                     </ByValueCondition>
                  </Condition>
               </ConditionGroup>
            </StartTrigger>
         </Act>"""

    story_acts = """         <Act name="DummyAct">
            <ManeuverGroup maximumExecutionCount="1" name="DummyManueverGroup">
               <Actors selectTriggeringEntities="false">
                  <EntityRef entityRef="Ego"/>
               </Actors>
            </ManeuverGroup>
            <StartTrigger>
               <ConditionGroup>
                  <Condition name="StartTrigger" delay="0" conditionEdge="none">
                     <ByValueCondition>
                        <SimulationTimeCondition value="0" rule="greaterThan"/>
                     </ByValueCondition>
                  </Condition>
               </ConditionGroup>
            </StartTrigger>
         </Act>""" + rear_brake_act_xml

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OpenSCENARIO>
   <FileHeader revMajor="1" revMinor="1" date="2020-05-12T10:00:00" description="Generated from state_bridge config" author="phase2"/>
   <ParameterDeclarations></ParameterDeclarations>
   <CatalogLocations>
      <VehicleCatalog>
         <Directory path="../xosc/Catalogs/Vehicles"/>
      </VehicleCatalog>
      <ControllerCatalog>
         <Directory path="../xosc/Catalogs/Controllers"/>
      </ControllerCatalog>
   </CatalogLocations>
   <RoadNetwork>
      <LogicFile filepath="{_escape_xml(logic_file)}"/>
      <SceneGraphFile filepath="{_escape_xml(scene_graph)}"/>
   </RoadNetwork>
   <Entities>
{chr(10).join(entities_xml)}
   </Entities>
   <Storyboard>
      <Init>
         <Actions>
{chr(10).join(init_actions)}
         </Actions>
      </Init>
      <Story name="MyStory">
{story_acts}
      </Story>
      <StopTrigger>
         <ConditionGroup>
            <Condition name="StopCondition" delay="0" conditionEdge="rising">
               <ByValueCondition>
                  <SimulationTimeCondition value="30" rule="greaterThan"/>
               </ByValueCondition>
            </Condition>
         </ConditionGroup>
      </StopTrigger>
   </Storyboard>
</OpenSCENARIO>"""

    if output_path is None and esmini_bin_dir:
        output_path = os.path.join(
            esmini_bin_dir, "resources", "xosc", "generated_phase2.xosc"
        )
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".xosc")
        os.close(fd)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return os.path.abspath(output_path)
