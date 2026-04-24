'''
   This script shows how to fetch and parse OSI message on UDP socket from esmini
   Prerequisites:
      Python 3

   Python dependencies:
      pip install protobuf==3.20.2

   To run it:
   1. Open two terminals
   2. From terminal 1, run: ./scripts/udp_driver/testUDPDriver-print-osi-info.py
        or python ./scripts/udp_driver/testUDPDriver-print-osi-info.py
        or python3 ./scripts/udp_driver/testUDPDriver-print-osi-info.py
        depending on platform and file type associations
   3. From terminal 2, run: ./bin/esmini --window 60 60 800 400 --osc ./scripts/udp_driver/one_car_on_road.xosc --osi_receiver_ip 127.0.0.1

   Note: The example involves printing of some static data, e.g. stationary objects, that is only posted in the first OSI frame. Hence,
         make sure to start the script BEFORE esmini, in order to not miss that very first OSI message.

   For complete driver control definitions, see esmini/EnvironmentSimulator/Modules/Controllers/ControllerUDPDriver.hpp
'''
import math
import time
import sys
import struct
import numpy as np
from socket import timeout
from udp_osi_common import *
from openpyxl import Workbook



# --- Excel file path ---
excel_file = r"C:\Users\kiran\Documents\esmini-demo\scripts\udp_driver\osi_data.xlsx"


k_theta = 0.5   # relative motion sensitivity
k_r = 0.05      # distance decay
MAX_DISTANCE = 50.0  # maximum distance to consider a vehicle for risk

'''
# Driver Class to send back control signals to the simualtion
class Driver():
    def __init__(self):
        self.steering = 0.0
        self.speed = 0.0
        self.target_speed = 50 / 3.6  # km/h
        self.throttle = 0.0
        self.brake = 0.0
    
    def trajectory_function(self, x):
        # linear zigzag shape 
        # First a line from 0,0 to 100,100 (y=x)
        
        # Then a line from 100,100 to 200,0 (y=-x+200)
        # Then repeat infinitely along x. 
        if x % 200 < 100:
            return x % 200
        else:
            return -(x % 200) + 200

    def step(self, speed, x, y, h):

        lookahead = max(5.0, 1.2 * speed)  # look ahead some distance proportional to speed
        # Lateral position on the trajectory
        y_target = self.trajectory_function(x+lookahead)  # look ahead 10 m

        # Calculate angle to target point from current location
        angle = math.atan((y_target - y) / ((x+lookahead) - x))
        # Subtract current heading/yaw of the vehicle and apply some scaling factor
        self.steering = 0.5 * (angle - h)

        # Give throttle or brake to reach target_speed
        if self.target_speed - speed > 0:
            self.throttle =  0.05 * (self.target_speed - speed)
            self.brake = 0.0
        else:
            self.throttle =  0.0
            self.brake = 0.05 * (speed - self.target_speed)

        # slow down with increased steering angle
        self.throttle -= 0.4 * abs(self.steering)
        self.brake += 0.4 * abs(self.steering)
'''

# Fucntion to calculate total ego-centric risk
def calculate_total_risk(msg):
    """
    Calculates total risk induced by all other moving objects on ego vehicle (id=0).
    Only vehicles within MAX_DISTANCE are considered.
    """
    # Find ego vehicle (ego vehicle is assigned the id 0)
    ego = None
    for o in msg.moving_object:
        if o.id.value == 0:
            ego = o
            break

    if ego is None:
        return 0.0

    # Ego vehicle states (all the relative values are calculated in reference to Ego vehicle)
    xe, ye = ego.base.position.x, ego.base.position.y
    vex, vey = ego.base.velocity.x, ego.base.velocity.y
    ve = math.hypot(vex, vey)

    if ve < 1e-3:
        return 0.0

    total_risk = 0.0

    # Loop over other vehicles
    for o in msg.moving_object:
        if o.id.value == 0:
            continue  # skip ego itself

        # Other vehicle state
        xo, yo = o.base.position.x, o.base.position.y
        vox, voy = o.base.velocity.x, o.base.velocity.y
        vo = math.hypot(vox, voy) #euclidean velocity since the vehicle has velocity in both lateral and longitudinal directions

        # Relative position
        dx = xo - xe
        dy = yo - ye
        d = math.hypot(dx, dy) #euclidean distance btw ego and other vehicles

        # checking if the relative Distance is above threshold 50 meter (can be updated accordingly) and if the distance and velocity difference
        # are too small, the calculation is skipped to avoid division by zero and irrelevant velocity values 
        if d > MAX_DISTANCE or d < 1e-3 or vo < 1e-3:
            continue

        # Cosine projections along relative position
        cos_theta_e = (vex * dx + vey * dy) / (ve * d)
        cos_theta_o = (vox * dx + voy * dy) / (vo * d)

        # Clamp to [-1,1] for numerical stability
        cos_theta_e = max(-1.0, min(1.0, cos_theta_e))
        cos_theta_o = max(-1.0, min(1.0, cos_theta_o))

        # Relative motion influence
        delta_v = ve * cos_theta_e - vo * cos_theta_o
        xi_RM = math.exp(k_theta * delta_v)

        # Distance decay
        xi_D = math.exp(-k_r * d)

        # Add contribution to total risk (in the journal three terms are considered; one which takes into account vehicle massess is neglected for simplicity)
        total_risk += xi_RM * xi_D

    return total_risk


def print_osi_stuff(msg, risk):

    print("OSI message timestamp: {:.2f} seconds".format(msg.timestamp.seconds + msg.timestamp.nanos * 1e-9))

    # Print some static content typically only available in first message
    print("{} lanes".format(len(msg.lane)))
    for i, l in enumerate(msg.lane):
        clf = l.classification
        print("  [{}] id {} type: {}".format(i, l.id.value, clf.type))
        print("    centerline:")
        for c_line in clf.centerline:
            print("    x: {:.2f} y: {:.2f}".format(c_line.x, c_line.y))

    print('{} stationary objects'.format(len(msg.stationary_object)))
    for i, s in enumerate(msg.stationary_object):
        print('  [{}] id {} type {}'.format(i, s.id.value, s.classification.type))
        print('    pos.x {:.2f} pos.y {:.2f} rot.h {:.2f}'.format(s.base.position.x, s.base.position.y, s.base.orientation.yaw))

    # Print dynamic content from the message
    print('{} moving objects'.format(len(msg.moving_object)))
    for i, o in enumerate(msg.moving_object):
        print('  [{}] id {}'.format(i, o.id.value))
        print('    pos.x {:.2f} pos.y {:.2f} rot.h {:.2f}'.format(o.base.position.x, o.base.position.y, o.base.orientation.yaw))
        print('    vel.x {:.2f} vel.y {:.2f} rot_rate.h {:.2f}'.format(o.base.velocity.x, o.base.velocity.y, o.base.orientation_rate.yaw))
        print('    acc.x {:.2f} acc.y {:.2f} rot_acc.h {:.2f}'.format(o.base.acceleration.x, o.base.acceleration.y, o.base.orientation_acceleration.yaw))

        lane_id = o.assigned_lane_id[0].value if len(msg.lane) > 0 and len(o.assigned_lane_id) > 0 else -1
        left_lane_id = -1
        right_lane_id = -1
        for l in msg.lane:
            if l.id.value == o.assigned_lane_id[0].value:
                left_lane_id = l.classification.left_adjacent_lane_id[0].value if len(l.classification.left_adjacent_lane_id) > 0 else -1
                right_lane_id = l.classification.right_adjacent_lane_id[0].value if len(l.classification.right_adjacent_lane_id) > 0 else -1
                break
        print('    lane id {} left adj lane id {} right adj lane id {}'.format(lane_id, left_lane_id, right_lane_id))
        
    # Print total ego-centric risk
    print("Total ego-centric risk: {:.4f}".format(risk))
    print("="*80)

# --- Function to write OSI info and risk to Excel --- for further analysis
def write_osi_to_excel(ws, msg, total_risk):
    timestamp = msg.timestamp.seconds + msg.timestamp.nanos * 1e-9
    row_data = [timestamp, total_risk]

    # Add moving object data (ID, pos, yaw, vel, acc)
    for o in msg.moving_object:
        row_data.extend([
            o.id.value,
            o.base.position.x,
            o.base.position.y,
            o.base.orientation.yaw,
            o.base.velocity.x,
            o.base.velocity.y,
            o.base.acceleration.x,
            o.base.acceleration.y
        ])
    ws.append(row_data)


if __name__ == "__main__":

    id = 0
    # Create UDP socket objects
    udpSender0 = UdpSender(port = 53995)
    osiReceiver = OSIReceiver()
    #driver = Driver()
    done = False
    counter = 0
    frame = 0


    # --- Setup Excel workbook ---
    wb = Workbook()
    ws = wb.active
    ws.title = "OSI_Data"

    # Prepare headers
    headers = ["Timestamp", "Total_Ego_Risk"]
    for i in range(20):
        headers.extend([
            f"ID_{i}", f"X_{i}", f"Y_{i}", f"Yaw_{i}", f"Vx_{i}", f"Vy_{i}", f"Ax_{i}", f"Ay_{i}"
        ])
    ws.append(headers)

    excel_file = r"C:\Users\kiran\Documents\esmini-demo\scripts\udp_driver\osi_data.xlsx"
    

    udpSender0.send(
        struct.pack(
            'iiiiddd',
            1,                  # version
            input_modes['driverInput'],
            0,                  # object ID (Car 1)
            frame,              # frame number
            # below values won't affect the ego vehicle states if no controller is assigned to the ego vehicle in the OpenSCENARIO file
            0.5,  # throttle
            0.0,  # brake 
            0.0,  # steering angle
        )
        )


    while not done: #while loop will run for every timestep to calculated risk and print OSI data
        # Read OSI
        try:
            msg = osiReceiver.receive()
            #print_osi_stuff(msg)


        
        except timeout:
            print('osiReceive Timeout')
            continue
        except KeyboardInterrupt:
            print('Ctrl+C pressed, quit')
            done = True


        # --- Calculate ego-centric risk ---
        total_risk = calculate_total_risk(msg)

        # --- Print OSI data + risk ---
        print_osi_stuff(msg, total_risk)

                # --- Write to Excel ---
        write_osi_to_excel(ws, msg, total_risk)
        wb.save(excel_file)  # Save after each timestep


        

        frame += 1

    

    # --- Save Excel file ---
    wb.save(excel_file)
    print(f"Excel data saved to: {excel_file}")

    # Close and quit
    udpSender0.close()
    osiReceiver.close()
