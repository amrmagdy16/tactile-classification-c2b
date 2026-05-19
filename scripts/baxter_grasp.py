#!/usr/bin/env python3
"""
Baxter gripper controller for C2b grasp-release protocol.
Connects via rosbridge WebSocket, commands left gripper to grasp materials repeatedly.

VERIFIED AGAINST:
  - EndEffectorCommand.msg: uint32 id, string command, string args, string sender, uint32 sequence
  - JointCommand.msg: int32 mode (1=POSITION), float64[] command, string[] names
  - baxter_cli._send_gripper_position(): id=gripper_id, command='go', args=json.dumps({'position': pos})
  - Topics: /robot/end_effector/left_gripper/command, /robot/limb/left/joint_command
  - Gripper ID: 65538 (default)
"""

import time
import json
import sys
from pathlib import Path
import roslibpy

# ---- LOAD CONFIG ----
config_path = Path(__file__).parent.parent / "config" / "experiment_config.json"
with open(config_path) as f:
    config = json.load(f)

BAXTER_HOST = config["baxter"]["host"]
BAXTER_PORT = config["baxter"]["port"]
GRIPPER_ID  = config["baxter"]["gripper_id"]       # 65538

GRIP_OPEN   = config["baseline_protocol"]["grip_open_position"]    # 100.0
GRIP_CLOSED = config["baseline_protocol"]["grip_closed_position"]  # CALIBRATE IN LAB
HOLD_TIME   = config["baseline_protocol"]["hold_time_s"]           # 1.0
INTER_TRIAL = config["baseline_protocol"]["inter_trial_pause_s"]   # 2.0

ARM_POSITIONS = config["baxter"]["arm_joint_positions"]  # LEFT_TABLE_LOW
ARM_NAMES     = config["baxter"]["arm_joint_names"]      # left_s0 through left_w2

# ---- COMMAND LINE ----
if len(sys.argv) < 2:
    print("Usage: python baxter_grasp.py <n_trials>")
    print("  e.g. python baxter_grasp.py 10   (pilot)")
    print("  e.g. python baxter_grasp.py 50   (full baseline)")
    sys.exit(1)

N_TRIALS = int(sys.argv[1])

# ---- CONNECT ----
print(f"=== C2b Baxter Grasp Controller ===")
print(f"Connecting to {BAXTER_HOST}:{BAXTER_PORT}...")

client = roslibpy.Ros(host=BAXTER_HOST, port=BAXTER_PORT)
client.run()

if not client.is_connected:
    print("ERROR: Connection failed. Check you're on the lab network.")
    sys.exit(1)

print(f"Connected!")
print(f"  Gripper ID:     {GRIPPER_ID}")
print(f"  Open position:  {GRIP_OPEN}")
print(f"  Close position: {GRIP_CLOSED}  <-- MUST BE CALIBRATED")
print(f"  Hold time:      {HOLD_TIME}s")
print(f"  Inter-trial:    {INTER_TRIAL}s")
print(f"  Trials:         {N_TRIALS}")
print()

# ---- PUBLISHERS ----
# Topic and message type match baxter_cli exactly
gripper_pub = roslibpy.Topic(
    client,
    '/robot/end_effector/left_gripper/command',      # same as baxter_cli
    'baxter_core_msgs/EndEffectorCommand'
)

arm_pub = roslibpy.Topic(
    client,
    '/robot/limb/left/joint_command',                # same as baxter_cli
    'baxter_core_msgs/JointCommand'
)

def send_gripper_position(position):
    """
    Send gripper position command.
    Matches baxter_cli._send_gripper_position() exactly:
      msg = {'id': gripper_id, 'command': 'go', 'args': json.dumps({'position': pos})}
    """
    position = max(0.0, min(100.0, float(position)))  # clamp like baxter_cli does
    msg = {
        'id': GRIPPER_ID,
        'command': 'go',                              # CMD_GO from EndEffectorCommand.msg
        'args': json.dumps({'position': position}),   # JSON-encoded position
        'sender': 'c2b_grasp_script',                 # optional identifier
        'sequence': 0                                  # optional sequence number
    }
    gripper_pub.publish(roslibpy.Message(msg))

def move_arm_to_table():
    """
    Move left arm to table_low pose.
    Matches JointCommand.msg: mode=1 (POSITION_MODE), command=float64[], names=string[]
    """
    msg = {
        'mode': 1,                    # POSITION_MODE from JointCommand.msg
        'command': ARM_POSITIONS,     # [-0.55, -0.95, 0.10, 1.70, 0.00, 1.05, 0.00]
        'names': ARM_NAMES            # ['left_s0', ..., 'left_w2']
    }
    arm_pub.publish(roslibpy.Message(msg))

# ---- GRASP PROTOCOL ----
def grasp_cycle(n_trials):
    # Step 1: Position arm at table height (once for all trials)
    print("Step 1: Moving arm to table_low...")
    move_arm_to_table()
    print("  Waiting 4s for arm to settle...")
    time.sleep(4.0)
    
    # Step 2: Open gripper
    print("Step 2: Opening gripper...")
    send_gripper_position(GRIP_OPEN)
    time.sleep(1.5)
    
    # Step 3: Wait for user to place material
    print(f"\n>>> Place material between gripper fingers <<<")
    print(f">>> Then press Enter to start {n_trials} grasps <<<\n")
    input("Press Enter when ready...")
    print()
    
    # Step 4: Run grasp-release cycles
    for i in range(n_trials):
        print(f"  Grasp {i+1}/{n_trials}...", end='', flush=True)
        
        # Open (release previous grasp)
        send_gripper_position(GRIP_OPEN)
        time.sleep(1.2)  # wait for gripper to fully open
        
        # Close (grasp material)
        send_gripper_position(GRIP_CLOSED)
        time.sleep(HOLD_TIME)  # hold the grasp — sensor captures contact
        
        # Open (release)
        send_gripper_position(GRIP_OPEN)
        time.sleep(INTER_TRIAL)  # pause between trials
        
        print(" done")
    
    print(f"\n=== All {n_trials} grasps completed ===")

# ---- MAIN ----
try:
    grasp_cycle(N_TRIALS)
except KeyboardInterrupt:
    print("\n[Interrupted] Opening gripper for safety...")
    send_gripper_position(GRIP_OPEN)
    time.sleep(1.0)
finally:
    print("Cleaning up...")
    gripper_pub.unadvertise()
    arm_pub.unadvertise()
    client.terminate()
    print("Disconnected from Baxter.")
