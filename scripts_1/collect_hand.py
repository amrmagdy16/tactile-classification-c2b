#!/usr/bin/env python3
"""
Hand-operated data collection for C2b — NO BAXTER NEEDED.

You press each material against the sensor by hand. The script detects the
contact, records a fixed window, then waits for you to release and present
the next contact. Guarantees one trial per press because it explicitly
waits for release (magnitude drops back near baseline) before re-arming.

Usage:
  python scripts/collect_hand.py <material> <n_trials> [condition]
  e.g. python scripts/collect_hand.py steel 20 baseline

Workflow per trial:
  - Script says "waiting for contact"
  - You press material onto the sensor
  - Script detects contact, records N frames, says "saved"
  - You lift the material off
  - Script waits until magnitude drops (release confirmed), then re-arms
"""

import time
import json
import sys
import numpy as np
from pathlib import Path
from dmrobotics import Sensor

# ---- LOAD CONFIG ----
config_path = Path(__file__).parent.parent / "config" / "experiment_config.json"
with open(config_path) as f:
    config = json.load(f)

SERIAL_ID  = config["sensor"]["serial_id"]
THRESHOLD  = config["baseline_protocol"]["contact_threshold"]
FRAMES     = config["baseline_protocol"]["frames_per_trial"]
RESET_EVERY = config["sensor"]["reset_interval_trials"]
DOWNSAMPLE = config["data"]["downsample_spatial"]
FACTOR     = config["data"]["downsample_factor"]

# Release threshold: must drop below this before next contact arms.
# Set a bit below the contact threshold to avoid bouncing.
RELEASE_THRESHOLD = THRESHOLD * 0.5

# ---- COMMAND LINE ----
if len(sys.argv) < 3:
    print("Usage: python collect_hand.py <material> <n_trials> [condition]")
    print("  e.g. python collect_hand.py steel 20 baseline")
    sys.exit(1)

MATERIAL  = sys.argv[1]
N_TRIALS  = int(sys.argv[2])
CONDITION = sys.argv[3] if len(sys.argv) > 3 else "baseline"

save_path = Path(config["data"]["save_dir"]) / MATERIAL / CONDITION
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== C2b Hand Collection (no Baxter) ===")
print(f"Material:  {MATERIAL}")
print(f"Condition: {CONDITION}")
print(f"Trials:    {N_TRIALS}")
print(f"Frames:    {FRAMES} per trial")
print(f"Contact threshold: {THRESHOLD}")
print(f"Release threshold: {RELEASE_THRESHOLD:.3f}")
print(f"Save:      {save_path}")
print()

# ---- CONNECT SENSOR ----
sensor = Sensor(SERIAL_ID)
print(f"Sensor {SERIAL_ID} connected.")
sensor.reset()
print("Sensor reset, waiting 1.5s...")
time.sleep(1.5)

def downsample(a, f=8):
    return a[::f, ::f, :]

def current_magnitude():
    d = sensor.getDeformation2D()
    return np.mean(np.linalg.norm(d, axis=2))

print("\n=== TECHNIQUE ===")
print("Press the material firmly and evenly onto the sensor, hold ~1 second,")
print("then lift it fully off before the next press. Try to press with")
print("similar force each time for consistency.\n")
input("Press Enter to start...")
print()

trial = 0
try:
    while trial < N_TRIALS:
        # Reset sensor periodically for clean baseline
        if trial > 0 and trial % RESET_EVERY == 0:
            print(f"  [{trial}/{N_TRIALS}] Resetting sensor — keep hand OFF...")
            sensor.reset()
            time.sleep(1.5)

        # --- Wait for contact ---
        print(f"  Trial {trial+1}/{N_TRIALS}: waiting for contact...", end='', flush=True)
        while True:
            if current_magnitude() > THRESHOLD:
                break
            time.sleep(0.01)

        # --- Record N frames ---
        buf_def = []
        buf_shear = []
        for _ in range(FRAMES):
            buf_def.append(sensor.getDeformation2D().copy())
            buf_shear.append(sensor.getShear().copy())
            # read as fast as sensor allows (~120 Hz)

        # --- Save ---
        if DOWNSAMPLE:
            def_data = np.array([downsample(f, FACTOR) for f in buf_def])
            shear_data = np.array([downsample(f, FACTOR) for f in buf_shear])
        else:
            def_data = np.array(buf_def)
            shear_data = np.array(buf_shear)

        fname = save_path / f"trial_{trial:04d}"
        np.save(str(fname) + "_def.npy", def_data)
        np.save(str(fname) + "_shear.npy", shear_data)

        peak = np.max(np.mean(np.linalg.norm(def_data, axis=3), axis=(1, 2)))
        print(f" recorded, saved {def_data.shape}  peak={peak:.3f}")
        trial += 1

        # --- Wait for release before re-arming ---
        print("    lift material off...", end='', flush=True)
        while current_magnitude() > RELEASE_THRESHOLD:
            time.sleep(0.01)
        time.sleep(0.3)  # small debounce
        print(" released.")

except KeyboardInterrupt:
    print(f"\n[Interrupted] {trial} trials saved before stop.")

finally:
    sensor.disconnect()
    metadata = {
        "material": MATERIAL,
        "condition": CONDITION,
        "trials_collected": trial,
        "frames_per_trial": FRAMES,
        "collection_method": "hand",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n=== Done: {trial}/{N_TRIALS} trials ===")
    print(f"Saved to: {save_path}")
