#!/usr/bin/env python3
import time, numpy as np, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dmrobotics import Sensor

config = json.load(open(Path(__file__).parent.parent / "config" / "experiment_config.json"))
SERIAL_ID = config["sensor"]["serial_id"]
THRESHOLD = config["baseline_protocol"]["contact_threshold"]
FRAMES = config["baseline_protocol"]["frames_per_trial"]
RESET_EVERY = config["sensor"]["reset_interval_trials"]
DOWNSAMPLE = config["data"]["downsample_spatial"]
FACTOR = config["data"]["downsample_factor"]

if len(sys.argv) < 2:
    print("Usage: python collect_data.py <material> [condition]")
    sys.exit(1)

MATERIAL, CONDITION = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "baseline"
TRIALS = config["baseline_protocol"]["trials_per_material"]

save_path = Path(config["data"]["save_dir"]) / MATERIAL / CONDITION
save_path.mkdir(parents=True, exist_ok=True)

print(f"=== C2b Data Collection ===\nMaterial: {MATERIAL}\nCondition: {CONDITION}\nTrials: {TRIALS}\nSave: {save_path}\n")

sensor = Sensor(SERIAL_ID)
sensor.reset()
time.sleep(1.5)

trial, recording, buf_def, buf_shear, waiting = 0, False, [], [], True

def down(a, f=8): return a[::f, ::f, :]

print(f"Ready for {TRIALS} contacts...\n")

try:
    while trial < TRIALS:
        deformation, shear = sensor.getDeformation2D(), sensor.getShear()
        mag = np.mean(np.linalg.norm(deformation, axis=2))
        
        if waiting and mag > THRESHOLD:
            recording, waiting, buf_def, buf_shear = True, False, [], []
            print(f"  Trial {trial+1}: contact (mag={mag:.3f})", end='')
        
        if recording:
            buf_def.append(deformation.copy())
            buf_shear.append(shear.copy())
            
            if len(buf_def) >= FRAMES:
                def_data = np.array([down(f, FACTOR) for f in buf_def]) if DOWNSAMPLE else np.array(buf_def)
                shear_data = np.array([down(f, FACTOR) for f in buf_shear]) if DOWNSAMPLE else np.array(buf_shear)
                
                fname = save_path / f"trial_{trial:04d}"
                np.save(str(fname) + "_def.npy", def_data)
                np.save(str(fname) + "_shear.npy", shear_data)
                
                print(f" -> saved ({def_data.shape})")
                trial += 1
                recording, waiting = False, True
                
                if trial % RESET_EVERY == 0 and trial < TRIALS:
                    print(f"  [{trial}/{TRIALS}] Resetting sensor...")
                    sensor.reset()
                    time.sleep(1.5)
except KeyboardInterrupt:
    print(f"\n[Interrupted] Saved {trial} trials.")
finally:
    sensor.disconnect()
    print(f"\nComplete: {trial}/{TRIALS}\nSaved to: {save_path}")
    json.dump({"material": MATERIAL, "condition": CONDITION, "trials": trial, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, 
              open(save_path / "metadata.json", "w"), indent=2)
