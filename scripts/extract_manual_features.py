import numpy as np
import os
import glob

DATA_DIR = "data/raw/sponge/manual_test/"

def extract_features(def_file, shear_file):
    # 1. Load the 4D data cubes
    def_data = np.load(def_file)
    shear_data = np.load(shear_file)
    
    # 2. Collapse the spatial dimensions to get the time-series curves
    def_mag_time = np.mean(np.linalg.norm(def_data, axis=3), axis=(1,2))
    shear_mag_time = np.mean(np.linalg.norm(shear_data, axis=3), axis=(1,2))
    
    # --- FEATURE 1: Peak Deformation (Correlates to Hardness) ---
    peak_def = np.max(def_mag_time)
    peak_idx = np.argmax(def_mag_time)
    
    # --- FEATURE 2: Peak Shear (Correlates to Friction/Texture) ---
    # How much shear was happening at the exact moment of maximum squeeze?
    peak_shear = shear_mag_time[peak_idx]
    
    # --- FEATURE 3: Loading Slope (Correlates to Stiffness) ---
    # How fast did the sensor reach peak deformation? (Steeper = stiffer material)
    if peak_idx > 0:
        loading_slope = peak_def / peak_idx
    else:
        loading_slope = 0
        
    # --- FEATURE 4: Viscoelastic Relaxation (Correlates to Softness/Foam Type) ---
    # How much does the material "give" or relax while being held steady?
    # We compare the peak to the end of the recording (frame 149)
    end_hold_mag = def_mag_time[-1]
    relaxation = peak_def - end_hold_mag
    
    return {
        "Peak Deformation": peak_def,
        "Peak Shear": peak_shear,
        "Loading Slope": loading_slope,
        "Relaxation": relaxation
    }

print("Extracting features from manual sponge trials...\n")
print(f"{'Trial':<10} | {'Peak Def':<12} | {'Peak Shear':<12} | {'Slope':<10} | {'Relaxation':<12}")
print("-" * 65)

# Find all the deformation files
def_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_def.npy")))

for def_file in def_files:
    # Find the matching shear file
    shear_file = def_file.replace("_def.npy", "_shear.npy")
    trial_name = os.path.basename(def_file).split("_def")[0]
    
    try:
        features = extract_features(def_file, shear_file)
        print(f"{trial_name:<10} | {features['Peak Deformation']:<12.4f} | {features['Peak Shear']:<12.4f} | {features['Loading Slope']:<10.4f} | {features['Relaxation']:<12.4f}")
    except Exception as e:
        print(f"Error processing {trial_name}: {e}")
