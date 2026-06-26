import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- Configuration ---
# Point this to the exact folder you just listed in your terminal
DATA_DIR = "data/raw/sponge/manual_test/"
TRIAL_NUM = "0000"  # Change this to look at different trials

def_file = os.path.join(DATA_DIR, f"trial_{TRIAL_NUM}_def.npy")
shear_file = os.path.join(DATA_DIR, f"trial_{TRIAL_NUM}_shear.npy")

print(f"Loading files:\n{def_file}\n{shear_file}")

try:
    # 1. Load the data
    def_data = np.load(def_file)
    shear_data = np.load(shear_file)
    print(f"Deformation shape: {def_data.shape}")
    
    # --- Check 1: The Temporal Loading Curve ---
    # Calculate the mean magnitude of deformation across the spatial grid for every frame
    # We take the norm (magnitude) of the X/Y components, then average across Height/Width
    mag = np.mean(np.linalg.norm(def_data, axis=3), axis=(1,2))
    
    plt.figure(figsize=(10, 4))
    plt.plot(mag, label="Mean Deformation Magnitude", color='red')
    plt.title(f"Sponge Manual Press - Trial {TRIAL_NUM} Loading Curve")
    plt.xlabel("Frame (Time @ 120Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    
    # --- Check 2: The Spatial Contact Patch ---
    # Find the frame where you pressed the hardest (Peak Deformation)
    peak_frame_idx = np.argmax(mag)
    print(f"Peak force occurred at frame: {peak_frame_idx}")
    
    # Extract that specific frame
    peak_def_frame = def_data[peak_frame_idx]  # Shape will be (30, 40, 2)
    
    # To visualize it as an image, we calculate the magnitude per pixel
    # and normalize it to a 0-255 scale for OpenCV
    patch_mag = np.linalg.norm(peak_def_frame, axis=2)
    patch_normalized = cv2.normalize(patch_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Make it bigger so you can actually see it on screen (30x40 is tiny)
    patch_large = cv2.resize(patch_normalized, (320, 240), interpolation=cv2.INTER_NEAREST)
    
    # Apply a heatmap color map (Red is high pressure, Blue is low)
    heatmap = cv2.applyColorMap(patch_large, cv2.COLORMAP_JET)
    
    # Show the results
    print("Close the Matplotlib graph window to show the Heatmap image.")
    plt.show()  # This pauses execution until you close the graph
    
    cv2.imshow("Peak Deformation Heatmap", heatmap)
    print("Press any key on the heatmap window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError:
    print(f"Error: Could not find the files at {DATA_DIR}. Check your paths!")
except Exception as e:
    print(f"An error occurred: {e}")
