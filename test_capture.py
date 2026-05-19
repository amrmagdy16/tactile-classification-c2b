import time
import numpy as np
from dmrobotics import Sensor

# Use the serial ID that works with your sensor
DEV_SERIAL = "S2508080077"

print(f"Connecting to sensor {DEV_SERIAL}...")
sensor = Sensor(DEV_SERIAL)
print("Sensor connected. Recording deformation and shear data.")
print("Press Ctrl+C to stop.\n")

deformation_list = []
shear_list = []

try:
    while True:
        # Read the two modalities you need for C2b
        deform = sensor.getDeformation2D()
        shear = sensor.getShear()
        
        deformation_list.append(deform)
        shear_list.append(shear)
        
        print(f"Frames: {len(deformation_list)} | "
              f"Deformation shape: {deform.shape} | "
              f"Shear shape: {shear.shape}   ", end='\r')
        
        time.sleep(0.05)   # ~20 Hz, adapt if needed

except KeyboardInterrupt:
    print("\n\nStopping capture...")
    
    # Convert to numpy arrays
    deform_data = np.array(deformation_list)
    shear_data = np.array(shear_list)
    
    print(f"Deformation data shape : {deform_data.shape}")
    print(f"Shear data shape      : {shear_data.shape}")
    print(f"Data type             : {deform_data.dtype}")
    print(f"Deformation range     : min={deform_data.min():.4f}, max={deform_data.max():.4f}")
    print(f"Shear range           : min={shear_data.min():.4f}, max={shear_data.max():.4f}")
    
    # Save to disk
    np.save("test_deformation.npy", deform_data)
    np.save("test_shear.npy", shear_data)
    print("\nData saved to test_deformation.npy and test_shear.npy")
    
    sensor.disconnect()
