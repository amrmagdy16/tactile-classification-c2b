import time
import numpy as np
from dmrobotics import Sensor

dev_serial_id = "S2508080042"   # confirm YOUR sensor's ID
sensor = Sensor(dev_serial_id)

sensor.reset()                  # establish clean baseline
time.sleep(0.5)

for i in range(10):
    deformation = sensor.getDeformation2D()
    shear = sensor.getShear()
    print(f"frame {i}")
    print("  deformation -> type:", type(deformation),
          "shape:", np.asarray(deformation).shape,
          "dtype:", np.asarray(deformation).dtype)
    print("  shear       -> type:", type(shear),
          "shape:", np.asarray(shear).shape,
          "dtype:", np.asarray(shear).dtype)
    print("  deformation range:", np.asarray(deformation).min(), "to", np.asarray(deformation).max())
    print("  shear range:", np.asarray(shear).min(), "to", np.asarray(shear).max())
    time.sleep(0.1)

sensor.disconnect()
