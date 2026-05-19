import numpy as np
import matplotlib.pyplot as plt

# Load data
deform = np.load('test_deformation.npy')   # (frames, 240, 320, 2)
shear = np.load('test_shear.npy')          # (frames, 240, 320, 2)

# Compute magnitude per pixel, then average over all pixels per frame
d_mag = np.linalg.norm(deform, axis=-1)   # (frames, 240, 320)
s_mag = np.linalg.norm(shear, axis=-1)    # (frames, 240, 320)

d_avg = d_mag.mean(axis=(1, 2))           # (frames,)
s_avg = s_mag.mean(axis=(1, 2))           # (frames,)

# Also get max per frame (useful for contact detection)
d_max = d_mag.max(axis=(1, 2))            # (frames,)

frames = np.arange(len(d_avg))
time_seconds = frames / 20.0              # approximate

# Create the plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(time_seconds, d_avg, 'b-', linewidth=0.8, label='Mean Deformation Mag')
axes[0].plot(time_seconds, d_max, 'b--', linewidth=0.5, alpha=0.6, label='Max Deformation Mag')
axes[0].set_ylabel('Deformation')
axes[0].set_title('Deformation Magnitude over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_seconds, s_avg, 'r-', linewidth=0.8, label='Mean Shear Mag')
axes[1].set_ylabel('Shear Magnitude')
axes[1].set_title('Shear Magnitude over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Combined view (both on same plot)
axes[2].plot(time_seconds, d_avg, 'b-', linewidth=0.8, label='Deformation')
axes[2].plot(time_seconds, s_avg, 'r-', linewidth=0.8, label='Shear')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Magnitude')
axes[2].set_title('Deformation vs Shear (Combined)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sensor_signals.png', dpi=150)
print("Plot saved as sensor_signals.png")
