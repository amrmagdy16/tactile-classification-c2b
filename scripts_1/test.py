import numpy as np
from pathlib import Path

sample = np.load("data/raw/ceramic_cup/baseline/trial_0002_def.npy")
print("Shape:", sample.shape)   # Should be (T, H, W, C) – e.g., (150, 16, 16, 1) or similar
print("Max value:", sample.max())
print("Any non-zero?", np.any(sample))
