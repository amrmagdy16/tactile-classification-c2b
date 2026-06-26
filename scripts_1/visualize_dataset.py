#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for Tactile Time-Series.
Validates signal presence, time-series variance, and 2D spatial contact patches.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import warnings

warnings.filterwarnings('ignore')
DATA_DIR = Path("data/raw")

def load_sample_data():
    """Loads one random trial per material, plus aggregate peak stats."""
    materials = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle'])
    
    sample_trials = {}
    aggregate_stats = {'material': [], 'peak_def': [], 'peak_shear': []}
    
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists():
            continue
            
        def_files = sorted(tdir.glob("trial_*_def.npy"))
        sh_files = sorted(tdir.glob("trial_*_shear.npy"))
        
        if not def_files:
            continue
            
        # 1. Grab one random trial for the timeline/spatial plots
        idx = random.randint(0, len(def_files) - 1)
        d_data = np.load(def_files[idx])
        s_data = np.load(sh_files[idx])
        
        # Calculate magnitudes (T, H, W)
        d_mag = np.linalg.norm(d_data, axis=3)
        s_mag = np.linalg.norm(s_data, axis=3)
        
        sample_trials[m] = {'def': d_mag, 'shear': s_mag}
        
        # 2. Extract peak values for ALL trials to check distribution
        for df, sf in zip(def_files, sh_files):
            d_all = np.linalg.norm(np.load(df), axis=3)
            s_all = np.linalg.norm(np.load(sf), axis=3)
            
            d_ts = np.mean(d_all, axis=(1, 2))
            s_ts = np.mean(s_all, axis=(1, 2))
            
            aggregate_stats['material'].append(m)
            aggregate_stats['peak_def'].append(np.max(d_ts))
            aggregate_stats['peak_shear'].append(np.max(s_ts))
            
    return sample_trials, aggregate_stats, materials

def main():
    print("Loading data for visualization...")
    sample_trials, aggregate_stats, materials = load_sample_data()
    
    if not sample_trials:
        print("No data found to visualize.")
        return

    # Set up the massive Matplotlib figure
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.manager.set_window_title('Tactile Dataset Validation Dashboard')
    
    # ==========================================
    # PLOT 1: 1D Time-Series Comparison
    # ==========================================
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    
    for m in materials:
        if m not in sample_trials: continue
        
        # Collapse to 1D
        d_ts = np.mean(sample_trials[m]['def'], axis=(1, 2))
        s_ts = np.mean(sample_trials[m]['shear'], axis=(1, 2))
        
        # Normalize to see shape clearly
        d_ts_norm = d_ts / (np.max(d_ts) + 1e-8)
        
        ax1.plot(d_ts_norm, label=m, linewidth=2, alpha=0.8)
        ax2.plot(s_ts, label=m, linewidth=2, alpha=0.8) # Keep shear raw to see magnitude differences

    ax1.set_title("Normalized Deformation over Time (The Loading Curve)")
    ax1.set_xlabel("Frames (Time)")
    ax1.set_ylabel("Normalized Force")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Raw Shear over Time (Poisson Effect / Slippage)")
    ax2.set_xlabel("Frames (Time)")
    ax2.set_ylabel("Absolute Shear Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # PLOT 2: Statistical Distribution (Boxplots)
    # ==========================================
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(x='material', y='peak_def', data=aggregate_stats, ax=ax3, palette='Set2')
    ax3.set_title("Variance Check: Peak Deformation Spread")
    ax3.set_ylabel("Raw Peak Deformation")
    ax3.tick_params(axis='x', rotation=45)

    # ==========================================
    # PLOT 3: 2D Spatial Contact Patch at Peak
    # ==========================================
    # We will plot the 2D patches for up to 5 materials
    num_mats = min(len(materials), 5)
    
    # Create a sub-gridspec for the heatmaps in the 4th quadrant
    sub_grid = ax1.get_gridspec()[1, 1].subgridspec(1, num_mats, wspace=0.1)
    
    for i, m in enumerate(materials[:num_mats]):
        if m not in sample_trials: continue
        
        d_mag = sample_trials[m]['def']
        d_ts = np.mean(d_mag, axis=(1, 2))
        peak_frame = int(np.argmax(d_ts))
        
        peak_2d_image = d_mag[peak_frame]
        
        ax_heat = fig.add_subplot(sub_grid[0, i])
        im = ax_heat.imshow(peak_2d_image, cmap='magma', vmin=0, vmax=np.max(peak_2d_image))
        ax_heat.set_title(m, fontsize=10)
        ax_heat.axis('off')
    
    fig.text(0.75, 0.48, '2D Spatial Imprint at Maximum Squeeze', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = "notes/dataset_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Dashboard saved as an image to:{output_path}")

if __name__ == "__main__":
    main()
