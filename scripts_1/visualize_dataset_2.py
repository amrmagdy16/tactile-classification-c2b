#!/usr/bin/env python3
"""
Comprehensive EDA Suite for Tactile Time-Series Data.
Generates:
1. Time-series variance (Mean ± Std)
2. Spatial dynamics (Onset, Peak, Release)
3. Statistical Variability (Violin + Strip plots)
4. Feature Space embeddings (PCA & t-SNE)
5. Console summaries & CSV reports
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42) # Reproducibility

DATA_DIR = Path("data/raw")
OUT_DIR = Path("notes/eda_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. DATA INTEGRITY & LOADING
# ------------------------------------------------------------------
def load_and_check_data():
    materials = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle'])
    
    ts_data = {m: {'def': [], 'shear': [], 'def_norm': [], 'shear_norm': []} for m in materials}
    spatial_reps = {} # Store one clean trial per class for spatial dynamics
    feature_rows = []
    
    print("="*50)
    print("1. RUNNING DATA INTEGRITY CHECKS")
    print("="*50)
    
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists(): continue
            
        def_files = sorted(tdir.glob("trial_*_def.npy"))
        sh_files = sorted(tdir.glob("trial_*_shear.npy"))
        
        valid_trials = 0
        for i, (df, sf) in enumerate(zip(def_files, sh_files)):
            d_data, s_data = np.load(df), np.load(sf)
            
            # --- INTEGRITY CHECKS ---
            if np.isnan(d_data).any() or np.isinf(d_data).any():
                print(f"  [WARNING] NaN/Inf found in {m} trial {i}. Skipping.")
                continue
            if np.std(d_data) < 1e-6:
                print(f"  [WARNING] Constant array (dead sensor) in {m} trial {i}. Skipping.")
                continue
                
            # Collapse spatial to 1D TS
            d_mag = np.linalg.norm(d_data, axis=3)
            s_mag = np.linalg.norm(s_data, axis=3)
            
            d_ts = np.mean(d_mag, axis=(1, 2))
            s_ts = np.mean(s_mag, axis=(1, 2))
            
            # Normalization
            peak_idx = int(np.argmax(d_ts))
            max_d = d_ts[peak_idx] + 1e-8
            max_s = np.max(s_ts) + 1e-8
            
            ts_data[m]['def'].append(d_ts)
            ts_data[m]['shear'].append(s_ts)
            ts_data[m]['def_norm'].append(d_ts / max_d)
            ts_data[m]['shear_norm'].append(s_ts / max_s) # Shape only
            
            # Save 1st valid trial for Spatial Dynamics
            if valid_trials == 0:
                spatial_reps[m] = d_mag
                
            # Feature extraction for PCA/t-SNE
            feature_rows.append({
                'Material': m,
                'Peak_Def': max_d,
                'Peak_Shear': max_s,
                'Def_Mean': np.mean(d_ts),
                'Shear_Mean': np.mean(s_ts),
                'Def_Std': np.std(d_ts),
                'Ratio_Peak': max_d / max_s
            })
            valid_trials += 1
            
        print(f"  {m}: Passed {valid_trials}/{len(def_files)} trials.")
        
    return ts_data, spatial_reps, pd.DataFrame(feature_rows), materials

# ------------------------------------------------------------------
# 2. STATISTICAL REPORTING (CSV & Console)
# ------------------------------------------------------------------
def generate_statistics(df):
    print("\n" + "="*50)
    print("2. STATISTICAL SUMMARY (Kruskal-Wallis)")
    print("="*50)
    
    # Kruskal-Wallis Test (Non-parametric ANOVA)
    mats = df['Material'].unique()
    groups = [df[df['Material'] == m]['Peak_Def'].values for m in mats]
    stat, p_val = kruskal(*groups)
    
    print(f"H-statistic: {stat:.2f}, p-value: {p_val:.4e}")
    if p_val < 0.05:
        print("Conclusion: STRONG statistical difference between materials.")
    else:
        print("Conclusion: NO statistical difference (check your hardware!).")
        
    # Generate Numerical Summary
    summary = df.groupby('Material').agg(['mean', 'std']).round(3)
    summary.to_csv(OUT_DIR / "eda_01_numerical_stats.csv")
    print(f"\nSaved CSV Report to: notes/eda_results/eda_01_numerical_stats.csv")

# ------------------------------------------------------------------
# 3. PLOTTING FUNCTIONS
# ------------------------------------------------------------------
def plot_timeseries(ts_data, materials):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.canvas.manager.set_window_title('Time-Series Analysis')
    
    metrics = [('def', 'Raw Deformation'), ('def_norm', 'Normalized Deformation (Shape)'),
               ('shear', 'Raw Shear'), ('shear_norm', 'Normalized Shear (Shape)')]
               
    for i, (key, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        for m in materials:
            data = np.array(ts_data[m][key])
            if len(data) == 0: continue
            
            mean_ts = np.mean(data, axis=0)
            std_ts = np.std(data, axis=0)
            frames = range(len(mean_ts))
            
            p = ax.plot(frames, mean_ts, label=m, linewidth=2)
            ax.fill_between(frames, mean_ts - std_ts, mean_ts + std_ts, alpha=0.2, color=p[0].get_color())
            
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Magnitude')
        ax.grid(alpha=0.3)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_02_timeseries_variance.png", dpi=300)
    plt.close()

def plot_variability(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, feature in enumerate(['Peak_Def', 'Peak_Shear']):
        sns.violinplot(x='Material', y=feature, data=df, inner=None, color='lightgray', ax=axes[i])
        sns.stripplot(x='Material', y=feature, data=df, size=4, jitter=True, alpha=0.7, ax=axes[i], palette='Set1')
        axes[i].set_title(f'Variability: {feature}', fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(alpha=0.3, axis='y')
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_03_variability_stats.png", dpi=300)
    plt.close()

def plot_feature_space(df):
    X = df.drop('Material', axis=1).values
    y = df['Material'].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set1', s=80, alpha=0.8, ax=axes[0])
    axes[0].set_title(f'PCA (Explains {sum(var_exp)*100:.1f}% Variance)', fontweight='bold')
    
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='Set1', s=80, alpha=0.8, ax=axes[1])
    axes[1].set_title('t-SNE (Manifold Learning)', fontweight='bold')
    
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_04_feature_space.png", dpi=300)
    plt.close()

def plot_spatial_dynamics(spatial_reps, materials):
    n_mats = len(materials)
    fig, axes = plt.subplots(n_mats, 3, figsize=(9, 2.5 * n_mats))
    
    for i, m in enumerate(materials):
        d_mag = spatial_reps[m]
        ts = np.mean(d_mag, axis=(1, 2))
        
        peak_f = np.argmax(ts)
        # Find Onset (first frame > 10% of peak)
        onset_f = np.argmax(ts > (0.1 * ts[peak_f])) 
        # Release (Frame 145, or right before end)
        release_f = 145 
        
        frames = [('Onset', onset_f), ('Peak', peak_f), ('Release', release_f)]
        vmax = np.max(d_mag[peak_f]) # Normalize colors to peak
        
        for j, (title, f_idx) in enumerate(frames):
            ax = axes[i, j] if n_mats > 1 else axes[j]
            im = ax.imshow(d_mag[f_idx], cmap='magma', vmin=0, vmax=vmax)
            ax.axis('off')
            if i == 0: ax.set_title(f"{title} (Frame {f_idx})", fontweight='bold')
            if j == 0:
                ax.text(-10, d_mag.shape[1]//2, m, va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)
                
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eda_05_spatial_dynamics.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    ts_data, spatial_reps, df_features, materials = load_and_check_data()
    
    if df_features.empty:
        print("ERROR: No valid data to plot!")
        return
        
    generate_statistics(df_features)
    
    print("\nRendering Visualizations...")
    plot_timeseries(ts_data, materials)
    print("  -> Saved Time-Series (eda_02)")
    plot_variability(df_features)
    print("  -> Saved Variability (eda_03)")
    plot_feature_space(df_features)
    print("  -> Saved Feature Space (eda_04)")
    plot_spatial_dynamics(spatial_reps, materials)
    print("  -> Saved Spatial Dynamics (eda_05)")
    
    print("\n" + "="*50)
    print("EDA COMPLETE. Check notes/eda_results/ folder.")
    print("="*50)

if __name__ == "__main__":
    main()
